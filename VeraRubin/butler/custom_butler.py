# vera rubin v1.0
# butler.custom_butler.py

# Loading modules
import subprocess
import pathlib
import logging
import os, sys

from lsst.daf.butler import Butler, DatasetType, DatasetRef, CollectionType

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.tools import setup_logger, _run

# Top-level
# ---------------------------------------------------------------------
def main_local_repo(
    LOCAL_REPO: str, 
    REMOTE_REPO: str,
    visits_datasetRef: list[DatasetRef],
    remote_collection: str = "LSSTComCam/DP1",
    LOGDIR: str = "/projects/BR/logs",
    make_repo: bool = True,
    chain_name: str = "local_main_chain"
) -> bool:
    """
    Create a local Butler repo and import visits and supporting datasets from a remote repo.
    Important: DO NOT pre-register SkyMap; we copy it from remote after importing visits to preserve UUIDs.

    visits_datasetRef: list of DatasetRef objects (from the remote butler) describing each visit to transfer.
    """
    # Setup logging directory and logger
    log_path = pathlib.Path(LOGDIR)
    log_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "ug+rw", LOGDIR], check=True)

    logfile = log_path / "pipeline.log"
    logger = setup_logger(str(logfile))
    logger.info(f"Created LOGDIR at {LOGDIR}")
    logger.info(f"Starting pipeline for local repo: {LOCAL_REPO}")

    # create an empty repo if requested
    if make_repo:
        logger.info(f"Creating empty repo at {LOCAL_REPO}")
        try:
            create_empty_repo(path=LOCAL_REPO, logger=logger)
            logger.info("Empty repo created successfully.")
        except Exception as e:
            logger.exception("Could not create empty repo")
            raise

    # register instruments (collect unique instrument names from DatasetRefs)
    #instruments = set(map(lambda x: x.dataId["instrument"], visits_datasetRef))
    instruments: set[str] = {ref.dataId["instrument"] for ref in visits_datasetRef}
    logger.info(f"Registering instruments: {sorted(instruments)}")
    try:
        instrument_register_from_remote(
            local_repo=LOCAL_REPO,
            remote_repo=REMOTE_REPO,
            instruments=instruments,
            remote_collection=remote_collection,
            logger=logger
        )
        logger.info("Instrument registration complete.")
    except Exception:
        logger.exception("Could not register instruments")
        raise

    # Transfer visits (iterate datasetRefs)
    collections_str = remote_collection  # pass to CLI
    logger.info("Starting visits transfer loop")
    remote_butler = Butler(REMOTE_REPO, collections=remote_collection)
    for ref in visits_datasetRef:
        try:
            dataId = ref.dataId
            visit = dataId["visit"]
            band = dataId["band"]
            instrument=dataId["instrument"]

            logger.info(f"\n[VISIT] Processing visit {visit}")

            # discover datasets
            out_dic = discover_datasets_for_visit(REMOTE_REPO, datasetRef=ref, optional=True, logger=logger)
            required_present = out_dic['required_present']
            optional_present = out_dic['optional_present']
            missing_required = out_dic['missing_required']

            if missing_required:
                logger.warning(f"[WARN] Required datasets missing for visit {visit}: {missing_required}")

            # Register datasetTypes found in remote before copying (so registry knows types)
            all_dt = list(required_present | optional_present)
            dtypes = [
                dt for dt in remote_butler.registry.queryDatasetTypes()
                if dt.name in all_dt]
            logger.info(f"Registering DatasetTypes: {[dt.name for dt in dtypes]}")
            
            if dtypes:
                try:
                    register_datasetTypes(LOCAL_REPO, dtypes, logger=logger)
                    logger.info("DatasetType registration complete.")
                except Exception:
                    logger.exception("Could not register DatasetTypes")
                    raise
            
            # Transfer all dataset types for this visit
            transfer_all_visit_datasets(
                remote_repo=REMOTE_REPO,
                local_repo=LOCAL_REPO,
                visit_id=visit,
                required_present=required_present,
                optional_present=optional_present,
                instrument=instrument,
                band=band,
                collections=collections_str,
                logger=logger,
                )
            logger.info(f"Transfer succeeded for dataId: {dataId}")
        except Exception:
            logger.exception(f"Transfer failed for DatasetRef: {ref}")
            raise
    
    # After all visits copied, copy skyMap datasets (preserve UUIDs, avoid conflicts)
    # Register skyMap DatasetType BEFORE transfer
    remote_dt = [dt for dt in remote_butler.registry.queryDatasetTypes()
             if dt.name == "skyMap"]
    if remote_dt:
        logger.info("Registering DatasetType 'skyMap' in local repo before transfer.")
        register_datasetTypes(LOCAL_REPO, remote_dt, logger=logger)
    else:
        logger.warning("Remote repo does NOT contain datasetType 'skyMap'.")
        raise

    # transfer skyMap(s)
    try:
        logger.info("Copying skyMap(s) from remote to local (preserving UUIDs)...")
        skymap_register_from_remote(REMOTE_REPO, LOCAL_REPO, remote_collections=remote_collection, logger=logger)
        logger.info("SkyMap copy finished.")
    except Exception:
        logger.exception("SkyMap copy failed.")
        raise

    # Create a chained collection that includes skymaps and the imported runs (so butler.get finds skyMap)
    # Determine available collections in local registry
    local_butler = Butler(LOCAL_REPO, writeable=True)
    local_collections = list(local_butler.registry.queryCollections())
    # typical collections: ['skymaps', 'ingest', 'my_imported_run', ...]
    # We'll create chain containing 'skymaps' plus everything that startswith 'LSSTComCam/runs' or similar
    # Conservative approach: include all local collections (but you can change selection)
    try:
        logger.info(f"Creating chained collection '{chain_name}' that includes existing local collections.")
        ensure_chained_collection(LOCAL_REPO, chain_name, members=local_collections, logger=logger)
    except Exception:
        logger.exception("Could not create or set chained collection.")
        raise

    logger.info("Pipeline finished successfully.")
    return True


# Low-level helpers
# ---------------------------------------------------------------------
def create_empty_repo(path: str, logger: logging.Logger = None) -> bool:
    """Create a new local Butler repo (using Butler.makeRepo)."""
    if os.path.exists(os.path.join(path, "butler.yaml")):
        raise FileExistsError(f"Repository already exists: {path}")
    os.makedirs(path, exist_ok=True)

    if logger:
        logger.info(f"Creating Butler repo at {path}")
    else:
        print(f"Creating Butler repo at {path}")

    Butler.makeRepo(path)
    return True

def transfer_all_visit_datasets(
    remote_repo: str,
    local_repo: str,
    visit_id: int,
    required_present: set[str],
    optional_present: set[str],
    instrument: str,
    band: str,
    collections: str | list[str],
    logger: logging.Logger = None
) -> bool:
    """
    Copy all required + optional datasets for a visit.
    Calls your existing `transfer_visits` function repeatedly.
    """

    all_types = list(sorted(required_present)) + list(sorted(optional_present))
    for dt in all_types:
        if logger:
            logger.info(f"[TRANSFER] Copying datasetType '{dt}' for visit={visit_id}")
        transfer_visits(remote_repo=remote_repo, local_repo=local_repo, visits=[visit_id],
                            band=band, instrument=instrument, collections=collections, dataset=dt, logger=logger)
    return True


# High-level operations
# ---------------------------------------------------------------------
def discover_datasets_for_visit(
        REMOTE_REPO: str,
        datasetRef: DatasetRef,
        optional: bool = True,
        logger: logging.Logger = None) -> dict:
    """
    Identify all datasetTypes available for a visit in the remote repo.
    Returns a dictionary: {'required_present': set[str], 'optional_present': set[str], 'missing_required': set[str], 'missing_optional': set[str],}
    """

    required_set = {
        "visit_summary", "visit_summary_schema", "visit_summary_metadata",
        "calexp", "calexp_metadata", "calexp_background", "calexp_psf", "calexp_wcs", "calexp_noise",
        "raw", "postISRCCD", "visit_image"
    }

    optional_set = {
        "postISRCCD", "postISRCCD_metadata", "calexp_background_model",
        "src", "src_schema", "src_schema_validity", "srcMatches", "srcMatchFull"
    }

    try:
        if logger:
            logger.info(f"Getting the remote Butler registry on: {REMOTE_REPO}.")
        remote = Butler(REMOTE_REPO).registry
        dt_available = {dt.name: dt for dt in remote.queryDatasetTypes()}
        collections = remote.queryCollections()  # Cache collections
    except Exception as e:
        if logger:
            logger.exception(f"Could not have access to Butler registry on: {REMOTE_REPO}")
            logger.error(f"Command failed: {e}")
        raise

    out_dic = {
        'required_present': set(),
        'optional_present': set(),
        'missing_required': set(),
        'missing_optional': set(),
    }
    visit_dataId = datasetRef.dataId

    # Required datasets
    for dt_name in required_set:
        if dt_name not in dt_available:
            out_dic['missing_required'].add(dt_name)
            continue
        
        refs = list(remote.queryDatasets(
            dt_available[dt_name],
            dataId=visit_dataId,
            collections=collections
        ))

        if refs:
            out_dic['required_present'].add(dt_name)
        else:
            out_dic['missing_required'].add(dt_name)

    # Optional datasets
    if optional:
        for dt_name in optional_set:
            if dt_name not in dt_available:
                out_dic['missing_optional'].add(dt_name)
                continue

            refs = list(remote.queryDatasets(
                dt_available[dt_name],
                dataId=visit_dataId,
                collections=collections
            ))

            if refs:
                out_dic['optional_present'].add(dt_name)
            else:
                out_dic['missing_optional'].add(dt_name)

    if logger:
        logger.info(f"[DISCOVER] Visit {visit_dataId}")
        logger.info(f"  Required present : {sorted(out_dic['required_present'])}")
        logger.info(f"  Missing required : {sorted(out_dic['missing_required'])}")
        logger.info(f"  Optional present : {sorted(out_dic['optional_present'])}")
        logger.info(f"  Missing optional : {sorted(out_dic['missing_optional'])}")

    return out_dic

def instrument_register_from_remote(
    local_repo: str,
    remote_repo: str,
    instruments: set[str],
    remote_collection: str = "LSSTComCam/DP1",
    logger: logging.Logger = None,
) -> bool:
    """
    Register instruments in the local Butler repo using definitions
    retrieved from a remote repo.

    Parameters
    ----------
    instruments: set of instrument names (strings)
    remote_collection: collection to open the remote Butler with
    """
    if not instruments:
        raise RuntimeError("No instrument names provided.")
    if logger:
        logger.info(f"Detected instruments to register: {sorted(instruments)}")
    
    if logger:
        logger.info(f"Opening remote Butler: {remote_repo} (collections={remote_collection})")
    remote = Butler(remote_repo, collections=remote_collection)


    # Get instrument records from remote registry
    remote_inst_records = [
        rec for rec in remote.registry.queryDimensionRecords("instrument")
        if rec.name in instruments
    ]

    found_names = {rec.name for rec in remote_inst_records}
    missing = instruments - found_names
    if missing and logger:
        logger.warning(f"Some instruments not found in remote registry: {sorted(missing)}")
        # still continue with those found; you can choose to raise instead

    # Full instrument class names (e.g., lsst.obs.lsst.LsstCam)
    instrument_full = [rec.class_name for rec in remote_inst_records]
    if logger:
        logger.debug(f"Instrument classes from remote: {instrument_full}")

    # Local registry
    local_registry = Butler(local_repo).registry
    local_instrument_names = {rec.name for rec in local_registry.queryDimensionRecords("instrument")}

    for rec, class_name in zip(remote_inst_records, instrument_full):
        if rec.name in local_instrument_names:
            if logger:
                logger.info(f"Instrument '{rec.name}' already registered locally — skipping.")
            continue

        cmd = ["butler", "register-instrument", local_repo, class_name]
        _run(cmd, logger=logger)
        if logger:
            logger.info(f"Registered instrument '{rec.name}' -> {class_name}")

    return True

def register_datasetTypes(local_repo: str, 
                          datasettypes: list[DatasetType],
                          logger: logging.Logger = None,
                          check: bool = True) -> bool:
    """
    Register given DatasetType objects into the local repository.

    Parameters:
        local_repo (str): Path to the local Butler repository.
        datasettypes (list[DatasetType]): List of DatasetType objects to register.
        check (bool): If True, print the list of registered DatasetTypes after registration (default True).

    Returns:
        bool: True if registration completed successfully.
    """
    lbutler = Butler(local_repo, writeable=True)
    lreg = lbutler.registry

    # Get names of already registered DatasetTypes
    already = {d.name for d in lreg.queryDatasetTypes()}

    # Register types if not already present
    for dt in datasettypes:
        if dt.name in already:
            if logger:
                logger.debug(f"DatasetType '{dt.name}' already registered — skipping.")
            continue
        lreg.registerDatasetType(dt)
        if logger:
            logger.info(f"Registered DatasetType: {dt.name}")

    # Optional check: print registered DatasetTypes
    if check and logger:
        logger.debug("Current DatasetTypes in registry:")
        for d in lreg.queryDatasetTypes():
            logger.debug(f"  - {d.name}")

    return True

def skymap_register_from_remote(
    remote_repo: str,
    local_repo: str,
    remote_collections: str | list[str] = "LSSTComCam/DP1",
    logger: logging.Logger = None) -> bool:
    """
    1) Detect SkyMap dimension record.
    2) Find which RUN actually contains the skyMap dataset.
    3) Copy it using 'butler transfer-datasets' (preserves UUIDs).
    """
    # Normalize input
    if isinstance(remote_collections, str):
        remote_collections = [remote_collections]

    # Open remote repo
    remote = Butler(remote_repo, collections=remote_collections)

    # SkyMap DIMENSION RECORD
    skymap_dimrecs = list(remote.registry.queryDimensionRecords("skymap"))
    if not skymap_dimrecs:
        raise RuntimeError(
            f"Remote repo '{remote_repo}' contains the 'skymap' dimension "
            "but no SkyMap records are stored in the given collections."
        )

    skymap_name = skymap_dimrecs[0].name
    if logger: logger.info(f"[SKYMAP] SkyMap dimension found: {skymap_name}")

    # FIND ACTUAL SKYMAP DATASET (search all collections)
    all_remote_collections = list(remote.registry.queryCollections())
    found_refs = []
    for coll in all_remote_collections:
        try:
            refs = list(remote.registry.queryDatasets(
                datasetType="skyMap", dataId={"skymap": skymap_name}, collections=coll))
            if refs:
                found_refs.extend(refs)
                if logger: logger.info(f"[SKYMAP] Found skyMap dataset in collection: {coll}")
        except Exception:
            # ignore incompatible collections
            pass

    if not found_refs:
        raise RuntimeError(
            "SkyMap DimensionRecord exists, BUT no actual skyMap dataset exists "
            "in any collection of the remote repository."
        )

    run_name = found_refs[0].run
    if logger: logger.info(f"[SKYMAP] SkyMap dataset stored in RUN: {run_name}")
        
    # TRANSFER SKYMAP DATASET
    cmd = ["butler", "transfer-datasets", remote_repo, local_repo, "--dataset-type", "skyMap", "--collections", run_name]

    if logger: logger.info("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    if logger: logger.info("[SKYMAP] SkyMap successfully transferred.")

    return True

def ensure_chained_collection(
        local_repo: str,
        chain_name: str,
        members: list[str],
        logger: logging.Logger = None) -> bool:
    """
    Creation/update of a CHAINED collection:
      - Avoids inserting the chain into itself
      - Avoids ingest/transfer/curated/other system collections
    """
    reg = Butler(local_repo, writeable=True).registry

    # Create chain if it does not exist
    if chain_name not in reg.queryCollections():
        reg.registerCollection(chain_name, CollectionType.CHAINED)
        if logger: logger.info(f"Registered CHAINED collection: {chain_name}")

    # Filter dangerous or irrelevant collections
    safe_members = []
    for m in members:
        if m == chain_name:  # Never include the chain itself
            continue
        if m.startswith("ingest") or m.startswith("transfer") or m.startswith("_"):  # Skip system collections
            continue
        if "butler" in m.lower():  # Skip curated/registry internals
            continue
        safe_members.append(m)

    # Apply chain
    reg.setCollectionChain(chain_name, safe_members)
    if logger:
        logger.info(f"Set chain for '{chain_name}' -> {safe_members}")
    return True

def transfer_visits(remote_repo: str,
                        local_repo: str,
                        visits: list[int] | int,
                        band: str, instrument: str,
                        detector=None,
                        day_obs=None,
                        physical_filter=None, skymap=None,
                        collections: str | list[str] = None,
                        dataset: str = None,
                        logger: logging.Logger = None) -> bool:
    """
    Transfer (use CLI 'butler transfer-datasets' to preserve UUIDs)
    one-or-more visits from remote_repo to local_repo via `butler transfer-datasets`.
    `visits` may be a single int or an iterable of ints.

    Returns:
        bool: True if the transfer completed successfully.
    """
    # Build where clause
    if isinstance(visits, int):
        visits_list = [visits]
    else:
        visits_list = list(visits)

    if not visits_list:
        raise ValueError("No visits provided to transfer_visits().")
    elif logger:
        logger.info("[INFO] Transferring visits...")

    # Create visit IDs string
    visit_ids_str = "(" + ",".join(map(str, visits_list)) + ")"

    # Build the query parts
    query_parts = [f"instrument='{instrument}'", f"visit IN {visit_ids_str}", f"band='{band}'"]
    if detector is not None:
        if isinstance(detector, (list, tuple, set)):
            det_str = "(" + ",".join(map(str, detector)) + ")"
            query_parts.append(f"detector IN {det_str}")
        else:
            query_parts.append(f"detector={int(detector)}")

    if day_obs is not None:
        if isinstance(day_obs, (list, tuple, set)):
            day_str = "(" + ",".join(map(str, day_obs)) + ")"
            query_parts.append(f"day_obs IN {day_str}")
        else:
            query_parts.append(f"day_obs={int(day_obs)}")

    if physical_filter:
        query_parts.append(f"physical_filter='{physical_filter}'")

    if skymap:
        query_parts.append(f"skymap='{skymap}'")
    query_string = " AND ".join(query_parts)

    # Build the command
    cmd = ["butler", "transfer-datasets", remote_repo, local_repo, "--where", query_string]

    if collections:
        if isinstance(collections, (list, tuple, set)):
            collections_str = ",".join(collections)
        else:
            collections_str = collections
        cmd.extend(["--collections", collections_str])

    if dataset:
        cmd.extend(["--dataset-type", dataset])

    _run(cmd, logger=logger)
    if logger:
        logger.info(f"Completed transfer-datasets for dataset={dataset}, visits={visit_ids_str}")
    return True




