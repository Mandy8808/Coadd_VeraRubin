# vera rubin v1.0
# butler.custom_butler.py

import subprocess
import pathlib
import logging
import os, sys

from lsst.daf.butler import Butler, DatasetType, DatasetRef

# Top-level
# ---------------------------------------------------------------------
def main_local_repo(
    LOCAL_REPO: str, 
    REMOTE_REPO: str,
    visits_datasetRef: list[DatasetRef],
    skymap_register: bool = True,
    remote_collection: str = "LSSTComCam/DP1",
    LOGDIR: str = "/projects/BR/logs",
    make_repo: bool = True
) -> bool:
    """
    Main pipeline to create local repo, register instruments/datatypes/skymap,
    and transfer requested visits.

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

    # register dataset types
    #dataset_types = set(map(lambda x: x.datasetType, visits_datasetRef))
    unique_dt_by_name = {}
    for ref in visits_datasetRef:
        dt = ref.datasetType
        unique_dt_by_name[dt.name] = dt
    dataset_types = list(unique_dt_by_name.values())
    logger.info(f"Registering DatasetTypes: {[dt.name for dt in dataset_types]}")
    try:
        register_datasetTypes(LOCAL_REPO, dataset_types, logger=logger, check=True)
        logger.info("DatasetType registration complete.")
    except Exception:
        logger.exception("Could not register DatasetTypes")
        raise

    # Optional: register skymap before transferring data
    if skymap_register:
        logger.info("Registering skymap from remote")
        try:
            skymap_register_from_remote(
                local_repo=LOCAL_REPO,
                remote_repo=REMOTE_REPO,
                collections=remote_collection,
                logger=logger
            )
            logger.info("Skymap registration complete.")
        except Exception:
            logger.exception("Could not register skymap")
            raise

    # Transfer visits (iterate datasetRefs)
    logger.info("Starting visits transfer loop")
    for ref in visits_datasetRef:
        try:
            dataId = ref.dataId
            visits_val = dataId.get("visit")
            # ensure visits is a list for transfer_visits
            visits_arg = [visits_val] if isinstance(visits_val, (int, str)) else list(visits_val)

            transfer_visits(
                REMOTE_REPO,
                LOCAL_REPO,
                visits=visits_arg,
                band=dataId["band"],
                instrument=dataId["instrument"],
                physical_filter=dataId.get("physical_filter"),
                detector=dataId.get("detector"),
                day_obs=dataId.get("day_obs"),
                skymap=dataId.get("skymap"),
                collections=remote_collection,
                dataset=ref.datasetType.name,
                logger=logger
            )
            logger.info(f"Transfer succeeded for dataId: {dataId}")
        except Exception:
            logger.exception(f"Transfer failed for DatasetRef: {ref}")
            raise

    logger.info("Pipeline finished successfully.")
    return True

# Low-level helpers
# ---------------------------------------------------------------------
def setup_logger(logfile_path: str, name: str = 'pipeline.log') -> logging.Logger:
    """Create a logger that writes DEBUG to file and INFO to console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Log to file
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def _run(cmd: list[str], logger: logging.Logger = None) -> subprocess.CompletedProcess:
    """
    Run a subprocess command, log stdout/stderr and raise if it fails.
    Returns the CompletedProcess on success.
    """
    msg = msg = "[CMD] " + " ".join(cmd)
    if logger:
        logger.info(msg)
    else:
        print(msg)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if logger:
            logger.info(result.stdout.strip())
            if result.stderr.strip():
                logger.warning(result.stderr.strip())

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Command failed: {e}")
            logger.error(e.stderr)
        else:
            print("Command failed:", e)
            print(e.stderr)
        raise

def get_butler_location(butler: Butler) -> str | bool:
    """Return repository path for a local Butler or parsed location for remote Butler."""
    try:
        return butler.repo   # Local Butler
    except AttributeError:
        import re
        # Remote Butler: parse str(butler)
        s = str(butler)
        match = re.search(r"RemoteButler\((.+?)\)", s)
        if match:
            return match.group(1)
        else:
            return False  # Could not determine location

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
        

# High-level operations
# ---------------------------------------------------------------------

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
    if missing:
        if logger:
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
    local_repo: str,
    remote_repo: str,
    collections: str = "LSSTComCam/DP1",
    HSCskymap: bool = False,
    OBS_DECAM_DIR: bool = False,
    logger: logging.Logger = None
) -> bool:
    """
    Register skymap(s) from remote into local repo.

    Parameters
    ----------
    local_repo : str
        Path to the local Butler repository where the SkyMap will be registered.
    remote_repo : str
        Path to the remote repository from which the SkyMap definition will be taken.
    collections : str, optional
        Collection to use when querying the remote repository for SkyMap dimension
        records (default: "LSSTComCam/DP1").
    HSCskymap : bool, optional
        If True, also register the HSC rings skymap (default: False).
    OBS_DECAM_DIR : bool, optional
        If True, use the `makeSkyMap.py` from `OBS_DECAM_DIR` environment variable (default: False).

    Returns
    -------
    bool
        True if registration succeeded.
    """

    # Load remote repo
    remote = Butler(remote_repo, collections=collections)

    # Query skymaps
    skymaps = list(remote.registry.queryDimensionRecords("skymap"))
    if not skymaps:
        raise RuntimeError("The remote repository has no SkyMap registered.")

    if len(skymaps) > 1 and logger:
        logger.warning(f"Multiple SkyMaps found in remote; using first: {skymaps[0].name}")

    skymap_name = skymaps[0].name
    if logger:
        logger.info(f"Detected remote SkyMap: {skymap_name}")

    # DECam skymap logic
    if OBS_DECAM_DIR:
        if "OBS_DECAM_DIR" not in os.environ:
            raise RuntimeError("OBS_DECAM_DIR is required but not set.")
        run_config = os.path.join(os.environ["OBS_DECAM_DIR"], "config/makeSkyMap.py")
        _run(["butler", "register-skymap", local_repo, "-C", run_config, "-c", f"name='{skymap_name}'"], logger=logger)
    else:
        _run(["butler", "register-skymap", local_repo, "-c", f"name='{skymap_name}'"], logger=logger)

    # Optional: register HSC skymap
    if HSCskymap:
        if "OBS_SUBARU_DIR" not in os.environ:
            raise RuntimeError("OBS_SUBARU_DIR is required for HSC skymap but not set.")
        hsc_config = os.path.join(os.environ["OBS_SUBARU_DIR"], "config/makeSkyMap.py")
        _run(["butler", "register-skymap", local_repo, "-C", hsc_config, "-c", "name='hsc_rings_v1'"], logger=logger)

    if logger:
        logger.info(f"SkyMap '{skymap_name}' registered successfully.")
    if logger and OBS_DECAM_DIR:
        logger.info("[OK] DECam skymap logic registered.")
    if logger and HSCskymap:
        logger.info("[OK] Additional HSC ring skymap registered.")
    
    return True

def transfer_visits(
    remote_repo: str,
    local_repo: str,
    visits: list[int] | int,
    band: str,
    instrument: str = 'LSSTComCam',
    detector: int | list[int] = None,
    day_obs: int | list[int] = None,
    physical_filter: str = None,
    skymap: str = None,
    collections: str | list[str] = None,
    dataset: str = None,
    logger: logging.Logger = None,
) -> bool:
    """
    Transfer one-or-more visits from remote_repo to local_repo via `butler transfer-datasets`.
    `visits` may be a single int or an iterable of ints.

    Returns:
        bool: True if the transfer completed successfully.
    """
    if isinstance(visits, int):
        visits_list = [visits]
    else:
        visits_list = list(visits)
    
    if not visits_list:
        raise ValueError("No visits provided to transfer_visits().")
    elif logger:
        logger.info("[INFO] Transferring visits...")

    # Convert collections list to comma-separated string if necessary
    if isinstance(collections, (list, tuple, set)):
        collections = ",".join(collections)

    # Create visit IDs string
    visit_ids_str = "(" + ",".join(map(str, visits)) + ")"

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
    cmd = [
        "butler", "transfer-datasets",
        remote_repo, local_repo,
        "--where", query_string,]
    if collections:
        cmd.extend(["--collections", collections])
    if dataset:
        cmd.extend(["--dataset-type", dataset])

    # Run the command
    _run(cmd, logger=logger)
    if logger:
        logger.info(f"Completed transfer for visits {visits_list}")
    return True


