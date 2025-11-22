# vera rubin v1.0
# butler.custom_butler.py

import subprocess
import pathlib
import logging
import os, sys

from lsst.daf.butler import Butler, DatasetType


########################################
def main_local_repo(
    LOCAL_REPO: str, 
    REMOTE_REPO: str,
    example_dataId: dict,
    visits: list[int],
    dataset_type: str = "visit_image",
    remote_collection: str = "LSSTComCam/DP1",
    LOGDIR: str = "/projects/BR/logs",
    make_repo: bool = True,
    info: bool = True,
) -> bool:
    """
    Create a local Butler repository, register its instrument and dataset types,
    and transfer visits from a remote repository.

    Parameters
    ----------
    LOCAL_REPO : str
        Path to the local Butler repo to be created.
    REMOTE_REPO : str
        Path to the remote Butler repo containing the reference data.
    example_dataId : dict
        A valid data ID used to infer the instrument and DatasetType.
    visits : list[int]
        List of visits to transfer.
    dataset_type : str, optional
        Dataset type used for metadata inspection and DatasetType registration.
    remote_collection : str, optional
        Collection to query in the remote repository.
    info : bool, optional
        Verbosity flag.
    """

    # Setup logging directory and logger
    log_path = pathlib.Path(LOGDIR)
    log_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "ug+rw", LOGDIR], check=True)

    logfile = log_path / "pipeline.log"
    logger = setup_logger(str(logfile))
    logger.info(f"Created LOGDIR at {LOGDIR}")
    
    logger.info(f"Starting pipeline for local repo: {LOCAL_REPO}")

    if make_repo:
        # Create empty repo
        logger.info(f"Creating empty repo at {LOCAL_REPO}")
        try:
            create_empty_repo(path=LOCAL_REPO)
            logger.info("Empty repo created successfully.")
        except Exception as e:
            logger.error(f"Could not create empty repo: {e}")
            raise

    # Register instrument
    logger.info(f"Registering instrument from remote repo {REMOTE_REPO}")
    try:
        remot_butler = instrument_register_from_remote(
            local_repo=LOCAL_REPO,
            remote_repo=REMOTE_REPO,
            example_dataId=example_dataId,
            dataset_type=dataset_type,
            remote_collection=remote_collection,
            out_remot_butler=True,
            info=info
        )
        logger.info("Instrument registered.")
    except Exception as e:
        logger.error(f"Could not register instrument: {e}")
        raise

    # Register DatasetType
    logger.info(f"Registering DatasetType '{dataset_type}'")
    try:
        datasetRef = remot_butler.registry.findDataset(dataset_type, example_dataId)
        dt = datasetRef.datasetType
        register_datasetTypes(LOCAL_REPO, [dt], check=True)
        logger.info(f"DatasetType '{dataset_type}' registered.")
    except Exception as e:
        logger.error(f"Could not register DatasetType: {e}")
        raise

    # Transfer visits
    logger.info(f"Transferring visits: {visits}")
    try:
        transfer_visits(
            REMOTE_REPO,
            LOCAL_REPO,
            visits=visits,
            band=example_dataId["band"],
            instrument=example_dataId["instrument"],
            physical_filter=example_dataId.get("physical_filter"),
            collections=remote_collection,
            dataset=dataset_type
        )
        logger.info("Transfer completed.")
    except Exception as e:
        logger.error(f"Could not transfer visits: {e}")
        raise

    logger.info("Pipeline finished successfully.")
    return True

def setup_logger(logfile_path: str, name: str = 'pipeline.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

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

def _run(cmd, logger=None):
    if logger:
        logger.info("[CMD] " + " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if logger:
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Command failed: {e}")
            logger.error(e.stderr)
        raise

def get_butler_location(butler):
    """
    Returns the repository path for a local Butler or the URL for a RemoteButler.
    """
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
            return None  # Could not determine location
        

def create_empty_repo(path: str):
    """Create a new local Butler repo (using Butler.makeRepo)."""
    if os.path.exists(os.path.join(path, "butler.yaml")):
        raise FileExistsError(f"Repository already exists: {path}")
    os.makedirs(path, exist_ok=True)
    Butler.makeRepo(path)
    return True

def instrument_register_from_remote(
    local_repo: str,
    remote_repo: str,
    example_dataId: dict,
    dataset_type: str = "visit_image",
    remote_collection: str = "LSSTComCam/DP1",
    info: bool = True,
    out_remot_butler: bool = False
) -> bool:
    """
    Register an instrument in a local Butler repository using metadata extracted
    from a ilustrative dataset in a remote repository.

    Parameters
    ----------
    local_repo : str
        Path to the local Butler repository where the instrument will be registered.
    remote_repo : str
        Path to the remote repository hosting datasets with the desired instrument.
    example_dataId : dict
        A valid data ID used to fetch a dataset whose metadata contains the instrument
        identification (e.g. {"visit": 172, "detector": 0, "instrument": "LSSTComCam"}).
    dataset_type : str, optional
        Dataset type to retrieve for metadata inspection. Default is "visit_image".
    remote_collection : str, optional
        Collection to use when opening the remote repository. Default: "LSSTComCam/DP1".
    info : bool, optional
        If True, prints progress and diagnostic information.

    Returns
    -------
    bool
        True if the instrument was successfully registered.

    Raises
    ------
    RuntimeError
        If the instrument cannot be identified or is not present in the remote registry.
    """
    
    remote = Butler(remote_repo, collections=remote_collection)   # Load remote repository

    # Retrieve one dataset to inspect metadata
    try:
        exp = remote.get(dataset_type, example_dataId)
    except Exception as e:
        raise RuntimeError(f"Could not fetch dataset for metadata extraction: {e}")
    md = exp.getMetadata()

    # Extract instrument name from typical metadata keys
    instrument = (
        md.get("LSST BUTLER DATAID INSTRUMENT")
        or md.get("INSTRUME")
        or md.get("instrument"))

    if not instrument:
        raise RuntimeError(
            "Could not determine instrument name from dataset metadata. "
            "You may need to inspect metadata keys manually.")

    if info: print(f"[INFO] Detected instrument: {instrument}")

    # Retrieve the full instrument definition from the remote registry
    inst_records = [
        rec for rec in remote.registry.queryDimensionRecords("instrument")
        if rec.name == instrument]

    if not inst_records:
        raise RuntimeError(
            f"Instrument '{instrument}' not found in remote registry.")

    instrument_full = inst_records[0].class_name

    if info:
        print(f"[INFO] Full instrument class: {instrument_full}")

    # Check if the instrument is already registered locally
    local_registry = Butler(local_repo).registry
    local_instruments = {
        rec.name for rec in local_registry.queryDimensionRecords("instrument")}

    if instrument in local_instruments:
        if info:
            print(f"[INFO] Instrument '{instrument}' is already registered in local repo → skipping.")
        return True  # Nothing else to do

    # Register instrument in the local repo
    cmd = ["butler", "register-instrument", local_repo, instrument_full]

    if info: print("[CMD]", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Instrument registration failed: {e}")

    if info: print("[OK] Instrument registered successfully.")

    if out_remot_butler:
        return remote

    return True

def register_datasetTypes(local_repo: str, 
                          datasettypes: list[DatasetType],
                          check: bool = True) -> bool:
    """
    Register a list of DatasetTypes in a local Butler repository.

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
    registered_names = [d.name for d in lreg.queryDatasetTypes()]

    # Register types if not already present
    for dt in datasettypes:
        if dt.name in registered_names:
            continue
        lreg.registerDatasetType(dt)

    # Optional check: print registered DatasetTypes
    if check:
        for dt in lreg.queryDatasetTypes():
            print(dt.name)

    return True

def skymap_register_from_remote(
    local_repo: str,
    remote_repo: str,
    collections: str = "LSSTComCam/DP1",
    HSCskymap: bool = False,
    OBS_DECAM_DIR: bool = False,
    info: bool = True
) -> bool:
    """
    Register a SkyMap in a local Butler repository based on a SkyMap found in a remote repository.

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
    info : bool, optional
        Whether to print informational messages.

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

    # Use first skymap found
    if len(skymaps) > 1 and info:
        print(f"[WARN] Multiple SkyMaps found. Using the first one: {skymaps[0].name}")
    skymap_name = skymaps[0].name

    if info: print(f"[INFO] Detected remote SkyMap: {skymap_name}")

    # DECam skymap logic
    if OBS_DECAM_DIR:
        if "OBS_DECAM_DIR" not in os.environ:
            raise RuntimeError("OBS_DECAM_DIR is required but not set.")
        run_config = os.path.join(os.environ["OBS_DECAM_DIR"], "config/makeSkyMap.py")
        _run(["butler", "register-skymap", local_repo, "-C", run_config, "-c", f"name='{skymap_name}'"])
    else:
        _run(["butler", "register-skymap", local_repo, "-c", f"name='{skymap_name}'"])

    # Optional: register HSC skymap
    if HSCskymap:
        if "OBS_SUBARU_DIR" not in os.environ:
            raise RuntimeError("OBS_SUBARU_DIR is required for HSC skymap but not set.")
        hsc_config = os.path.join(os.environ["OBS_SUBARU_DIR"], "config/makeSkyMap.py")
        _run(["butler", "register-skymap", local_repo, "-C", hsc_config, "-c", "name='hsc_rings_v1'"])

    if info:
        print(f"[OK] SkyMap '{skymap_name}' registered successfully.")
        if HSCskymap: print("[OK] Additional HSC ring skymap registered.")
    return True

def transfer_visits(
    remote_repo: str,
    local_repo: str,
    visits: list[int],
    band: str,
    instrument: str = 'LSSTComCam',
    detector: int | list[int] = None,
    day_obs: int | list[int] = None,
    physical_filter: str = None,
    skymap: str = None,
    collections: str | list[str] = None,
    dataset: str = None,
    info: bool = True
) -> bool:
    """
    Transfer datasets of specified visits from a remote repository to a local repository using Butler.

    Parameters:
        remote_repo (str): Path to the remote repository.
        local_repo (str): Path to the local repository.
        visits (list[int]): List of visit IDs to transfer.
        band (str): Observing band (e.g., 'u', 'g', 'r', 'i').
        instrument (str): Instrument name (default 'LSSTCam').
        detector (int | list[int], optional): Detector(s) to filter (default None).
        day_obs (int | list[int], optional): Observation day(s) to filter (default None).
        physical_filter (str, optional): Physical filter name (default None).
        skymap (str, optional): Skymap identifier (default None).
        collections (str | list[str], optional): Collections to include (default None).
        dataset (str, optional): Dataset type to transfer (default None).
        info (bool): If True, prints informational messages during transfer (default True).

    Returns:
        bool: True if the transfer completed successfully.
    """
    if info:
        print("[INFO] Transferring visits...")

    # Convert collections list to comma-separated string if necessary
    if isinstance(collections, list):
        collections = ",".join(collections)

    # Create visit IDs string
    visit_ids_str = "(" + ",".join(map(str, visits)) + ")"

    # Build the query parts
    query_parts = [f"instrument='{instrument}'", f"visit IN {visit_ids_str}", f"band='{band}'"]

    if detector:
        if isinstance(detector, list):
            detector_str = "(" + ",".join(map(str, detector)) + ")"
            query_parts.append(f"detector IN {detector_str}")
        else:
            query_parts.append(f"detector={detector}")

    if day_obs:
        if isinstance(day_obs, list):
            day_obs_str = "(" + ",".join(map(str, day_obs)) + ")"
            query_parts.append(f"day_obs IN {day_obs_str}")
        else:
            query_parts.append(f"day_obs={day_obs}")

    if physical_filter:
        query_parts.append(f"physical_filter='{physical_filter}'")

    if skymap:
        query_parts.append(f"skymap='{skymap}'")

    query_string = " AND ".join(query_parts)

    # Build the command
    cmd = [
        "butler", "transfer-datasets",
        remote_repo,
        local_repo,
        "--where", query_string,
    ]

    if collections:
        cmd.extend(["--collections", collections])
    if dataset:
        cmd.extend(["--dataset-type", dataset])

    # Run the command
    _run(cmd)

    print("[OK] Completed the transfer")
    return True



