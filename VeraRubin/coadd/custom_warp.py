# vera rubin v1.0
# coadd.custom_warp.py

import subprocess
import lsst.geom
import logging
import os, sys

from lsst.drp.tasks.make_direct_warp import MakeDirectWarpTask, MakeDirectWarpConfig, WarpDetectorInputs
from lsst.pipe.tasks.selectImages import WcsSelectImagesTask
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import TractInfo, PatchInfo
from lsst.daf.butler import Butler,DatasetRef
from collections import defaultdict
from pathlib import Path


# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visit.visit import visit_dataset
from tools.tools import setup_logger

###################################################################################################

# Top-level
# ---------------------------------------------------------------------
def custom_warp(
    butler: Butler,
    loc: list | tuple,
    band: str = "u",
    instrument: str = "LSSTComCam",
    datasetType: str = "visit_image",
    skymap_name: str = "lsst_cells_v1",
    # Optional advanced features
    filter_by_patch: bool = True,        # use tract/patch to restrict visits
    filter_by_region: bool = False,      # use visit_detector_region (if available)
    detectors: list = None,              # allowed detectors
    LOGDIR: str = "warps",
) -> list:
    """
    Custom warping function for LSST coaddition.

    This function selects visits overlapping a specified patch and runs the MakeDirectWarpTask
    to generate warped exposures for coaddition.
    """
    # Setup logging directory and logger
    log_path = Path.cwd() / LOGDIR
    log_path.mkdir(parents=True, exist_ok=True)
    # subprocess.run(["chmod", "ug+rw", LOGDIR], check=True)

    logfile = log_path / "warp.log"
    logger = setup_logger(str(logfile))
    logger.info(f"Created LOGDIR at {LOGDIR}")
    
    logger.info(f"Starting warping process...")

    # select visits
    logger.info(f"Selecting visits overlapping the patch...")
    visit_refs, tract_info, patch_info = select_visits(
        butler=butler,
        loc=loc,
        band=band,
        instrument=instrument,
        datasetType=datasetType,
        skymap_name=skymap_name,
        filter_by_patch=filter_by_patch,
        filter_by_region=filter_by_region,
        detectors=detectors,
        logger=logger
    )

    # run warping task
    logger.info(f"Running warping task for selected visits...")
    visit_warps = runDirectWarpTask(
        butler=butler,
        dataset_refs=visit_refs,
        tract_info=tract_info,
        patch_info=patch_info,
        datasetType=datasetType,
        skymap_name=skymap_name,
        use_visit_summary=True,
        out=True,
        logger=logger
    )

    logger.info(f"Warping process completed.")
    return visit_warps, visit_refs

# Low-level helpers
# ---------------------------------------------------------------------
def select_visits(
    butler: Butler,
    loc: list | tuple,
    band: str = "u",
    instrument: str = "LSSTComCam",
    datasetType: str = "visit_image",
    skymap_name: str = "lsst_cells_v1",
    # Optional advanced features
    filter_by_patch: bool = True,        # use tract/patch to restrict visits
    filter_by_region: bool = False,      # use visit_detector_region (if available)
    detectors: list = None,              # allowed detectors
    logger: logging.Logger = None,):
    """
    Return DatasetRefs for images overlapping the patch.
    https://github.com/lsst/pipe_tasks/blob/main/python/lsst/pipe/tasks/selectImages.py
    """
    ra_deg, dec_deg = loc
    point_sky = lsst.geom.SpherePoint(ra_deg, dec_deg, lsst.geom.degrees)

    # Load sky map
    sky_map = butler.get("skyMap", skymap=skymap_name)
    tract_info = sky_map.findTract(point_sky)
    patch_info = tract_info.findPatch(point_sky)

    tract_id = tract_info.getId()
    patch_index = patch_info.getIndex()
    if logger: logger.info(f"[SELECT] Tract={tract_id}, Patch={patch_index}")
    
    # Query all visits for this band + instrument
    visit_refs = visit_dataset(butler=butler,
                                  band=band, loc_data=loc,
                                  use_patch_area=filter_by_patch,
                                  filter_by_region=filter_by_region,
                                  detectors=detectors, 
                                  instrument=instrument)
    if logger: logger.info(f"[SELECT] Candidate visits: {len(visit_refs)}")

    # Load WCS + BBox to filter images by footprint intersection
    wcsList, bboxList, validRefs = [], [], []
    for ref in visit_refs:
        try:
            exp = butler.get(datasetType, dataId=ref.dataId)
            wcsList.append(exp.getWcs())
            bboxList.append(exp.getBBox())
            validRefs.append(ref)
        except Exception as e:
            if logger: logger.exception( f"[WARN] Failed reading {ref.dataId}: {e}")

     # Prepare polygon vertices
    poly = patch_info.getOuterSkyPolygon()
    coordList = [lsst.geom.SpherePoint(v) for v in poly.getVertices()]
    
    # Filter with WCS intersection
    task = WcsSelectImagesTask()
    indices = task.run(wcsList=wcsList, bboxList=bboxList, coordList=coordList, dataIds=validRefs,)
    visit_refs = [validRefs[i] for i in indices]
    if logger: logger.info(f"[SELECT] Overlapping visits: {len(visit_refs)}")

    return visit_refs, tract_info, patch_info

# High-level operations
# ---------------------------------------------------------------------
def runDirectWarpTask(
    butler: Butler,
    dataset_refs: list[DatasetRef] | tuple[DatasetRef],
    tract_info: TractInfo,
    patch_info: PatchInfo,
    datasetType: str = "visit_image",
    skymap_name: str = "lsst_cells_v1",
    use_visit_summary: bool = True,
    out: bool = False,
    logger: logging.Logger | None = None,
    ) -> dict[int, list] | None:
    """
    Run MakeDirectWarpTask per visit

    Comment:
    This function is an adaptation of runQuantum tha appear on the class MakeDirectWarpTask:
    https://github.com/lsst/drp_tasks/blob/main/python/lsst/drp/tasks/make_direct_warp.py

    Returns:
    dict[visit_id, list[ExposureF]] or None
        Warps per visit if out=True, else None.
    """

    if logger: logger.info(f"[INF] Build WarpDetectorInputs")
    
    # Build WarpDetectorInputs (visit → detector → inputs)
    inputs: dict[int, dict[int, WarpDetectorInputs]] = defaultdict(dict)   # visit_id -> [Detector ID -> WarpDetectorInputs]
    registry = butler.registry
    
    for ref in dataset_refs:
        handle = butler.getDeferred(datasetType, dataId=ref.dataId)  # lazy copy (handle)
        # exp = butler.get(datasetType, dataId=ref.dataId)  # Load exposure for this ref

        expanded_data_id = registry.expandDataId(ref.dataId)  # EXPANDED dataId is REQUIRED for IdGenerator
        
        visit_id = handle.dataId["visit"]  # Visit ID
        det_id = handle.dataId['detector'] # Detector ID

        # Store in structures
        inputs[visit_id][det_id] = WarpDetectorInputs(
            exposure_or_handle=handle,
            data_id=expanded_data_id,
            background_revert=None,
            background_apply=None,
            background_ratio_or_handle=None
        )
    if not inputs:
        logger.exception("[ERROR] No input warps provided for co-addition")
        raise

    if logger: logger.info(f"[INF] Warping visits: {sorted(inputs.keys())}")

    # Load SkyMap
    sky_map = butler.get("skyMap", skymap=skymap_name)
    
    # Configure task
    config = MakeDirectWarpConfig()
    config.doSelectPreWarp = False
    
    # Leave visit_summary enabled (default DRP behavior)
    if not use_visit_summary:
        config.useVisitSummaryPsf = False
        config.useVisitSummaryPhotoCalib = False
        config.useVisitSummaryWcs = False
    task = MakeDirectWarpTask(config=config)
    
    # Output container
    visit_warps = {} if out else None
    
    # Run per visit (IMPORTANT: SkyInfo recreated every time)
    for visit_id, detector_inputs in inputs.items():
        if logger: logger.info(f"Processing visit {visit_id}")
        
        # Recreate SkyInfo
        sky_info = makeSkyInfo(sky_map, tract_info.getId(), patch_info.getIndex())
        
        # Use first detector to fetch visit_summary
        first_input = next(iter(detector_inputs.values()))
        visit_summary = butler.get('visit_summary', dataId=first_input.data_id)

        # Run warp
        results = task.run(inputs=detector_inputs,
                           sky_info=sky_info,
                           visit_summary=visit_summary,)

        if out:
            if logger: logger.info(f"WARP SAVED: visit={visit_id}")
            visit_warps[visit_id] = results.warp
        else:
            pass
            # no work because we need the tract, and path dimmension
            #dataId_out = outputRefs_warps(registry, first_input.data_id, sky_info, skymap_name="lsst_cells_v1", dtype_name="directWarp", logger=logger)
            #butler.put(results.warp, dataId_out)

        # Explicit cleanup
        del results
        del visit_summary
        del sky_info

    # Final cleanup
    inputs.clear()
    del inputs
    del sky_map
    
    return visit_warps if out else None
