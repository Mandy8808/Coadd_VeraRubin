


########################## OLD FUNCTIONS
def coadd_exposures_pipeline_0(
    exposures, ref_exp=None, warping_kernel="lanczos3", save_path=None, coadd_name="coadd.fits", info=False
):
    """
    Coadd exposures using LSST-style weighting and WCS alignment.

    Parameters
    ----------
    exposures : list of ExposureF
        List of exposures to coadd.
    ref_exp : ExposureF, optional
        Reference exposure for WCS and output size. If None, first exposure is used.
    warping_kernel : str
        Interpolation kernel for warping. Options: "lanczos3", "bilinear", etc.
    save_path : str, optional
        Directory to save coadd FITS. If None, coadd is not saved.
    coadd_name : str
        Filename for saved coadd.

    Returns
    -------
    coadd_exp : ExposureF
        Coadded exposure aligned in WCS.

    See: sec 5.3 on:
    https://dp1.lsst.io/tutorials/notebook/306/notebook-306-2.html
    """

    if len(exposures) == 0:
        raise ValueError("Exposure list is empty.")

    if ref_exp is None:
        # If no reference exposure is provided, use the first exposure
        ref_exp = exposures[0]

    # Get reference dimensions and WCS
    dims = ref_exp.getMaskedImage().getDimensions()
    ref_wcs = ref_exp.getWcs() # take the referential WCS

    # Initialize empty MaskedImage for coadd
    coadd_mi = MaskedImageF(dims)  # Create an empty MaskedImage to store the final coadd
    coadd_array = coadd_mi.getImage().getArray()
    coadd_weight = np.zeros((dims.getY(), dims.getX()), dtype=np.float32)

    progressbar(0, len(exposures), bar_length=20, progress_char="#")
    for ind, exp in enumerate(exposures, start=1):
        
        # Make an empty warped exposure with the same dimensions and WCS as reference    
        warped_exp = ExposureF(dims, ref_wcs)
        
        # Warp the current exposure to match the reference WCS
        warpExposure(warped_exp, exp, WarpingControl(warpingKernelName=warping_kernel))

        # Get the MaskedImage of the warped exposure
        warped_mi = warped_exp.getMaskedImage()

        # Extract image, variance, and mask arrays from the MaskedImage
        img = warped_mi.getImage().getArray()

        if info:
            fraction_nan = np.isnan(img).sum() / img.size
            print(f"\n Warp {ind}: {fraction_nan:.3%} NaN pixels \n ")

            c1 = ref_wcs.pixelToSky(0.5*dims.getX(), 0.5*dims.getY())
            c2 = exp.getWcs().pixelToSky(0.5*dims.getX(), 0.5*dims.getY())
            print("Offset (arcsec):", c1.separation(c2).asArcseconds())

        var = warped_mi.getVariance().getArray()
        # mask = warped_mi.getMask().getArray()

        # optional: match photometric scale between exposures
        # coadd LSST style where apply a scale factor \sigma: photometric calibration o background level
        try:
            ref_flux0 = ref_exp.getCalib().getFluxMag0()[0]
            exp_flux0 = exp.getCalib().getFluxMag0()[0]
            scale = exp_flux0 / ref_flux0
        except Exception:
            scale = 1.0

        img = img * scale
        var = var * (scale ** 2)

        # Mask invalid pixels:
        # Identify valid pixels: positive variance, unmasked, and not NaN
        # valid = (var > 0) & (mask == 0) & (~np.isnan(img))
        valid = (var > 0) & (~np.isnan(img))
        w = np.zeros_like(var, dtype=np.float32)
        w[valid] = 1.0 / var[valid]

        img_safe = np.where(valid, img, 0.0)  # Replacing NaN pixels in the image with zero to avoid propagating NaN
        coadd_array += img_safe * w  # Accumulate weighted sum of pixel values
        coadd_weight += w  # Accumulate total weight

        if info:
            temp0 = np.where(coadd_weight == 0, 1.0, coadd_weight)
            temp = coadd_array/temp0 - img_safe
            
            print("Total difference -> ", np.sum(temp),
                  "Maximum difference -> ", np.max(np.abs(temp)),
                  "Minumum difference -> ", np.min(np.abs(temp)))

            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                temp,
                origin="lower",
                cmap="gray",
                vmin=np.nanpercentile(temp, 0.01),
                vmax=np.nanpercentile(temp, 99.9),
            )
            plt.title(f"Residual after warp {ind}")
            plt.xlabel("X [pix]")
            plt.ylabel("Y [pix]")
            plt.colorbar(im, label="Intensity", pad=0.05, shrink=0.8)
            plt.show()

        progressbar(ind, len(exposures), bar_length=20, progress_char="#")

    # Normalize coadd by dividing by total weight (weighted mean)
    coadd_weight_safe = np.where(coadd_weight == 0, 1.0, coadd_weight)  # avoid division by zero
    #  coadd_mi.getImage().getArray()[:, :] /= coadd_weight_safe
    coadd_array /= coadd_weight_safe

    # Update variance of coadd: inverse of total weight
    coadd_mi.getVariance().getArray()[:, :] = coadd_weight_safe

    # Build final ExposureF from MaskedImage and reference WCS
    coadd_exp = ExposureF(coadd_mi, ref_wcs)

    # Save coadd to FITS if a path is specified
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, coadd_name)
        coadd_exp.writeFits(full_path)
        print(f"[INFO] LSST-style coadd saved to {full_path}")

    return coadd_exp

def coadd_exposures_pipeline_old(exposures, ref_exp=None, warping_kernel="lanczos3", save_path=None, coadd_name="coadd.fits"):
    """
    Coadd exposures using LSST-style weighting and WCS alignment.

    Parameters
    ----------
    exposures : list of ExposureF
        List of exposures to coadd.
    ref_exp : ExposureF, optional
        Reference exposure for WCS and output size. If None, first exposure is used.
    warping_kernel : str
        Interpolation kernel for warping. Options: "lanczos3", "bilinear", etc.
    save_path : str, optional
        Directory to save coadd FITS. If None, coadd is not saved.
    coadd_name : str
        Filename for saved coadd.

    Returns
    -------
    coadd_exp : ExposureF
        Coadded exposure aligned in WCS.

    See: sec 5.3 on:
    https://dp1.lsst.io/tutorials/notebook/306/notebook-306-2.html
    """
    if len(exposures) == 0:
        raise ValueError("Exposure list is empty.")

    if ref_exp is None:
        # If no reference exposure is provided, use the first exposure
        ref_exp = exposures[0]

    # Initialize coadd MaskedImage
    dims = ref_exp.getMaskedImage().getDimensions()  # Get dimensions of reference exposure
    coadd_mi = MaskedImageF(dims)  # Create an empty MaskedImage to store the final coadd
    
    # Initialize warping control parameters
    control = WarpingControl(warpingKernelName=warping_kernel)
    
    # Get direct references to the coadd image and weight arrays
    coadd_array = coadd_mi.getImage().getArray()
    coadd_weight = np.zeros((dims.getY(), dims.getX()), dtype=np.float32)
    
    progressbar(0, len(exposures), bar_length=20, progress_char='#')
    for ind, exp in enumerate(exposures, start=1):  # Warp each exposure to the reference WCS
        # Making a empty ExposureF with the same dimensions and WCS as the reference
        warped_exp = ExposureF(dims, ref_exp.getWcs())
        warpExposure(warped_exp, exp, control)  # Warp the current exposure to match the reference WCS
        
        warped_mi = warped_exp.getMaskedImage()  # Get the MaskedImage of the warped exposure
        
        # Extract image, variance, and mask arrays from the MaskedImage
        img = warped_mi.getImage().getArray()
        var = warped_mi.getVariance().getArray()
        # mask = warped_mi.getMask().getArray()

        # Identify valid pixels: positive variance, unmasked, and not NaN
        # valid = (var > 0) & (mask == 0) & (~np.isnan(img))
        valid = (var > 0) & (~np.isnan(img))
        
        # Initialize weight array (inverse variance), zero for invalid pixels
        w = np.zeros_like(var, dtype=np.float32)
        w[valid] = 1.0 / var[valid]

        # Replace NaN pixels in the image with zero to avoid propagating NaN
        img_safe = np.where(valid, img, 0.0)
        
        coadd_array += img_safe * w  # Accumulate weighted sum of pixel values
        coadd_weight += w  # Accumulate total weight

        progressbar(ind, len(exposures), bar_length=20, progress_char='#')
        
    # Normalize coadd by dividing by total weight (weighted mean)
    coadd_weight_safe = np.where(coadd_weight == 0, 1.0, coadd_weight)  # avoid division by zero
    coadd_mi.getImage().getArray()[:, :] /= coadd_weight_safe

    # Update variance of coadd: inverse of total weight
    coadd_mi.getVariance().getArray()[:, :] = 1.0 / coadd_weight_safe

    # Build final ExposureF from MaskedImage and reference WCS
    coadd_exp = ExposureF(coadd_mi, ref_exp.getWcs())

    # Save coadd to FITS if a path is specified
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, coadd_name)
        coadd_exp.writeFits(full_path)
        print(f"[INFO] LSST-style coadd saved to {full_path}")

    return coadd_exp



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
    order_by_time: bool = True,          # sort visits by timestamp
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

    message(logger, f"[SELECT] Tract={tract_id}, Patch={patch_index}")

    # Prepare polygon vertices
    poly = patch_info.getOuterSkyPolygon()
    coordList = [lsst.geom.SpherePoint(v) for v in poly.getVertices()]

    # Query all visits for this band + instrument
    with butler.query() as q:
        q = q.datasets(datasetType)
        query_parts = [f"band='{band}'", f"instrument='{instrument}'",]

        # Optional: restrict to tract
        if filter_by_patch:
            query_parts.append(f"tract={tract_info.getId()}")

        # Optional: use detector footprint region
        if filter_by_region:
            query_parts.append(f"visit_detector_region.region OVERLAPS POINT({ra_deg}, {dec_deg})")

        # Optional: filter detectors
        if detectors:
            query_parts.append(f"detector IN ({','.join(str(d) for d in detectors)})")
        
        q = q.where(" AND ".join(query_parts))
        if order_by_time:
            q = q.order_by("visit.timespan.begin")
        visit_refs = list(q)

    message(logger, f"[SELECT] Candidate visits: {len(visit_refs)}")

    # Load WCS + BBox to filter images by footprint intersection
    wcsList, bboxList, validRefs = [], [], []
    for ref in visit_refs:
        try:
            exp = butler.get(datasetType, dataId=ref.dataId)
            wcsList.append(exp.getWcs())
            bboxList.append(exp.getBBox())
            validRefs.append(ref)
        except Exception as e:
            message(logger, f"[WARN] Failed reading {ref.dataId}: {e}")

    # Filter with WCS intersection
    task = WcsSelectImagesTask()
    indices = task.run(wcsList=wcsList, bboxList=bboxList, coordList=coordList, dataIds=validRefs,)
    visit_refs = [validRefs[i] for i in indices]
    message(logger, f"[SELECT] Overlapping visits: {len(visit_refs)}")

    return visit_refs, tract_info, patch_info


def group_detector_inputs(dataset_refs, butler, datasetType="visit_image"):
    """
    Return:
    {id_1: {0: WarpDetectorInputs, 3: WarpDetectorInputs, ...}, id_2: {3: WarpDetectorInputs, 6: WarpDetectorInputs, ...}, ...}
    """
    grouped = defaultdict(dict)

    for ref in dataset_refs:
        # Load exposure for this ref
        exp = butler.get(datasetType, dataId=ref.dataId)

        # Detector ID
        det_id = exp.getDetector().getId()

        # Visit ID
        visit_id = ref.dataId["visit"]

        # Expand DataCoordinate so WarpDetectorInputs works
        expanded_id = butler.registry.expandDataId(ref.dataId)

        # Build WarpDetectorInputs
        wdi = WarpDetectorInputs(
            exposure_or_handle=exp,
            data_id=expanded_id,
            background_revert=None,
            background_apply=None,
            background_ratio_or_handle=None
        )

        # Store in structure
        grouped[visit_id][det_id] = wdi

    return dict(grouped)

def show_all(exp):
    """Plot image, variance, and mask using the Exposure WCS."""
    
    # Get arrays
    img = exp.getImage().getArray()
    var = exp.getVariance().getArray()
    mask = exp.getMask().getArray()

    # Convert LSST WCS -> Astropy WCS
    awcs = WCS(exp.getWcs().getFitsMetadata().toDict())
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': awcs})

    p5, p95 = np.nanpercentile(img, [5, 95])
    axs[0].imshow(img, origin='lower', cmap='gray', vmin=p5, vmax=p95)
    axs[0].coords.grid(color='white', ls='dotted')
    axs[0].set_title("Image")
    
    axs[1].imshow(var, origin='lower', cmap='magma')
    axs[1].coords.grid(color='white', ls='dotted')
    axs[1].set_title("Variance")

    axs[2].imshow(mask != 0, origin='lower', cmap='gray')
    axs[2].coords.grid(color='white', ls='dotted')
    axs[2].set_title("Mask")

    plt.tight_layout()
    plt.show()



from lsst.skymap import TractInfo, PatchInfo
from lsst.daf.butler import DatasetType
from lsst.daf.butler import CollectionType


def outputRefs_warps(registry,
                     first_input,  # data_id
                     sky_info,  # makeSkyInfo
                     skymap_name="lsst_cells_v1",
                     dtype_name="directWarp",
                     run_name="directWarp_run",
                     logger=None,):
    # check that the collection exists
    try:
        registry.getCollectionType(run_name)
    except Exception:
        message(logger, f"[INFO] Registering the collection '{run_name}'")
        registry.registerCollection(run_name, CollectionType.RUN)
    
    # ensure datasetType exists
    try:
        dt = registry.getDatasetType(dtype_name)
    except Exception:
        message(logger, f"[INFO] Registering datasetType '{dtype_name}'")
        universe = registry.dimensions
        dims = universe.conform(
            ("skymap", "instrument", "visit", "band", "physical_filter", "tract", "patch")
        )

        dt = DatasetType(
            name=dtype_name,
            dimensions=dims,
            storageClass="ExposureF",
            universe=universe,
        )
        registry.registerDatasetType(dt)

    dataId = dict(
        skymap          = skymap_name,
        instrument      = first_input["instrument"],
        visit           = first_input["visit"],
        band            = first_input["band"],
        physical_filter = first_input["physical_filter"],
        tract           = sky_info.tractInfo.getId(),
        patch           = sky_info.patchInfo.getSequentialIndex(),
    )

    return DatasetRef(
        datasetType=dt,
        dataId=dataId,
        run="directWarp_run",
    )