


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