# vera rubin v1.0
# coadd.custom_inject_coadd.py

import sys, os
import numpy as np
import matplotlib.pyplot as plt

from lsst.afw.image import ExposureF, MaskedImageF, ImageF
from lsst.afw.math import warpExposure, WarpingControl

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.tools import progressbar

###################################################################################################
def load_exposures(paths_or_exposures):
    """
    Load a list of exposures from file paths or use given ExposureF objects.
    """
    exposures = []
    for item in paths_or_exposures:
        if isinstance(item, str):
            if not os.path.exists(item):
                raise FileNotFoundError(f"{item} not found.")
            exp = ExposureF(item)
        elif isinstance(item, ExposureF):
            exp = item
        else:
            raise TypeError("Each item must be a path (str) or ExposureF object.")
        exposures.append(exp)
    return exposures

def coadd_exposures_pipeline(
    exposures,
    ref_exp=None,
    warping_kernel="lanczos3",
    save_path=None,
    coadd_name="coadd.fits",
    info=False,
    plot_debug=False,
    same_check=True,
):
    """
    Coadd exposures using LSST-style weighting and WCS alignment.
    Also returns a coverage map showing how many exposures contribute per pixel.

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
    info : bool
        If True, print diagnostic information for each exposure.
    plot_debug : bool
        If True, display residual images at each step.

    Returns
    -------
    coadd_exp : ExposureF
        Final coadded exposure aligned in WCS.
    coverage_map : np.ndarray
        2D array counting how many exposures contribute to each pixel.
    """

    if len(exposures) == 0:
        raise ValueError("Exposure list is empty.")

    if ref_exp is None:
        ref_exp = exposures[0]

    # Get reference geometry and WCS
    dims = ref_exp.getMaskedImage().getDimensions()
    ref_wcs = ref_exp.getWcs()

    # Initialize empty accumulators
    coadd_mi = MaskedImageF(dims)
    coadd_array = coadd_mi.getImage().getArray()
    coadd_weight = np.zeros((dims.getY(), dims.getX()), dtype=np.float32)
    # Use float64 for accumulation (higher precision)
    #coadd_array = np.zeros((dims.getY(), dims.getX()), dtype=np.float64)
    #coadd_weight = np.zeros_like(coadd_array)
    coverage_map = np.zeros_like(coadd_weight, dtype=np.int32)

    print(f"[INFO] Starting coaddition of {len(exposures)} exposures...")

    # progressbar(0, len(exposures), bar_length=20, progress_char="#")
    for ind, exp in enumerate(exposures, start=1):

        # Skip invalid exposures
        if exp is None:
            print(f"[WARNING] Exposure {ind} is None. Skipping.")
            continue

        if same_check:
            # Compare WCS by checking if their centers match closely (within tolerance)
            ref_center = ref_wcs.pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
            exp_center = exp.getWcs().pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
            offset_arcsec = ref_center.separation(exp_center).asArcseconds()
            if offset_arcsec < 1e-3:  # ~0.001 arcsec ≈ identical WCS
                warped_exp = exp.clone() # If WCS is identical → skip warping (avoids resampling noise)
            else:
                warped_exp = ExposureF(dims, ref_wcs)
                warpExposure(warped_exp, exp, WarpingControl(warpingKernelName=warping_kernel))
        else:
            warped_exp = ExposureF(dims, ref_wcs)
            warpExposure(warped_exp, exp, WarpingControl(warpingKernelName=warping_kernel))

        warped_mi = warped_exp.getMaskedImage()
        img = warped_mi.getImage().getArray()
        var = warped_mi.getVariance().getArray()

        # Debug info
        if info:
            fraction_nan = np.isnan(img).sum() / img.size
            print(f"Warp {ind}: {fraction_nan:.3%} NaN pixels")

            c1 = ref_wcs.pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
            c2 = exp.getWcs().pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
            print("Offset (degree):", c1.separation(c2).asArcseconds()/3600)

        # Apply photometric scale factor if available
        try:
            ref_flux0 = ref_exp.getCalib().getFluxMag0()[0]
            exp_flux0 = exp.getCalib().getFluxMag0()[0]
            scale = exp_flux0 / ref_flux0
        except Exception:
            scale = 1.0

        img *= scale
        var *= scale ** 2

        # Mask invalid pixels
        valid = (var > 0) & (~np.isnan(img))
        w = np.zeros_like(var, dtype=np.float32)
        w[valid] = 1.0 / var[valid]

        img_safe = np.where(valid, img, 0.0)
        
        # Accumulate weighted flux and total weight
        #coadd_array += img_safe * w
        coadd_array += (img_safe) # * w
        coadd_weight += w
        coverage_map += valid.astype(np.int32)

        # Optional: visualize residual between partial coadd and current image
        if plot_debug:
            temp0 = np.where(coadd_weight == 0, 1.0, coadd_weight)
            #temp = coadd_array / temp0 - img_safe
            temp = img_safe - ref_exp.getImage().getArray()
            print("→ Total diff:", np.nansum(temp),
                  "Max:", np.nanmax(np.abs(temp)),
                  "Min:", np.nanmin(np.abs(temp)))

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

        print(f"[{ind}/{len(exposures)}] exposure coadded.")
        # progressbar(ind, len(exposures), bar_length=20, progress_char="#")

    # Final normalization
    coadd_weight_safe = np.where(coadd_weight == 0, 1.0, coadd_weight)
    coadd_array /= coadd_weight_safe
    #coadd_final = coadd_array / coadd_weight_safe
    
    # Variance of coadd = inverse of total weight
    coadd_mi.getVariance().getArray()[:, :] = 1.0 / coadd_weight_safe

    # Convert back to float32 for LSST ExposureF
    #coadd_mi = MaskedImageF(
    #    ImageF(np.float32(coadd_final)),
    #    None, #np.zeros_like(np.float32(coadd_final)),
    #    ImageF(np.float32(1.0 / coadd_weight_safe))
    #)

    # Build final coadded exposure
    coadd_exp = ExposureF(coadd_mi, ref_wcs)

    # Save FITS if requested
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, coadd_name)
        coadd_exp.writeFits(full_path)
        print(f"[INFO] LSST-style coadd saved to {full_path}")

        # Also save coverage map
        import astropy.io.fits as fits
        cov_path = os.path.join(save_path, "coverage_map.fits")
        fits.writeto(cov_path, coverage_map, overwrite=True)
        print(f"[INFO] Coverage map saved to {cov_path}")

    print("Coaddition complete.")
    print(f"Pixels with no coverage: {(coverage_map == 0).sum()} / {coverage_map.size}")

    return coadd_exp, coverage_map


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


def leave_one_out_residual(coadd_exp, exp, warping_kernel="lanczos3"):
    """
    Compare an individual exposure against a coadd by warping it to the coadd's WCS.

    Parameters
    ----------
    coadd_exp : ExposureF
        The final coadd exposure (already built).
    exp : ExposureF
        An individual exposure to compare against the coadd.
    warping_kernel : str
        Warping kernel for resampling.

    Returns
    -------
    warped_exp : ExposureF
        The exposure warped to the coadd's WCS.
    residual : np.ndarray
        Difference image = coadd - warped exposure.
    """

    dims = coadd_exp.getMaskedImage().getDimensions()
    ref_wcs = coadd_exp.getWcs()

    # Warp individual exposure to coadd WCS
    warped_exp = ExposureF(dims, ref_wcs)
    warpExposure(warped_exp, exp, WarpingControl(warpingKernelName=warping_kernel))

    # Image arrays
    coadd_arr = coadd_exp.getMaskedImage().getImage().getArray()
    warped_arr = warped_exp.getMaskedImage().getImage().getArray()

    # Compute raw residual: coadd - warped exposure
    residual = coadd_arr - warped_arr

    # Expected variance per pixel: Var(resid) = 1/sum_weights_coadd + 1/weight_exp
    coadd_var = coadd_exp.getVariance().getArray()
    exp_var = exp.getVariance().getArray()
    expected_var = coadd_var + exp_var
    expected_std = np.sqrt(expected_var)

    # Normalized residual
    residual_norm = residual / expected_std
    return warped_exp, residual, residual_norm
