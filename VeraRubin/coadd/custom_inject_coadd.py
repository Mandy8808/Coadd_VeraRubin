# vera rubin v1.0
# coadd.custom_inject_coadd.py

import sys, os
import numpy as np

from lsst.afw.image import ExposureF, MaskedImageF
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

def coadd_exposures_pipeline(exposures, ref_exp=None, warping_kernel="lanczos3", save_path=None, coadd_name="coadd.fits"):
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
    coadd_mi.getImage().getArray()[:, :] /= 1 #coadd_weight_safe

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



# ---------------------------
# Example usage
# ---------------------------
# exposures_list = ["visit1.fits", "visit2.fits", exposure_obj3]
# exposures = load_exposures(exposures_list)
# coadded_exp = coadd_exposures_wcs(exposures, save_path="./coadds", coadd_name="band_r_coadd.fits")