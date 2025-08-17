# vera rubin v1.0
# source_injection/injection.py
# see:
# https://pipelines.lsst.io/v/daily/modules/lsst.source.injection/index.html
# https://dp1.lsst.io/tutorials/notebook/105/notebook-105-4.html
# https://github.com/alxogm/SL-MEX-1/blob/main/stamp_inyect_1.ipynb
import os, sys
import numpy as np
import astropy.units as u
import lsst.afw.image as afwImage


from lsst.source.injection import VisitInjectConfig, VisitInjectTask, generate_injection_catalog
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sky.sky import tract_patch, patch_center
from visit.visit import Visit

def measure_quality(calexp):
    """
    Estimate the approximate SNR (signal-to-noise ratio) of an LSST calexp image.

    Parameters:
    -----------
    calexp : lsst.afw.image.ExposureF
        A single calibrated LSST exposure (calexp) containing:
        - image array
        - mask array
        - variance array

    Returns:
    --------
    float
        Approximate global SNR for the image.
    """

    # Extract arrays from calexp
    image = calexp.image.array        # Pixel values after calibration
    variance = calexp.variance.array  # Per-pixel noise estimate
    mask = calexp.mask.array          # Pixel mask flags

    # Create a boolean mask for "good" pixels
    # Exclude bad pixels, saturated pixels, and those with zero variance
    GOOD = (mask == 0) & (variance > 0)

    # Estimate the background using the median of good pixels
    background = np.median(image[GOOD])

    # Signal: mean of pixels above the background
    signal = np.mean(image[GOOD & (image > background)])

    # Noise: sqrt of mean variance of good pixels
    noise = np.sqrt(np.mean(variance[GOOD]))

    # Compute SNR, handle division by zero
    SNR = signal / noise if noise > 0 else 0
    return SNR


def create_crowded_injection_catalog(
        ra_list,
        dec_list,
        stamp_paths,
        mags,
        min_sep=0.0005,
        separation_spherical=True):
    """
    Create an injection catalog for multiple 'Stamp'-type sources while ensuring a minimum separation 
    between them to avoid overlaps.

    Parameters
    ----------
    ra_list : list of float
        Right Ascensions of the sources in degrees.
    dec_list : list of float
        Declinations of the sources in degrees.
    stamp_paths : list of str
        File paths to the stamp images used for injection.
    mags : list of float
        Apparent magnitudes of the injected sources.
    min_sep : float, optional
        Minimum allowed separation between injected sources in degrees. Default is 0.0005.
    separation_spherical : bool, optional
        If True, compute angular separations using spherical geometry (more accurate for larger separations).
        If False, use plain Euclidean separation (fine for very small separations).

    Returns
    -------
    astropy.table.Table
        Table containing the injection catalog with non-overlapping sources.
    """
    
    n_sources = len(ra_list)
    
    # Validate input lengths to avoid mismatched parameters
    if not (len(dec_list) == len(stamp_paths) == len(mags) == n_sources):
        raise ValueError("All input lists must have the same length.")
    
    accepted = []  # Stores only the sources that pass the separation check
    for i in range(n_sources):
        ra_i, dec_i = ra_list[i], dec_list[i]
        too_close = False
        
        # Compare this source to all previously accepted ones
        for src in accepted:
            if separation_spherical:
                sep = SkyCoord(ra_i*u.deg, dec_i*u.deg).separation(
                    SkyCoord(src['ra']*u.deg, src['dec']*u.deg)
                ).deg
            else:
                sep = np.sqrt((ra_i - src['ra'])**2 + (dec_i - src['dec'])**2)  # Approximation valid for small separations
            
            if sep < min_sep:
                too_close = True
                break
        
        if not too_close:
            accepted.append({
                "ra": ra_i,
                "dec": dec_i,
                "mag": mags[i],
                "stamp": stamp_paths[i]
            })

    # Return an empty table if no sources are accepted
    if not accepted:
        return Table(names=["injection_id", "ra", "dec", "source_type", "mag", "stamp"])
    
    # Try using LSST's generate_injection_catalog
    try:
        cat_list = []
        for idx, src in enumerate(accepted):
            ra_lim = [src['ra'] - min_sep/2, src['ra'] + min_sep/2]
            dec_lim = [src['dec'] - min_sep/2, src['dec'] + min_sep/2]
            
            # number=1 → inject exactly one source in this RA/Dec bounding box ([ra_min, ra_max], [dec_min, dec_max])
            # seed controls the random placement; adding idx ensures reproducibility but variation between sources
            cat = generate_injection_catalog(
                ra_lim=ra_lim,
                dec_lim=dec_lim,
                number=1,
                seed=3210 + idx,
                source_type="Stamp",
                mag=[src['mag']],
                stamp=[src['stamp']]
            )
            cat_list.append(cat)
        
        return Table.vstack(cat_list)
    
    except Exception as e:
        print(f"generate_injection_catalog failed: {e}")
    
    # Fallback: create the catalog manually without LSST helper function
    return Table({
        "injection_id": list(range(len(accepted))),
        "ra": [src['ra'] for src in accepted],
        "dec": [src['dec'] for src in accepted],
        "source_type": ["Stamp"] * len(accepted),
        "mag": [src['mag'] for src in accepted],
        "stamp": [src['stamp'] for src in accepted]
    })


def apply_correction_from_data(data,
        hdr,
        rotation_angle,
        keep_size=False,
        interp_order=3,
        update_wcs=True,
        mode="constant",
        cval=0.0, c=1):
    """
    Rotate a 2D image array and optionally update its WCS (World Coordinate System) in the FITS header.

    Parameters
    ----------
    data : 2D ndarray
        Input image data.
    hdr : astropy.io.fits.Header
        FITS header associated with the image.
    rotation_angle : float
        Rotation angle in degrees (anticlockwise).
    keep_size : bool, optional
        If True, output image keeps same shape as input (may crop edges). Default False.
    interp_order : int, optional
        Spline interpolation order used in rotation. Default is 3 (cubic).
    update_wcs : bool, optional
        If True, modifies the WCS in the header to match the rotated image. Default True.
    mode : str, optional
        How to handle values outside the input boundaries. Default 'constant'.
    cval : float, optional
        Value to fill past edges if mode='constant'. Default 0.0.

    Returns
    -------
    rotated_data : 2D ndarray
        Rotated image data.
    hdr_new : astropy.io.fits.Header
        Updated FITS header with rotated WCS if update_wcs=True, else original header.
    """
    from scipy import ndimage
    # from reproject import reproject_interp
    
    # Rotate the image array
    rotated_data = ndimage.rotate(
        data,
        rotation_angle,
        reshape=not keep_size,
        order=interp_order,
        mode=mode,
        cval=cval,
        prefilter=True
    )

    hdr_new = hdr.copy()

    # Update WCS if present
    if update_wcs and 'CTYPE1' in hdr:
        from astropy.wcs import WCS
        
        w = WCS(hdr)  # Initialize WCS object

        # Original and rotated image shapes
        ny, nx = data.shape
        ny2, nx2 = rotated_data.shape

        # Original and new image centers (FITS is 1-based, the pixel (1,1) is on the upper left)
        cx, cy = (nx + 1) / 2.0, (ny + 1) / 2.0
        cx2, cy2 = (nx2 + 1) / 2.0, (ny2 + 1) / 2.0

        # 2x2 rotation matrix (anticlockwise)
        theta = np.deg2rad(rotation_angle)
        R = np.array([[np.cos(theta), - c * np.sin(theta)],
                      [c * np.sin(theta),  np.cos(theta)]])

        # Adjust reference pixel (CRPIX) for center shift
        v = np.array([w.wcs.crpix[0] - cx, w.wcs.crpix[1] - cy])  # vector from the original center to CRPIX (in pixels, FITS 1-based convention)
        v_rot = R @ v  # rotate this vector
        w.wcs.crpix = [v_rot[0] + cx2, v_rot[1] + cy2]  # shifting it to the new center

        # Rotate linear transformation (CD or PC matrix)
        # w.wcs.pc or cd linear part (matrix) of WCS which transform pixel offsets to intermediate coordinates (before sky projection and CDELT scaling).
        # When compute R @ PC, we inject the same rotation applied to the imagen to the system axes of the WCS, in order that the "header" describe
        # correctly as the pixel point out the sky after the array rotation.
        if w.wcs.has_cd():
            w.wcs.cd = R @ w.wcs.cd
        else:
            w.wcs.pc = R @ w.wcs.pc

        # Update header with rotated WCS, preserving other keywords
        hdr_new = w.to_header()
        for key in hdr:
            try:
                # copy only the non problematic FITS
                if key not in hdr_new:
                    val = hdr[key]
                    if isinstance(val, (int, float, str, bool, type(None))):
                        hdr_new[key] = val
            except ValueError:
                print("Warning: Ignoring illegal keyword: ", key)

        # rotated_data, footprint = reproject_interp((data, hdr), hdr_new, shape_out=(ny2, nx2))
                
    return rotated_data, hdr_new

def apply_correction_to_stamp(stamp_file, rotation_angle, output_path=None,
                              keep_size=False, interp_order=3, update_wcs=True, c=1):
    """
    Rotate/Shift a FITS (Flexible Image Transport System) stamp image by a given angle (anticlockwise), optionally updating the WCS.

    Parameters:
    -----------
    stamp_file : str
        Path to the input FITS stamp.
    rotation_angle : float
        Rotation angle in degrees (anticlockwise).
    output_path : str
        Path to save the rotated FITS file.
    keep_size : bool
        If True, keeps the original image size (cropping or padding as needed).
    interp_order : int
        Interpolation order for rotation (0=nearest, 1=linear, 3=cubic).
    update_wcs : bool
        If True, rotate and update the WCS information in the header.

    Returns:
    --------
    str or astropy.io.fits.HDUList
        Path to the rotated FITS file if output_path is given, else HDU object.

    see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    """
    try:
        # Load the original data and header
        with fits.open(stamp_file) as hdul:
            data = hdul[0].data  # Give the pixels of the first extension
            hdr = hdul[0].header.copy()  # Copy its metadata (header)
            # Note: The header may contain WCS (World Coordinate System) keywords such as:
            # NAXIS1, NAXIS2: size of the image in pixels along X and Y axes.
            # CRPIX1, CRPIX2: pixel coordinates of the reference point in the image.
            #                 This is the “anchor” pixel used for coordinate transformations.
            # CRVAL1, CRVAL2: celestial coordinates (RA/Dec in degrees) corresponding to the reference pixel (CRPIX1, CRPIX2).
            #                  They define the position on the sky that the reference pixel represents.
            # CDELT1, CDELT2: pixel scale in degrees per pixel along X and Y axes.
            #                  They are used to convert pixel offsets to sky coordinate offsets.
            # CTYPE1, CTYPE2: type of projection used for mapping the sky onto the image (e.g., 'TAN' for gnomonic, 'CAR' for Cartesian).
            #                  Determines how RA/Dec are computed for pixels away from the reference pixel.
            # etc.
            # Using these WCS parameters, the celestial coordinates (RA, Dec) of any pixel (x, y) can be calculated as:
            #   RA  = CRVAL1 + (x - CRPIX1) * CDELT1 (with projection applied by CTYPE1)
            #   Dec = CRVAL2 + (y - CRPIX2) * CDELT2 (with projection applied by CTYPE2)
            # This allows transforming pixel positions to accurate sky coordinates.
        
        rotated_data, hdr_new = apply_correction_from_data(
            data, hdr, rotation_angle,
            keep_size=keep_size, interp_order=interp_order, update_wcs=update_wcs, c=c
        )

        # Create new HDU with rotated data and header
        hdu = fits.PrimaryHDU(data=rotated_data, header=fits.Header(hdr_new)) # Dumps the modified WCS back to the header. Thus the FITS you are going to write will already have the rotated WCS.
        # IMPORTANT: This methodology ajust only the linear part (orientation/scale). If the "header" has distortions (SIP A_*, B_*, PV/TPV…), 
        #            they are not recalculated here. For distorted mappings, reprojection with tools like reproject would be needed.

        # Remove NAXIS1/NAXIS2 to let FITS recalc from data shape
        for key in ['NAXIS1', 'NAXIS2']:
            hdu.header.pop(key, None)

        # Add metadata about rotation
        hdu.header['ROT_ANG'] = (rotation_angle, 'Rotation applied (deg, anticlockwise)')
        hdu.header['ROT_KEEP'] = (keep_size, 'True if original image size kept')
        hdu.header['ROT_INT'] = (interp_order, 'Interpolation order used')

        # Save to file if requested
        if output_path:
            hdu.writeto(output_path, overwrite=True)
            return output_path
        else:
            return hdu

    except Exception as e:
        print(f"Error rotating stamp '{stamp_file}': {e}")
        return None
    

def inject_stamp(visit_image, inj_cat):
    """
    Inject artificial sources ("stamps") into an LSST exposure.

    This function uses the LSST VisitInjectTask to add synthetic sources 
    defined in an input injection catalog into a given calibrated exposure.

    Parameters
    ----------
    visit_image : lsst.afw.image.ExposureF
        The calibrated LSST exposure (calexp/visit image) to inject sources into.
    inj_cat : lsst.afw.table.SourceCatalog
        Injection catalog with source positions, magnitudes, etc.

    Returns
    -------
    injected_exposure : lsst.afw.image.ExposureF or None
        The exposure with injected sources, or None if injection failed.
    
    https://dp1.lsst.io/tutorials/notebook/105/notebook-105-4.html
    """
    # Create the injection task
    inject_config = VisitInjectConfig()
    inject_task = VisitInjectTask(config=inject_config)

    try:
        injected_output = inject_task.run(
            injection_catalogs=inj_cat,
            input_exposure=visit_image.clone(),  # clone to avoid modifying the original
            psf=visit_image.getPsf(),
            photo_calib=visit_image.getPhotoCalib(),
            wcs=visit_image.wcs
        )
    except Exception as e:
        print(f"  -> Error during injection in {visit_image.dataId}: {e}")
        return None
    
    # Extract the exposure with injected sources
    injected_exposure = injected_output.output_exposure
    return injected_exposure

def inject_stamp(
        butler, 
        loc_data,
        band,
        stamp_paths,
        mags,
        ra_list,
        dec_list,
        sky_coordinates=True,
        use_patch_area=False,
        detectors=None,
        timespan=None,
        visit_ids=None,
        num_select=None, 
        min_sep=0.0005,
        separation_spherical=True,
        keep_size=False,
        interp_order=3,
        update_wcs=True,
        c=1,
        name='visit_image',
        info=True):
    """
    Inject artificial sources (stamps) into multiple LSST visit images.

    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        Butler instance for accessing LSST data.
    loc_data : tuple
        If `sky_coordinates=True`: (ra, dec) in degrees.  
        If `sky_coordinates=False`: (tract, patch).
    band : str
        Filter band (e.g. "r", "i").
    stamp_paths : list of str
        File paths to stamp images (PSF-like cutouts to inject).
    mags : list of float
        Magnitudes of the sources to inject.
    ra_list, dec_list : list of float
        RA/Dec positions for the injected sources (degrees).
    sky_coordinates : bool, optional
        If True, loc_data is interpreted as (ra, dec).  
        If False, loc_data is interpreted as (tract, patch).
    use_patch_area : bool, optional
        If False (default), uses only the central coordinate of the patch.  
        If True, uses the full patch area (as in coadd construction).
    detectors : list of int, optional
        Restrict query to specific detectors.
    timespan : lsst.daf.butler.Timespan, optional
        Restrict query to a given time range.
    visit_ids : list of int, optional
        Restrict to specific visit IDs.
    num_select : int, optional
        Limit the number of visits selected (after sorting by SNR).
    min_sep : float, optional
        Minimum separation between injected sources [deg]. Default = 0.0005.
    separation_spherical : bool, optional
        If True, use spherical separation for spacing. Otherwise, Euclidean.
    keep_size, interp_order, update_wcs, c : passed to `apply_correction_to_stamp`
        Options for stamp rotation and resampling.
    name : str, optional
        Dataset type to fetch from Butler. Default "visit_image".
    info : bool, optional
        Print debug info.

    Returns
    -------
    injected_exposure : lsst.afw.image.ExposureF
        The last visit exposure with injected sources.
    visit_calexp_data : list of lsst.afw.image.ExposureF
        List of selected visit exposures before injection.
    """

    # Identify coordinates
    if sky_coordinates:
        ra_deg, dec_deg = loc_data
    else:
        tract, patch = loc_data
        ra_deg, dec_deg = patch_center(butler, tract, patch, sequential_index=True)

    # Identify visit images around location
    visit_obj = Visit(loc_data=(ra_deg, dec_deg), butler=butler, band=band)
    visit_calexp_dataset = visit_obj.query_visit_image(
        detectors=detectors,
        visit_ids=visit_ids,
        use_patch_area=use_patch_area,
        timespan=timespan
    )

    # Collect visit exposures
    visit_calexp_data = []
    visit_calexp_ids = []
    for ref in visit_calexp_dataset:
        try:
            exp = butler.get(name, dataId=ref.dataId)
            visit_calexp_data.append(exp)
            visit_calexp_ids.append(ref.dataId["visit"])
        except Exception as e:
            print(f"Error fetching calexp {ref.dataId}: {e}")

    # Sort visits by SNR
    snr_list = [measure_quality(calexp) for calexp in visit_calexp_data]
    sorter_snr = np.argsort(snr_list)[::-1]  # descending
    visit_calexp_data = [visit_calexp_data[i] for i in sorter_snr]
    visit_calexp_ids = [visit_calexp_ids[i] for i in sorter_snr]

    # Optionally restrict number of visits
    if num_select is not None:
        visit_calexp_data = visit_calexp_data[:num_select]
        visit_calexp_ids = visit_calexp_ids[:num_select]

    # Compute relative rotation angles w.r.t first visit
    ref_wcs = visit_calexp_data[0].getWcs()  # referential visit
    rotation_angle_list = [
        calexp.getWcs().getRelativeRotationToWcs(ref_wcs).asDegrees()
        for calexp in visit_calexp_data[1:]
    ]

    # Rotate stamps accordingly
    stamp_list = [[] for _ in range(len(rotation_angle_list))]  # [[stamp1R1, stamp2R1, ...], [stamp1R2, stamp2R2, ...], ...]
    for i, angle in enumerate(rotation_angle_list):
        for stamp_file in stamp_paths:
            stamp_R = apply_correction_to_stamp(
                stamp_file,
                angle,
                keep_size=keep_size,
                interp_order=interp_order,
                update_wcs=update_wcs,
                c=c
            )
            stamp_list[i].append(stamp_R)

    # Build injection catalogs
    catalog_list = []
    for stamps in stamp_list:
        inj_cat = create_crowded_injection_catalog(
            ra_list,
            dec_list,
            stamps,
            mags,
            min_sep=min_sep,
            separation_spherical=separation_spherical
        )
        catalog_list.append(inj_cat)

    # Perform injection
    injected_exposure = None
    for i, inj_cat in enumerate(catalog_list):
        visit_image = visit_calexp_data[i]
        injected_exposure = inject_stamp(visit_image, inj_cat)

    return injected_exposure, visit_calexp_data
    
