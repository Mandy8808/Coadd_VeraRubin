
# vera rubin v1.0
# coadd.exposure_tools.py

import os
import numpy as np

from astropy.io import fits
from lsst.afw.image import ExposureF

###################################################################################################
def load_exposures(items):
    """
    Load a mixed list of:
      - FITS image paths
      - FITS stamps (with optional MASK extension)
      - LSST ExposureF file paths
      - ExposureF objects

    Returns a list of dictionaries with standardized content:
    {
        "type": "fits" or "lsst",
        "data": np.ndarray,
        "mask": np.ndarray(bool),
        "header": fits.Header or None,
        "exposure": ExposureF or None
    }
    """

    results = []
    for item in items:
        # ExposureF object already given
        if "ExposureF" in str(type(item)):
            exp = item
            img = exp.getMaskedImage().getImage().getArray()
            mask = np.isfinite(img)
            results.append({
                "type": "lsst",
                "data": img,
                "mask": mask,
                "header": None,
                "exposure": exp
            })
            continue

        # Case 2: String → must be a file
        if isinstance(item, str):
            if not os.path.exists(item):
                raise FileNotFoundError(f"{item} not found.")

            # Look for FITS by extension
            if item.lower().endswith(".fits") or item.lower().endswith(".fits.gz"):
                try:
                    with fits.open(item) as hdul:
                        data = hdul[0].data
                        header = hdul[0].header.copy()

                        # Optional MASK extension
                        if "MASK" in [hdu.name for hdu in hdul]:
                            mask = hdul["MASK"].data.astype(bool)
                        else:
                            mask = np.isfinite(data)

                    results.append({
                        "type": "fits",
                        "data": data,
                        "mask": mask,
                        "header": header,
                        "exposure": None
                    })
                    continue

                except Exception:
                    # If FITS loading fails, try LSST ExposureF
                    pass

            # Try LSST ExposureF
            try:
                exp = ExposureF(item)
                img = exp.getMaskedImage().getImage().getArray()
                mask = np.isfinite(img)
                results.append({
                    "type": "lsst",
                    "data": img,
                    "mask": mask,
                    "header": None,
                    "exposure": exp
                })
                continue
            except Exception as e:
                raise RuntimeError(f"Could not interpret file {item} as FITS or ExposureF: {e}")
        # Anything else
        else:
            raise TypeError(
                "Items must be FITS paths, ExposureF paths, or ExposureF objects."
            )
    return results

def save_exposure(injected_exposure, 
                           output_root="./data",
                           band="r", 
                           visit_id=None,
                           prefix="calexp",
                           overwrite=True,
                           info=True):
    """
    Save an injected ExposureF to disk using the structure expected by
    custom_coadd_multiband_local().

    Parameters
    ----------
    injected_exposure : lsst.afw.image.ExposureF
        Exposure to save (e.g. simulated or modified image).
    output_root : str, optional
        Root directory where to store the FITS file.
    band : str, optional
        Photometric band (used in the filename, e.g. 'r', 'i', 'z').
    visit_id : int or str, optional
        Visit identifier (used in the filename). If None, a sequential ID is assigned.
    prefix : str, optional
        File prefix. Defaults to 'calexp' to match expected pattern.
    overwrite : bool, optional
        Whether to overwrite an existing file.
    info : bool, optional
        Print path and status messages.
    """

    os.makedirs(output_root, exist_ok=True)

    # Auto-generate a visit ID if not provided
    if visit_id is None:
        # Count existing files for that band to generate a sequential ID
        existing = [f for f in os.listdir(output_root) if f"_{band}_" in f]
        visit_id = len(existing) + 1

    # File path
    file_name = f"{prefix}_{band}_{int(visit_id):03d}.fits"
    file_path = os.path.join(output_root, file_name)

    # Overwrite protection
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to replace it.")

    # Save as FITS
    injected_exposure.writeFits(file_path)

    if info:
        print(f"[INFO] Exposure saved: {file_path}")

    return file_path

def fits_to_exposure(data, hdr, variance_init=1.0):
    """
    Convert a NumPy array + FITS header into an LSST ExposureF,
    initializing variance and mask planes.
    
    Parameters
    ----------
    data : np.ndarray
        Image data (2D array).
    hdr : astropy.io.fits.Header
        FITS header with WCS keywords.
    variance_init : float, optional
        Default variance value to assign to all pixels.
    
    Returns
    -------
    exposure : lsst.afw.image.ExposureF
        LSST Exposure with WCS, mask, and variance.
    """
    import lsst.geom as geom
    from lsst.afw.geom import makeSkyWcs
    from lsst.afw.image import ImageF, Mask, MaskedImageF
    
    # Create image plane
    image = ImageF(data.astype(np.float32, order="C"), deep=False)

    # Initialize mask (all zeros = good pixels)
    mask = Mask(image.getDimensions())
    mask.set(0)

    # Initialize variance plane (constant value, 1.0)
    variance = ImageF(image.getDimensions())
    variance.set(variance_init)

    # Build masked image (image, mask, variance)
    masked = MaskedImageF(image, mask, variance)

    # Wrap into an ExposureF
    exposure = ExposureF(masked)

    # Build LSST SkyWcs from FITS header keywords
    crpix = geom.Point2D(hdr["CRPIX1"], hdr["CRPIX2"])
    crval = geom.SpherePoint(hdr["CRVAL1"] * geom.degrees,
                             hdr["CRVAL2"] * geom.degrees)
    cd_matrix = np.array([[hdr["CD1_1"], hdr["CD1_2"]],
                          [hdr["CD2_1"], hdr["CD2_2"]]])

    skyWcs = makeSkyWcs(crpix, crval, cd_matrix)
    exposure.setWcs(skyWcs)

    return exposure

def exposure_to_fits_datahdr(exposure):
    """
    Convert an LSST Exposure to FITS-like (data, header).

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Input exposure.

    Returns
    -------
    data : 2D numpy.ndarray
        Image array from the exposure.
    hdr : astropy.io.fits.Header
        FITS header including WCS metadata.
    """
    import numpy as np
    from astropy.io import fits

    # Extract image array
    mi = exposure.getMaskedImage()
    data = mi.getImage().getArray()

    # Build WCS header
    wcs = exposure.getWcs()
    hdr = fits.Header()
    hdr.update(wcs.getFitsMetadata())

    return data, hdr

def cutout_exposure(ref_exposure, center_coord, radius_pixels, info=True):
    """
    Cut out an LSST exposure centered on a celestial position with a given radius.
    If the cutout exceeds the image boundaries, missing regions are filled with zeros.
    Updates the WCS of the cutout to remain consistent.

    Parameters
    ----------
    ref_exposure : `lsst.afw.image.Exposure`
        Input exposure.
    center_coord : `lsst.afw.coord.Coord`
        Cutout center in celestial coordinates.
    radius_pixels : float
        Radius of the cutout in pixels.

    Returns
    -------
    cutout_exp : `lsst.afw.image.Exposure`
        Cutout exposure with updated WCS.
    """
    import lsst.geom as geom
    import lsst.afw.image as afwImage
    import numpy as np

    wcs = ref_exposure.getWcs()
    bbox = ref_exposure.getBBox()

    # Convert celestial center to pixel coordinates
    center_point = wcs.skyToPixel(center_coord)

    # Desired cutout BBox
    size = int(2 * radius_pixels)
    min_x = int(center_point.getX() - radius_pixels)
    min_y = int(center_point.getY() - radius_pixels)
    desired_bbox = geom.Box2I(geom.Point2I(min_x, min_y), geom.Extent2I(size, size))

    # Checking if box is subset of desired_bbox
    # when this happened intersect_bbox is clipped to box
    intersect_bbox = desired_bbox.clippedTo(bbox)
    is_subset = (intersect_bbox != bbox)

    if is_subset:
        # Case when intersect_bbox is subset of bbox
        x0, y0 = 0, 0
    else:
        # Case when bbox is subset of intersect_bbox
        x0 = intersect_bbox.getMinX() - desired_bbox.getMinX()
        y0 = intersect_bbox.getMinY() - desired_bbox.getMinY()

    # Create empty arrays for image, mask, and variance
    dtype = ref_exposure.getMaskedImage().getImage().getArray().dtype
    empty_image = np.zeros((size, size), dtype=dtype)
    empty_mask = np.zeros((size, size), dtype=np.uint16)
    empty_variance = np.zeros((size, size), dtype=dtype)

    # Copy image, mask, and variance from the original
    mi = ref_exposure.getMaskedImage()
    empty_image[y0:y0 + intersect_bbox.getHeight() - 1, x0:x0 + intersect_bbox.getWidth() - 1] = \
            mi.getImage().getArray()[intersect_bbox.getMinY():intersect_bbox.getMaxY(),
                                 intersect_bbox.getMinX():intersect_bbox.getMaxX()]
    empty_mask[y0:y0 + intersect_bbox.getHeight() - 1, x0:x0 + intersect_bbox.getWidth() - 1] = \
            mi.getMask().getArray()[intersect_bbox.getMinY():intersect_bbox.getMaxY(),
                                intersect_bbox.getMinX():intersect_bbox.getMaxX()]
    empty_variance[y0:y0 + intersect_bbox.getHeight() - 1, x0:x0 + intersect_bbox.getWidth() - 1] = \
            mi.getVariance().getArray()[intersect_bbox.getMinY():intersect_bbox.getMaxY(),
                                    intersect_bbox.getMinX():intersect_bbox.getMaxX()]
        
    # Create MaskedImage and Exposure
    mi_cutout = afwImage.MaskedImageF(empty_image.shape[1], empty_image.shape[0])
    mi_cutout.getImage().getArray()[:, :] = empty_image
    mi_cutout.getMask().getArray()[:, :] = empty_mask
    mi_cutout.getVariance().getArray()[:, :] = empty_variance

    cutout_exp = afwImage.ExposureF(mi_cutout)

    # Shift WCS
    shift = geom.Extent2D(-desired_bbox.getMinX(), -desired_bbox.getMinY())
    shifted_wcs = wcs.copyAtShiftedPixelOrigin(shift)
    cutout_exp.setWcs(shifted_wcs)
    
    if info:
        center_point_cutout = cutout_exp.getWcs().skyToPixel(center_coord)
        print("Referential point in exposure:", center_point)
        print("Desired bbox:", desired_bbox)
        print("Intersection bbox:", intersect_bbox)
        print("Referential point in cutout:", center_point_cutout)
        print("Cutout shape:", cutout_exp.getMaskedImage().getImage().getArray().shape)

    return cutout_exp

def cutout_fits(data, hdr, center_coord, radius_pixels, info=True, ext=0, center_pixels=True):
    """
    Cut out a FITS image centered on a celestial position with a given radius.
    If the cutout exceeds the image boundaries, missing regions are filled with zeros.
    Updates the WCS of the cutout to remain consistent.

    Parameters
    ----------
    data : 2D numpy.ndarray
        Input FITS image data.
    hdr : astropy.io.fits.Header
        FITS header associated with the image.
    center_coord : astropy.coordinates.SkyCoord or tuple(float, float)
        Cutout center in celestial coordinates (if center_pixels=False)
        or in pixel coordinates (if center_pixels=True).
    radius_pixels : float
        Radius of the cutout in pixels.
    info : bool, optional
        If True, print debugging information.
    ext : int, optional
        FITS extension (default: 0).
    center_pixels : bool, optional
        If True, `center_coord` is interpreted as pixel coordinates.
        Otherwise, it is assumed to be a `SkyCoord`.

    Returns
    -------
    cutout_data : 2D numpy.ndarray
        Cutout image data with missing regions filled with zeros.
    cutout_hdr : astropy.io.fits.Header
        FITS header updated with the cutout WCS.
    """
    from astropy.wcs import WCS

    wcs = WCS(hdr)

    if center_pixels:
        # Center given directly in pixels
        center_x, center_y = center_coord
    else:
        # Convert sky coordinate to pixel position
        center_x, center_y = wcs.world_to_pixel(center_coord)

    # Desired cutout bbox
    size = int(2 * radius_pixels)
    min_x = int(center_x - radius_pixels)
    min_y = int(center_y - radius_pixels)
    max_x = min_x + size
    max_y = min_y + size

    # Initialize empty cutout
    cutout_data = np.zeros((size, size), dtype=data.dtype)

    # Intersection with original image bounds
    src_x0 = max(0, min_x)
    src_y0 = max(0, min_y)
    src_x1 = min(data.shape[1], max_x)
    src_y1 = min(data.shape[0], max_y)

    # Destination indices in cutout
    dest_x0 = src_x0 - min_x
    dest_y0 = src_y0 - min_y
    dest_x1 = dest_x0 + (src_x1 - src_x0)
    dest_y1 = dest_y0 + (src_y1 - src_y0)

    # Copy valid region into cutout
    cutout_data[dest_y0:dest_y1, dest_x0:dest_x1] = \
        data[src_y0:src_y1, src_x0:src_x1]

    # Adjust WCS: shift reference pixel
    cutout_wcs = wcs.deepcopy()
    cutout_wcs.wcs.crpix[0] -= min_x
    cutout_wcs.wcs.crpix[1] -= min_y

    # Create new header with updated WCS
    cutout_hdr = hdr.copy()
    cutout_hdr.update(cutout_wcs.to_header())

    if info:
        print("Original shape:", data.shape)
        print("Cutout shape:", cutout_data.shape)
        print("Center in original (px):", (center_x, center_y))
        print("BBox: x[%d:%d], y[%d:%d]" % (min_x, max_x, min_y, max_y))
        print("Intersection source region:", (src_x0, src_x1, src_y0, src_y1))
        print("Intersection destination region:", (dest_x0, dest_x1, dest_y0, dest_y1))

    return cutout_data, cutout_hdr

############## OLD functions
def load_exposures_old(paths_or_exposures):
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