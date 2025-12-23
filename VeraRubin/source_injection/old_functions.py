
############ OLD FUNCIONS VERSIONS
def apply_correction_from_exposureF_old(
        data, hdr, rotation_angle,
        warping_kernel='lanczos4',
        keep_size=False,
        update_wcs=True
    ):
    """
    Rotate an astronomical image (2D numpy array) using LSST's ExposureF 
    representation and optionally update its WCS information in the FITS header.

    Parameters
    ----------
    data : numpy.ndarray
        2D image data (e.g. from a FITS file).
    hdr : astropy.io.fits.Header
        FITS header associated with the image. Must contain standard WCS keywords 
        (CRPIX, CRVAL, CD or PC matrix).
    rotation_angle : float
        Rotation angle in degrees (counter-clockwise). The value is normalized to [0, 360).
    warping_kernel : str, optional
        Resampling kernel used by the LSST Warper (default: 'lanczos4').
    keep_size : bool, optional
        If True, output image keeps same shape as input (may crop edges). Default False.
    update_wcs : bool, optional
        If True, the FITS header WCS will be rotated consistently with the image 
        transformation. If False, the original header is returned unchanged.

    Returns
    -------
    rotated_data : numpy.ndarray
        The rotated image data.
    hdr_new : astropy.io.fits.Header
        A new FITS header with updated WCS if `update_wcs=True`, otherwise a copy 
        of the input header.

    Notes
    -----
    - The function relies on LSST's `ImageF`, `MaskedImageF`, `ExposureF`, and `makeSkyWcs`.
    - Rotation is performed around the image center.
    - When updating the WCS, both the reference pixel (CRPIX) and the linear 
      transformation (CD/PC matrix) are rotated consistently with the image data.

    see: 
    https://github.com/rubin-dp0/tutorial-notebooks/blob/main/DP02_14_Injecting_Synthetic_Sources.ipynb
    https://community.lsst.org/t/rotating-dp0-2-exposures-with-wcs/10085/4
    https://github.com/lsst/atmospec/blob/1e7d6e8e5655cc13d71b21ba866001e6d49ee04e/python/lsst/atmospec/utils.py#L259-L301
    
    """
    import lsst.geom as geom
    import lsst.afw.math as afwMath
    import lsst.afw.geom as afwGeom
    from lsst.afw.image import ExposureF, ImageF, MaskedImageF

    # Convert numpy array into an LSST ExposureF
    image = ImageF(data.astype(np.float32, order="C"), deep=False)
    masked = MaskedImageF(image)
    exposure = ExposureF(masked)

    # Build LSST SkyWcs from FITS header keywords ---
    crpix = geom.Point2D(hdr["CRPIX1"], hdr["CRPIX2"])
    crval = geom.SpherePoint(hdr["CRVAL1"] * geom.degrees, hdr["CRVAL2"] * geom.degrees)

    # Support both CD or PC+CDELT conventions
    if "CD1_1" in hdr:
        cd_matrix = np.array([[hdr["CD1_1"], hdr["CD1_2"]],
                              [hdr["CD2_1"], hdr["CD2_2"]]])
    else:
        cd_matrix = np.array([
            [hdr["PC1_1"] * hdr["CDELT1"], hdr["PC1_2"] * hdr["CDELT1"]],
            [hdr["PC2_1"] * hdr["CDELT2"], hdr["PC2_2"] * hdr["CDELT2"]]
        ])
        
    skyWcs = afwGeom.makeSkyWcs(crpix, crval, cd_matrix)
    exposure.setWcs(skyWcs)

    # Normalize rotation angle
    rotation_angle = rotation_angle % 360.0

    # Build a rotation around the image center
    # To rotate around to the point C, we used the idea
    # -> Translate to origin -> Rotate -> Translate back: T(C) \cdot R(\theta) \cdot T(-C), 
    # -> T1 (translate to origin) → rot_origin (rotate) → T2 (translate back).
    # where T is the translation
    dims = exposure.getDimensions()
    center = geom.Point2D(0.5 * dims.getX(), 0.5 * dims.getY())

    # Construct rotation around origin
    rot_origin = geom.AffineTransform.makeRotation(rotation_angle * geom.degrees)

    # Build translation transforms (use Extent2D objects)
    T1 = geom.AffineTransform.makeTranslation(geom.Extent2D(-center.getX(), -center.getY()))
    T2 = geom.AffineTransform.makeTranslation(geom.Extent2D(center.getX(), center.getY()))

    # Compose the transforms: T2 * R * T1
    rot_around_center = T2 * rot_origin * T1
    
    # Create the corresponding afwGeom.Transform
    transform_p2top2 = afwGeom.makeTransform(rot_around_center)
    rotated_wcs = afwGeom.makeModifiedWcs(transform_p2top2, exposure.getWcs(), False)

    # Warp the exposure with LSST Warper
    # Preserve original bounding box or expand and fill invalid pixels with 0.0
    # http://doxygen.lsst.codes/stack/doxygen/xlink_v29.0.1_2025_04_17_04.49.14/classlsst_1_1afw_1_1math_1_1__warper_1_1_warper.html#a24012d6302090acffb399cb771f53881
    if keep_size:
        bbox = exposure.getBBox()
    else:
        bbox = None
    
    warper = afwMath.Warper(warping_kernel)
    rotated_exp = warper.warpExposure(rotated_wcs, exposure, destBBox=bbox)
    rotated_data = rotated_exp.getMaskedImage().getImage().getArray()
    rotated_data = np.nan_to_num(rotated_data, nan=0.0)

    # Optional debugging info
    try:
        c1 = rotated_wcs.pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
        c2 = exposure.getWcs().pixelToSky(0.5 * dims.getX(), 0.5 * dims.getY())
        print("Offset (deg):", c1.separation(c2).asArcseconds() / 3600)
    except Exception as e:
        print("Warning: could not compute WCS offset:", e)

    # Copy original header
    hdr_new = hdr.copy()

    # Update FITS WCS keywords if requested
    if update_wcs:
        wcs_rotated = rotated_exp.getWcs()

        # Ajustar el WCS al nuevo origen (0,0)
        offset = geom.Extent2D(geom.Point2I(0, 0) - rotated_exp.getXY0())
        wcs_adjusted = wcs_rotated.copyAtShiftedPixelOrigin(offset)

        # Obtener metadatos FITS del WCS ajustado
        wcs_md = wcs_adjusted.getFitsMetadata()

        # Rewriting the WCS
        for k, v in wcs_md.toDict().items():
            hdr_new[k] = v

        # Optional debugging info
        try:
            c_old = exposure.getWcs().pixelToSky(*np.array(exposure.getDimensions()) / 2)
            c_new = wcs_adjusted.pixelToSky(*np.array(rotated_exp.getDimensions()) / 2)
            print("WCS center shift (deg):", c_old.separation(c_new).asDegrees())
        except Exception as e:
            print("Warning: center comparison failed:", e)

    return rotated_data, hdr_new


def old_apply_correction_from_exposureF(
        data, hdr, rotation_angle,
        warping_kernel='lanczos4',
        keep_size=False,
        update_wcs=True
    ):
    """
    Rotate an astronomical image (2D numpy array) using LSST's ExposureF 
    representation and optionally update its WCS information in the FITS header.

    Parameters
    ----------
    data : numpy.ndarray
        2D image data (e.g. from a FITS file).
    hdr : astropy.io.fits.Header
        FITS header associated with the image. Must contain standard WCS keywords 
        (CRPIX, CRVAL, CD or PC matrix).
    rotation_angle : float
        Rotation angle in degrees (counter-clockwise). The value is normalized to [0, 360).
    warping_kernel : str, optional
        Resampling kernel used by the LSST Warper (default: 'lanczos4').
    keep_size : bool, optional
        If True, output image keeps same shape as input (may crop edges). Default False.
    update_wcs : bool, optional
        If True, the FITS header WCS will be rotated consistently with the image 
        transformation. If False, the original header is returned unchanged.

    Returns
    -------
    rotated_data : numpy.ndarray
        The rotated image data.
    hdr_new : astropy.io.fits.Header
        A new FITS header with updated WCS if `update_wcs=True`, otherwise a copy 
        of the input header.

    Notes
    -----
    - The function relies on LSST's `ImageF`, `MaskedImageF`, `ExposureF`, and `makeSkyWcs`.
    - Rotation is performed around the image center.
    - When updating the WCS, both the reference pixel (CRPIX) and the linear 
      transformation (CD/PC matrix) are rotated consistently with the image data.

    see: 
    https://github.com/rubin-dp0/tutorial-notebooks/blob/main/DP02_14_Injecting_Synthetic_Sources.ipynb
    https://community.lsst.org/t/rotating-dp0-2-exposures-with-wcs/10085/4
    https://github.com/lsst/atmospec/blob/1e7d6e8e5655cc13d71b21ba866001e6d49ee04e/python/lsst/atmospec/utils.py#L259-L301
    """
    import lsst.geom as geom
    import lsst.afw.math as afwMath
    import lsst.afw.geom as afwGeom

    from lsst.afw.image import ExposureF, ImageF, MaskedImageF
    from lsst.afw.geom import makeSkyWcs
    from lsst.geom import Point2D, SpherePoint
    
    # Convert numpy array into an LSST ExposureF
    image = ImageF(data.astype(np.float32, order="C"), deep=False)
    masked = MaskedImageF(image)
    exposure = ExposureF(masked)

    # Build LSST SkyWcs from FITS header keywords
    crpix = Point2D(hdr["CRPIX1"], hdr["CRPIX2"])
    crval = SpherePoint(hdr["CRVAL1"] * geom.degrees, hdr["CRVAL2"] * geom.degrees)
    cd_matrix = np.array([[hdr["CD1_1"], hdr["CD1_2"]],
                          [hdr["CD2_1"], hdr["CD2_2"]]])
        
    skyWcs = makeSkyWcs(crpix, crval, cd_matrix)
    exposure.setWcs(skyWcs)

    # Normalize rotation angle
    rotation_angle = rotation_angle % 360

    # Warp the exposure with LSST warper
    wcs = exposure.getWcs()
    warper = afwMath.Warper(warping_kernel)
    affine_rot_transform = geom.AffineTransform.makeRotation(rotation_angle * geom.degrees)
    transform_p2top2 = afwGeom.makeTransform(affine_rot_transform)
    rotated_wcs = afwGeom.makeModifiedWcs(transform_p2top2, wcs, False)

    # Preserve original bounding box or expand and fill invalid pixels with 0.0
    # http://doxygen.lsst.codes/stack/doxygen/xlink_v29.0.1_2025_04_17_04.49.14/classlsst_1_1afw_1_1math_1_1__warper_1_1_warper.html#a24012d6302090acffb399cb771f53881
    
    #min_point = geom.Point2I(0, 0)
    #extent = geom.Extent2I(keep_size[0], keep_size[1])
    rotated_exp = warper.warpExposure(rotated_wcs, exposure, destBBox=None)
    rotated_data = rotated_exp.getMaskedImage().getImage().getArray()
    rotated_data = np.nan_to_num(rotated_data, nan=0.0)

    # Start with a copy of the original header
    hdr_new = hdr.copy()

    # Update FITS WCS keywords if requested
    if update_wcs and 'CTYPE1' in hdr:
        from astropy.wcs import WCS
        w = WCS(hdr)

        # Original and rotated image dimensions
        ny, nx = data.shape
        ny2, nx2 = rotated_data.shape

        # Image centers (FITS convention: pixel (1,1) is upper left)
        cx, cy = (nx + 1) / 2.0, (ny + 1) / 2.0
        cx2, cy2 = (nx2 + 1) / 2.0, (ny2 + 1) / 2.0

        # Build 2x2 rotation matrix (counter-clockwise)
        theta = np.deg2rad(rotation_angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        # Adjust CRPIX (reference pixel) for the new center
        v = np.array([w.wcs.crpix[0] - cx, w.wcs.crpix[1] - cy])
        v_rot = R @ v
        w.wcs.crpix = [v_rot[0] + cx2, v_rot[1] + cy2]

        # Rotate linear WCS transformation (CD or PC matrix)
        if w.wcs.has_cd():
            w.wcs.cd = R @ w.wcs.cd
        else:
            w.wcs.pc = R @ w.wcs.pc

        # Create updated header and merge with original keywords
        hdr_new = w.to_header()
        hdr_new.update(hdr, useblanks=False, update=True)

    # If expanded, update NAXIS keywords
    if not keep_size:
        ny2, nx2 = rotated_data.shape
        hdr_new['NAXIS1'] = nx2
        hdr_new['NAXIS2'] = ny2

    return rotated_data, hdr_new