# v1.0
# sky file

import lsst.geom

#################### SkyMap ##################################################
def tract_patch(butler, ra_deg, dec_deg, sequential_index=True):
    """
    From sky coordinates (RA, Dec in degrees), identify the tract and patch
    that cover the desired region using the SkyMap.

    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        An initialized Butler instance.
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    sequential_index : bool, optional
        If True, return patch as sequential integer index. If False, return "x,y" string.

    Returns
    -------
    tract : int
        ID of the tract containing the point.
    patch : str
        Patch ID, either as integer (string) or "x,y" format depending on sequential_index.

    See also:
    https://dp1.lsst.io/tutorials/notebook/104/notebook-104-2.html
    """
    # with the sky coordinates, we identify the tract and patch that cover the desired region
    point_sky = lsst.geom.SpherePoint(ra_deg, dec_deg, lsst.geom.degrees)
    skymap = butler.get("skyMap")

    tract_info = skymap.findTract(point_sky)
    tract = tract_info.getId()

    patch_info = tract_info.findPatch(point_sky)

    if sequential_index:
        patch_index = patch_info.getSequentialIndex()
        patch_str = f"{patch_index}"
    else:
        x, y = patch_info.getIndex()
        patch_str = f"{x},{y}"

    return tract, patch_str

def patch_center(butler, tract, patch_str, sequential_index=True):
    """
    Given a tract and patch, return the sky coordinates (RA, Dec) of the patch center.

    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        Butler instance.
    tract : int
        Tract ID.
    patch_str : str or int
        Patch in "x,y" format (string), or an integer index if sequential_index=True.
    sequential_index : bool, optional
        Whether patch_str is a sequential patch index (default: True).

    Returns
    -------
    ra_deg : float
        Right Ascension of the patch center, in degrees.
    dec_deg : float
        Declination of the patch center, in degrees.
    """
    skymap = butler.get("skyMap")
    tract_info = skymap[tract]

    if sequential_index:
        patch_info = tract_info.getPatchInfo(int(patch_str))
    else:
        try:
            x_str, y_str = patch_str.split(",")
            patch_index = (int(x_str), int(y_str))
            patch_info = tract_info.getPatchInfo(patch_index)
        except Exception as e:
            raise ValueError(f"Invalid patch format '{patch_str}': {e}")

    centroid_vec = patch_info.getInnerSkyPolygon().getCentroid()
    sky_center = lsst.geom.SpherePoint(centroid_vec)

    ra_deg = sky_center.getLongitude().asDegrees()
    dec_deg = sky_center.getLatitude().asDegrees()

    return ra_deg, dec_deg
