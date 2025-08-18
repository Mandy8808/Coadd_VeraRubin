# v1.0
# plots/injection_plot.py
# set of functions that plot injection results

import numpy as np
import matplotlib.pyplot as plt
import lsst.afw.display as afwDisplay
import lsst.geom

from astropy.wcs import WCS

afwDisplay.setDefaultBackend('matplotlib')


#######################################################
def skywcs_to_astropy(lsst_wcs):
    """Convert LSST SkyWcs to Astropy WCS."""
    md = lsst_wcs.getFitsMetadata()  # returns PropertyList with FITS WCS keywords
    header = {}
    for key in md.names():
        try:
            header[key] = md.getScalar(key)
        except Exception:
            continue
    return WCS(header)

def fix_wcsaxes_labels(ax):
    """
    Automatically fix RA/Dec axis formatting in WCSAxes.

    Parameters
    ----------
    ax : WCSAxes
        Matplotlib axis with WCS projection.
    """
    from astropy import units as u
    
    for i, coord in enumerate(ax.coords):
        ctype = coord.coord_type.upper()  # usually "RA" or "DEC"
        # print(ctype)
        if "RA" in ctype:
            coord.set_format_unit(u.hourangle)
            coord.set_axislabel("RA", fontsize=12)
        elif "DEC" in ctype:
            coord.set_format_unit(u.deg)
            coord.set_axislabel("Dec", fontsize=12)
        else:
            # fallback: just leave as degrees
            coord.set_format_unit(u.deg)
            coord.set_axislabel(ctype)

def injection_steps(before, after, points, diference=True,
                    grid=False, percentiles=[5, 95],
                    cutout_radius_arcsec=None,
                    xlim_world=None, ylim_world=None):
    """
    Compare exposures before/after injection and plot with WCS coordinates.

    Automatically detects whether inputs are LSST ExposureF objects
    or Astropy (FITS/CCDData-like) objects.

    Parameters
    ----------
    before : lsst.afw.image.ExposureF or Astropy object
        Exposure before injection.
    after : lsst.afw.image.ExposureF or Astropy object
        Exposure after injection.
    points : list of [ra, dec]
        Positions of injected sources in degrees.
    grid : bool, optional
        If True, overlay a coordinate grid.
    percentiles : list, optional
        Percentiles for image scaling (default = [5, 95]).
    cutout_radius_arcsec : float, optional
        If set, zoom around the first injected point by this radius (arcsec).
    xlim_world : tuple, optional
        Manual RA limits (deg), e.g. (RA_min, RA_max).
    ylim_world : tuple, optional
        Manual Dec limits (deg), e.g. (Dec_min, Dec_max).
    """
    labelpoint = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Detect type automatically
    if hasattr(after, "getWcs"):  # LSST object
        try:
            wcs_for_plot = after.getWcs()
            # Convert to Astropy WCS for matplotlib
            wcs_for_plot = skywcs_to_astropy(wcs_for_plot)
            print("[INFO] Converted LSST SkyWcs -> Astropy WCS")
        except Exception as e:
            raise TypeError(f"Failed to convert LSST SkyWcs to Astropy WCS: {e}")
        before_data = before.image.array
        after_data = after.image.array
    elif hasattr(after, "header") and hasattr(after, "data"):  # Astropy object
        print("[INFO] Detected Astropy data with WCS")
        wcs_for_plot = WCS(after.header)
        before_data = before.data
        after_data = after.data
    else:
        raise TypeError("Unsupported input type. Must be LSST ExposureF or Astropy HDU/CCDData.")

    # Build image panels
    images = [
        (before_data, 'Before'),
        (after_data, 'After'),
        (after_data - before_data, 'Difference')
    ]

    # Plotting
    n = 3 if diference else 2
    fig = plt.figure(figsize=(n*5, 5))
    for i, (data, title_str) in enumerate(images[:n:], start=1):
        ax = fig.add_subplot(1, 3, i, projection=wcs_for_plot)

        # Plot injection points
        for label, (ra_deg, dec_deg) in zip(labelpoint, points):
            ax.scatter(ra_deg, dec_deg,
                       transform=ax.get_transform('world'),
                       edgecolor='red', facecolor='None')
            ax.text(ra_deg, dec_deg, label,
                    transform=ax.get_transform('world'),
                    color='yellow', fontsize=12, weight='bold')

        # Contrast scaling
        p5, p95 = np.percentile(data, percentiles)
        im = ax.imshow(data, origin='lower', cmap='gray', vmin=p5, vmax=p95)

        ax.set_title(title_str)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')

        if grid:
            ax.grid(color='white', ls='dotted')

        plt.colorbar(im, ax=ax, label='Intensity', pad=0.05, shrink=0.6)

        ax.set_xlim()

        # Fix RA/Dec formatting automatically
        # fix_wcsaxes_labels(ax)

        
        # Zoom options
        # Automatic
        if cutout_radius_arcsec is not None:
            # Take first point
            ra0, dec0 = points[0]
            radius_deg = cutout_radius_arcsec / 3600.0  # arcsec → degrees

            # World → pixel
            x_center, y_center = wcs_for_plot.world_to_pixel_values(ra0, dec0)

            # Approx pixel scale (deg/pixel)
            scale_deg = np.mean(np.abs(wcs_for_plot.pixel_scale_matrix.diagonal()))
            radius_pix = radius_deg / scale_deg

            ax.set_xlim(x_center - radius_pix, x_center + radius_pix)
            ax.set_ylim(y_center - radius_pix, y_center + radius_pix)

        # Manual zoom overrides automatic
        if xlim_world is not None:
            ax.set_xlim(*xlim_world)
        if ylim_world is not None:
            ax.set_ylim(*ylim_world)

    plt.tight_layout()
    return None