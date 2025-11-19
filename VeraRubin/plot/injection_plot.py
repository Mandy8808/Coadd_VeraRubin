# v1.0
# plots/injection_plot.py
# set of functions that plot injection results

import numpy as np
import matplotlib.pyplot as plt
import lsst.afw.display as afwDisplay
import lsst.geom

from astropy.wcs import WCS

afwDisplay.setDefaultBackend('matplotlib')

###################################################################################################
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
    return None

def injection_steps(before, after, points, diference=True,
                    grid=False, percentiles=[5, 95],
                    cutout_radius_arcsec=None,
                    xlim_world=None, ylim_world=None,
                    save_path=None, names=['Before', 'After', 'Difference']):
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
    save_path : str, optional
        If provided, the figure will be saved at this path instead of being displayed.
    names :  list, optional
        List of panel Name, default, ['Before', 'After', 'Difference']
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
        (before_data, names[0]),
        (after_data, names[1]),
        (after_data - before_data, names[2])
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
        p5, p95 = np.nanpercentile(data, percentiles)
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
    
    # Save or show the figure depending on argument
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()

    # Free memory after plotting
    plt.close(fig)
    return None

def pixel_intensity(image_array_list, y_positions_pixel, image_ref=False,
                    save_path=None, colormap='tab10', ind=0):
    """
    Plot the pixel intensity along the X-axis for given Y-pixel positions in an image.

    Parameters
    ----------
    image_array : numpy.ndarray
        2D image array containing pixel intensity values.
    y_positions_pixel : int or list of int
        Single Y-pixel position or a list of Y-pixel positions to analyze.
    image_ref : bool
        If True, the reference image will be shown at the first panel.
    save_path : str, optional
        If provided, the figure will be saved at this path instead of being displayed.
    colormap : str
        Name of the matplotlib colormap to assign distinct colors to each curve.
    ind : int, optional
        Index used as referential imagen from image_array_list
    """
    from matplotlib import cm

    # Ensure y_positions_pixel is a list (even if a single int was passed)
    if isinstance(y_positions_pixel, int):
        y_positions_pixel = [y_positions_pixel]

    # number of subplots to create: ref + len(y_positions_pixel)
    npanels = 1 + len(y_positions_pixel) if image_ref else len(y_positions_pixel)
      
    # Create subplots: one row with npanels columns
    fig, axes = plt.subplots(1, npanels, figsize=(npanels * 5, 4))

    # Ensure axes is iterable (if npanels == 1, make it a list)
    if npanels == 1:
        axes = [axes]
    
    # Colormap
    cmap = cm.get_cmap(colormap, len(y_positions_pixel))
    lines = ['-', ':', '.-','--']
    colors = []

    
    # Loop over each subplot axis
    axes_intensity = axes[1:] if image_ref else axes
    colors = []
    for i, ax in enumerate(axes_intensity):
        y_pixel = y_positions_pixel[i]

        # Plot the pixel intensity along the X-axis for the given Y position
        color = cmap(i)
        maxpixel = np.max(image_array_list)
        for j, image_array in enumerate(image_array_list):
            line = lines[j]
            ax.plot(image_array[y_pixel, :]/maxpixel, label=rf'Y-Pixel: {y_pixel}', color=color, ls=line)  # line = 
        colors.append(color)  # Save color for later use # old line[0].get_color()

        # Add legend without frame
        ax.legend(frameon=False)

        # Label axes
        ax.set_xlabel(r'X-pixel')
        ax.set_ylabel(r'Pixel Intensity Profile')
    
    if image_ref:
        vmin, vmax = np.nanpercentile(image_array_list[ind], 1), np.nanpercentile(image_array_list[ind], 99)
        axes[0].imshow(image_array_list[ind], origin='lower', cmap="gray", vmin=vmin, vmax=vmax)

        # draw the horizontal lines corresponding to the Y-pixel positions
        for (y_pixel, color) in zip(y_positions_pixel, colors):
            axes[0].axhline(y=y_pixel, color=color, linestyle='--', linewidth=2)
        
        axes[0].set_title("Reference Image", fontsize=12)
        axes[0].axis("off")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save or show the figure depending on argument
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")        
    else:
        plt.show()
    
    # Free memory after plotting
    plt.close(fig)
        
    return None

def plot_exposures_full(
        exposures,
        coadd_exp=None,
        center_coord=None,
        titles=None,
        axeslabels=None,
        save_path=None,
        show_second_row=False,
        show_histograms=False,
        add_colorbar=False,
        filter_nan_hist=True,
        exposures_scale='zscale_asinh',
        coadd_exp_scale='zscale_asinh',
        percentiles=[1, 99]):
    """
    Unified exposure/coadd plotting with:
      - optional sky coordinate marking
      - optional second row (variance images)
      - optional third row (histograms)
      - optional colorbars per image
      - optional NaN filtering before histograms

    Parameters
    ----------
    exposures : list of ExposureF
    coadd_exp : ExposureF, optional
    center_coord : lsst.afw.coord.Coord, optional
    titles : list[str], optional
    axeslabels : list[str], optional
    save_path : str, optional
    show_second_row : bool
    show_histograms : bool
    add_colorbar : bool
    filter_nan_hist : bool
        If True, removes NaN/Inf values before histogramming.
    exposures_scale : str
    coadd_exp_scale : str
    percentiles : list
    """
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch

    # helpers
    def get_array(exp, variance=False):
        """Return science or variance array, fallback to exp if plain numpy."""
        try:
            if variance:
                return exp.getMaskedImage().getVariance().getArray()
            return exp.getMaskedImage().getImage().getArray()
        except Exception as e:
            print(f"Warning: The data is not an exposure ({e})")
            return exp   # assume it's already a numpy array

    def show_image(ax, img, title, normalization, cmap, percentiles):
        """Render an image with chosen scaling."""
        if normalization == 'zscale_asinh':
            norm = ImageNormalize(img, interval=ZScaleInterval(),
                                  stretch=AsinhStretch())
            vmin, vmax = None, None
        else:  # percentile scaling
            norm = None
            vmin, vmax = (
                np.nanpercentile(img, percentiles[0]),
                np.nanpercentile(img, percentiles[1])
            )

        im = ax.imshow(img, origin='lower', cmap=cmap,
                       norm=norm, vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=11)
        ax.axis("off")
        return im

    n_exps = len(exposures)
    ncols = n_exps + (1 if coadd_exp is not None else 0)

    # Count rows
    nrows = 1
    if show_second_row:
        nrows += 1
    if show_histograms:
        nrows += 1

    # Frames
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    # enforce 2D array for axes
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Row indices
    SCIENCE_ROW = 0
    VAR_ROW = 1 if show_second_row else None
    HIST_ROW = (nrows - 1) if show_histograms else None

    # ROWS
    for i, exp in enumerate(exposures):
        img = get_array(exp)

        title = titles[i] if titles and i < len(titles) else f"Exposure {i+1}"
        im = show_image(axes[SCIENCE_ROW, i], img, title,
                        exposures_scale, "gray", percentiles)

        if add_colorbar:
            fig.colorbar(im, ax=axes[SCIENCE_ROW, i], fraction=0.046, pad=0.04)

        # mark coordinate
        if center_coord:
            try:
                cx, cy = exp.getWcs().skyToPixel(center_coord)
                # Check if the point lies within the image boundaries
                if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                    axes[SCIENCE_ROW, i].plot(cx, cy, 'r*', markersize=8)
            except Exception as e:
                print(f"[WARNING] skyToPixel failed: {e}")

    # coadd in science row
    if coadd_exp is not None:
        img = get_array(coadd_exp)
        im = show_image(axes[SCIENCE_ROW, -1], img, "Coadd",
                        coadd_exp_scale, "gray", percentiles)

        if add_colorbar:
            fig.colorbar(im, ax=axes[SCIENCE_ROW, -1], fraction=0.046, pad=0.04)

        if center_coord:
            try:
                cx, cy = coadd_exp.getWcs().skyToPixel(center_coord)
                if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                    axes[SCIENCE_ROW, -1].plot(cx, cy, 'r*', markersize=8)
            except Exception as e:
                print(f"[WARNING] skyToPixel failed (coadd): {e}")

    # VARIANCE ROW
    if show_second_row:
        for i, exp in enumerate(exposures):
            varimg = get_array(exp, variance=True)
            im = show_image(axes[VAR_ROW, i], varimg, f"Var {i+1}",
                            "percentile", "inferno", percentiles)

            if add_colorbar:
                fig.colorbar(im, ax=axes[VAR_ROW, i], fraction=0.046, pad=0.04)

        if coadd_exp is not None:
            varimg = get_array(coadd_exp, variance=True)
            im = show_image(axes[VAR_ROW, -1], varimg, "Coadd Var",
                            "percentile", "inferno", percentiles)

            if add_colorbar:
                fig.colorbar(im, ax=axes[VAR_ROW, -1], fraction=0.046, pad=0.04)

    # HISTOGRAM ROW
    if show_histograms:
        for i, exp in enumerate(exposures):
            img = get_array(exp)

            # filter NaNs and inf if requested
            if filter_nan_hist:
                data = img[np.isfinite(img)]
            else:
                data = img.flatten()

            if data.size == 0:
                axes[HIST_ROW, i].text(0.5, 0.5, "No finite data",
                                       ha='center', va='center')
            else:
                axes[HIST_ROW, i].hist(data, bins=100, histtype="step")

            axes[HIST_ROW, i].set_title(f"Hist {i+1}")
            axes[HIST_ROW, i].set_xlabel("Pixel value")

        # coadd histogram
        if coadd_exp is not None:
            img = get_array(coadd_exp)

            if filter_nan_hist:
                data = img[np.isfinite(img)]
            else:
                data = img.flatten()

            if data.size == 0:
                axes[HIST_ROW, -1].text(0.5, 0.5, "No finite data",
                                        ha='center', va='center')
            else:
                axes[HIST_ROW, -1].hist(data, bins=100, histtype="step")

            axes[HIST_ROW, -1].set_title("Coadd Hist")
            axes[HIST_ROW, -1].set_xlabel("Pixel value")

    # Axis labels
    if axeslabels and len(axeslabels) == 2:
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c].set_xlabel(axeslabels[0])
                axes[r, c].set_ylabel(axeslabels[1])

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)  # free memory
    return None

def normalize_exposures(exp_ref, exp_coadd, method="rescale"):
    """
    Normalize two LSST Exposure images to the same intensity range,
    returning full ExposureF objects with metadata preserved.

    Parameters
    ----------
    exp_ref : lsst.afw.image.ExposureF
        Reference exposure
    exp_coadd : lsst.afw.image.ExposureF
        Coadd exposure
    method : str, optional
        Normalization method:
        - "rescale": Rescale intensity of each image independently to [0,1]
        - "global_rescale": Rescale both using the global min and max
        - "zscore": Normalize to zero mean and unit variance

    Returns
    -------
    exp_ref_norm : lsst.afw.image.ExposureF
        Normalized reference exposure
    exp_coadd_norm : lsst.afw.image.ExposureF
        Normalized coadd exposure
    """

    # Convert Exposure -> NumPy arrays
    img_ref = exp_ref.getImage().getArray()
    img_coadd = exp_coadd.getImage().getArray()

    if method == "rescale":
        ref_norm = (img_ref - np.min(img_ref)) / (np.max(img_ref) - np.min(img_ref))
        coadd_norm = (img_coadd - np.min(img_coadd)) / (np.max(img_coadd) - np.min(img_coadd))

    elif method == "global_rescale":
        global_min = min(np.min(img_ref), np.min(img_coadd))
        global_max = max(np.max(img_ref), np.max(img_coadd))
        ref_norm = (img_ref - global_min) / (global_max - global_min)
        coadd_norm = (img_coadd - global_min) / (global_max - global_min)

    elif method == "zscore":
        ref_norm = (img_ref - np.mean(img_ref)) / np.std(img_ref)
        coadd_norm = (img_coadd - np.mean(img_coadd)) / np.std(img_coadd)

    else:
        raise ValueError("Invalid method. Choose 'rescale', 'global_rescale', or 'zscore'.")

    # Build normalized ExposureF with metadata preserved
    def build_normalized_exposure(exp_original, norm_array):
        # Make a deep copy of the original exposure
        exp_copy = exp_original.clone()

        # Replace only the image plane with normalized values
        exp_copy.getImage().getArray()[:, :] = norm_array.astype(np.float32)

        return exp_copy

    exp_ref_norm = build_normalized_exposure(exp_ref, ref_norm)
    exp_coadd_norm = build_normalized_exposure(exp_coadd, coadd_norm)

    return exp_ref_norm, exp_coadd_norm


########################################## OLD FUNCTIONS
def plot_exposures_and_coadd(exposures,
                             coadd_exp=None,
                             save_path=None,
                             show_second_row=False,
                             exposures_scale='zscale_asinh',
                             coadd_exp_scale='zscale_asinh'):
    """
    Plot individual exposures and optionally the coadd result, with optional variance row.

    Parameters
    ----------
    exposures : list of ExposureF
        List of exposures to plot.
    coadd_exp : ExposureF, optional
        Final coadded exposure to plot alongside.
    save_path : str, optional
        If provided, saves the figure to this path.
    show_second_row : bool
        If True, adds a second row showing variance maps.
    exposures_scale : str
        Scale used to normalize the exposures images: 'zscale_asinh' or 'percentile', default: 'zscale_asinh'.
    coadd_exp_scale : str
        Scale used to normalize the coadd image: 'zscale_asinh' or 'percentile', default: 'zscale_asinh'.
    """
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    
    n_exps = len(exposures)
    nrows = 2 if show_second_row else 1
    total_plots = n_exps + (1 if coadd_exp is not None else 0)

    # Helper: show science or variance image
    def show_image(ax, exp, title, normalization, cmap,
                   percentiles=[1, 99], origin='lower', image=True):
        img = exp.getMaskedImage().getImage().getArray() if image else exp.getMaskedImage().getVariance().getArray()
        if normalization == 'zscale_asinh':
            norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
            vmin, vmax = None, None
        elif normalization == 'percentile':
            norm = None
            vmin, vmax = np.nanpercentile(img, percentiles[0]), np.nanpercentile(img, percentiles[1])
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        im = ax.imshow(img, origin=origin, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        return im
    
    fig, axes = plt.subplots(nrows=nrows, ncols=total_plots,
                             figsize=(4*total_plots, 4*nrows))

    # Ensure axes is always 2D for consistent indexing
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Plot exposures
    for i in range(n_exps):
        show_image(axes[0, i], exposures[i],
                   f"Exposure {i+1} ({exposures_scale})",
                   normalization=exposures_scale, cmap="gray")
        if show_second_row:
            show_image(axes[1, i], exposures[i],
                       f"Var {i+1} (Percentile)", image=False,
                       normalization='percentile', cmap="inferno")

    # Plot coadd, if provided
    if coadd_exp is not None:
        show_image(axes[0, -1], coadd_exp,
                   f"Coadd ({coadd_exp_scale})",
                   normalization=coadd_exp_scale, cmap="gray")
        if show_second_row:
            show_image(axes[1, -1], coadd_exp,
                       "Coadd Var (Percentile)", image=False,
                       normalization='percentile', cmap="inferno")
        
    plt.tight_layout()

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)  # free memory
    return None

def plot_exposures_with_point(exposures, center_coord=None,
                              titles=None, axeslabels=None,
                              save_path=None, percentiles=[0.1, 99.9]):
    """
    Plot one or multiple LSST exposures side by side, optionally marking a sky coordinate.

    Parameters
    ----------
    exposures : list of `lsst.afw.image.Exposure`
        List of exposures to plot.
    center_coord : `lsst.afw.coord.Coord`, optional
        Celestial coordinate to mark in each exposure (red star).
    titles : list of str, optional
        Titles for each subplot.
    axeslabels : list of str, optional
        Axis labels as [xlabel, ylabel].
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    ncol = len(exposures)

    fig, axes = plt.subplots(1, ncol, figsize=(ncol*5, 4))

    if ncol == 1:
        axes = [axes]

    for ind, (ax, exp) in enumerate(zip(axes, exposures)):
        # imag = exp.getImage().getArray()
        try:
            imag = exp.getImage().getArray()
        except Exception as e:
            print(f"Warning: The data is not an exposure ({e})")
            imag = exp
        
        ax.imshow(imag, origin='lower', cmap='gray',
                  vmin=np.nanpercentile(imag, percentiles[0]),
                  vmax=np.nanpercentile(imag, percentiles[1]))

        if titles and ind < len(titles):
            ax.set_title(titles[ind])

        if axeslabels and len(axeslabels) == 2:
            ax.set_xlabel(axeslabels[0])
            ax.set_ylabel(axeslabels[1])

        if center_coord:
            center_point = exp.getWcs().skyToPixel(center_coord)

            # Check if the point lies within the image boundaries
            if (0 <= center_point.getX() < imag.shape[1] and
                0 <= center_point.getY() < imag.shape[0]):
                ax.plot(center_point.getX(), center_point.getY(),
                        'r*', markersize=8)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")
    else:
        plt.show()
        
    plt.close(fig)  # free memory
    return None
