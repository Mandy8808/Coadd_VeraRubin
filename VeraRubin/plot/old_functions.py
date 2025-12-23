


def old_plot_exposures_full(
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
        percentiles=[1, 99],
        grid=True,
        proyection=None):
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

    def show_image(ax, img, title, normalization, cmap, percentiles, proyection=None):
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

        #if not proyection:
        #    ax.axis("off")
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), subplot_kw={'projection': proyection})

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
                        exposures_scale, "gray", percentiles, proyection=proyection)

        if add_colorbar:
            fig.colorbar(im, ax=axes[SCIENCE_ROW, i], fraction=0.046, pad=0.04)

        if grid and proyection:
            axes[SCIENCE_ROW, i].coords.grid(color='white', ls='dotted')

        # mark coordinate
        if center_coord:
            try:
                cx, cy = exp.getWcs().skyToPixel(center_coord)
                bbox = exp.getBBox()
                x0 = bbox.getMinX()
                y0 = bbox.getMinY()

                x_plot = cx - x0
                y_plot = cy - y0
                
                # Check if the point lies within the image boundaries
                if 0 <= x_plot < img.shape[1] and 0 <= y_plot < img.shape[0]:
                    axes[SCIENCE_ROW, i].plot(x_plot, y_plot, 'r*', markersize=8)
            except Exception as e:
                print(f"[WARNING] skyToPixel failed: {e}")

    # coadd in science row
    if coadd_exp is not None:
        img = get_array(coadd_exp)
        im = show_image(axes[SCIENCE_ROW, -1], img, "Coadd",
                        coadd_exp_scale, "gray", percentiles, proyection=proyection)

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
