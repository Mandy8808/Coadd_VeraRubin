# v1.0
# plots/statistics_plots.py
# set of functions that plot visit results

import numpy as np
import matplotlib.pyplot as plt
import lsst.afw.display as afwDisplay

from astropy.time import Time
afwDisplay.setDefaultBackend('matplotlib')

###################################################################################################
def mjds_to_dates(mjd_list):
    """
    Convert a list of MJDs (Modified Julian Dates) to UTC calendar dates.
    """
    mjd_list = np.array(mjd_list)
    times = Time(mjd_list, format='mjd', scale='tai')
    return [t.to_datetime().date() for t in times]

def filt_plot(df_metrics, visits_selected, filt_cut_select, incr=1000,
              xysc=('id_plot', 'psfSigma_mean', 'psfSigma_std', 'airmass'),
              cmap='viridis', alpha=0.2, ec='red', linewidths=1.5, save=False):
    """
    Plot PSF, id and airmass give a filter data for the mean and std PSF and an airmass upperbound

    Parameters
    ----------
    df_metrics : Data Frame which columns correspond to:
                'id_plot', 'psfSigma_mean', 'psfSigma_std', 'airmass', ...
    visits_selected : Data Frame containing the selected visits
    filt_cut_select : Dictionary containing filter cut values for the plot
    incr : Increment value for marker size
    xysc : Tuple containing the names of the columns to be plotted on the x and y axes
    cmap : Colormap to be used for the scatter plot
    alpha : Alpha value for the scatter plot markers
    ec : Edge color for the scatter plot markers
    linewidths : Linewidths for the scatter plot markers
    save : Boolean or string indicating whether to save the plot
    """
    xticklabels = df_metrics["visit_id"]
    my_filter = df_metrics["band"].unique()
    size_visit = 8 if my_filter[0] in list('ugz') else 5
    
    fig, ax = plt.subplots(figsize=(15, 5))

    # Full visit
    x, y, s, c = [df_metrics[name] for name in xysc]
    sc = ax.scatter(x, y, s=s*incr, c=c, cmap=cmap, alpha=alpha)
    cbar = plt.colorbar(sc, ax=ax, label='airmass')

    # Select_visit
    x, y, s, c = [visits_selected[name] for name in xysc]
    ax.scatter(x, y, s=s*incr, fc='none', linewidths=linewidths, ec=ec) 

    xc, yc, s_c, cc = [filt_cut_select.get(name, None) for name in xysc]
    if yc:
        yc = float(yc.strip().split()[-1])
        ax.axhline(y=yc, lw=3, ls='--', color='k', alpha=0.5)
    if cc:
        cc = float(cc.strip().split()[-1])
        # norm = sc.norm(cc)
        # cbar.ax.hlines(norm, *cbar.ax.get_xlim(), color='red', linewidth=2)
        cbar.ax.hlines(cc, *cbar.ax.get_xlim(), color='red', linewidth=2)
    if s_c:
        pass

    ax.set_xticks(range(len(df_metrics)))
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=size_visit)

    ax.set_xticklabels(xticklabels, rotation = 90)
    ax.set_xlabel('Visit ID', fontsize=14)
    ax.set_ylabel('PSF size', fontsize=14)

    ax.set_xlim(0, len(df_metrics))

    fig.suptitle(my_filter[0] + " band", fontsize=16)

    # Save output
    if save:
        filename = save if isinstance(save, str) else 'filt_output.pdf'
        fig.savefig(
            filename, format='pdf', metadata=None,
            pad_inches=0.1, dpi=1000,
            bbox_inches='tight'
            )
        
    plt.show()
    return None

######### Display coadds
def plot_custom_coadd(fig, axes, custom_coadds, bands_to_plot=None, algorithm='asinh',
                      min='zscale', max=None, unit=None):
    """
    Plot custom coadds for selected bands.
    """
    if not custom_coadds:
        print("Warning: No custom coadds provided.")
        return fig, axes

    if bands_to_plot is None:
        bands_to_plot = list(custom_coadds.keys())
    elif isinstance(bands_to_plot, str):
        bands_to_plot = [bands_to_plot]

    axes = np.atleast_1d(axes)
    if len(bands_to_plot) != len(axes):
        print("Warning: Number of bands to plot does not match number of axes.")
        return fig, axes

    titles_Custom = [f"Custom Coadd ({b})" for b in bands_to_plot]

    for ax, title, band in zip(axes, titles_Custom, bands_to_plot):
        plt.sca(ax)
        display = afwDisplay.Display(frame=fig)
        display.scale(algorithm, min, max, unit)
        display.mtv(custom_coadds[band].image)
        display.show_colorbar(False)
        ax.axis('off')
        ax.set_title(title)

    return fig, axes

def plot_original_coadd(fig, axes, butler, my_tract, my_patch, bands_to_plot=None, skymap_name='lsst_cells_v1',
                        algorithm='asinh', min='zscale', max=None, unit=None):
    """
    Plot original coadds from Butler.
    """
    
    if bands_to_plot is None:
        print("Warning: No bands to plot specified.")
        return fig, axes
    elif isinstance(bands_to_plot, str):
        bands_to_plot = list(bands_to_plot)
    
    axes = np.atleast_1d(axes)
    if len(bands_to_plot) != len(axes):
        print("Warning: Number of bands to plot does not match number of axes.")
        return fig, axes

    titles_Original = [f"Original Coadd ({b})" for b in bands_to_plot]

    coaddId = {
        'band': None,
        'tract': my_tract,
        'patch': my_patch,
        'skymap': skymap_name
    }

    for ax, title, band in zip(axes, titles_Original, bands_to_plot):
        coaddId['band'] = band
        original_coadd = butler.get_dataset('deep_coadd', coaddId)
        plt.sca(ax)
        display = afwDisplay.Display(frame=fig)
        display.scale(algorithm, min, max, unit)
        display.mtv(original_coadd.image)
        display.show_colorbar(False)
        ax.axis('off')
        ax.set_title(title)

    return fig, axes

def plot_compare(butler=None, my_tract=None, my_patch=None,
                 custom_coadd=None, bands_to_plot=None, skymap_name='lsst_cells_v1',
                 algorithm='asinh', min='zscale', max=None, unit=None):
    """
    Compare original and custom coadds for the same bands.
    see: https://pipelines.lsst.io/modules/lsst.afw.display/index.html
    """
    if bands_to_plot is None:
        if custom_coadd:
            bands_to_plot = list(custom_coadd.keys())
        else:
            print("Warning: No bands specified.")
            return None
    elif isinstance(bands_to_plot, str):
        bands_to_plot = list(bands_to_plot)

    ncols = len(bands_to_plot)

    # Caso: Original + Custom
    if butler and custom_coadd:
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5*ncols, 8))
        axes = np.atleast_2d(axes)
        plot_original_coadd(fig, axes[0], butler, my_tract, my_patch, bands_to_plot, skymap_name, algorithm, min, max, unit)
        plot_custom_coadd(fig, axes[1], custom_coadd, bands_to_plot, algorithm, min, max, unit)
    # Only Original
    elif butler and not custom_coadd:
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5*ncols, 4))
        plot_original_coadd(fig, axes, butler, my_tract, my_patch, bands_to_plot, skymap_name, algorithm, min, max, unit)
    # Only Custom
    elif custom_coadd and not butler:
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5*ncols, 4))
        plot_custom_coadd(fig, axes, custom_coadd, 
                          bands_to_plot=bands_to_plot, algorithm=algorithm, min=min, max=max, unit=unit)
    else:
        print("Warning: No Butler or custom coadd provided.")
        return None

    plt.tight_layout()
    plt.show()
    return None

######### Display RGB coadds
def normalize_image(img, p1=None, p99=None):
    """Normalize an image array to [0, 1] using percentiles."""
    if p1 is None or p99 is None:
        p1, p99 = np.percentile(img, (1, 99))
    img_clip = np.clip((img - p1) / (p99 - p1), 0, 1)
    return img_clip, p1, p99

def make_rgb_image(coadd_dict, bands, p1_vals=None, p99_vals=None):
    """Create RGB composite from coadd dict and bands."""
    rgb = []
    for i, b in enumerate(bands):
        img = coadd_dict[b].image.array.astype(float)
        if p1_vals is not None and p99_vals is not None:
            img_norm, _, _ = normalize_image(img, p1_vals[i], p99_vals[i])
        else:
            img_norm, _, _ = normalize_image(img)
        rgb.append(img_norm)
    return np.dstack(rgb)

def compare_rgb_coadds(custom_coadd=None, butler=None, my_tract=None, my_patch=None,
                       skymap_name='lsst_cells_v1', bands=('g', 'r', 'i'),
                       titles=("Original Coadd", "Custom Coadd"), normalize_together=True):
    """
    Compare RGB composites between the original and custom coadds.
    """
    original_dict = None
    custom_dict = None

    # Load original coadd from Butler if provided
    if butler is not None:
        original_dict = {}
        coaddId = {
            'band': None,
            'tract': my_tract,
            'patch': my_patch,
            'skymap': skymap_name
        }
        for band in bands:
            coaddId['band'] = band
            original_dict[band] = butler.get_dataset('deep_coadd', coaddId)

    # Load custom coadd if provided
    if custom_coadd is not None:
        custom_dict = custom_coadd

    # Determine normalization percentiles if needed
    if normalize_together and original_dict is not None and custom_dict is not None:
        p1_vals = []
        p99_vals = []
        for b in bands:
            all_pixels = np.concatenate([
                original_dict[b].image.array.ravel(),
                custom_dict[b].image.array.ravel()
            ])
            p1, p99 = np.percentile(all_pixels, (1, 99))
            p1_vals.append(p1)
            p99_vals.append(p99)
    else:
        p1_vals = p99_vals = None

    # Build RGB images
    original_rgb = make_rgb_image(original_dict, bands, p1_vals, p99_vals) if original_dict else None
    custom_rgb = make_rgb_image(custom_dict, bands, p1_vals, p99_vals) if custom_dict else None

    # Decide layout
    imgs = []
    lbls = []
    if original_rgb is not None:
        imgs.append(original_rgb)
        lbls.append(titles[0])
    if custom_rgb is not None:
        imgs.append(custom_rgb)
        lbls.append(titles[1])

    ncols = len(imgs)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]  # make iterable

    for ax, img, title in zip(axes, imgs, lbls):
        ax.imshow(img, origin='lower')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Displayed RGB comparison using bands {bands} "
          f"(normalize_together={normalize_together}).")

######### Display CCDs and cutout region
def display_ccds_and_cutout(butler, my_tract, my_patch, skymap_name='lsst_cells_v1'):
    """
    Display the CCDs and cutout region for a given tract and patch.
    """
    # Get the skymap
    skymap = butler.get_dataset('skymap', skymap_name)

    # Get the CCDs for the specified tract and patch
    ccds = skymap.get_ccds(tract=my_tract, patch=my_patch)

    # Plot the CCDs
    fig, ax = plt.subplots(figsize=(10, 10))
    for ccd in ccds:
        ax.add_patch(plt.Rectangle((ccd.x0, ccd.y0), ccd.width, ccd.height,
                                     edgecolor='red', facecolor='none'))
    ax.set_title(f"CCDs for Tract {my_tract}, Patch {my_patch}")
    plt.show()