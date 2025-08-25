# verarubin/__init__.py

# Butler
from .butler.butler import ExpButler

# Coadd
from .coadd.custom_coadd import custom_coadd_filter, custom_coadd_multiband, load_custom_coadd_from_file
from .coadd.custom_inject_coadd import coadd_exposures_pipeline, load_exposures

# Plots
from .plot.plot_conf import general, FigParam, LineParam, axesParam, labelParam, legendParam, fontParam, get_colors
from .plot.statistics_plot import StatisticsPlots
from .plot.visit_plot import mjds_to_dates, filt_plot
from .plot.visit_plot import plot_custom_coadd, plot_original_coadd, plot_compare
from .plot.visit_plot import normalize_image, make_rgb_image, compare_rgb_coadds
from .plot.injection_plot import skywcs_to_astropy, fix_wcsaxes_labels, injection_steps, pixel_intensity,\
                            plot_exposures_and_coadd

# Sky
from .sky.sky import tract_patch, patch_center, get_patch_center_radius, RA_to_degree, Dec_to_degree

# Visit
from visit.visit import Visit, combine_visits_selected

# Injection
from source_injection.injection import make_serializable, measure_quality, create_crowded_injection_catalog, apply_correction_from_data
from source_injection.injection import apply_correction_to_stamp, inject_stamp, main_inject_stamp, visit_dataset, apply_correction_from_exposureF

# Tools
from tools.tools import progressbar

__all__ = [
    # ExpButler
    'ExpButler',
    # Coadd
    'custom_coadd_filter', 'custom_coadd_multiband', 'load_custom_coadd_from_file',
    'coadd_exposures_pipeline', 'load_exposures',
    # Plots
    'general', 'FigParam', 'LineParam', 'axesParam', 'labelParam', 'legendParam', 'fontParam', 'get_colors',
    'StatisticsPlots', 'mjds_to_dates', 'filt_plot', 'plot_custom_coadd', 'plot_original_coadd', 'plot_compare',
    'normalize_image', 'make_rgb_image', 'compare_rgb_coadds', 'injection_steps', 'skywcs_to_astropy', 'fix_wcsaxes_labels',
    'pixel_intensity', 'plot_exposures_and_coadd',
    # Sky
    'tract_patch', 'patch_center', 'get_patch_center_radius', 'RA_to_degree', 'Dec_to_degree',
    # Visit
    'Visit', 'combine_visits_selected',
    # Injection
    'make_serializable', 'measure_quality', 'create_crowded_injection_catalog', 'apply_correction_from_data',
    'apply_correction_to_stamp', 'inject_stamp', 'main_inject_stamp', 'visit_dataset', 'apply_correction_from_exposureF',
    # Tools
    'progressbar'
]