# verarubin/__init__.py

# Butler
from .butler.butler import ExpButler
from .butler.custom_butler import main_local_repo, create_empty_repo, transfer_all_visit_datasets,\
                           discover_datasets_for_visit, instrument_register_from_remote, register_datasetTypes,\
                           skymap_register_from_remote, ensure_chained_collection, transfer_visits
# Coadd
from .coadd.custom_coadd import custom_coadd_filter, custom_coadd_multiband, load_custom_coadd_from_file
from .coadd.custom_inject_coadd import coadd_exposures_pipeline, coadd_exposures_pipeline, leave_one_out_residual, validate_rotation
from .coadd.custom_warp import custom_warp, select_visits, runDirectWarpTask

# exposure
from exposure.exposure import load_exposures, save_exposure, normalize_exposures, exposure_to_fits_datahdr, cutout_exposure
# Fits
from fits.fits import fits_to_exposure, cutout_fits

# Plots
from .plot.plot_conf import general, FigParam, LineParam, axesParam, labelParam, legendParam, fontParam, get_colors
from .plot.statistics_plot import StatisticsPlots
from .plot.array_plot import pixel_intensity
from .plot.butler_plot import filt_plot, display_ccds_and_cutout, plot_compare
from .plot.coadd_plot import plot_custom_coadd, plot_original_coadd, normalize_image, make_rgb_image, compare_rgb_coadds
from .plot.exposure_plot import fix_wcsaxes_labels, extract_array, normalize_axes, render_image, overlay_sky_point,\
    plot_histogram, injection_steps, plot_exposures_full

# Sky
from .sky.sky import tract_patch, patch_center, get_patch_center_radius, RA_to_degree, Dec_to_degree, skywcs_to_astropy

# Injection
from source_injection.injection import make_serializable, measure_quality, create_crowded_injection_catalog, apply_correction_from_data
from source_injection.injection import apply_correction_to_stamp, inject_stamp, main_inject_stamp, apply_correction_from_exposureF

# Tools
from tools.tools import progressbar, setup_logger, _run, get_butler_location, mjds_to_dates

# Visit
from visit.visit import Visit, combine_visits_selected, visit_dataset

__all__ = [
    # ExpButler
    'ExpButler', 
    'main_local_repo', 'create_empty_repo', 'transfer_all_visit_datasets',
    'discover_datasets_for_visit', 'instrument_register_from_remote', 'register_datasetTypes',
    'skymap_register_from_remote', 'ensure_chained_collection', 'transfer_visits',
    # Coadd
    'custom_coadd_filter', 'custom_coadd_multiband', 'load_custom_coadd_from_file',
    'coadd_exposures_pipeline', 'leave_one_out_residual', 'validate_rotation',
    'custom_warp', 'select_visits', 'runDirectWarpTask',
    # Exposure
    'load_exposures', 'save_exposure', 'normalize_exposures', 'exposure_to_fits_datahdr', 'cutout_exposure',
    # Fits
    'fits_to_exposure', 'cutout_fits',
    # Plots
    'general', 'FigParam', 'LineParam', 'axesParam', 'labelParam', 'legendParam', 'fontParam', 'get_colors', 'StatisticsPlots',
    'pixel_intensity', 'filt_plot', 'display_ccds_and_cutout', 'plot_compare', 'plot_custom_coadd', 'plot_original_coadd',
    'normalize_image', 'make_rgb_image', 'compare_rgb_coadds', 'fix_wcsaxes_labels', 'extract_array', 'normalize_axes', 'render_image',
    'overlay_sky_point', 'plot_histogram', 'injection_steps', 'plot_exposures_full',
    # Sky
    'tract_patch', 'patch_center', 'get_patch_center_radius', 'RA_to_degree', 'Dec_to_degree', 'skywcs_to_astropy',
    # Injection
    'make_serializable', 'measure_quality', 'create_crowded_injection_catalog', 'apply_correction_from_data',
    'apply_correction_to_stamp', 'inject_stamp', 'main_inject_stamp', 'apply_correction_from_exposureF',
    # Tools
    'progressbar', 'setup_logger', '_run', 'get_butler_location', 'mjds_to_dates',
    # Visit
    'Visit', 'combine_visits_selected', 'visit_dataset'
]