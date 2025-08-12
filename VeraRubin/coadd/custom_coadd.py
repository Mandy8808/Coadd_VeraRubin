
# vera rubin v1.0
# coadd.custom_coadd.py
import os, sys
import getpass
import json

from lsst.ctrl.mpexec import SimplePipelineExecutor
from lsst.pipe.base import Pipeline

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visit.visit import Visit, combine_visits_selected
from sky.sky import tract_patch
from plot.visit_plot import filt_plot

def custom_coadd_filter(loc_data, bands='ugrizy', sky_coordinates=True, repository="dp1",
                        collections="LSSTComCam/DP1", butler=None,
                        type_coadd='deep_coadd',
                        statistics={'std', 'mean'}, 
                        selection={'u': ['psfSigma', 'airmass'], 'g': ['psfSigma', 'airmass'], 'r': ['psfSigma', 'airmass'],
                                   'i': ['psfSigma', 'airmass'], 'z': ['psfSigma', 'airmass'], 'y': ['psfSigma', 'airmass']},
                        filt_cut={'u': None, 'g': None, 'r': None, 'i': None, 'z': None, 'y': None}, 
                        n_visits={'u': None, 'g': None, 'r': None, 'i': None, 'z': None, 'y': None},
                        my_collection_identifier='custom_coadd',
                        repo_name='local_repo', info=True, out=True, plot=False):
    """
    Run filtering and coaddition over multiple photometric bands.

    This function applies filtering criteria to visits per band using Visit.filt_visit,
    and optionally runs a custom coaddition pipeline for the selected visits.

    Parameters
    ----------
    loc_data : tuple
        Either (RA, Dec) in degrees or (tract, patch) integers depending on `sky_coordinates`.
    bands : str or list of str
        Photometric bands to process (e.g., 'ugrizy' or ['r', 'i']).
    sky_coordinates : bool
        If True, loc_data is interpreted as sky coordinates (RA, Dec).
    repository : str
        Input repository root.
    collections : str
        Input collections string for the Butler.
    butler : lsst.daf.butler.Butler, optional
        Existing Butler instance. If None, one will be created.
    type_coadd : str
        Type of coadd to inspect for visit selection.
    statistics : set
        Statistics to compute ('mean', 'std', etc.).
    selection : dict
        Selection keys per band (dict of lists of column names).
    filt_cut : dict
        Optional filtering conditions per band (e.g., {'r': {'airmass_mean': '< 1.5'}}).
    n_visits : dict
        Max number of visits per band (e.g., {'r': 10}).
    my_collection_identifier : str
        Name for the output collection.
    repo_name : str
        Local repo directory to store outputs.
    info : bool
        If True, print debug information.
    out : bool
        If True, run and return custom coadd results.

    Returns
    -------
    visits_selected_list : list of pd.DataFrame
        Filtered visits per band.
    df_metrics_list : list of pd.DataFrame
        Computed statistics per band.
    coadd_results : dict (if out=True)
        Deep coadd results per band.
    """
    
    # Initialize Butler if not provided
    if butler:
        butler = butler
    else:
        from butler.butler import ExpButler
        butler = ExpButler(repository=repository, collections=collections)._create_butler()

    # Ensure bands is a list
    if isinstance(bands, str):
        bands = list(bands)
            
    visits_selected_list = []
    df_metrics_list = []

    # Visit filtering per band
    for band in bands:
        visit_instance = Visit(loc_data, band, butler=butler, sky_coordinates=sky_coordinates)
        df_metrics, visits_selected = visit_instance.filt_visit(
            statistics=statistics,
            type_coadd=type_coadd,
            selection=selection.get(band, []),
            filt_cut=filt_cut.get(band),
            n_visits=n_visits.get(band)
        )
        if plot:
            filt_plot(df_metrics, visits_selected, filt_cut.get(band),
              incr=1000, save=False)
        df_metrics_list.append(df_metrics)
        visits_selected_list.append(visits_selected)

    # Combine visits into single DataFrame
    visits_selected_comb = combine_visits_selected(visits_selected_list)

    # Run coadd if requested
    if out:
        coadd_results = custom_coadd_multiband(
            butler, visits_selected_comb, loc_data,
            sky_coordinates=sky_coordinates,
            my_collection_identifier=my_collection_identifier,
            repo_name=repo_name,
            bands=bands,
            info=info,
            out=True
        )
        return visits_selected_list, df_metrics_list, coadd_results
    else:
        custom_coadd_multiband(
            butler, visits_selected_comb, loc_data,
            sky_coordinates=sky_coordinates,
            my_collection_identifier=my_collection_identifier,
            repo_name=repo_name,
            bands=bands,
            info=info,
            out=False
        )
        return visits_selected_list, df_metrics_list, None


def custom_coadd_multiband(butler, visits_selected, loc_data,
                           my_collection_identifier='custom_coadd',
                           repo_name='local_repo', bands=None, info=True, out=True,
                           sky_coordinates=True, skymap='lsst_cells_v1'):
    """
    Run a custom coaddition pipeline for multiple bands.

    Parameters
    ----------
    butler : lsst.daf.butler.Butler
        LSST Butler with the input repository.
    visits_selected : pandas.DataFrame
        DataFrame with columns ['visit_id', 'band'].
    loc_data : tuple
                Either (RA, Dec) in degrees, or (tract, patch) integers.
    my_collection_identifier : str, optional
        Output collection name under u/<user>/.
    repo_name : str, optional
        Local folder to store the writable repo.
    bands : list of str, optional
        Bands to process (e.g. ['r', 'i', 'z']).
    info : bool, optional
        Print debug information if True.
    out : bool, optional
        If True, return dictionary of coadd outputs by band.
    sky_coordinates : bool, optional
        If True, interpret loc_data as (RA, Dec); else as (tract, patch).
    skymap : str, optional
        Name of the skymap to use for the coadd.
    
    Returns
    -------
    dict
        Dictionary mapping each band to its corresponding deep coadd, if `out=True`.
        If `out=False`, returns None.
    """

    # Set up user-specific output collection
    my_username = getpass.getuser()
    my_outputCollection = f'u/{my_username}/{my_collection_identifier}'

    # Get tract and patch either from sky coords or directly
    if sky_coordinates:
        ra_deg, dec_deg = loc_data
        my_tract, my_patch = tract_patch(butler, ra_deg, dec_deg, sequential_index=True)
    else:
        my_tract, my_patch = loc_data

    # Define pipeline URI and steps
    # Custom pipeline with a Uniform Resource Identifier (URI) that consists of the name of the DP1 DRP pipeline's YAML file 
    # followed by # and then a comma-separated list of the four processing steps to be executed.
    drp_yaml_file = "$DRP_PIPE_DIR/pipelines/LSSTComCam/DRP-v2-compat.yaml"
    uri = drp_yaml_file + "#makeDirectWarp,assembleDeepCoadd,makePsfMatchedWarp,selectDeepCoaddVisits"
    pipeline = Pipeline.from_uri(uri)

    # Override configs
    # The DP1 dataset available to users does not contain all intermediate data products.
    # To work around this, use pipeline configuration overrides to ensure that custom coaddition 
    # will draw needed metadata information directly from the calibrated exposures rather than other intermediate products
    pipeline.addConfigOverride('makeDirectWarp', 'useVisitSummaryPsf', False)
    pipeline.addConfigOverride('makeDirectWarp', 'useVisitSummaryPhotoCalib', False)
    pipeline.addConfigOverride('makeDirectWarp', 'useVisitSummaryWcs', False)
    pipeline.addConfigOverride('makeDirectWarp', 'connections.calexp_list', 'visit_image')

    # Auto-detect bands if not provided
    if bands is None:
        bands = visits_selected['band'].unique()

    # Extract visit IDs for all bands
    visit_ids_all = []
    for band in bands:
        if band in visits_selected['band'].unique():
            visit_ids_band = visits_selected.query(f"band == '{band}'")['visit_id'].tolist()
            visit_ids_all.extend(visit_ids_band)

    if not visit_ids_all:
        raise ValueError("No visits to process.")

    # Construct query string for input dataset filtering
    visit_ids_str = "(" + ",".join(map(str, visit_ids_all)) + ")"
    query_string = (
        f"tract = {my_tract} AND patch = {my_patch} "
        f"AND visit IN {visit_ids_str} AND skymap = '{skymap}'"
    )

    if info:
        print(f"[INFO] Output collection: {my_outputCollection}")
        print(f"[INFO] Query: {query_string}")
        print(f"[INFO] Bands: {bands}")


    # Set up executor
    executor = SimplePipelineExecutor.from_pipeline(
        pipeline, butler=butler,
        output=my_outputCollection,
        where=query_string
    )

    # Making a writable local repository within your current directory. 
    # The call to use_local_butler creates a new directory that will contain the local Butler repository.
    local_repo_path = os.path.join(os.getenv("HOME"), repo_name)
    out_butler = executor.use_local_butler(local_repo_path)

    # Saving the custom coadd
    try:
        # Run pipeline
        executor.run(register_dataset_types=True)
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return {}

    # Optionally return deep coadd results
    if out:
        coadd_results = {}
        for band in bands:
            try:
                coadd = out_butler.get(
                    'deep_coadd_predetection',
                    tract=my_tract,
                    patch=int(my_patch),
                    band=band,
                    instrument='LSSTComCam',
                    skymap=skymap,
                    collections=executor.quantum_graph.metadata["output_run"]
                )
                coadd_results[band] = coadd
            except Exception as e:
                print(f"[WARNING] Could not get coadd for band '{band}': {e}")
        # saving the coadd-metadata on a file
        coadd_info = {
            'repo_path': local_repo_path,
            'collection': executor.quantum_graph.metadata["output_run"],
            'coadd_type': 'deep_coadd_predetection',
            'tract': my_tract,
            'patch': str(my_patch),
            'bands': list(bands),
            'instrument': 'LSSTComCam',
            'skymap': skymap
            }

        save_path = os.path.join(local_repo_path, "custom_coadd_info.txt")
        with open(save_path, "w") as f:
            json.dump(coadd_info, f, indent=4)
    
        print(f"[INFO] Saved coadd metadata to: {save_path}")
        
        return coadd_results
    else:
        # saving the coadd-metadata on a file
        coadd_info = {
            'repo_path': local_repo_path,
            'collection': executor.quantum_graph.metadata["output_run"],
            'coadd_type': 'deep_coadd_predetection',
            'tract': my_tract,
            'patch': str(my_patch),
            'bands': list(bands),
            'instrument': 'LSSTComCam',
            'skymap': skymap
            }

        save_path = os.path.join(local_repo_path, "custom_coadd_info.txt")
        with open(save_path, "w") as f:
            json.dump(coadd_info, f, indent=4)
    
        print(f"[INFO] Saved coadd metadata to: {save_path}")
        print(f"[INFO] Coadd pipeline completed, results saved.")

    return None


def load_custom_coadd_from_file(info_txt_path):
    from butler.butler import ExpButler

    with open(info_txt_path, "r") as f:
        info = json.load(f)

    butler =  ExpButler(repository=info['repo_path'], collections=info['collection'])._create_butler()

    results = {}
    for band in info['bands']:
        try:
            coadd = butler.get(
                info['coadd_type'],
                tract=info['tract'],
                patch=int(info['patch']),
                band=band,
                instrument=info['instrument'],
                skymap=info['skymap'],
                collections=info['collection']
            )
            results[band] = coadd
        except Exception as e:
            print(f"[WARNING] Could not load coadd for band {band}: {e}")

    return results