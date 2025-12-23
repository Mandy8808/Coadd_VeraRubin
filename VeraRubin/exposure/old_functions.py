


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

def old_save_exposure(injected_exposure, 
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