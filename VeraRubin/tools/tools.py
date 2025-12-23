# vera rubin v1.0
# tools.tools.py
# set of utility functions for various tasks

import numpy as np
import subprocess
import logging
import sys

from lsst.daf.butler import Butler
from astropy.time import Time


###################################################################################################
def mjds_to_dates(mjd_list):
    """
    Convert a list of MJDs (Modified Julian Dates) to UTC calendar dates.
    """
    mjd_list = np.array(mjd_list)
    times = Time(mjd_list, format='mjd', scale='tai')
    return [t.to_datetime().date() for t in times]

def get_butler_location(butler: Butler) -> str | bool:
    """Return repository path for a local Butler or parsed location for remote Butler."""
    try:
        return butler.repo   # Local Butler
    except AttributeError:
        import re
        # Remote Butler: parse str(butler)
        s = str(butler)
        match = re.search(r"RemoteButler\((.+?)\)", s)
        if match:
            return match.group(1)
        else:
            return False  # Could not determine location

def _run(cmd: list[str], logger: logging.Logger = None) -> subprocess.CompletedProcess:
    """
    Run a subprocess command, log stdout/stderr and raise if it fails.
    Returns the CompletedProcess on success.
    """
    msg = "[CMD] " + " ".join(cmd)
    if logger:
        logger.info(msg)
    else:
        print(msg)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if logger:
            logger.info(result.stdout.strip())
            if result.stderr.strip():
                logger.warning(result.stderr.strip())
        else:
            print(result)
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Command failed: {e}")
            logger.error(e.stderr)
        else:
            print("Command failed:", e)
            print(e.stderr)
        raise

def setup_logger(logfile_path: str, name: str = 'pipeline.log') -> logging.Logger:
    """Create a logger that writes DEBUG to file and INFO to console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Log to file
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def progressbar(current_value, total_value, bar_length=20, progress_char='#'): 
    """
    Display a progress bar in the console.
    
    Parameters
    ----------
    current_value : int
        Current progress value.
    total_value : int
        Total value for completion.
    bar_length : int, optional
        Length of the progress bar.
    progress_char : str, optional
        Character used to fill the progress bar.
    """
    if total_value == 0:
        print("Error: total_value cannot be 0")
        return
    
    # Calculate the percentage and progress
    percentage = int((current_value / total_value) * 100)
    progress = int((bar_length * current_value) / total_value)
    
    # Build the progress bar string
    loadbar = f"Progress: [{progress_char * progress}{'.' * (bar_length - progress)}] {percentage}%"
    
    # Print the progress bar (overwrite line until finished)
    end_char = '\r' if current_value < total_value else '\n'
    print(loadbar, end=end_char)