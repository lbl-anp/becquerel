"""Read in a becquerel Spectrum HDF5 file."""

from .. import io
from ..core import calibration
from .parsers import BecquerelParserError, override_calibration


def read(filename, verbose=False, cal_kwargs=None):
    """Parse the becquerel Spectrum HDF5 file and return a dictionary of data.

    Parameters
    ----------
    filename : str | pathlib.Path
        The filename of the HDF5 file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.
    cal_kwargs : dict or None (optional)
        Kwargs to override the Calibration parameters read from file.

    Returns
    -------
    data : dict
        Dictionary of data that can be used to instantiate a Spectrum.
    cal : Calibration
        Energy calibration stored in the file.
    """
    print(f"Reading HDF5 file {filename}")
    if not io.h5.is_h5_filename(filename):
        raise BecquerelParserError(f"File is not an HDF5: {filename}")

    # group datasets and attributes into one dictionary
    dsets, attrs, skipped = io.h5.read_h5(filename)
    data = {**dsets, **attrs}

    # read energy calibration from file
    if cal_kwargs is None:
        cal_kwargs = {}
    if "energy_cal" in skipped:
        with io.h5.open_h5(filename, "r") as h5:
            group = h5["energy_cal"]
            cal = calibration.Calibration.read(group)
            cal = override_calibration(cal, **cal_kwargs)
    else:
        cal = None

    return data, cal
