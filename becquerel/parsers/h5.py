"""Read in a becquerel Spectrum HDF5 file."""

from .. import io
from ..core import calibration
from .parsers import BecquerelParserError


def read(filename, verbose=False):
    """Parse the becquerel Spectrum HDF5 file and return a dictionary of data.

    Parameters
    ----------
    filename : str
        The filename of the HDF5 file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.

    Returns
    -------
    data : dict
        Dictionary of data that can be used to instantiate a Spectrum.
    cal : Calibration
        Energy calibration stored in the file.
    """
    print("Reading HDF5 file " + filename)
    if not io.h5.is_h5_filename(filename):
        raise BecquerelParserError("File is not an HDF5: " + filename)

    # group datasets and attributes into one dictionary
    dsets, attrs, skipped = io.h5.read_h5(filename)
    data = {**dsets, **attrs}

    # read energy calibration from file
    if "energy_cal" in skipped:
        with io.h5.open_h5(filename, "r") as h5:
            group = h5["energy_cal"]
            cal = calibration.Calibration.read(group)
    else:
        cal = None

    return data, cal
