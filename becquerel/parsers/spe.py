"""Read in an Ortec SPE file."""

import os
import warnings
import dateutil.parser
import numpy as np
from ..core import calibration
from .parsers import BecquerelParserError, BecquerelParserWarning


def read(filename, verbose=False):
    """Parse the ASCII SPE file and return a dictionary of data.

    ORTEC's SPE file format is given on page 73 of this document:
        http://www.ortec-online.com/download/ortec-software-file-structure-manual.pdf

    Parameters
    ----------
    filename : str
        The filename of the CNF file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.

    Returns
    -------
    data : dict
        Dictionary of data that can be used to instantiate a Spectrum.
    cal : Calibration
        Energy calibration stored in the file.
    """
    print("SpeFile: Reading file " + filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".spe":
        raise BecquerelParserError("File extension is incorrect: " + ext)

    # initialize a dictionary of spectrum data to populate as we parse
    data = {}

    # parse the file line-by-line
    collection_start = None
    livetime = None
    realtime = None
    counts = []
    channels = []
    cal_coeff = []
    with open(filename) as f:
        # read & remove newlines from end of each line
        lines = [line.strip() for line in f.readlines()]
        i = 0
        while i < len(lines):
            # check whether we have reached a keyword and parse accordingly
            if lines[i] == "$DATE_MEA:":
                i += 1
                collection_start = dateutil.parser.parse(lines[i])
                if verbose:
                    print(collection_start)
            elif lines[i] == "$MEAS_TIM:":
                i += 1
                livetime = float(lines[i].split(" ")[0])
                realtime = float(lines[i].split(" ")[1])
                if verbose:
                    print(livetime, realtime)
            elif lines[i] == "$DATA:":
                i += 1
                first_channel = int(lines[i].split(" ")[0])
                # I don't know why it would be nonzero
                if first_channel != 0:
                    raise BecquerelParserError(
                        f"First channel is not 0: {first_channel}"
                    )
                num_channels = int(lines[i].split(" ")[1])
                if verbose:
                    print(first_channel, num_channels)
                j = first_channel
                while j <= num_channels + first_channel:
                    i += 1
                    counts = np.append(counts, int(lines[i]))
                    channels = np.append(channels, j)
                    j += 1
            elif lines[i] == "$MCA_CAL:":
                i += 1
                n_coeff = int(lines[i])
                i += 1
                for j in range(n_coeff):
                    cal_coeff.append(float(lines[i].split(" ")[j]))
                if verbose:
                    print(cal_coeff)
            elif lines[i].startswith("$"):
                key = lines[i][1:].rstrip(":")
                i += 1
                values = []
                while i < len(lines) and not lines[i].startswith("$"):
                    values.append(lines[i])
                    i += 1
                if i < len(lines):
                    if lines[i].startswith("$"):
                        i -= 1
                if len(values) == 1:
                    values = values[0]
                data[key] = values
            else:
                warnings.warn(
                    f"Line {i + 1} unknown: " + lines[i],
                    BecquerelParserWarning,
                )
            i += 1

    # check the data that were read
    if collection_start is None:
        raise BecquerelParserError("Start time not found")
    if livetime is None:
        raise BecquerelParserError("Live time not found")
    if realtime is None:
        raise BecquerelParserError("Real time not found")

    if realtime <= 0.0:
        raise BecquerelParserError(f"Real time not parsed correctly: {realtime}")
    if livetime <= 0.0:
        raise BecquerelParserError(f"Live time not parsed correctly: {livetime}")
    if livetime > realtime:
        raise BecquerelParserError(f"Livetime > realtime: {livetime} > {realtime}")

    # finish populating data dict
    data["realtime"] = realtime
    data["livetime"] = livetime
    data["start_time"] = collection_start
    data["counts"] = counts

    # create an energy calibration object
    cal = None
    if len(cal_coeff) > 0 and not np.allclose(cal_coeff, 0):
        cal = calibration.Calibration.from_polynomial(cal_coeff)

    return data, cal
