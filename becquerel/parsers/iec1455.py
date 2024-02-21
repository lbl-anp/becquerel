"""Read in an IEC 1455 file."""

import os
import re
import warnings

import dateutil.parser
import numpy as np

from ..core import calibration
from .parsers import BecquerelParserError, BecquerelParserWarning


def _read_nonzero_number_pairs(str_array):
    """Read a pair of non-zero numbers from a string array"""
    out = []

    # check for even length
    if len(str_array) % 2 == 0:
        for i in range(0, len(str_array), 2):
            a = float(str_array[i])
            b = float(str_array[i + 1])
            if a != 0 or b != 0:
                out += [a, b]
    else:
        warnings.warn(
            "Cannot read number pairs from odd-length string array",
            BecquerelParserWarning,
        )

    return out


def read(filename, verbose=False, cal_kwargs=None):
    """Parse the ASCII IEC 1455 file and return a dictionary of data.

    IEC 1455 International Standard Format
    Nuclear instrumentation -
    MCA histogram data interchange format for nuclear spectroscropy
    First edition 1995

    This is an ASCII file with a variable number of 70 bytes (characters)
    records (lines). Each line starts with a A004 prefix followed by
    different values depending on the record number.

    Record number  Record content
    ----------------------------------------------------------------------------
    1              System identification, sub-system identification,
                   analog-to-digital converter number, segment number, digital
                   offset
    2              Live time, real time, number of channels
    3              Acquisition start time, sample collection time
    4              Energy calibration coefficients
    5              Peak Full-Width-Half-Maximum (FWHM) calibration coefficients
    6              Sample description - 1
    7              Sample description - 2
    8              Sample description - 3
    9              Sample description - 4
    10             Spare
    11-22          Energy and channel pairs
    23-34          Energy and resolution pairs
    35-46          Energy and efficiency pairs
    47-58          User defined
    59-End         Spectral data

    Parameters
    ----------
    filename : str
        The filename of the IEC 1455 file to read.
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
    print("IEC1455File: Reading file " + filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".iec":
        raise BecquerelParserError("File extension is incorrect: " + ext)

    # initialize a dictionary of spectrum data to populate as we parse
    data = {}

    # parse the file line-by-line
    counts = []
    cal_coeff = []
    fwhm_coeff = []  # currently unused
    energy_channel_pairs = []
    energy_res_pairs = []  # currently unused
    energy_eff_pairs = []  # currently unused
    with open(filename) as f:
        record = 1
        # loop over lines
        for line in f:
            # read line and check for prefix
            if line.startswith("A004") is False:
                warnings.warn(
                    "Cannot parse line, ignoring record: " + line,
                    BecquerelParserWarning,
                )
                continue

            # strip A004 prefix and tokenize
            line = line[4:]
            tok = line.split()

            # parse records
            if record == 1:
                if len(tok) == 5:  # all information
                    data["system_identification"] = tok[0]
                    data["sub_system_identification"] = tok[1]
                    data["ADC_number"] = int(tok[2])
                    data["segment_number"] = int(tok[3])
                    data["digital_offset"] = int(tok[4])
                elif len(tok) == 4 and line.startswith(
                    " "
                ):  # missing system identification
                    data["sub_system_identification"] = tok[0]
                    data["ADC_number"] = int(tok[1])
                    data["segment_number"] = int(tok[2])
                    data["digital_offset"] = int(tok[3])
                else:
                    warnings.warn(
                        "Cannot parse record 1, ignoring information: " + line,
                        BecquerelParserWarning,
                    )
            elif record == 2:
                if len(tok) == 3:
                    data["livetime"] = float(tok[0])
                    data["realtime"] = float(tok[1])
                    num_channels = int(tok[2])
                else:
                    raise BecquerelParserError(
                        "Record 2 does not contain 3 values: " + line
                    )
            elif record == 3:
                if len(tok) == 2:
                    data["start_time"] = dateutil.parser.parse(tok[0] + " " + tok[1])
                elif len(tok) == 4:
                    data["start_time"] = dateutil.parser.parse(tok[0] + " " + tok[1])
                    data["sample_collection_time"] = dateutil.parser.parse(
                        tok[2] + " " + tok[3]
                    )
                else:
                    raise BecquerelParserError("Cannot parse record 2: " + line)
            elif record == 4:
                # fix nasty formatting of Interwinner export
                line = re.sub(r"(e[+-][0-9]{2})", r"\1 ", line.lower())
                tok = line.split()
                for coeff in tok:
                    cal_coeff.append(float(coeff))
            elif record == 5:
                # fix nasty formatting of Interwinner export
                line = re.sub(r"(e[+-][0-9]{2})", r"\1 ", line.lower())
                tok = line.split()
                for coeff in tok:
                    fwhm_coeff.append(float(coeff))
            elif record >= 6 and record <= 9:
                line = line.strip()
                if len(line) > 0:
                    if "sample_description" in data:
                        data["sample_description"] += " " + line
                    else:
                        data["sample_description"] = line
            elif record >= 11 and record <= 22:
                energy_channel_pairs += _read_nonzero_number_pairs(tok)
            elif record >= 23 and record <= 34:
                energy_res_pairs += _read_nonzero_number_pairs(tok)
            elif record >= 35 and record <= 46:
                energy_eff_pairs += _read_nonzero_number_pairs(tok)
            elif record >= 59:
                for c in tok[1:]:  # skip channel index
                    counts.append(int(c))

            # increment record number
            record += 1

    # check the data that were read
    if "start_time" not in data:
        raise BecquerelParserError("Start time not found")
    if "livetime" not in data:
        raise BecquerelParserError("Live time not found")
    if "realtime" not in data:
        raise BecquerelParserError("Real time not found")

    if data["realtime"] <= 0.0:
        raise BecquerelParserError(
            f"Real time not parsed correctly: {data['realtime']}"
        )
    if data["livetime"] <= 0.0:
        raise BecquerelParserError(
            f"Live time not parsed correctly: {data['livetime']}"
        )
    if data["livetime"] > data["realtime"]:
        raise BecquerelParserError(
            f"Livetime > realtime: {data['livetime']} > {data['realtime']}"
        )

    if len(counts) < num_channels:
        raise BecquerelParserError(
            f"Could not read data for all {num_channels} channels"
        )
    if len(counts) > num_channels:
        counts = counts[:num_channels]
        warnings.warn(
            f"Data for more than {num_channels} were read, ignoring them",
            BecquerelParserWarning,
        )

    # finish populating data dict
    data["counts"] = counts

    # create an energy calibration object either using polynomial
    # coefficients or energy-channel pairs
    if cal_kwargs is None:
        cal_kwargs = {}
    cal = None
    if len(cal_coeff) > 0 and not np.allclose(cal_coeff, 0):
        cal = calibration.Calibration.from_polynomial(cal_coeff, **cal_kwargs)
    elif len(energy_channel_pairs) >= 4:
        if len(energy_channel_pairs) == 4:  # 2 calibration points
            cal = calibration.Calibration("p[0] * x", [1], **cal_kwargs)
        elif len(energy_channel_pairs) == 6:  # 3 calibration points
            cal = calibration.Calibration("p[0] + p[1] * x", [0, 1], **cal_kwargs)
        else:
            cal = calibration.Calibration(
                "p[0] + p[1] * x + p[2] * x**2", [0, 1, 0.01], **cal_kwargs
            )
        cal.fit_points(energy_channel_pairs[1::2], energy_channel_pairs[::2])

    return data, cal
