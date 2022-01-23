"""Read in a Canberra CNF file.

The CNF format specification appears to be proprietary, so this code is
based on the CNF parsing code in the xylib project:
    https://github.com/wojdyr/xylib

The relevant code is in these files:
    https://github.com/wojdyr/xylib/blob/master/xylib/util.cpp
    https://github.com/wojdyr/xylib/blob/master/xylib/canberra_cnf.cpp
"""

import datetime
import os
import struct
import numpy as np
from ..core import calibration
from .parsers import BecquerelParserError


def _from_little_endian(data, index, n_bytes):
    """Convert bytes starting from index from little endian to an integer."""
    return sum(data[index + j] << 8 * j for j in range(n_bytes))


def _convert_date(data, index):
    """Convert 64-bit number starting at index into a date."""
    # 'data' is the number of 100ns intervals since 17 November 1858, which
    # is the start of the Modified Julian Date (MJD) epoch.
    # 3506716800 is the number of seconds between the start of the MJD epoch
    # to the start of the Unix epoch on 1 January 1970.
    d = _from_little_endian(data, index, 8)
    t = (d / 10000000.0) - 3506716800
    return datetime.datetime.utcfromtimestamp(t)


def _convert_time(data, index):
    """Convert 64-bit number starting at index into a time."""
    # 2^(64) - number of 100ns intervals since timing started
    d = _from_little_endian(data, index, 8)
    d = (pow(2, 64) - 1) & ~d
    return d * 1.0e-7


def _from_pdp11(data, index):
    """Convert 32-bit floating point in DEC PDP-11 format to a double.

    For a description of the format see:
    http://home.kpn.nl/jhm.bonten/computers/bitsandbytes/wordsizes/hidbit.htm
    """
    if (data[index + 1] & 0x80) == 0:
        sign = 1
    else:
        sign = -1
    exb = ((data[index + 1] & 0x7F) << 1) + ((data[index] & 0x80) >> 7)
    if exb == 0:
        if sign == -1:
            return np.NaN
        else:
            return 0.0
    h = (
        data[index + 2] / 256.0 / 256.0 / 256.0
        + data[index + 3] / 256.0 / 256.0
        + (128 + (data[index] & 0x7F)) / 256.0
    )
    return sign * h * pow(2.0, exb - 128.0)


def _read_energy_calibration(data, index):
    """Read the four energy calibration coefficients."""
    coeff = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        coeff[i] = _from_pdp11(data, index + 2 * 4 + 28 + 4 * i)
    if coeff[1] == 0.0:
        return None
    return coeff


def read(filename, verbose=False):
    """Parse the CNF file and return a dictionary of data.

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
    print("Reading CNF file " + filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".cnf":
        raise BecquerelParserError("File extension is incorrect: " + ext)

    # read all of the file into memory
    file_bytes = []
    with open(filename, "rb") as f:
        byte = f.read(1)
        while byte:
            byte_int = struct.unpack("1B", byte)
            file_bytes.append(byte_int[0])
            byte = f.read(1)
    if verbose:
        print("file size (bytes):", len(file_bytes))

    # initialize a dictionary of spectrum data to populate as we parse
    data = {}

    # step through the stored data and parse it
    # skip first 112 bytes of file
    i = 112
    # scan for offsets
    offset_acq = 0
    offset_sam = 0
    offset_eff = 0
    offset_enc = 0
    offset_chan = 0
    for i in range(112, 128 * 1024, 48):
        offset = _from_little_endian(file_bytes, i + 10, 4)
        if (
            (file_bytes[i + 1] == 0x20 and file_bytes[i + 2] == 0x01)
            or (file_bytes[i + 1] == 0)
            or (file_bytes[i + 2] == 0)
        ):
            if file_bytes[i] == 0:
                if offset_acq == 0:
                    offset_acq = offset
                else:
                    offset_enc = offset
            elif file_bytes[i] == 1:
                if offset_sam == 0:
                    offset_sam = offset
            elif file_bytes[i] == 2:
                if offset_eff == 0:
                    offset_eff = offset
            elif file_bytes[i] == 5:
                if offset_chan == 0:
                    offset_chan = offset
            else:
                pass
            if (
                offset_acq != 0
                and offset_sam != 0
                and offset_eff != 0
                and offset_chan != 0
            ):
                break
    if offset_enc == 0:
        offset_enc = offset_acq

    # extract sample information
    if (
        (offset_sam + 48 + 80) >= len(file_bytes)
        or file_bytes[offset_sam] != 1
        or file_bytes[offset_sam + 1] != 0x20
    ):
        if verbose:
            print(offset_sam + 48 + 80, len(file_bytes))
            print(file_bytes[offset_sam], file_bytes[offset_sam + 1])
        raise BecquerelParserError("Sample information not found")
    else:
        sample_name = ""
        for j in range(offset_sam + 48, offset_sam + 48 + 64):
            sample_name += chr(file_bytes[j])
        if verbose:
            print("sample name: ", sample_name)
        sample_id = ""
        for j in range(offset_sam + 112, offset_sam + 112 + 64):
            sample_id += chr(file_bytes[j])
        if verbose:
            print("sample id:   ", sample_id)
        sample_type = ""
        for j in range(offset_sam + 176, offset_sam + 176 + 16):
            sample_type += chr(file_bytes[j])
        if verbose:
            print("sample type: ", sample_type)
        sample_unit = ""
        for j in range(offset_sam + 192, offset_sam + 192 + 64):
            sample_unit += chr(file_bytes[j])
        if verbose:
            print("sample unit: ", sample_unit)
        user_name = ""
        for j in range(offset_sam + 0x02D6, offset_sam + 0x02D6 + 32):
            user_name += chr(file_bytes[j])
        if verbose:
            print("user name:   ", user_name)
        sample_desc = ""
        for j in range(offset_sam + 0x036E, offset_sam + 0x036E + 256):
            sample_desc += chr(file_bytes[j])
        if verbose:
            print("sample desc: ", sample_desc)
        data["sample_name"] = sample_name
        data["sample_id"] = sample_id
        data["sample_type"] = sample_type
        data["sample_unit"] = sample_unit
        data["user_name"] = user_name
        data["sample_description"] = sample_desc

    # extract acquisition information
    if (
        (offset_acq + 48 + 128 + 10 + 4) >= len(file_bytes)
        or file_bytes[offset_acq] != 0
        or file_bytes[offset_acq + 1] != 0x20
    ):
        if verbose:
            print(offset_acq + 48 + 128 + 10 + 4, len(file_bytes))
            print(file_bytes[offset_acq], file_bytes[offset_acq + 1])
        raise BecquerelParserError("Acquisition information not found")
    else:
        offset1 = _from_little_endian(file_bytes, offset_acq + 34, 2)
        offset2 = _from_little_endian(file_bytes, offset_acq + 36, 2)
        offset_pha = offset_acq + 48 + 128
        if (
            chr(file_bytes[offset_pha + 0]) != "P"
            and chr(file_bytes[offset_pha + 1]) != "H"
            and chr(file_bytes[offset_pha + 2]) != "A"
        ):
            raise BecquerelParserError("PHA keyword not found")
        num_channels = 256 * _from_little_endian(file_bytes, offset_pha + 10, 2)
        if num_channels < 256 or num_channels > 16384:
            raise BecquerelParserError("Unexpected number of channels: ", num_channels)
        if verbose:
            print("Number of channels: ", num_channels)

    # extract date and time information
    offset_date = offset_acq + 48 + offset2 + 1
    if offset_date + 24 >= len(file_bytes):
        raise BecquerelParserError("Problem with date offset")
    collection_start = _convert_date(file_bytes, offset_date)
    realtime = _convert_time(file_bytes, offset_date + 8)
    if verbose:
        print("realtime: ", realtime)
    livetime = _convert_time(file_bytes, offset_date + 16)
    if verbose:
        print("livetime: ", livetime)
        print(f"{collection_start:%Y-%m-%d %H:%M:%S}")

    # check data and time information
    if realtime <= 0.0:
        raise BecquerelParserError(f"Realtime not parsed correctly: {realtime}")
    if livetime <= 0.0:
        raise BecquerelParserError(f"Livetime not parsed correctly: {livetime}")
    if livetime > realtime:
        raise BecquerelParserError(f"Livetime > realtime: {livetime} > {realtime}")

    # extract energy calibration information
    offset_cal = offset_enc + 48 + 32 + offset1
    if offset_cal >= len(file_bytes):
        raise BecquerelParserError("Problem with energy calibration offset")
    cal_coeff = _read_energy_calibration(file_bytes, offset_cal)
    if verbose:
        print("calibration coefficients:", cal_coeff)
    if cal_coeff is None:
        if verbose:
            print("Energy calibration - second try")
        cal_coeff = _read_energy_calibration(file_bytes, offset_cal - offset1)
        if verbose:
            print("calibration coefficients:", cal_coeff)
    if cal_coeff is None:
        raise BecquerelParserError("Energy calibration not found")

    # extract channel count data
    if (
        offset_chan + 512 + 4 * num_channels > len(file_bytes)
        or file_bytes[offset_chan] != 5
        or file_bytes[offset_chan + 1] != 0x20
    ):
        raise BecquerelParserError("Channel data not found")
    channels = np.array([], dtype=float)
    counts = np.array([], dtype=float)
    for i in range(0, 2):
        y = _from_little_endian(file_bytes, offset_chan + 512 + 4 * i, 4)
        if y == int(realtime) or y == int(livetime):
            y = 0
        counts = np.append(counts, y)
        channels = np.append(channels, i)
    for i in range(2, num_channels):
        y = _from_little_endian(file_bytes, offset_chan + 512 + 4 * i, 4)
        counts = np.append(counts, y)
        channels = np.append(channels, i)

    # finish populating data dict
    data["realtime"] = realtime
    data["livetime"] = livetime
    data["start_time"] = collection_start
    data["counts"] = counts

    # clean up null characters in any strings
    for key in data.keys():
        if isinstance(data[key], str):
            data[key] = data[key].replace("\x00", " ")
            data[key] = data[key].replace("\x01", " ")
            data[key] = data[key].strip()

    # create an energy calibration object
    cal = calibration.Calibration.from_polynomial(cal_coeff)

    return data, cal
