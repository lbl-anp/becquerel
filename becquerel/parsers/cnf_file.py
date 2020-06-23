"""Read in a Canberra CNF file.

The CNF format specification appears to be proprietary, so this code is
based on the CNF parsing code in the xylib project:
    https://github.com/wojdyr/xylib

The relevant code is in these files:
    https://github.com/wojdyr/xylib/blob/master/xylib/util.cpp
    https://github.com/wojdyr/xylib/blob/master/xylib/canberra_cnf.cpp

"""

from __future__ import print_function
import datetime
import os
import struct
import numpy as np
from .spectrum_file import SpectrumFile, SpectrumFileParsingError


class CnfFileParsingError(SpectrumFileParsingError):
    """Failed while parsing a CNF file."""

    pass


def _from_little_endian(data, index, n_bytes):
    """Convert bytes starting from index from little endian to an integer."""
    return sum([data[index + j] << 8 * j for j in range(n_bytes)])


def _convert_date(data, index):
    """Convert 64-bit number starting at index into a date."""
    # 'data' is the number of 100ns intervals since 17 November 1858, which
    # is the start of the Modified Julian Date (MJD) epoch.
    # 3506716800 is the number of seconds between the start of the MJD epoch
    # to the start of the Unix epoch on 1 January 1970.
    d = _from_little_endian(data, index, 8)
    t = (d / 10000000.) - 3506716800
    try:
        return datetime.datetime.utcfromtimestamp(t)
    except:
        raise CnfFileParsingError('Date conversion failed')


def _convert_time(data, index):
    """Convert 64-bit number starting at index into a time."""
    # 2^(64) - number of 100ns intervals since timing started
    d = _from_little_endian(data, index, 8)
    d = (pow(2, 64) - 1) & ~d
    return d * 1.e-7


def _from_pdp11(data, index):
    """Convert 32-bit floating point in DEC PDP-11 format to a double.

    For a description of the format see:
    http://home.kpn.nl/jhm.bonten/computers/bitsandbytes/wordsizes/hidbit.htm
    """
    if (data[index + 1] & 0x80) == 0:
        sign = 1
    else:
        sign = -1
    exb = ((data[index + 1] & 0x7f) << 1) + ((data[index] & 0x80) >> 7)
    if exb == 0:
        if sign == -1:
            return np.NaN
        else:
            return 0.
    h = data[index + 2] / 256. / 256. / 256. \
        + data[index + 3] / 256. / 256. \
        + (128 + (data[index] & 0x7f)) / 256.
    return sign * h * pow(2., exb - 128.)


def _read_energy_calibration(data, index):
    """Read the four energy calibration coefficients."""
    coeff = [0., 0., 0., 0.]
    for i in range(4):
        coeff[i] = _from_pdp11(data, index + 2 * 4 + 28 + 4 * i)
    if coeff[1] == 0.:
        return None
    return coeff


class CnfFile(SpectrumFile):
    """CNF binary file parser.

    Just instantiate a class with a filename:
        spec = CnfFile(filename)

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.energies
        spec.bin_edges_kev
        spec.energy_bin_widths
        spec.energy_bin_edges (deprecated)

    """

    def __init__(self, filename):
        """Initialize the CNF file."""
        super(CnfFile, self).__init__(filename)
        _, ext = os.path.splitext(self.filename)
        if ext.lower() != '.cnf':
            raise CnfFileParsingError('File extension is incorrect: ' + ext)
        # read in the data
        self.read()
        self.apply_calibration()

    def read(self, verbose=False):
        """Read in the file."""
        print('CnfFile: Reading file ' + self.filename)
        self.realtime = 0.0
        self.livetime = 0.0
        self.channels = np.array([], dtype=np.float)
        self.data = np.array([], dtype=np.float)
        self.cal_coeff = []
        data = []
        with open(self.filename, 'rb') as f:
            data_str = f.read(1)
            while data_str:
                data_int = struct.unpack('1B', data_str)
                data.append(data_int[0])
                data_str = f.read(1)
        if verbose:
            print('file size (bytes):', len(data))
        # skip first 112 bytes of file
        i = 112
        # scan for offsets
        offset_acq = 0
        offset_sam = 0
        offset_eff = 0
        offset_enc = 0
        offset_chan = 0
        for i in range(112, 128 * 1024, 48):
            offset = _from_little_endian(data, i + 10, 4)
            if (data[i + 1] == 0x20 and data[i + 2] == 0x01) \
                    or (data[i + 1] == 0) or (data[i + 2] == 0):
                if data[i] == 0:
                    if offset_acq == 0:
                        offset_acq = offset
                    else:
                        offset_enc = offset
                elif data[i] == 1:
                    if offset_sam == 0:
                        offset_sam = offset
                elif data[i] == 2:
                    if offset_eff == 0:
                        offset_eff = offset
                elif data[i] == 5:
                    if offset_chan == 0:
                        offset_chan = offset
                else:
                    pass
                if offset_acq != 0 and offset_sam != 0 \
                        and offset_eff != 0 and offset_chan != 0:
                    break
        if offset_enc == 0:
            offset_enc = offset_acq

        # extract sample information
        if (offset_sam + 48 + 80) >= len(data) or data[offset_sam] != 1 \
                or data[offset_sam + 1] != 0x20:
            if verbose:
                print(offset_sam + 48 + 80, len(data))
                print(data[offset_sam], data[offset_sam + 1])
            raise CnfFileParsingError('Sample information not found')
        else:
            sample_name = ''
            for j in range(offset_sam + 48, offset_sam + 48 + 64):
                sample_name += chr(data[j])
            if verbose:
                print('sample name: ', sample_name)
            sample_id = ''
            for j in range(offset_sam + 112, offset_sam + 112 + 64):
                sample_id += chr(data[j])
            if verbose:
                print('sample id:   ', sample_id)
            sample_type = ''
            for j in range(offset_sam + 176, offset_sam + 176 + 16):
                sample_type += chr(data[j])
            if verbose:
                print('sample type: ', sample_type)
            sample_unit = ''
            for j in range(offset_sam + 192, offset_sam + 192 + 64):
                sample_unit += chr(data[j])
            if verbose:
                print('sample unit: ', sample_unit)
            user_name = ''
            for j in range(offset_sam + 0x02d6, offset_sam + 0x02d6 + 32):
                user_name += chr(data[j])
            if verbose:
                print('user name:   ', user_name)
            sample_desc = ''
            for j in range(offset_sam + 0x036e, offset_sam + 0x036e + 256):
                sample_desc += chr(data[j])
            if verbose:
                print('sample desc: ', sample_desc)
            self.spectrum_id = sample_name
            self.sample_description = sample_desc
            self.detector_description = ''
            self.location_description = sample_id

        # extract acquisition information
        if (offset_acq + 48 + 128 + 10 + 4) >= len(data) \
                or data[offset_acq] != 0 \
                or data[offset_acq + 1] != 0x20:
            if verbose:
                print(offset_acq + 48 + 128 + 10 + 4, len(data))
                print(data[offset_acq], data[offset_acq + 1])
            raise CnfFileParsingError('Acquisition information not found')
        else:
            offset1 = _from_little_endian(data, offset_acq + 34, 2)
            offset2 = _from_little_endian(data, offset_acq + 36, 2)
            offset_pha = offset_acq + 48 + 128
            if chr(data[offset_pha + 0]) != 'P' \
                    and chr(data[offset_pha + 1]) != 'H' \
                    and chr(data[offset_pha + 2]) != 'A':
                raise CnfFileParsingError('PHA keyword not found')
            self.num_channels = 256 * \
                _from_little_endian(data, offset_pha + 10, 2)
            if self.num_channels < 256 or self.num_channels > 16384:
                raise CnfFileParsingError(
                    'Unexpected number of channels: ', self.num_channels)
            if verbose:
                print('Number of channels: ', self.num_channels)

        # extract date and time information
        offset_date = offset_acq + 48 + offset2 + 1
        if offset_date + 24 >= len(data):
            raise CnfFileParsingError('Problem with date offset')
        self.collection_start = _convert_date(data, offset_date)
        self.realtime = _convert_time(data, offset_date + 8)
        if verbose:
            print('realtime: ', self.realtime)
        self.livetime = _convert_time(data, offset_date + 16)
        if verbose:
            print('livetime: ', self.livetime)
            print('{:%Y-%m-%d %H:%M:%S}'.format(self.collection_start))
        self.collection_stop = self.collection_start \
            + datetime.timedelta(seconds=self.realtime)
        if verbose:
            print('{:%Y-%m-%d %H:%M:%S}'.format(self.collection_stop))

        # extract energy calibration information
        offset_cal = offset_enc + 48 + 32 + offset1
        if offset_enc + 48 + 32 + offset1 >= len(data):
            raise CnfFileParsingError('Problem with energy calibration offset')
        coeff = _read_energy_calibration(data, offset_enc + 48 + 32 + offset1)
        if verbose:
            print('calibration coefficients:', coeff)
        if coeff is None:
            if verbose:
                print('Energy calibration - second try')
            coeff = _read_energy_calibration(data, offset_enc + 48 + 32)
            if verbose:
                print('calibration coefficients:', coeff)
        if coeff is None:
            raise CnfFileParsingError('Energy calibration not found')
        self.cal_coeff = coeff

        # extract channel data
        if offset_chan + 512 + 4 * self.num_channels > len(data) \
                or data[offset_chan] != 5 or data[offset_chan + 1] != 0x20:
            raise CnfFileParsingError('Channel data not found')
        self.channels = np.array([], dtype=float)
        self.data = np.array([], dtype=float)
        for i in range(0, 2):
            y = _from_little_endian(data, offset_chan + 512 + 4 * i, 4)
            if y == int(self.realtime) or y == int(self.livetime):
                y = 0
            self.data = np.append(self.data, y)
            self.channels = np.append(self.channels, i)
        for i in range(2, self.num_channels):
            y = _from_little_endian(data, offset_chan + 512 + 4 * i, 4)
            self.data = np.append(self.data, y)
            self.channels = np.append(self.channels, i)

        # finished
        if self.realtime <= 0.0:
            raise CnfFileParsingError(
                'Realtime not parsed correctly: {}'.format(self.realtime))
        if self.livetime <= 0.0:
            raise CnfFileParsingError(
                'Livetime not parsed correctly: {}'.format(self.livetime))
        if self.livetime > self.realtime:
            raise CnfFileParsingError(
                'Livetime > realtime: {} > {}'.format(
                    self.livetime, self.realtime))
