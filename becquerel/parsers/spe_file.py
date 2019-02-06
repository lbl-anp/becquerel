"""Read in an Ortec SPE file."""

from __future__ import print_function
import datetime
import os
import warnings
import dateutil.parser
import numpy as np
from .spectrum_file import (SpectrumFile, SpectrumFileParsingError,
                            SpectrumFileParsingWarning)


class SpeFileParsingError(SpectrumFileParsingError):
    """Failed while parsing an SPE file."""

    pass


class SpeFileWritingError(SpectrumFileParsingError):
    """Failed while writing an SPE file."""

    pass


class SpeFile(SpectrumFile):
    """SPE ASCII file parser.

    Just instantiate a class with a filename:
        spec = SpeFile(filename)

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.energies
        spec.bin_edges_kev
        spec.energy_bin_widths
        spec.energy_bin_edges (deprecated)

    ORTEC's SPE file format is given on page 73 of this document:
        http://www.ortec-online.com/download/ortec-software-file-structure-manual.pdf

    """

    def __init__(self, filename):
        """Initialize the SPE file."""
        super(SpeFile, self).__init__(filename)
        _, ext = os.path.splitext(self.filename)
        if ext.lower() != '.spe':
            raise SpeFileParsingError('File extension is incorrect: ' + ext)
        # SPE-specific members
        self.first_channel = 0
        self.ROIs = []
        self.energy_cal = []
        self.shape_cal = []
        # read in the data
        self.read()
        self.apply_calibration()

    def read(self, verbose=False):
        """Read in the file."""
        print('SpeFile: Reading file ' + self.filename)
        self.realtime = 0.0
        self.livetime = 0.0
        self.channels = np.array([], dtype=np.float)
        self.data = np.array([], dtype=np.float)
        self.cal_coeff = []
        with open(self.filename, 'r') as f:
            # read & remove newlines from end of each line
            lines = [line.strip() for line in f.readlines()]
            i = 0
            while i < len(lines):
                # check whether we have reached a keyword and parse accordingly
                if lines[i] == '$SPEC_ID:':
                    i += 1
                    self.spectrum_id = lines[i]
                    if verbose:
                        print(self.spectrum_id)
                elif lines[i] == '$SPEC_REM:':
                    self.sample_description = ''
                    i += 1
                    while i < len(lines) and lines[i][0] != '$':
                        self.sample_description += lines[i] + '\n'
                        i += 1
                    self.sample_description = self.sample_description[:-1]
                    i -= 1
                    if verbose:
                        print(self.sample_description)
                elif lines[i] == '$DATE_MEA:':
                    i += 1
                    self.collection_start = dateutil.parser.parse(lines[i])
                    if verbose:
                        print(self.collection_start)
                elif lines[i] == '$MEAS_TIM:':
                    i += 1
                    self.livetime = float(lines[i].split(' ')[0])
                    self.realtime = float(lines[i].split(' ')[1])
                    if verbose:
                        print(self.livetime, self.realtime)
                elif lines[i] == '$DATA:':
                    i += 1
                    self.first_channel = int(lines[i].split(' ')[0])
                    # I don't know why it would be nonzero
                    if self.first_channel != 0:
                        raise SpeFileParsingError(
                            'First channel is not 0: {}'.format(
                                self.first_channel))
                    self.num_channels = int(lines[i].split(' ')[1])
                    if verbose:
                        print(self.first_channel, self.num_channels)
                    j = self.first_channel
                    while j <= self.num_channels + self.first_channel:
                        i += 1
                        self.data = np.append(self.data, int(lines[i]))
                        self.channels = np.append(self.channels, j)
                        j += 1
                elif lines[i] == '$ROI:':
                    self.ROIs = []
                    i += 1
                    while i < len(lines) and lines[i][0] != '$':
                        self.ROIs.append(lines[i])
                        i += 1
                    i -= 1
                    if verbose:
                        print(self.ROIs)
                elif lines[i] == '$ENER_FIT:':
                    i += 1
                    self.energy_cal.append(float(lines[i].split(' ')[0]))
                    self.energy_cal.append(float(lines[i].split(' ')[1]))
                    if verbose:
                        print(self.energy_cal)
                elif lines[i] == '$MCA_CAL:':
                    i += 1
                    n_coeff = int(lines[i])
                    i += 1
                    for j in range(n_coeff):
                        self.cal_coeff.append(float(lines[i].split(' ')[j]))
                    if verbose:
                        print(self.cal_coeff)
                elif lines[i] == '$SHAPE_CAL:':
                    i += 1
                    n_coeff = int(lines[i])
                    i += 1
                    for j in range(n_coeff):
                        self.shape_cal.append(float(lines[i].split(' ')[j]))
                    if verbose:
                        print(self.shape_cal)
                elif lines[i].startswith('$'):
                    key = lines[i][1:].rstrip(':')
                    i += 1
                    values = []
                    while i < len(lines) and not lines[i].startswith('$'):
                        values.append(lines[i])
                        i += 1
                    if i < len(lines):
                        if lines[i].startswith('$'):
                            i -= 1
                    self.metadata[key] = values
                else:
                    warnings.warn(
                        'Line {} unknown: '.format(i + 1) + lines[i],
                        SpectrumFileParsingWarning)
                i += 1
        if self.realtime <= 0.0:
            raise SpeFileParsingError(
                'Realtime not parsed correctly: {}'.format(self.realtime))
        if self.livetime <= 0.0:
            raise SpeFileParsingError(
                'Livetime not parsed correctly: {}'.format(self.livetime))
        if self.livetime > self.realtime:
            raise SpeFileParsingError(
                'Livetime > realtime: {} > {}'.format(
                    self.livetime, self.realtime))
        self.collection_stop = self.collection_start + \
            datetime.timedelta(seconds=self.realtime)

    def _spe_format(self):
        """Format of this spectrum for writing to file."""
        s = ''
        s += '$SPEC_ID:\n'
        s += self.spectrum_id + '\n'
        s += '$SPEC_REM:\n'
        s += self.sample_description + '\n'
        if self.collection_start is not None:
            s += '$DATE_MEA:\n'
            s += '{:%m/%d/%Y %H:%M:%S}\n'.format(self.collection_start)
        s += '$MEAS_TIM:\n'
        s += '{:.0f} {:.0f}\n'.format(self.livetime, self.realtime)
        s += '$DATA:\n'
        s += '{:.0f} {:d}\n'.format(self.first_channel, self.num_channels)
        for j in range(self.num_channels):
            s += '       {:.0f}\n'.format(self.data[j])
        s += '$ROI:\n'
        for line in self.ROIs:
            s += line + '\n'
        if len(self.energy_cal) > 0:
            s += '$ENER_FIT:\n'
            s += '{:f} {:f}\n'.format(self.energy_cal[0], self.energy_cal[1])
        if len(self.cal_coeff) > 0:
            s += '$MCA_CAL:\n'
            n_coeff = len(self.cal_coeff)
            s += '{:d}\n'.format(n_coeff)
            s += '{:E}'.format(self.cal_coeff[0])
            for j in range(1, n_coeff):
                s += ' {:E}'.format(self.cal_coeff[j])
            s += '\n'
        if len(self.shape_cal) > 0:
            s += '$SHAPE_CAL:\n'
            n_coeff = len(self.shape_cal)
            s += '{:d}\n'.format(n_coeff)
            s += '{:E}'.format(self.shape_cal[0])
            for j in range(1, n_coeff):
                s += ' {:E}'.format(self.shape_cal[j])
            s += '\n'
        if len(self.metadata.keys()) > 0:
            for key, values in self.metadata.items():
                s += '$' + key + ':\n'
                for val in values:
                    s += str(val) + '\n'
        return s[:-1]

    def write(self, filename):
        """Write back to a file."""
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.spe':
            raise SpeFileWritingError('File extension is incorrect: ' + ext)
        with open(filename, 'w') as outfile:
            print(self._spe_format(), file=outfile)
