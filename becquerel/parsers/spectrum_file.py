"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import warnings
import numpy as np
from scipy.interpolate import interp1d
warnings.simplefilter('always', DeprecationWarning)


class SpectrumFileParsingWarning(UserWarning):
    """Warnings displayed by SpectrumFile."""

    pass


class SpectrumFileParsingError(Exception):
    """Failed while parsing a spectrum file."""

    pass


class SpectrumFile(object):
    """Spectrum file parser base class.

    Just instantiate a class with a filename:
        spec = SpectrumFile(filename)

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.energies
        spec.bin_edges_kev
        spec.energy_bin_widths
        spec.energy_bin_edges (deprecated)

    """

    def __init__(self, filename):
        """Initialize the spectrum."""
        self.filename = filename
        assert os.path.exists(self.filename)
        # fields to read from file
        self.spectrum_id = ''
        self.sample_description = ''
        self.detector_description = ''
        self.location_description = ''
        self.hardware_status = ''
        self.collection_start = None
        self.collection_stop = None
        self.realtime = 0.0
        self.livetime = 0.0
        self.num_channels = 0
        # miscellaneous metadata
        self.metadata = {}
        # arrays to be read from file
        self.channels = np.array([], dtype=np.float)
        self.data = np.array([], dtype=np.float)
        self.cal_coeff = []
        # arrays to be calculated using calibration
        self.energies = np.array([], dtype=np.float)
        self.bin_edges_kev = None

    def __str__(self):
        """String form of the spectrum."""

        print_channels = False

        s = ''
        s += 'Filename:              {:s}\n'.format(self.filename)
        s += 'Spectrum ID:           {:s}\n'.format(self.spectrum_id)
        s += 'Sample description:    {:s}\n'.format(self.sample_description)
        s += 'Detector description:  {:s}\n'.format(self.detector_description)
        s += 'Location Description:  {:s}\n'.format(self.location_description)
        s += 'Hardware Status:       {:s}\n'.format(self.hardware_status)
        if self.collection_start is not None:
            s += 'Collection Start:      {:%Y-%m-%d %H:%M:%S}\n'.format(
                self.collection_start)
        else:
            s += 'Collection Start:      None\n'
        if self.collection_stop is not None:
            s += 'Collection Stop:       {:%Y-%m-%d %H:%M:%S}\n'.format(
                self.collection_stop)
        else:
            s += 'Collection Stop:       None\n'
        s += 'Livetime:              {:.2f} sec\n'.format(self.livetime)
        s += 'Realtime:              {:.2f} sec\n'.format(self.realtime)
        if len(self.metadata.keys()) > 0:
            s += 'Metadata:\n'
            for key, value in self.metadata.items():
                s += '    {} : {}\n'.format(key, value)
        s += 'Number of channels:    {:d}\n'.format(self.num_channels)
        if len(self.cal_coeff) > 0:
            s += 'Calibration coeffs:    '
            s += ' '.join(['{:E}'.format(x) for x in self.cal_coeff])
            s += '\n'
        s += 'Data:                  \n'
        if print_channels:
            for ch, dt in zip(self.channels, self.data):
                s += '    {:5.0f}    {:5.0f}\n'.format(ch, dt)
        else:
            s += '    [length {}]\n'.format(len(self.data))
        return s

    @property
    def energy_bin_edges(self):
        warnings.warn('The use of energy_bin_edges is deprecated, ' +
                      'use bin_edges_kev instead', DeprecationWarning)
        return self.bin_edges_kev

    @property
    def energy_bin_widths(self):
        """Retrieve the calibrated width of all the bins."""
        return self.bin_width(self.channels)

    def read(self, verbose=False):
        """Read in the file."""
        raise NotImplementedError('read method not implemented')

    def write(self, filename):
        """Write back to a file."""
        raise NotImplementedError('write method not implemented')

    def apply_calibration(self):
        """Calculate energies corresponding to channels."""
        self.energies = self.channel_to_energy(self.channels)
        n_edges = len(self.energies) + 1
        channel_edges = np.linspace(-0.5, self.channels[-1] + 0.5, num=n_edges)
        self.bin_edges_kev = self.channel_to_energy(channel_edges)

        # check that calibration makes sense, remove calibration if not
        if np.any(np.diff(self.energies) <= 0):
            warnings.warn(
                'Spectrum will be initated without an energy calibration;' +
                'invalid calibration, energies not monotonically increasing.',
                SpectrumFileParsingWarning)
            self.bin_edges_kev = None

    def channel_to_energy(self, channel):
        """Apply energy calibration to the given channel(s)."""
        chan = np.array(channel, dtype=float)
        en = np.zeros_like(chan)
        for j in range(len(self.cal_coeff)):
            en += self.cal_coeff[j] * pow(chan, j)
        return en

    def energy_to_channel(self, energy):
        """Invert the energy calibration to find the channel(s)."""
        energy = np.array(energy, dtype=float)
        return interp1d(self.energies, self.channels)(energy)

    def bin_width(self, channel):
        """Calculate the width of the bin in keV at the channel(s)."""
        en0 = self.channel_to_energy(channel - 0.5)
        en1 = self.channel_to_energy(channel + 0.5)
        return en1 - en0
