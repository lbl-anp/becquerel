"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import numpy as np
import becquerel.parsers as parsers
# from ..parsers import SpeFile, SpcFile, CnfFile


class SpectrumError(Exception):
    """Exception raised by Spectrum."""

    pass


class UncalibratedError(SpectrumError):
    """Exception raised when an uncalibrated spectrum is treated as calibrated.
    """

    pass


class Spectrum(object):
    """
    Represents an energy spectrum.

    Initialize a Spectrum directly, or with Spectrum.from_file(filename).

    Attributes:
      data: np.array of counts in each channel
      channels: [Read-only] np.array of channel index as integers
      is_calibrated: [Read-only] bool
      energies_kev: [Read-only] np.array of energy bin centers, if calibrated
      bin_edges_kev: np.array of energy bin edges, if calibrated
    """

    def __init__(self, data, bin_edges_kev=None):
        """Initialize the spectrum.

        Args:
          data: an iterable of counts per channel
          bin_edges_kev: an iterable of bin edge energies.
            Defaults to None for an uncalibrated spectrum.
            If not none, should have length of (len(data) + 1).

        Raises:
          SpectrumError: for bad input arguments
        """

        if len(data) == 0:
            raise SpectrumError('Empty spectrum data')
        self.data = np.array(data, dtype=float)

        if bin_edges_kev is None:
            self.bin_edges_kev = None
        elif len(bin_edges_kev) != len(data) + 1:
            raise SpectrumError('Bad length of bin edges vector')
        elif np.any(np.diff(bin_edges_kev) <= 0):
            raise SpectrumError(
                'Bin edge energies must be strictly increasing')
        else:
            self.bin_edges_kev = np.array(bin_edges_kev, dtype=float)

        self.infilename = None
        self._infileobject = None

    @property
    def channels(self):
        """Channel index.

        Returns:
          np.array of int's from 0 to (len(self.data) - 1)
        """

        return np.arange(len(self.data), dtype=int)

    @property
    def energies_kev(self):
        """Convenience function for accessing the energies of bin centers.

        Returns:
          np.array of floats, same length as self.data

        Raises:
          UncalibratedError: if spectrum is not calibrated
        """

        if self.bin_edges_kev is None:
            raise UncalibratedError('Spectrum is not calibrated')
        else:
            return self.bin_centers_from_edges(self.bin_edges_kev)

    @property
    def is_calibrated(self):
        """Is the spectrum calibrated?

        Returns:
          A bool.
          True if spectrum has defined energy bin edges. False otherwise.
        """

        return self.bin_edges_kev is not None

    @classmethod
    def from_file(cls, infilename):
        """Construct a Spectrum object from a filename.

        Args:
          infilename: a string representing the path to a parsable file.

        Returns:
          A Spectrum object.

        Raises:
          IOError: for a bad filename.
        """

        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data,
                        bin_edges_kev=spect_file_obj.energy_bin_edges)
        spect_obj._infileobject = spect_file_obj

        # TODO Get more attributes from self.infileobj

        return spect_obj

    @staticmethod
    def bin_centers_from_edges(edges_kev):
        """Calculate bin centers from bin edges.

        Args:
          edges_kev: an iterable representing bin edge energies in keV.

        Returns:
          np.array of length (len(edges_kev) - 1),
          representing bin center energies.
        """

        edges_kev = np.array(edges_kev)
        centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
        return centers_kev

    def integrate(self, left_ch, right_ch):
        """Integrate over a region of interest.

        Args:
          left_ch: channel number of left side of region
          right_ch: channel number of right side of region

        Returns:
          a float of counts between left_ch and right_ch, inclusive
        """

        # TODO inputs as floats and compute partial bins

        left_ind = int(np.round(left_ch))
        right_ind = int(np.round(right_ch))

        integral = np.sum(self.data[left_ind:right_ind + 1])
        return integral

    def __add__(self, other):
        return self._add_sub(other, sub=False)

    def __sub__(self, other):
        return self._add_sub(other, sub=True)

    def __mul__(self, other):
        return self._mul_div(other, div=False)

    def __div__(self, other):
        return self._mul_div(other, div=True)

    def __truediv__(self, other):
        return self._mul_div(other, div=True)

    def _add_sub(self, other, sub=False):
        """Add or subtract two spectra. Handle errors."""

        if not isinstance(other, Spectrum):
            raise TypeError(
                'Spectrum addition/subtraction must involve a Spectrum object')
        if len(self.data) != len(other.data):
            raise SpectrumError(
                'Cannot add/subtract spectra of different lengths')

        # TODO: if both spectra are calibrated with different calibrations,
        #   should one be rebinned to match energy bins?
        if not self.is_calibrated and not other.is_calibrated:
            if sub:
                data = self.data - other.data
            else:
                data = self.data + other.data
            spect_obj = Spectrum(data)
        else:
            raise NotImplementedError(
                'Addition/subtraction for calibrated spectra not implemented')
        return spect_obj

    def _mul_div(self, scaling_factor, div=False):
        """Multiply or divide a spectrum by a scalar. Handle errors."""

        try:
            scaling_factor = float(scaling_factor)
        except (TypeError, ValueError):
            raise TypeError('Spectrum must be multiplied/divided by a scalar')
        else:
            if (scaling_factor == 0 or
                    np.isinf(scaling_factor) or
                    np.isnan(scaling_factor)):
                raise SpectrumError(
                    'Scaling factor must be nonzero and finite')
            if div:
                multiplier = 1 / scaling_factor
            else:
                multiplier = scaling_factor
            data = self.data * multiplier
            spect_obj = Spectrum(data, bin_edges_kev=self.bin_edges_kev)
            return spect_obj


def _get_file_object(infilename):
    """
    Parse a file and return an object according to its extension.

    Args:
      infilename: a string representing a path to a parsable file.

    Raises:
      AssertionError: for a bad filename.  # TODO let this be an IOError
      NotImplementedError: for an unparsable file extension.
      ...?

    Returns:
      a file object of type SpeFile, SpcFile, or CnfFile
    """

    _, extension = os.path.splitext(infilename)
    if extension.lower() == '.spe':
        spect_file_obj = parsers.SpeFile(infilename)
    elif extension.lower() == '.spc':
        spect_file_obj = parsers.SpcFile(infilename)
    elif extension.lower() == '.cnf':
        spect_file_obj = parsers.CnfFile(infilename)
    else:
        raise NotImplementedError(
            'File type {} can not be read'.format(extension))
    return spect_file_obj
