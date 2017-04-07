"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import numpy as np
from uncertainties import UFloat, unumpy
import becquerel.parsers as parsers
# from ..parsers import SpeFile, SpcFile, CnfFile


class SpectrumError(Exception):
    """Exception raised by Spectrum."""

    pass


class UncalibratedError(SpectrumError):
    """Raised when an uncalibrated spectrum is treated as calibrated."""

    pass


class Spectrum(object):
    """
    Represents an energy spectrum.

    Initialize a Spectrum directly, or with Spectrum.from_file(filename).

    Data Attributes:
      data: np.array of UFloat objects, of counts in each channel
      data_vals: np.array of floats of counts
      data_uncs: np.array of uncertainties for each bin
      bin_edges_kev: np.array of energy bin edges, if calibrated
      livetime: int or float of livetime, in seconds

    Properties:
      channels: (read-only) np.array of channel index as integers
      is_calibrated: (read-only) bool
      energies_kev: (read-only) np.array of energy bin centers, if calibrated
    """

    def __init__(self, data, uncs=None, bin_edges_kev=None,
                 input_file_object=None, livetime=None):
        """Initialize the spectrum.

        Args:
          data: an iterable of counts per channel. may be a np.array of UFloats
          uncs (optional): an iterable of uncertainty on the counts for each
            channel.
            If data is NOT an uncertainties.UFloat type, and uncs is not given,
            the uncertainties are assumed to be sqrt(N), with a minimum
            uncertainty of 1 (e.g. for 0 counts).
          bin_edges_kev (optional): an iterable of bin edge energies
            If not none, should have length of (len(data) + 1)
          input_file_object (optional): a parser file object
          livetime (optional): the livetime of the spectrum [s]

        Raises:
          SpectrumError: for bad input arguments
        """

        if len(data) == 0:
            raise SpectrumError('Empty spectrum data')
        are_ufloats = [isinstance(d, UFloat) for d in data]
        if all(are_ufloats):
            if uncs is None:
                self._data = np.array(data)
            else:
                raise SpectrumError('Specify uncertainties via uncs arg ' +
                                    'or via UFloats, but not both')
        elif any(are_ufloats):
            raise SpectrumError(
                'Spectrum data should be all UFloats or no UFloats')
        else:
            if uncs is None:
                uncs = np.maximum(np.sqrt(data), 1)
            self._data = unumpy.uarray(data, uncs)

        if bin_edges_kev is None:
            self.bin_edges_kev = None
        elif len(bin_edges_kev) != len(data) + 1:
            raise SpectrumError('Bad length of bin edges vector')
        elif np.any(np.diff(bin_edges_kev) <= 0):
            raise SpectrumError(
                'Bin edge energies must be strictly increasing')
        else:
            self.bin_edges_kev = np.array(bin_edges_kev, dtype=float)

        self._infileobject = input_file_object
        if input_file_object is not None:
            self.infilename = input_file_object.filename
            self.livetime = input_file_object.livetime
        else:
            self.infilename = None
            self.livetime = livetime
            # TODO what if livetime and input_file_object are both specified?

    @property
    def data(self):
        """Counts in each channel, with uncertainty.

        Returns:
          an np.ndarray of uncertainties.ufloats
        """

        return self._data

    @property
    def data_vals(self):
        """Counts in each channel, no uncertainties.

        Returns:
          an np.ndarray of floats
        """

        return unumpy.nominal_values(self._data)

    @property
    def data_uncs(self):
        """Uncertainties in each channel.

        Returns:
          an np.ndarray of floats
        """

        return unumpy.std_devs(self._data)

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
          bool, True if spectrum has defined energy bin edges. False otherwise
        """

        return self.bin_edges_kev is not None

    @classmethod
    def from_file(cls, infilename):
        """Construct a Spectrum object from a filename.

        Args:
          infilename: a string representing the path to a parsable file

        Returns:
          A Spectrum object

        Raises:
          AssertionError: for a bad filename  # TODO make this an IOError
        """

        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data,
                        bin_edges_kev=spect_file_obj.energy_bin_edges,
                        input_file_object=spect_file_obj)

        # TODO Get more attributes from self.infileobj

        return spect_obj

    @staticmethod
    def bin_centers_from_edges(edges_kev):
        """Calculate bin centers from bin edges.

        Args:
          edges_kev: an iterable representing bin edge values

        Returns:
          np.array of length (len(edges_kev) - 1), representing bin center
            values with the same units as the input
        """

        edges_kev = np.array(edges_kev)
        centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
        return centers_kev

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

        if not isinstance(scaling_factor, UFloat):
            try:
                scaling_factor = float(scaling_factor)
            except (TypeError, ValueError):
                raise TypeError(
                    'Spectrum must be multiplied/divided by a scalar')
            if (scaling_factor == 0 or
                    np.isinf(scaling_factor) or
                    np.isnan(scaling_factor)):
                raise SpectrumError(
                    'Scaling factor must be nonzero and finite')
        else:
            if (scaling_factor.nominal_value == 0 or
                    np.isinf(scaling_factor.nominal_value) or
                    np.isnan(scaling_factor.nominal_value)):
                raise SpectrumError(
                    'Scaling factor must be nonzero and finite')
        if div:
            multiplier = 1 / scaling_factor
        else:
            multiplier = scaling_factor
        data = self.data * multiplier
        spect_obj = Spectrum(data, bin_edges_kev=self.bin_edges_kev)
        return spect_obj

    def norm_subtract(self, other):
        """Normalize another spectrum to this one by livetime, and subtract.

        new = self - (self.livetime / other.livetime) * other

        Args:
          other: the Spectrum object to be normalized and subtracted

        Raises:
          TypeError: if other is not a Spectrum instance
          SpectrumError: if the spectra are different lengths

        Returns:
          a new Spectrum with the normalized and subtracted data
        """

        if not isinstance(other, Spectrum):
            raise TypeError(
                'Spectrum addition/subtraction must involve a Spectrum object')
        if len(self.data) != len(other.data):
            raise SpectrumError(
                'Cannot add/subtract spectra of different lengths')

        norm_other = other * (self.livetime / other.livetime)

        return self - norm_other


def _get_file_object(infilename):
    """
    Parse a file and return an object according to its extension.

    Args:
      infilename: a string representing a path to a parsable file

    Raises:
      AssertionError: for a bad filename  # TODO let this be an IOError
      NotImplementedError: for an unparsable file extension
      ...?

    Returns:
      a file object of type SpeFile, SpcFile, or CnfFile
    """

    _, extension = os.path.splitext(infilename)
    if extension.lower() == '.spe':
        return parsers.SpeFile(infilename)
    elif extension.lower() == '.spc':
        return parsers.SpcFile(infilename)
    elif extension.lower() == '.cnf':
        return parsers.CnfFile(infilename)
    else:
        raise NotImplementedError(
            'File type {} can not be read'.format(extension))
