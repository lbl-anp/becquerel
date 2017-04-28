"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
from copy import deepcopy
import numpy as np
from uncertainties import UFloat, unumpy
from becquerel import parsers
from becquerel.core import utils
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
      bin_edges_kev: np.array of energy bin edges, if calibrated
      livetime: int or float of livetime, in seconds

    Properties:
      counts: (read-only) np.array of UFloat objects, of counts in each channel
      counts_vals: (read-only) np.array of floats of counts
      counts_uncs: (read-only) np.array of uncertainties for each bin
      channels: (read-only) np.array of channel index as integers
      is_calibrated: (read-only) bool
      energies_kev: (read-only) np.array of energy bin centers, if calibrated
      bin_widths: (read-only) np.array of energy bin widths, if calibrated

    Methods:
      apply_calibration: use an EnergyCal object to calibrate this spectrum
      norm_subtract: livetime-normalize and subtract another Spectrum object
      combine_bins: make a new Spectrum with counts combined into bigger bins.
    """

    def __init__(self, counts=None, cps=None, uncs=None, bin_edges_kev=None,
                 input_file_object=None, livetime=None):
        """Initialize the spectrum.

        Either counts or cps must be specified. Other args are optional.

        Args:
          counts: counts per channel. array-like of ints, floats or UFloats
          cps: counts per second per channel. array-like of floats or UFloats
          uncs (optional): an iterable of uncertainty on the counts for each
            channel.
            If counts is NOT a UFloat type, and uncs is not given,
            the uncertainties are assumed to be sqrt(N), with a minimum
            uncertainty of 1 (e.g. for 0 counts).
          bin_edges_kev (optional): an iterable of bin edge energies
            If not none, should have length of (len(counts) + 1)
          input_file_object (optional): a parser file object
          livetime (optional): the livetime of the spectrum [s]

        Raises:
          SpectrumError: for bad input arguments
        """

        if not (counts is None) ^ (cps is None):
            raise SpectrumError('Must specify one of counts or CPS')
        if counts is not None:
            self._counts = self._input_data_check(
                counts, uncs, lambda x: np.maximum(np.sqrt(x), 1))
            self.livetime = livetime
            self._cps = None
            data = counts
        else:
            self._counts = None
            if livetime is None:
                self.livetime = np.nan
            else:
                self.livetime = livetime
                # TODO should this be allowed?
                #   all calculations with CPS return livetime=np.nan anyway...
            self._cps = self._input_data_check(cps, uncs, lambda x: np.nan)
            data = cps

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
            if self.livetime is None:
                self.livetime = input_file_object.livetime
        else:
            self.infilename = None

    def _input_data_check(self, data, uncs, default_unc_func):
        """Check counts or cps for basic problems"""

        if len(data) == 0:
            raise SpectrumError('Empty spectrum counts')

        return utils.handle_uncs(data, uncs, default_unc_func)

    @property
    def counts(self):
        """Counts in each channel, with uncertainty.

        Returns:
          an np.array of uncertainties.ufloats
        """

        return self._counts

    @property
    def counts_vals(self):
        """Counts in each channel, no uncertainties.

        Returns:
          an np.array of floats
        """

        return unumpy.nominal_values(self._counts)

    @property
    def counts_uncs(self):
        """Uncertainties on the counts in each channel.

        Returns:
          an np.array of floats
        """

        return unumpy.std_devs(self._counts)

    @property
    def cps(self):
        """Counts per second in each channel, with uncertainty.

        Returns:
          an np.array of uncertainties.ufloats
        """

        if self._cps is not None:
            return self._cps
        else:
            try:
                return self.counts / self.livetime
            except TypeError:
                raise SpectrumError(
                    'Unknown livetime; cannot calculate CPS from counts')

    @property
    def cps_vals(self):
        """Counts per second in each channel, no uncertainties.

        Returns:
          an np.array of floats
        """

        return unumpy.nominal_values(self.cps)

    @property
    def cps_uncs(self):
        """Uncertainties on the counts per second in each channel.

        Returns:
          an np.array of floats
        """

        return unumpy.std_devs(self.cps)

    @property
    def cpskev(self):
        return self.cps / self.bin_widths

    @property
    def cpskev_vals(self):
        return unumpy.nominal_values(self.cpskev)

    @property
    def cpskev_uncs(self):
        return unumpy.std_devs(self.cpskev)

    @property
    def channels(self):
        """Channel index.

        Returns:
          np.array of int's from 0 to (len(self.counts) - 1)
        """

        return np.arange(len(self.counts), dtype=int)

    @property
    def energies_kev(self):
        """Convenience function for accessing the energies of bin centers.

        Returns:
          np.array of floats, same length as self.counts

        Raises:
          UncalibratedError: if spectrum is not calibrated
        """

        if not self.is_calibrated:
            raise UncalibratedError('Spectrum is not calibrated')
        else:
            return utils.bin_centers_from_edges(self.bin_edges_kev)

    @property
    def bin_widths(self):
        """The width of each bin, in keV.

        Returns:
          np.array of floats, same length as self.counts

        Raises:
          UncalibratedError: if spectrum is not calibrated
        """

        if not self.is_calibrated:
            raise UncalibratedError('Spectrum is not calibrated')
        else:
            return np.diff(self.bin_edges_kev)

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

        if spect_file_obj.cal_coeff:
            spect_obj = cls(spect_file_obj.data,
                            bin_edges_kev=spect_file_obj.energy_bin_edges,
                            input_file_object=spect_file_obj)
        else:
            spect_obj = cls(spect_file_obj.data,
                            input_file_object=spect_file_obj)

        # TODO Get more attributes from self.infileobj

        return spect_obj

    def copy(self):
        """Make a deep copy of this Spectrum object.

        Returns:
          a Spectrum object identical to this one
        """

        return deepcopy(self)

    def __len__(self):
        if self.counts is not None:
            return len(self.counts)
        else:
            return len(self.cps)

    def __add__(self, other):
        """Add spectra together.

        The livetimes combine (if they exist) and the resulting spectrum
        is still Poisson-distributed.

        The two spectra may both be uncalibrated, or both be calibrated
        with the same energy calibration.

        Args:
          other: another Spectrum object to add counts from

        Raises:
          TypeError: if other is not a Spectrum
          SpectrumError: if spectra are different lengths,
            or if only one is calibrated
          NotImplementedError: if spectra are calibrated differently

        Returns:
          a summed Spectrum object
        """

        self._add_sub_error_checking(other)

        if self.counts is not None and other.counts is not None:
            kwargs = {'counts': self.counts + other.counts}
            if self.livetime and other.livetime:
                kwargs['livetime'] = self.livetime + other.livetime
        else:
            kwargs = {'cps': self.cps + other.cps}
        spect_obj = Spectrum(
            bin_edges_kev=self.bin_edges_kev, **kwargs)
        return spect_obj

    def __sub__(self, other):
        """Normalize spectra and subtract.

        The resulting spectrum does not have a meaningful livetime or
        counts vector, and is NOT Poisson-distributed.

        The two spectra may both be uncalibrated, or both be calibrated
        with the same energy calibration.

        Args:
          other: another Spectrum object, to normalize and subtract

        Raises:

        Returns:
          a subtracted Spectrum object
        """

        self._add_sub_error_checking(other)

        cps = self.cps - other.cps
        spect_obj = Spectrum(cps=cps, bin_edges_kev=self.bin_edges_kev)

        return spect_obj

    def _add_sub_error_checking(self, other):
        """Handle errors for spectra addition or subtraction.

        Args:
          other: a spectrum

        Raises:
          TypeError: if other is not a Spectrum
          SpectrumError: if spectra are different lengths,
            or if only one is calibrated
          NotImplementedError: if spectra are calibrated differently
        """

        if not isinstance(other, Spectrum):
            raise TypeError(
                'Spectrum addition/subtraction must involve a Spectrum object')
        if len(self) != len(other):
            raise SpectrumError(
                'Cannot add/subtract spectra of different lengths')
        if self.is_calibrated ^ other.is_calibrated:
            raise SpectrumError(
                'Cannot add/subtract uncalibrated spectrum to/from a ' +
                'calibrated spectrum. If both have the same calibration, ' +
                'please use the "calibrate_like" method')
        if self.is_calibrated and other.is_calibrated:
            if not np.all(self.bin_edges_kev == other.bin_edges_kev):
                raise NotImplementedError(
                    'Addition/subtraction for arbitrary calibrated spectra ' +
                    'not implemented')
                # TODO: if both spectra are calibrated but with different
                #   calibrations, should one be rebinned to match?

    def __mul__(self, other):
        return self._mul_div(other, div=False)

    def __div__(self, other):
        return self._mul_div(other, div=True)

    def __truediv__(self, other):
        return self._mul_div(other, div=True)

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

        if self.counts is not None:
            data_arg = {'counts': self.counts * multiplier}
        else:
            data_arg = {'cps': self.cps * multiplier}
        spect_obj = Spectrum(bin_edges_kev=self.bin_edges_kev, **data_arg)
        return spect_obj

    def downsample(self, f):
        """Downsample counts and create a new spectrum.

        Args:
          f: factor by which to downsample. Must be greater than 1.

        Raises:
          SpectrumError: if f < 1

        Returns:
          a new Spectrum instance, downsampled from this spectrum
        """

        if self.counts is None:
            raise SpectrumError('Cannot downsample from CPS')
        if f < 1:
            raise SpectrumError('Cannot upsample a spectrum; f must be > 1')

        # TODO handle uncertainty?
        old_counts = self.counts_vals.astype(int)
        new_counts = np.random.binomial(old_counts, 1. / f)

        return Spectrum(counts=new_counts, bin_edges_kev=self.bin_edges_kev)

    def apply_calibration(self, cal):
        """Use an EnergyCal to generate bin edge energies for this spectrum.

        Args:
          cal: an object derived from EnergyCalBase
        """

        n_edges = len(self.channels) + 1
        channel_edges = np.linspace(-0.5, self.channels[-1] + 0.5, num=n_edges)
        self.bin_edges_kev = cal.ch2kev(channel_edges)

    def calibrate_like(self, other):
        """Apply another Spectrum object's calibration (bin edges vector).

        Bin edges are copied, so the two spectra do not have the same object
        in memory.

        Args:
          other: spectrum to copy the calibration from
        """

        if other.is_calibrated:
            self.bin_edges_kev = other.bin_edges_kev.copy()
        else:
            raise UncalibratedError('Other spectrum is not calibrated')

    def rm_calibration(self):
        """Remove the calibration (if it exists) from this spectrum."""

        self.bin_edges_kev = None

    def combine_bins(self, f):
        """Make a new Spectrum with counts combined into bigger bins.

        If f is not a factor of the number of channels, the counts from the
        first spectrum will be padded with zeros.

        len(new.counts) == np.ceil(float(len(self.counts)) / f)

        Args:
          f: an int representing the number of bins to combine into one

        Returns:
          a Spectrum object with counts from this spectrum, but with
            fewer bins
        """

        f = int(f)
        if self.counts is None:
            key = 'cps'
        else:
            key = 'counts'
        data = getattr(self, key)
        if len(self) % f == 0:
            padded_counts = np.copy(data)
        else:
            pad_len = f - len(self) % f
            pad_counts = unumpy.uarray(np.zeros(pad_len), np.zeros(pad_len))
            padded_counts = np.concatenate((data, pad_counts))
        padded_counts.resize(int(len(padded_counts) / f), f)
        combined_counts = np.sum(padded_counts, axis=1)
        if self.is_calibrated:
            combined_bin_edges = self.bin_edges_kev[::f]
            if combined_bin_edges[-1] != self.bin_edges_kev[-1]:
                combined_bin_edges = np.append(
                    combined_bin_edges, self.bin_edges_kev[-1])
        else:
            combined_bin_edges = None

        kwargs = {key: combined_counts,
                  'bin_edges_kev': combined_bin_edges,
                  'input_file_object': self._infileobject,
                  'livetime': self.livetime}
        obj = Spectrum(**kwargs)
        return obj


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
