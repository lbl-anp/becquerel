"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
from copy import deepcopy
import datetime
import numpy as np
from uncertainties import UFloat, unumpy
from .. import parsers
from .utils import handle_uncs, handle_datetime, bin_centers_from_edges
from . import plotting

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

    Note on livetime:
      A livetime of None is the default for a counts-based spectrum, and
        indicates a missing or unknown livetime.
      A livetime of np.nan is the default for a CPS-based spectrum, and
        indicates that livetime is not a meaningful quantity for this type of
        spectrum.
      Either of these types of spectrum may be initialized with a livetime
        value. However, any operation that produces a CPS-based spectrum
        (such as a spectrum subtraction) will discard the livetime and set it
        to np.nan since livetime is then a meaningless quantity.
      Operations that produce a counts-based spectrum may or may not preserve a
        livetime value (for example, the sum of two spectra has a livetime
        equal to the sum of the two livetimes; but a scalar multiplication or
        division results in a livetime of None).

    Data Attributes:
      bin_edges_kev: np.array of energy bin edges, if calibrated
      livetime: int or float of livetime, in seconds. See note above
      realtime: int or float of realtime, in seconds. May be None
      infilename: the filename the spectrum was loaded from, if applicable
      start_time: a datetime.datetime object representing the acquisition start
      stop_time: a datetime.datetime object representing the acquisition end

    Properties (read-only):
      counts: counts in each channel, with uncertainty
      counts_vals: counts in each channel, no uncertainty
      counts_uncs: uncertainties on counts in each channel
      cps: counts per second in each channel, with uncertainty
      cps_vals: counts per second in each channel, no uncertainty
      cps_uncs: uncertainties on counts per second in each channel
      cpskev: CPS/keV in each channel, with uncertainty
      cpskev_vals: CPS/keV in each channel, no uncertainty
      cpskev_uncs: uncertainties on CPS/keV in each channel
      channels: np.array of channel index as integers
      is_calibrated: bool indicating calibration status
      energies_kev: np.array of energy bin centers, if calibrated
      bin_widths: np.array of energy bin widths, if calibrated

    Methods:
      apply_calibration: use an EnergyCal object to calibrate this spectrum
      calibrate_like: copy the calibrated bin edges from another spectrum
      rm_calibration: remove the calibrated bin edges
      combine_bins: make a new Spectrum with counts combined into bigger bins
      downsample: make a new Spectrum, downsampled from this one by a factor
      copy: return a deep copy of this Spectrum object
    """

    def __init__(self, counts=None, cps=None, uncs=None, bin_edges_kev=None,
                 input_file_object=None, livetime=None, realtime=None,
                 start_time=None, stop_time=None):
        """Initialize the spectrum.

        Either counts or cps must be specified. Other args are optional.
        If start_time, stop_time, and realtime are being provided, only two of
        these should be specified as arguments, and the third will be
        calculated from the other two.

        See note on livetime in class docstring. As a reminder:
          livetime = None indicates a missing or unknown livetime, and may
            occur in a counts-based spectrum.
          livetime = np.nan indicates that livetime is not meaningful, and may
            occur in a CPS-based spectrum.

        Args:
          counts: counts per channel. array-like of ints, floats or UFloats
          cps: counts per second per channel. array-like of floats or UFloats
          uncs (optional): an iterable of uncertainty on the counts for each
            channel.
            If counts is given, is NOT a UFloat type, and uncs is not given,
            the uncertainties are assumed to be sqrt(N), with a minimum
            uncertainty of 1 (e.g. for 0 counts).
            If cps is given, uncs defaults to an array of np.nan
          bin_edges_kev (optional): an iterable of bin edge energies
            If not none, should have length of (len(counts) + 1)
          input_file_object (optional): a parser file object
          livetime (optional): the livetime of the spectrum [s]
            Note that livetime is not preserved through CPS-based operations,
            such as any subtraction, or addition with a CPS-based spectrum.
          realtime (optional): the duration of the acquisition [s]
          start_time (optional): datetime or string representing the
            acquisition start
          stop_time (optional): datetime or string representing the acquisition
            end.

        Raises:
          ValueError: for bin edges not monotonically increasing;
            livetime > realtime; or start_time > stop_time
          SpectrumError: for bad input arguments
          UncertaintiesError: if uncertainties are overspecified or of mixed
            types
          TypeError: for bad type on input args (start_time, stop_time)
        """

        if not (counts is None) ^ (cps is None):
            raise SpectrumError('Must specify one of counts or CPS')
        if counts is not None:
            if len(counts) == 0:
                raise SpectrumError('Empty spectrum counts')
            self._counts = handle_uncs(
                counts, uncs, lambda x: np.maximum(np.sqrt(x), 1))
            if livetime is None:
                self.livetime = None
            else:
                self.livetime = float(livetime)
            self._cps = None
        else:
            self._counts = None
            if livetime is None:
                self.livetime = np.nan  # default for CPS-based spectra
            else:
                self.livetime = livetime
                # TODO should this be allowed?
                #   all calculations with CPS return livetime=np.nan anyway...
            if len(cps) == 0:
                raise SpectrumError('Empty spectrum counts')
            self._cps = handle_uncs(cps, uncs, lambda x: np.nan)

        if bin_edges_kev is None:
            self.bin_edges_kev = None
        elif len(bin_edges_kev) != len(self) + 1:
            raise SpectrumError('Bad length of bin edges vector')
        elif np.any(np.diff(bin_edges_kev) <= 0):
            raise ValueError(
                'Bin edge energies must be strictly increasing')
        else:
            self.bin_edges_kev = np.array(bin_edges_kev, dtype=float)

        if realtime is None:
            self.realtime = None
        else:
            self.realtime = float(realtime)
            if self.livetime is not None:
                if self.livetime > self.realtime:
                    raise ValueError(
                        'Livetime ({}) cannot exceed realtime ({})'.format(
                            self.livetime, self.realtime))

        self.start_time = handle_datetime(
            start_time, 'start_time', allow_none=True)
        self.stop_time = handle_datetime(
            stop_time, 'stop_time', allow_none=True)

        if (self.realtime is not None
                and self.stop_time is not None
                and self.start_time is not None):
            raise SpectrumError(
                'Specify no more than 2 out of 3 args: ' +
                'realtime, stop_time, start_time')
        elif self.start_time is not None and self.stop_time is not None:
            if self.start_time > self.stop_time:
                raise ValueError(
                    'Stop time ({}) must be after start time ({})'.format(
                        self.start_time, self.stop_time))
            self.realtime = (self.stop_time - self.start_time).total_seconds()
        elif self.start_time is not None and self.realtime is not None:
            self.stop_time = self.start_time + datetime.timedelta(
                seconds=self.realtime)
        elif self.realtime is not None and self.stop_time is not None:
            self.start_time = self.stop_time - datetime.timedelta(
                seconds=self.realtime)

        self._infileobject = input_file_object
        if input_file_object is not None:
            self.infilename = input_file_object.filename
            if self.livetime is None:
                self.livetime = input_file_object.livetime
            if self.realtime is None:
                self.realtime = input_file_object.realtime
            if self.start_time is None:
                self.start_time = input_file_object.collection_start
            if self.stop_time is None:
                self.stop_time = input_file_object.collection_stop
        else:
            self.infilename = None

    @property
    def counts(self):
        """Counts in each channel, with uncertainty.

        Some spectra, including subtraction results, have cps but no counts,
        in which case counts is None.

        Returns:
          an np.array of uncertainties.ufloats, or None
        """

        return self._counts

    @property
    def counts_vals(self):
        """Counts in each channel, no uncertainties.

        Returns:
          an np.array of floats, or None
        """

        if self.counts is None:
            return None
        else:
            return unumpy.nominal_values(self._counts)

    @property
    def counts_uncs(self):
        """Uncertainties on the counts in each channel.

        Returns:
          an np.array of floats, or None
        """

        if self.counts is None:
            return None
        else:
            return unumpy.std_devs(self._counts)

    @property
    def cps(self):
        """Counts per second in each channel, with uncertainty.

        If counts is defined, cps is calculated from counts and livetime.
        Otherwise, it is an independent data property.

        Raises:
          SpectrumError: if counts is defined, but not livetime

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
        """Counts per second per keV in each channel, with uncertainty.

        Raises:
          SpectrumError: if cps is not defined due to missing livetime
          UncalibratedError: if bin edges (and thus bin widths) are not defined

        Returns:
          an np.array of uncertainties.ufloats
        """

        return self.cps / self.bin_widths

    @property
    def cpskev_vals(self):
        """Counts per second per keV in each channel, no uncertainties.

        Returns:
          an np.array of floats
        """

        return unumpy.nominal_values(self.cpskev)

    @property
    def cpskev_uncs(self):
        """Uncertainties on the counts per second per keV in each channel.

        Returns:
          an np.array of floats
        """

        return unumpy.std_devs(self.cpskev)

    def counts_over(self, seconds):
        """Evaluate CPS over a given time interval, to get counts.

        Args:
          seconds: number of seconds to evaluate over

        Returns:
          an np.array of ufloats
        """

        return self.cps * seconds

    def counts_vals_over(self, seconds):
        """Evaluate CPS over a given time interval, to get counts."""

        return unumpy.nominal_values(self.counts_over(seconds))

    def counts_uncs_over(self, seconds):
        return unumpy.std_devs(self.cpskev)

    @property
    def channels(self):
        """Channel index.

        Returns:
          np.array of int's from 0 to (len(self.counts) - 1)
        """

        return np.arange(len(self), dtype=int)

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
            return bin_centers_from_edges(self.bin_edges_kev)

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

        kwargs = {'counts': spect_file_obj.data,
                  'input_file_object': spect_file_obj}

        if spect_file_obj.cal_coeff:
            kwargs['bin_edges_kev'] = spect_file_obj.energy_bin_edges

        # TODO Get more attributes from self.infileobj

        return cls(**kwargs)

    def copy(self):
        """Make a deep copy of this Spectrum object.

        Returns:
          a Spectrum object identical to this one
        """

        return deepcopy(self)

    def __len__(self):
        """The number of channels in the spectrum.

        Returns:
          an int
        """

        if self.counts is not None:
            return len(self.counts)
        else:
            return len(self.cps)

    def __add__(self, other):
        """Add spectra together.

        The livetimes sum (if they exist) and the resulting spectrum
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
          TypeError: if other is not a Spectrum
          SpectrumError: if spectra are different lengths,
            or if only one is calibrated
          NotImplementedError: if spectra are calibrated differently

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
        """Return a new Spectrum object with counts (or CPS) scaled up.

        Args:
          factor: factor to multiply by. May be a ufloat.

        Raises:
          TypeError: if factor is not a scalar value
          SpectrumError: if factor is 0 or infinite

        Returns:
          a new Spectrum object
        """

        return self._mul_div(other, div=False)

    def __div__(self, other):
        """Return a new Spectrum object with counts (or CPS) scaled down.

        Args:
          factor: factor to divide by. May be a ufloat.

        Raises:
          TypeError: if factor is not a scalar value
          SpectrumError: if factor is 0 or infinite

        Returns:
          a new Spectrum object
        """

        return self._mul_div(other, div=True)

    def __truediv__(self, other):
        """Return a new Spectrum object with counts (or CPS) scaled down.

        Args:
          factor: factor to divide by. May be a ufloat.

        Raises:
          TypeError: if factor is not a scalar value
          SpectrumError: if factor is 0 or infinite

        Returns:
          a new Spectrum object
        """

        return self._mul_div(other, div=True)

    def _mul_div(self, scaling_factor, div=False):
        """Multiply or divide a spectrum by a scalar. Handle errors.

        Raises:
          TypeError: if factor is not a scalar value
          ValueError: if factor is 0 or infinite

        Returns:
          a new Spectrum object
        """

        if not isinstance(scaling_factor, UFloat):
            try:
                scaling_factor = float(scaling_factor)
            except (TypeError, ValueError):
                raise TypeError(
                    'Spectrum must be multiplied/divided by a scalar')
            if (scaling_factor == 0 or
                    np.isinf(scaling_factor) or
                    np.isnan(scaling_factor)):
                raise ValueError(
                    'Scaling factor must be nonzero and finite')
        else:
            if (scaling_factor.nominal_value == 0 or
                    np.isinf(scaling_factor.nominal_value) or
                    np.isnan(scaling_factor.nominal_value)):
                raise ValueError(
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

    def downsample(self, f, handle_livetime=None):
        """Downsample counts and create a new spectrum.

        The spectrum is resampled from a binomial distribution. Each count in
        the spectrum is preserved (with a probability of 1/f) or discarded,
        resulting in a new spectrum with fewer counts, in which Poisson
        statistics are preserved (unlike a scalar division operation).

        Note, it is not possible to downsample a CPS-based spectrum because
        such a spectrum does not have "counts" to downsample.

        Args:
          f: factor by which to downsample. Must be greater than 1.
          handle_livetime (optional): Possible values:
            None (default): the resulting spectrum has livetime = None.
            'preserve': the resulting spectrum has the same livetime.
            'reduce': the resulting spectrum has livetime reduced by the
              downsampling factor.

        Raises:
          SpectrumError: if this spectrum is CPS-based
          ValueError: if f < 1, or if handle_livetime is an illegal value

        Returns:
          a new Spectrum instance, downsampled from this spectrum
        """

        if self.counts is None:
            raise SpectrumError('Cannot downsample from CPS')
        if f < 1:
            raise ValueError('Cannot upsample a spectrum; f must be > 1')

        if handle_livetime is None:
            new_livetime = None
        elif handle_livetime.lower() == 'preserve':
            new_livetime = self.livetime
        elif handle_livetime.lower() == 'reduce':
            new_livetime = self.livetime / f
        else:
            raise ValueError('Illegal value for handle_livetime: {}'.format(
                handle_livetime))

        # TODO handle uncertainty?
        old_counts = self.counts_vals.astype(int)
        new_counts = np.random.binomial(old_counts, 1. / f)

        return Spectrum(counts=new_counts,
                        bin_edges_kev=self.bin_edges_kev,
                        livetime=new_livetime)

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

        Raises:
          UncalibratedError: if other Spectrum is not calibrated
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


    def plot(self, *fmt, **kwargs):
        """Plot a spectrum with matplotlib's plot command.

        Args:
          fmt:    matplotlib like plot format string
          x_mode: define what is plotted on x axis ('energy' or 'channel'),
                  defaults to energy if available
          y_mode: define what is plotted on y axis ('counts', 'cps', 'cpskev' 
                  or 'eval_over'), defaults to counts
          ax:     matplotlib axes object, if not provided one is created with the
                  variables provided by 'figsize'
          yscale: matplotlib scale: 'linear', 'log', 'logit', 'symlog'
          title:  costum plot title
          kwargs: arguments that are directly passed to matplotlib's plot command

        Returns:
          matplotlib axes object
        """
        plotter = plotting.SpectrumPlotter(self, *fmt, **kwargs)
        return plotter.plot()


    def fill_between(self, **kwargs):
        """Plot a spectrum with matplotlib's fill_between command

        Args:
          x_mode: define what is plotted on x axis ('energy' or 'channel'),
                  defaults to energy if available
          y_mode: define what is plotted on y axis ('counts', 'cps', 'cpskev' 
                  or 'eval_over'), defaults to counts
          ax:     matplotlib axes object, if not provided one is created with the
                  variables provided by 'figsize'
          yscale: matplotlib scale: 'linear', 'log', 'logit', 'symlog'
          title:  costum plot title
          kwargs: arguments that are directly passed to matplotlib's fill_between
                  command

        Returns:
          matplotlib axes object
        """
        plotter = plotting.SpectrumPlotter(self, **kwargs)
        return plotter.fill_between()

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
