"""Quantities of a nuclear isotope, with decay and activation tools."""

from __future__ import print_function
import datetime
import copy
import numpy as np
from .isotope import Isotope
from ..core import utils
from collections import OrderedDict

UCI_TO_BQ = 3.7e4
N_AV = 6.022141e23


class IsotopeQuantityError(Exception):
    """Raised by the IsotopeQuantity class"""

    pass


def handle_isotope(isotope, error_name=None):
    """Handle string or Isotope input.

    Args:
      isotope: either a string of an isotope name, or an Isotope object

    Raises:
      TypeError: if isotope is not a string or Isotope
      IsotopeError: if string is bad

    Returns:
      an Isotope object
    """

    if isinstance(isotope, Isotope):
        return isotope
    elif utils.isstring(isotope):
        return Isotope(isotope)
    else:
        raise TypeError(
            '{} needs an Isotope instance or string, not {}'.format(
                error_name, isotope))


class IsotopeQuantity(object):
    """An amount of an isotope.

    Can be multiplied or divided by a scalar, to produce a copy with the same
    isotope and reference date but a scaled reference quantity.

    Two IsotopeQuantity instances are equal iff they are the same isotope
    and the quantities are np.isclose for any given datetime.

    Construction class methods:
      from_decays: activity based on number of decays in a given time interval
      from_comparison: activity by comparing to a measured known sample

    Data Attributes:
      isotope: an Isotope object, the isotope of this material
      half_life: the half life of the isotope, in seconds
      decay_const: the decay constant of the isotope, in 1/seconds
      is_stable: bool representing whether the isotope is considered stable
      ref_date: a datetime object representing the reference date/time
      ref_atoms: the number of atoms of the isotope, at the reference time.

    Methods:
      atoms_at: number of atoms at given time
      bq_at: activity in Bq at given time
      uci_at: activity in uCi at given time
      g_at: mass in grams at given time
      atoms_now, bq_now, uci_now, g_now: quantity at current time
      decays_from: number of decays during a time interval
      bq_from, uci_from: average activity during a time interval
      decays_during: number of decays during a Spectrum measurement
      bq_during, uci_during: average activity during a Spectrum measurement
      time_when: time at which activity or mass equals a given value
    """

    def __init__(self, isotope, date=None, stability=1e18, **kwargs):
        """Initialize.

        Specify one of bq, uci, atoms, g to define the quantity.

        Args:
          isotope: an Isotope object, of which this is a quantity,
            OR a string to instantiate the Isotope
          date: the reference date for the activity or mass
          stability: half-life above which an isotope is considered stable [s]
          bq: the activity at the reference date [Bq]
          uci: the activity at the reference date [uCi]
          atoms: the number of atoms at the reference date
          g: the mass at the reference date [g]

        Raises:
          TypeError: if isotope is not an Isotope object
          AttributeError: if isotope is missing half_life or decay_const
          IsotopeQuantityError: if no valid quantity kwarg specified
        """

        self._init_isotope(isotope, stability)
        self._init_date(date)
        self._ref_quantities = self._quantities_from_kwargs(**kwargs)

    def _init_isotope(self, isotope, stability):
        """Initialize the isotope.

        Args:
          isotope: an Isotope object, or a string that defines an Isotope
          stability: the half-life above which an isotope is considered
            stable [s]

        Raises:
          TypeError: if isotope is not an Isotope object
          AttributeError: if isotope is missing half_life or decay_const
        """

        self.isotope = handle_isotope(isotope, error_name='IsotopeQuantity')

        self.half_life = self.isotope.half_life
        self.decay_const = self.isotope.decay_const
        self.is_stable = self.half_life > stability

    def _init_date(self, date):
        """Initialize the reference date/time.

        Args:
          date: a date string or datetime.datetime object
        """

        self.ref_date = utils.handle_datetime(
            date, error_name='IsotopeQuantity date', allow_none=True)
        if self.ref_date is None:
            # assume a long-lived source in the current epoch
            self.ref_date = datetime.datetime.now()

    def _quantities_from_kwargs(self, **kwargs):
        """Parse kwargs and return a quantity as a OrderedDictionary. The first
        element in the dictionary is the provided quantity.

        Args (specify one):
          atoms: the number of atoms
          bq: the activity [Bq]
          uci: the activity [uCi]
          g: the mass [g]
          _init_empty: (internal use only) set True if the reference quantity
            will be set later

        Raises:
          IsotopeQuantityError: if no valid argument specified
        """

        assert len(kwargs) == 1
        ref_quantities = OrderedDict()
        if '_init_empty' in kwargs:
            return ref_quantities
        if ('bq' in kwargs or 'uci' in kwargs) and self.is_stable:
            raise IsotopeQuantityError(
                'Cannot initialize a stable IsotopeQuantity from activity')

        # dictionary with functions that define how to calculate all quantities
        # in a circular manner
        conversions = dict(
            atoms=lambda : ref_quantities["g"] / self.isotope.A * N_AV,
            bq=lambda : ref_quantities["atoms"] * self.decay_const,
            uci=lambda : ref_quantities["bq"] / UCI_TO_BQ,
            g=lambda : ref_quantities["uci"] * UCI_TO_BQ /
                       self.decay_const / N_AV * self.isotope.A
        )

        # rotates the order of the list so that the provided kwarg is at [0]
        order = ["atoms", "bq", "uci", "g"]
        if next(iter(kwargs)) not in order:
            raise IsotopeQuantityError("Unknown isotope quantity.")
        while order[0] not in kwargs:
            order.append(order.pop(0))
        first = order.pop(0)
        ref_quantities[first] = self._check_positive_qty(kwargs[first])
        for i in order:
            ref_quantities[i] = conversions[i]()
        return ref_quantities

    def _check_positive_qty(self, val):
        """Check that the quantity value is a nonnegative float or ufloat.

        Raises:
          ValueError: if val is negative
        """

        val *= 1.   # convert to float, or preserve ufloat, as appropriate
        if val < 0:
            raise ValueError(
                'Mass or activity must be a positive quantity: {}'.format(val))
        return val

    @property
    def ref_atoms(self):
        """
        Access the reference atoms directly (for backwards compatibility)
        """
        return self._ref_quantities["atoms"]

    @classmethod
    def from_decays(cls, isotope, n_decays, start_time, stop_time):
        """
        Create an IsotopeQuantity from a known number of decays in an interval.

        Args:
          isotope: string or Isotope instance
          n_decays: int or float of the number of decays in the time interval
          start_time: string or datetime of the beginning of the interval
          stop_time: string or datetime of the end of the interval

        Returns:
          an IsotopeQuantity, referenced to start_time

        Raises:
          TypeError: if start_time or stop_time is not a datetime or string
          ValueError: if timestamps are out of order
        """

        obj = cls(isotope, date=start_time, _init_empty=True)

        stop_time = utils.handle_datetime(stop_time)
        duration = (stop_time - obj.ref_date).total_seconds()
        if duration < 0:
            raise ValueError(
                'Start time must precede stop time: {}, {}'.format(
                    start_time, stop_time))
        atoms = float(n_decays) / (1 - np.exp(-obj.decay_const * duration))

        obj._ref_quantities = obj._quantities_from_kwargs(atoms=atoms)
        return obj

    @classmethod
    def from_comparison(cls, isotope_qty1, counts1, interval1,
                        counts2, interval2):
        """Calculate an IsotopeQuantity by comparison with a known sample.

        Assumes the samples are in identical geometry with the detector.

        Args:
          isotope_qty1: an IsotopeQuantity of the known sample
          counts1: net counts measured in the known sample
          interval1: (start_time, stop_time) of the known sample measurement
          counts2: net counts measured in the unknown sample
          interval2: (start_time, stop_time) of the unknown sample measurement

        Returns:
          an IsotopeQuantity of the unknown sample

        Raises:
          IsotopeQuantityError: if intervals are not length 2
          TypeError: if interval elements are not datetimes or date strings
          ValueError: if timestamps are out of order
        """

        norm = decay_normalize(isotope_qty1.isotope, interval1, interval2)
        ratio = (counts2 * norm) / counts1

        return isotope_qty1 * ratio

    # ----------------------------
    #   *_at()
    # ----------------------------

    def quantity_at(self, quantity, date):
        """Return a quantity at a given time.

        Args:
          date: the date to calculate for

        Returns:
          a float of the number of atoms at date

        Raises:
          TypeError: if date is not recognized
        """

        t1 = utils.handle_datetime(date)
        dt = (t1 - self.ref_date).total_seconds()
        return self._ref_quantities[quantity] * 2**(-dt / self.half_life)


    def atoms_at(self, date):
        """Calculate the number of atoms at a given time.

        Args:
          date: the date to calculate for

        Returns:
          a float of the number of atoms at date

        Raises:
          TypeError: if date is not recognized
        """

        return self.quantity_at("atoms", date)

    def bq_at(self, date):
        """Calculate the activity [Bq] at a given time.

        As atoms_at() except for return value.
        """

        return self.quantity_at("bq", date)

    def uci_at(self, date):
        """Calculate the activity [uCi] at a given time.

        As atoms_at() except for return value.
        """

        return self.quantity_at("uci", date)

    def g_at(self, date):
        """Calculate the mass [g] at a given time.

        As atoms_at() except for return value.
        """

        return self.quantity_at("g", date)

    # ----------------------------
    #   *_now()
    # ----------------------------

    def atoms_now(self):
        """Calculate the number of atoms now.

        Returns:
          a float of the number of atoms at datetime.datetime.now()
        """

        return self.quantity_at("atoms", datetime.datetime.now())

    def bq_now(self):
        """Calculate the activity [Bq] now.

        As atoms_now() except for return value.
        """

        return self.quantity_at("bq", datetime.datetime.now())

    def uci_now(self):
        """Calculate the activity [uCi] now.

        As atoms_now() except for return value.
        """

        return self.quantity_at("uci", datetime.datetime.now())

    def g_now(self):
        """Calculate the mass [g] now.

        As atoms_now() except for return value.
        """

        return self.quantity_at("g", datetime.datetime.now())

    # ----------------------------
    #   *_from()
    # ----------------------------

    def decays_from(self, start_time, stop_time):
        """The expected number of decays from start_time to stop_time.

        Args:
          start_time: a string or datetime.datetime object
          stop_time: a string or datetime.datetime object

        Returns:
          a float of the number of decays in the time interval

        Raises:
          TypeError: if start_time or stop_time is not recognized
        """

        return self.atoms_at(start_time) - self.atoms_at(stop_time)

    def bq_from(self, start_time, stop_time):
        """Average activity [Bq] from start_time to stop_time.

        As decays_from() except for return value.
        """

        t0 = utils.handle_datetime(start_time, error_name='start_time')
        t1 = utils.handle_datetime(stop_time, error_name='stop_time')

        return self.decays_from(t0, t1) / (t1 - t0).total_seconds()

    def uci_from(self, start_time, stop_time):
        """Average activity [uCi] from start_time to stop_time.

        As decays_from() except for return value.
        """

        return self.bq_from(start_time, stop_time) / UCI_TO_BQ

    # ----------------------------
    #   *_during()
    # ----------------------------

    def decays_during(self, spec):
        """Calculate the expected number of decays during a measured spectrum.

        Args:
          spec: a Spectrum object containing start_time and stop_time

        Returns:
          a float of the number of decays during the acquisition of spec

        Raises:
          TypeError: if spec does not have start_time or stop_time defined
        """

        return self.decays_from(spec.start_time, spec.stop_time)

    def bq_during(self, spec):
        """Average activity [Bq] during the spectrum.

        As decays_during(), except for return value.
        """

        return self.bq_from(spec.start_time, spec.stop_time)

    def uci_during(self, spec):
        """Average activity [uCi] during the spectrum.

        As decays_during(), except for return value.
        """

        return self.uci_from(spec.start_time, spec.stop_time)

    # ----------------------------
    #   (other)
    # ----------------------------

    def time_when(self, **kwargs):
        """Calculate the date/time when the mass/activity is a given value.

        Args (specify one):
          atoms: number of atoms
          bq: activity [Bq]
          uci: activity [uCi]
          g: mass [g]

        Returns:
          a datetime.datetime of the moment when the mass/activity equals the
            specified input

        Raises:
          IsotopeQuantityError: if isotope is stable
        """

        if self.is_stable:
            raise IsotopeQuantityError(
                'Cannot calculate time_when for stable isotope')

        assert len(kwargs) == 1
        key = next(iter(kwargs))
        target = kwargs[key]
        dt = -self.half_life * np.log2(target / self._ref_quantities[key])
        return self.ref_date + datetime.timedelta(seconds=dt)

    def __str__(self):
        """Return a string representation.

        Shows grams if isotope is stable, otherwise Bq.
        """

        if self.isotope.is_stable:
            s = '{} g of {}'.format(self.g_at(self.ref_date), self.isotope)
        else:
            s = '{} Bq of {} (at {})'.format(
                self.bq_at(self.ref_date), self.isotope, self.ref_date)
        return s

    def __mul__(self, other):
        """Multiply the quantity"""

        return self._mul_div(other, div=False)

    def __div__(self, other):
        """Divide the quantity"""

        return self._mul_div(other, div=True)

    def __truediv__(self, other):
        """Divide the quantity (python 3)"""

        return self._mul_div(other, div=True)

    def _mul_div(self, other, div=False):
        """Multiply or divide the quantity.

        Args:
          other: a scalar to multiply/divide by
          div: a bool, True if dividing, False if multiplying

        Returns:
          a new IsotopeQuantity, same reference date, scaled quantity
        """

        if div:
            factor = 1 / float(other)
        else:
            factor = float(other)
        key = next(iter(self._ref_quantities))
        return IsotopeQuantity(
            copy.deepcopy(self.isotope),
            **{"date": self.ref_date, key: self._ref_quantities[key] * factor}
        )

    def __eq__(self, other):
        """Equality operation"""

        if not isinstance(other, IsotopeQuantity):
            return False
        else:
            # This supports uncertanties too
            a = self._ref_quantities["atoms"]
            b = other.atoms_at(self.ref_date)
            return (self.isotope == other.isotope and
                    abs(a - b) <= 1e-9 * max(abs(a), abs(b))
                   )

class NeutronIrradiationError(Exception):
    """Exception from NeutronIrradiation class."""

    pass


class NeutronIrradiation(object):
    """Represents an irradiation period with thermal neutrons.

    Data attributes:
      start_time: beginning of irradiation
      stop_time: end of irradiation
      duration: number of seconds of irradiation
      n_cm2_s: neutron flux (if duration is nonzero)
      n_cm2: neutron fluence

    Methods:
      activate: Calculate an IsotopeQuantity from before or after irradiation
    """

    def __init__(self, start_time, stop_time, n_cm2=None, n_cm2_s=None):
        """Initialize.

        Either n_cm2 or n_cm2_s is a required input.

        Args:
          start_time: datetime or date string representing start of irradiation
          stop_time: datetime or date string representing end of irradiation
          n_cm2: the total fluence of neutrons over the irradiation.
          n_cm2_s: the flux of neutrons during the irradiation.

        Raises:
          TypeError: if timestamps are not parseable
          ValueError: if timestamps are out of order,
            or if flux/fluence not specified,
            or if flux and fluence both specified
        """

        self.start_time = utils.handle_datetime(
            start_time, error_name='NeutronIrradiation start_time')
        self.stop_time = utils.handle_datetime(
            stop_time, error_name='NeutronIrradiation stop_time')
        if self.stop_time < self.start_time:
            raise ValueError('Timestamps out of order: {}, {}'.format(
                self.start_time, self.stop_time))
        self.duration = (self.stop_time - self.start_time).total_seconds()

        if not ((n_cm2 is None) ^ (n_cm2_s is None)):
            raise ValueError('Must specify either n_cm2 or n_cm2_s, not both')
        elif n_cm2 is None:
            self.n_cm2_s = n_cm2_s
            self.n_cm2 = n_cm2_s * self.duration
        elif n_cm2_s is None and self.duration > 0:
            self.n_cm2_s = n_cm2 / self.duration
            self.n_cm2 = n_cm2
        else:
            self.n_cm2_s = None
            self.n_cm2 = n_cm2

    def __str__(self):
        """Return a string representation.

        Shows flux if duration is nonzero, otherwise shows fluence.
        """

        if self.duration == 0:
            return '{} neutrons/cm2 at {}'.format(self.n_cm2, self.start_time)
        else:
            return '{} n/cm2/s from {} to {}'.format(
                self.n_cm2_s, self.start_time, self.stop_time)

    def activate(self, barns, initial, activated):
        """
        Calculate an IsotopeQuantity from before or after a neutron activation.

        For a forward calculation (known initial quantity, to calculate the
        activated result), specify an initial IsotopeQuantity and an
        activated Isotope.

        For a backward calculation (known activation product, to calculate the
        initial quantity), specify an initial Isotope and an activated
        IsotopeQuantity.

        Forward equations:
          A1 = phi * sigma * N0 * (1 - exp(-lambda * t_irr))
          A1 = n * sigma * N0 * lambda
        Backward equations:
          N0 = A1 / (phi * sigma * (1 - exp(-lambda * t_irr)))
          N0 = A1 / (n * sigma * lambda)

        in all equations:
          A1 = activated activity [Bq] at end of irradiation,
          phi = flux [neutrons/cm2/s],
          sigma = activation cross-section [cm2],
          N0 = number of atoms of initial isotope,
          lambda = activity coefficient of activated isotope [1/s],
          t_irr = duration of irradiation [s]
          n = fluence of zero-duration irradiation [neutrons/cm2],

        Args:
          barns: cross section for activation [barns = 1e-24 cm^2]
          initial: the isotope being activated, an IsotopeQuantity or Isotope.
            Specify an IsotopeQuantity if the initial quantity is known.
            Specify an Isotope if the initial quantity is unknown
          activated: the activated isotope, an IsotopeQuantity or Isotope.
            Specify an IsotopeQuantity if the activated quantity is known.
            Specify an Isotope if the activated quantity is unknown

        Returns:
          an IsotopeQuantity, corresponding to either the initial isotope or
            the activated isotope, depending on which quantity was input

        Raises:
          NeutronIrradiationError: if initial and activated are overspecified
            or underspecified
          TypeError: if initial and activated are not Isotope or
            IsotopeQuantity objects
        """

        if (isinstance(initial, IsotopeQuantity) and
                isinstance(activated, IsotopeQuantity)):
            raise NeutronIrradiationError(
                "Two IsotopeQuantity's in args, nothing left to calculate!" +
                'Args: {}, {}'.format(initial, activated))
        elif (isinstance(initial, IsotopeQuantity) and
              isinstance(activated, Isotope)):
            forward = True
        elif (isinstance(initial, Isotope) and
              isinstance(activated, IsotopeQuantity)):
            forward = False
        elif isinstance(initial, Isotope) and isinstance(activated, Isotope):
            raise NeutronIrradiationError(
                'No IsotopeQuantity specified, not enough data. ' +
                'Args: {}, {}'.format(initial, activated))
        else:
            raise TypeError(
                'Input args should be Isotope or IsotopeQuantity objects: ' +
                '{}, {}'.format(initial, activated))

        if not initial.is_stable:
            raise NotImplementedError(
                'Activation not implemented for a radioactive initial isotope')

        cross_section = barns * 1.0e-24

        if forward:
            if self.duration == 0:
                activated_bq = (
                    self.n_cm2 * cross_section *
                    initial.atoms_at(self.stop_time) *
                    activated.decay_const)
            else:
                activated_bq = (
                    self.n_cm2_s * cross_section *
                    initial.atoms_at(self.stop_time) *
                    (1 - np.exp(-activated.decay_const * self.duration))
                )
            return IsotopeQuantity(activated,
                                   date=self.stop_time, bq=activated_bq)
        else:
            if self.duration == 0:
                initial_atoms = (
                    activated.bq_at(self.stop_time) /
                    (self.n_cm2 * cross_section * activated.decay_const))
            else:
                initial_atoms = (
                    activated.bq_at(self.stop_time) /
                    (self.n_cm2_s * cross_section * (1 - np.exp(
                        -activated.decay_const * self.duration))))
            return IsotopeQuantity(initial,
                                   date=self.start_time, atoms=initial_atoms)


def decay_normalize(isotope, interval1, interval2):
    """Calculate the ratio to normalize decays between time intervals.

    If interval2 averages 1 Bq, what is interval1's average?

    Args:
      isotope: Isotope object or string of the isotope that is decaying
      interval1: (start_time, stop_time) in datetimes or strings
      interval2: (start_time, stop_time) in datetimes or strings

    Returns:
      ratio of (expected decays in interval1) / (expected decays in interval2).
        In other words, multiply measured counts in interval2 by this ratio
        to get the expected counts in interval1.

    Raises:
      IsotopeQuantityError: if intervals are not of length 2
      ValueError: if timestamps are out of order
      TypeError: if timestamps are not parseable, or isotope is not an Isotope
    """

    isotope = handle_isotope(isotope, error_name='decay_normalize')
    if len(interval1) != 2:
        raise IsotopeQuantityError(
            'interval1 should be length 2: {}'.format(interval1))
    elif len(interval2) != 2:
        raise IsotopeQuantityError(
            'interval2 should be length 2: {}'.format(interval2))
    start1 = utils.handle_datetime(interval1[0], error_name='decay_normalize')
    stop1 = utils.handle_datetime(interval1[1], error_name='decay_normalize')
    start2 = utils.handle_datetime(interval2[0], error_name='decay_normalize')
    stop2 = utils.handle_datetime(interval2[1], error_name='decay_normalize')
    if stop1 < start1:
        raise ValueError('Timestamps in interval1 out of order: {}, {}'.format(
            start1, stop1))
    elif stop2 < start2:
        raise ValueError('Timestamps in interval2 out of order: {}, {}'.format(
            start2, stop2))

    # TODO base this on countrate, not counts

    iq = IsotopeQuantity.from_decays(isotope, 1.0, start2, stop2)
    return iq.decays_from(start1, stop1)


def decay_normalize_spectra(isotope, spec1, spec2):
    """Calculate the ratio to normalize decays between measurements.

    If spec2 averages 1 Bq, what is spec1's average?

    Args:
      isotope: Isotope object or string of the isotope that is decaying
      spec1: Spectrum object with start_time, stop_time
      spec2: Spectrum object with start_time, stop_time

    Returns:
      ratio of (expected decays during spec1) / (expected decays during spec2).
        In other words, multiply measured counts in spec2 by this ratio
        to get the expected counts in spec1.

    Raises:
      TypeError: if isotope is not an Isotope instance
      AttributeError: if spec1 or spec2 do not have start_time and stop_time
    """

    return decay_normalize(isotope,
                           (spec1.start_time, spec1.stop_time),
                           (spec2.start_time, spec2.stop_time))
