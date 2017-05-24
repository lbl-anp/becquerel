"""Quantities of a nuclear isotope, with decay and activation tools."""

from __future__ import print_function
import datetime
from six import string_types
import numpy as np
from .isotope import Isotope
from ..core import utils

UCI_TO_BQ = 3.7e4
N_AV = 6.022141e23


class IsotopeQuantityError(Exception):
    """Raised by the IsotopeQuantity class"""

    pass


class IsotopeQuantity(object):
    """An amount of an isotope."""

    def __init__(self, isotope, date=None, **kwargs):
        """Initialize.

        Specify one of bq, uci, atoms, g to define the quantity.

        Args:
          isotope: an Isotope object, of which this is a quantity
          date: the reference date for the activity or mass
          bq: the activity at the reference date [Bq]
          uci: the activity at the reference date [uCi]
          atoms: the number of atoms at the reference date
          g: the mass at the reference date [g]

        Raises:
          ...
        """

        self._init_isotope(isotope)
        self._init_date(date)
        self.ref_atoms = self._atoms_from_kwargs(**kwargs)

    def _init_isotope(self, isotope):
        """Initialize the isotope.

        Args:
          isotope: an Isotope object, or a string that defines an Isotope

        Raises:
          TypeError: if isotope is not an Isotope object
          AttributeError: if isotope is missing half_life or decay_const
        """

        if isinstance(isotope, Isotope):
            self.isotope = isotope
        elif isinstance(isotope, string_types):
            self.isotope = Isotope(isotope)
        else:
            raise TypeError('IsotopeQuantity needs an Isotope instance or ' +
                            'string, not {}'.format(isotope))

        self.half_life = self.isotope.half_life
        self.decay_const = self.isotope.decay_const

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

    def _atoms_from_kwargs(self, **kwargs):
        """Parse kwargs and return a quantity in atoms.

        Args (specify one):
          atoms: the number of atoms
          bq: the activity [Bq]
          uci: the activity [uCi]
          g: the mass [g]

        Raises:
          IsotopeQuantityError: if no valid argument specified
        """

        # TODO handle unit prefixes with or without pint
        # TODO handle ufloats

        if 'atoms' in kwargs:
            return self._check_positive_qty(float(kwargs['atoms']))
        elif 'g' in kwargs:
            return (self._check_positive_qty(float(kwargs['g'])) /
                    self.isotope.A * N_AV)
        elif 'bq' in kwargs and self.decay_const > 0:
            return (self._check_positive_qty(float(kwargs['bq'])) /
                    self.decay_const)
        elif 'uci' in kwargs and self.decay_const > 0:
            return (self._check_positive_qty(float(kwargs['uci'])) *
                    UCI_TO_BQ / self.decay_const)
        elif 'bq' in kwargs or 'uci' in kwargs:
            raise IsotopeQuantityError(
                'Cannot initialize a stable IsotopeQuantity from activity')
        else:
            raise IsotopeQuantityError('Missing arg for isotope activity')

    def _check_positive_qty(self, val):
        """Check that the quantity value is positive.

        Raises:
          ValueError: if val is negative
        """

        if val < 0:
            raise ValueError(
                'Mass or activity must be a positive quantity: {}'.format(val))
        return val

    # ----------------------------
    #   *_at()
    # ----------------------------

    def atoms_at(self, date):
        """Calculate the number of atoms at a given time.

        Args:
          date: the date to calculate for

        Returns:
          a float of the number of atoms at date

        Raises:
          TypeError: if date is not recognized
        """

        t1 = utils.handle_datetime(date)
        dt = (t1 - self.ref_date).total_seconds()
        return self.ref_atoms * 2**(-dt / self.half_life)

    def bq_at(self, date):
        """Calculate the activity [Bq] at a given time.

        As atoms_at() except for return value.
        """

        return self.atoms_at(date) * self.decay_const

    def uci_at(self, date):
        """Calculate the activity [uCi] at a given time.

        As atoms_at() except for return value.
        """

        return self.bq_at(date) / UCI_TO_BQ

    def g_at(self, date):
        """Calculate the mass [g] at a given time.

        As atoms_at() except for return value.
        """

        return self.atoms_at(date) / N_AV * self.isotope.A

    # ----------------------------
    #   *_now()
    # ----------------------------

    def atoms_now(self):
        """Calculate the number of atoms now.

        Returns:
          a float of the number of atoms at datetime.datetime.now()
        """

        return self.atoms_at(datetime.datetime.now())

    def bq_now(self):
        """Calculate the activity [Bq] now.

        As atoms_now() except for return value.
        """

        return self.bq_at(datetime.datetime.now())

    def uci_now(self):
        """Calculate the activity [uCi] now.

        As atoms_now() except for return value.
        """

        return self.uci_at(datetime.datetime.now())

    def g_now(self):
        """Calculate the mass [g] now.

        As atoms_now() except for return value.
        """

        return self.g_at(datetime.datetime.now())

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

        if not np.isfinite(self.half_life):
            raise IsotopeQuantityError(
                'Cannot calculate time_when for stable isotope')

        target = self._atoms_from_kwargs(**kwargs)
        dt = -self.half_life * np.log2(target / self.ref_atoms)
        return self.ref_date + datetime.timedelta(seconds=dt)

    def __str__(self):
        """Return a string representation"""

        if self.isotope.is_stable:
            s = '{} g of {}'.format(self.g_at(self.ref_date), self.isotope)
        else:
            s = '{} Bq of {} (at {})'.format(
                self.bq_at(self.ref_date), self.isotope, self.ref_date)
        return s


class NeutronIrradiation(object):
    """Represents an irradiation period with thermal neutrons."""

    def __init__(self, start_time, stop_time, n_cm2=None, n_cm2_s=None):
        """Initialize.

        Args:
          start_time
          stop_time
          n_cm2 OR n_cm2_s
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
        """Return a string representation"""

        if self.duration == 0:
            return '{} neutrons/cm2 at {}'.format(self.n_cm2, self.start_time)
        else:
            return '{} n/cm2/s from {} to {}'.format(
                self.n_cm2_s, self.start_time, self.stop_time)

    def activate(self, barns,
                 initial_iso_q=None, initial_iso=None,
                 activated_iso_q=None, activated_iso=None):
        """
        Calculate an IsotopeQuantity from before or after a neutron activation.

        Specify an IsotopeQuantity from before/after the irradiation,
        as well as an Isotope from after/before the irradiation to calculate
        a quantity for.

        Args:
          barns: cross section for activation [barns = 1e-24 cm^2]

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
        """

        if not ((initial_iso_q is None) ^ (activated_iso_q is None)):
            raise IsotopeQuantityError('Only one IsotopeQuantity should be given')
        if initial_iso_q is not None and activated_iso is None:
            raise IsotopeQuantityError('Activated isotope must be specified')
        elif activated_iso_q is not None and initial_iso is None:
            raise IsotopeQuantityError('Initial isotope must be specified')

        if initial_iso is None:
            initial_iso = initial_iso_q.isotope
        elif activated_iso is None:
            activated_iso = activated_iso_q.isotope

        if np.isfinite(initial_iso.half_life):
            raise NotImplementedError(
                'Activation not implemented for a radioactive initial isotope')

        cross_section = barns * 1.0e-24

        if activated_iso_q is None:
            # forward calculation
            if self.duration == 0:
                activated_bq = (
                    self.n_cm2 * cross_section *
                    initial_iso_q.atoms_at(self.stop_time) *
                    activated_iso.decay_const)
            else:
                activated_bq = (
                    self.n_cm2_s * cross_section *
                    initial_iso_q.atoms_at(self.stop_time) *
                    (1 - np.exp(-activated_iso.decay_const * self.duration))
                )
            return IsotopeQuantity(activated_iso,
                                   date=self.stop_time, bq=activated_bq)
        else:
            # backward calculation
            if self.duration == 0:
                initial_atoms = (
                    activated_iso_q.bq_at(self.stop_time) /
                    (self.n_cm2 * cross_section * activated_iso.decay_const))
            else:
                initial_atoms = (
                    activated_iso_q.bq_at(self.stop_time) /
                    (self.n_cm2_s * cross_section * (1 - np.exp(
                        -activated_iso.decay_const * self.duration))))
            return IsotopeQuantity(initial_iso,
                                   date=self.start_time, atoms=initial_atoms)
