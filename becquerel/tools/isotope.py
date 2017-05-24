"""Nuclear isotopes and isomers."""

from __future__ import print_function
import datetime
from builtins import super
import numpy as np
import pandas as pd
from six import string_types
import uncertainties
from . import element
from . import nndc
from . import df_cache
from ..core import utils

UCI_TO_BQ = 3.7e4
N_AV = 6.022141e23

# pylint: disable=no-self-use


def convert_float_ufloat(x):
    """Convert string to a float or a ufloat, including None and NaN.

    Args:
      x: a string giving the number
    """

    if isinstance(x, string_types):
        if '+/-' in x:
            tokens = x.split('+/-')
            return uncertainties.ufloat(float(tokens[0]), float(tokens[1]))
    try:
        return float(x)
    except TypeError:
        return None


def format_ufloat(x, fmt='{:.12f}'):
    """Convert ufloat to a string, including None and NaN.

    Args:
      x: a ufloat
    """

    if x is None:
        return ''
    try:
        return fmt.format(x)
    except TypeError:
        return str(x)


class WalletCardCache(df_cache.DataFrameCache):
    """A cache of all isotope wallet cards from NNDC."""

    name = 'all_wallet_cards'

    def write_file(self):
        """Format ufloat columns before writing so they keep precision."""

        for col in ['Abundance (%)', 'Mass Excess (MeV)']:
            self.df[col] = self.df[col].apply(format_ufloat)
        super().write_file()

    def read_file(self):
        """Ensure some columns are properly converted to float/ufloat."""

        super().read_file()
        for col in ['Abundance (%)', 'Mass Excess (MeV)']:
            self.df[col] = self.df[col].apply(convert_float_ufloat)

    def fetch(self):
        """Fetch wallet card data from NNDC for all isotopes."""

        self.df = pd.DataFrame()
        z_edges = (0, 40, 80, 120)
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            df_chunk = nndc.fetch_wallet_card(z_range=(z0, z1 - 1))
            self.df = self.df.append(df_chunk)
        self.loaded = True


wallet_cache = WalletCardCache()


class IsotopeError(element.ElementError):
    """Problem with isotope properties."""

    pass


def _split_element_mass(arg):
    """Split a string into an element name/symbol plus a mass number/isomer.

    For example, the string 'TC99M' would be split into ('TC', '99M'), the
    string '238U' would be split into ('U', '238'), and the string 'Hf-178m3'
    would be split into ('Hf', '178m3').

    Args:
      arg: a string of the form "EE[-]AAA[m[M]]" or "AAA[m[M]][-]EE".

    Returns:
      tuple of strings identified as the element symbol/name and mass
        number A, which may include isomer level as well.

    Raises:
      IsotopeError: if the string could not be split.
    """

    arg = str(arg)
    element_id = None
    mass_isomer = None
    if '-' in arg:
        # parse string of the form "EE-AAA[m[M]]" or "AAA[m[M]]-EE"
        # (i.e., those with a hyphen)
        tokens = arg.split('-')
        if len(tokens) != 2:
            raise IsotopeError(
                'Too many hyphens in isotope string: {}'.format(arg))
        if tokens[0].isalpha() and tokens[1][0].isdigit():
            element_id = tokens[0]
            mass_isomer = tokens[1]
        elif tokens[0][0].isdigit() and tokens[1].isalpha():
            element_id = tokens[1]
            mass_isomer = tokens[0]
        else:
            raise IsotopeError(
                'Could not find mass number for isotope: {}'.format(tokens))
    else:
        # parse string of the form "EEAAA[m[M]]" or "AAA[m[M]]EE"
        # (i.e., those without a hyphen)
        part1 = [arg[j:] for j in range(1, len(arg))]
        part2 = [arg[:j] for j in range(1, len(arg))]
        element_ids = []
        mass_isomers = []
        # make a list of possible element symbols and names
        for p1, p2 in zip(part1, part2):
            if p1[0].isdigit() and p2[0].isalpha():
                p_element = p2
                p_mass = p1
            elif p1[0].isalpha() and p2[0].isdigit():
                p_element = p1
                p_mass = p2
            else:
                continue
            try:
                element.Element(p_element)
            except element.ElementError:
                continue
            element_ids.append(p_element)
            mass_isomers.append(p_mass)
        if len(element_ids) == 0:
            raise IsotopeError(
                'Could not find element for isotope: {}'.format(arg))
        # if multiple element IDs were found, choose the longest
        element_id = ''
        mass_isomer = ''
        for elem, mass in zip(element_ids, mass_isomers):
            if len(elem) > len(element_id):
                element_id = elem
                mass_isomer = mass
    # ensure element name or symbol is valid
    try:
        element.Element(element_id)
    except element.ElementError:
        raise IsotopeError(
            'Element name or symbol is invalid: {}'.format(element_id))
    return element_id, mass_isomer


def _split_mass_isomer(arg):
    """Split a string into a mass number and isomer level.

    For example, the string '99m' would be split into ('99', 'm'), the
    string '238' would be split into ('238', ''), and the string '178m3'
    would be split into ('178', 'm3').

    Args:
      arg: a string of the form "AAA[m[M]]".

    Returns:
      tuple of substrings identified as the mass number A and isomer.

    Raises:
      IsotopeError: if the string could not be split.
    """

    arg = str(arg)
    aa = 0
    mm = ''
    if 'm' in arg.lower():
        tokens = arg.lower().split('m')
        if len(tokens) != 2:
            raise IsotopeError(
                'Too many ms in mass number: {} {}'.format(arg, tokens))
        try:
            aa = int(tokens[0])
        except ValueError:
            raise IsotopeError(
                'Mass number cannot be converted to int: {} {}'.format(
                    tokens[0], arg))
        mm = 'm'
        if len(tokens[1]) > 0:
            if not tokens[1].isdigit():
                raise IsotopeError(
                    'Metastable level must be a number: {} {}'.format(
                        tokens[1], arg))
            mm += tokens[1]
    else:
        try:
            aa = int(arg)
        except ValueError:
            raise IsotopeError(
                'Mass number cannot be converted to int: {}'.format(arg))
    return (aa, mm)


def parse_isotope(arg):
    """Parse an isotope string into a symbol, mass, and metastable state.

    Args:
      arg: A string identifying the isotope, such as "232TH", "U-238", or
        "Tc-99m".

    Returns:
      A tuple of the element symbol, mass number, and metastable state.

    Raises:
      IsotopeError: if there was a problem parsing the string.
    """

    element_id, mass_isomer = _split_element_mass(arg)
    mass_number, isomer_level = _split_mass_isomer(mass_isomer)
    return (element_id, mass_number, isomer_level)


class Isotope(element.Element):
    """Basic properties of a nuclear isotope, including isomers.

    Also provides string formatting:
    >>> iso = Isotope('178M2HF')
    >>> '{:%n(%s)-%a%m Z=%z A=%a}'.format(iso)
    'Hafnium(Hf)-178m2 Z=72 A=178'

    Properties (read-only):
      symbol: the mixed-case element symbol (e.g., "Ge")
      name: the mixed-case element name (e.g., "Germanium")
      Z: an integer giving the atomic number (e.g., 32)
      atomic_mass: a float giving the atomic mass in amu
      A: an integer giving the mass number (e.g., 68)
      N: an integer giving the neutron number (e.g., 36)
      M: an integer giving the isomer level (e.g., 0)
      m: string version of M, e.g., '', 'm', or 'm2'
      half_life: half-life of the isotope in seconds
      is_stable: a boolean of whether the isotope is stable
      abundance: the natural abundance percent
      j_pi: a string describing the spin and parity
      energy_level: the nuclear energy level in MeV
      mass_excess: the mass excess in MeV
      decay_modes: the isotope's decay modes and their branching ratios
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, *args):
        """Initialize an Isotope.

        Args:
          args: Either one argument that is a string describing the isotope,
            e.g., "Tc-99m", or two or three arguments: one of the set
            {`name`, `sym`, `Z`} and the mass number `A`, and optionally the
            isomer level `m`.

        Raises:
          IsotopeError: if there was a problem instantiating the isotope.
        """

        self.A = 0
        self.m = ''
        self.M = 0
        if len(args) == 1:
            if not isinstance(args[0], str):
                raise IsotopeError('Single argument must be a string')
            symbol, aa, mm = parse_isotope(args[0])
            super().__init__(symbol)
            self._init_A(aa)
            self._init_m(mm)
        elif len(args) == 2 or len(args) == 3:
            try:
                super().__init__(args[0])
            except element.ElementError:
                raise IsotopeError('Unable to create Isotope: {}'.format(args))
            self._init_A(args[1])
            if len(args) == 3:
                self._init_m(args[2])
        else:
            raise IsotopeError('One, two, or three arguments required')
        self.N = self.A - self.Z
        if self.N < 0:
            raise IsotopeError(
                'Neutron number N cannot be negative: {} {}'.format(
                    args, self.N))

    def _init_A(self, arg):
        """Initialize with an isotope A."""
        try:
            self.A = int(arg)
        except ValueError:
            raise IsotopeError(
                'Mass number cannot be converted to integer: {}'.format(arg))
        if self.A < 1:
            raise IsotopeError(
                'Mass number must be >= 1: {}'.format(self.A))

    def _init_m(self, arg):
        """Initialize with an isomer level number."""
        if arg == '' or arg is None or arg == 0:
            self.m = ''
            self.M = 0
        else:
            if isinstance(arg, int):
                if arg == 1:
                    self.m = 'm'
                    self.M = 1
                elif arg >= 2:
                    self.M = arg
                    self.m = 'm{}'.format(self.M)
                else:
                    raise IsotopeError(
                        'Metastable level must be >= 0: {}'.format(arg))
            elif isinstance(arg, str):
                self.m = arg.lower()
                if self.m[0] != 'm':
                    raise IsotopeError(
                        'Metastable level must start with "m": {}'.format(
                            self.m))
                if len(self.m) > 1:
                    if not self.m[1:].isdigit():
                        raise IsotopeError(
                            'Metastable level must be numeric: {} {}'.format(
                                self.m[0], self.m[1:]))
                    self.M = int(self.m[1:])
                else:
                    self.M = 1
            else:
                raise IsotopeError(
                    'Metastable level must be integer or string: {} {}'.format(
                        arg, type(arg)))

    def __str__(self):
        """Define behavior of str() on Isotope."""
        return '{}'.format(self)

    def __format__(self, formatstr):
        """Define behavior of string's format method.

        Format codes:
            '%s': element symbol
            '%n': element name
            '%z': element Z
            '%a': isotope A
            '%m': isomer level
        """

        str0 = str(formatstr)
        if len(str0) == 0:
            str0 = '%s-%a%m'
        str0 = str0.replace('%s', self.symbol)
        str0 = str0.replace('%n', self.name)
        str0 = str0.replace('%z', '{}'.format(self.Z))
        str0 = str0.replace('%a', '{}'.format(self.A))
        str0 = str0.replace('%m', self.m)
        return str0

    def __eq__(self, other):
        """Define equality of two isotopes."""
        if isinstance(other, Isotope):
            return (
                super().__eq__(other) and
                self.A == other.A and self.Z == other.Z and self.M == other.M)
        else:
            raise TypeError('Cannot compare to non-isotope')

    def _wallet_card(self):
        """Retrieve the wallet card data for this isotope.

        Returns:
          a DataFrame containing the wallet card data.

        Raises:
          IsotopeError: if no isotope data can be found.
        """

        global wallet_cache
        if not wallet_cache.loaded:
            wallet_cache.load()
        this_isotope = \
            (wallet_cache.df['Z'] == self.Z) & \
            (wallet_cache.df['A'] == self.A) & \
            (wallet_cache.df['M'] == self.M)
        df = wallet_cache.df[this_isotope]
        if len(df) == 0:
            raise IsotopeError(
                'No wallet card data found for isotope {}'.format(self))
        return df

    @property
    def half_life(self):
        """The isotope's half-life in seconds.

        Returns:
          the half-life in seconds, or np.inf if stable.
        """

        df = self._wallet_card()
        data = df['T1/2 (s)'].tolist()
        assert len(np.unique(data)) == 1
        return data[0]

    @property
    def decay_const(self):
        """The decay constant (lambda), in 1/seconds.

        Returns:
          the decay constant in 1/seconds, or 0 if stable.
        """

        return np.log(2) / self.half_life

    @property
    def is_stable(self):
        """Return a boolean of whether the isotope is stable.

        Returns:
          a boolean.
        """

        df = self._wallet_card()
        data = df['T1/2 (txt)'].tolist()
        assert len(np.unique(data)) == 1
        return 'STABLE' in data

    @property
    def abundance(self):
        """Return the natural abundance percent.

        Returns:
          the natural abundance (float or ufloat), or None if not stable.
        """

        df = self._wallet_card()
        data = df['Abundance (%)'].tolist()
        if not isinstance(data[0], uncertainties.core.Variable):
            if np.isnan(data[0]):
                return None
        return data[0]

    @property
    def j_pi(self):
        """Return the spin and parity of the isotope.

        Returns:
          a string giving the J and Pi, e.g., '7/2+'.
        """

        df = self._wallet_card()
        data = df['JPi'].tolist()
        assert len(np.unique(data)) == 1
        return data[0]

    @property
    def energy_level(self):
        """Return the nuclear energy level in MeV.

        Returns:
          the energy level in MeV.
        """

        df = self._wallet_card()
        data = df['Energy Level (MeV)'].tolist()
        assert len(np.unique(data)) == 1
        return data[0]

    @property
    def mass_excess(self):
        """Return the mass excess in MeV.

        Returns:
          the mass excess in MeV.
        """

        df = self._wallet_card()
        data = df['Mass Excess (MeV)'].tolist()
        if not isinstance(data[0], uncertainties.core.Variable):
            if np.isnan(data[0]):
                return None
        return data[0]

    @property
    def decay_modes(self):
        """Return the decay modes and their branching ratios.

        Returns:
          a list of the decay modes, and a list of the branching percents.
        """

        df = self._wallet_card()
        data1 = df['Decay Mode'].tolist()
        data2 = df['Branching (%)'].tolist()
        if len(data1) == 1 and np.isnan(data2[0]):
            return [], []
        return data1, data2


class IsotopeQuantity(object):
    """An amount of an isotope."""

    def __init__(self, isotope, date=None, creation_date=True, **kwargs):
        """Initialize.

        Specify one of bq, uci, atoms, g to define the quantity.

        Args:
          isotope: an Isotope object, of which this is a quantity
          date: the reference date for the activity or mass
          creation_date: True means that dates before the reference date will
            raise an error (because the source did not exist then)
          bq: the activity at the reference date [Bq]
          uci: the activity at the reference date [uCi]
          atoms: the number of atoms at the reference date
          g: the mass at the reference date [g]

        Raises:
          ...
        """

        self._init_isotope(isotope)
        self._init_date(date)
        self.creation_date = creation_date
        self.ref_atoms = self._atoms_from_kwargs(**kwargs)

    def _init_isotope(self, isotope):
        """Initialize the isotope.

        Right now this just does one error check, but in the future maybe
        isotope could be a string and it generates the Isotope instance from
        cached data?

        Args:
          isotope: an Isotope object

        Raises:
          TypeError: if isotope is not an Isotope object
          AttributeError: if isotope is missing half_life or decay_const
        """

        if not isinstance(isotope, Isotope):
            raise TypeError(
                'Initialize IsotopeQuantity with an Isotope instance')
        self.isotope = isotope
        self.half_life = isotope.half_life
        self.decay_const = isotope.decay_const

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
          IsotopeError: if no valid argument specified
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
            raise IsotopeError(
                'Cannot initialize a stable IsotopeQuantity from activity')
        else:
            raise IsotopeError('Missing arg for isotope activity')

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
          IsotopeError: if date is prior to the creation date
        """

        t1 = utils.handle_datetime(date)
        dt = (t1 - self.ref_date).total_seconds()
        if dt < 0 and self.creation_date:
            raise IsotopeError(
                'The source represented by this IsotopeQuantity was created at'
                + ' {} and thus did not exist at {}'.format(
                    self.ref_date, date))
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

        Raises:
          IsotopeError: if this quantity's creation date is in the future
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
          IsotopeError: if start_time is prior to the creation date
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
          IsotopeError: if the acquisition start was before the creation date
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
            specified input, OR None if it is before creation date

        Raises:
          IsotopeError: if isotope is stable
        """

        if not np.isfinite(self.half_life):
            raise IsotopeError('Cannot calculate time_when for stable isotope')

        target = self._atoms_from_kwargs(**kwargs)
        dt = -self.half_life * np.log2(target / self.ref_atoms)
        if dt < 0 and self.creation_date:
            return None
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
            raise IsotopeError('Only one IsotopeQuantity should be given')
        if initial_iso_q is not None and activated_iso is None:
            raise IsotopeError('Activated isotope must be specified')
        elif activated_iso_q is not None and initial_iso is None:
            raise IsotopeError('Initial isotope must be specified')

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
