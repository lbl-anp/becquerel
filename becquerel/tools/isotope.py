"""Nuclear isotopes and isomers."""

from __future__ import print_function
from future.builtins import super
import numpy as np
import uncertainties
from . import element
from .wallet_cache import wallet_cache


# pylint: disable=no-self-use


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
