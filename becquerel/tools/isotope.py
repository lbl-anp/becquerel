"""Nuclear isotopes and isomers."""

from __future__ import print_function
import datetime
from builtins import super
import numpy as np
from . import element
from ..core import utils

BQ_TO_UCI = 3.7e4


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
                elem = element.Element(p_element)
            except element.ElementError:
                continue
            element_ids.append(elem.symbol)
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
        elem = element.Element(element_id)
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

    # split string into element and mass
    element_id, mass_isomer = _split_element_mass(arg)
    # determine mass number A and isomer level m
    mass_number, isomer_level = _split_mass_isomer(mass_isomer)
    return (element_id, mass_number, isomer_level)


class Isotope(element.Element):
    """Basic properties of a nuclear isotope, including isomers.

    Also provides string formatting:
    >>> iso = Isotope('178M2HF')
    >>> '{:%n(%s)-%a%m Z=%z A=%a}'.format(iso)
    'Hafnium(Hf)-178m2 Z=72 A=178'

    Properties:
      symbol (read-only): the mixed-case element symbol (e.g., "Ge")
      name (read-only): the mixed-case element name (e.g., "Germanium")
      Z (read-only): an integer giving the atomic number (e.g., 32)
      atomic_mass (read-only): a float giving the atomic mass in amu
      A (read-only): an integer giving the mass number (e.g., 68)
      N (read-only): an integer giving the neutron number (e.g., 36)
      M (read-only): an integer giving the isomer level (e.g., 0)
      m (read-only): string version of M, e.g., '', 'm', or 'm2'
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


def activity_from_kwargs(error_name='activity', **kwargs):
    """Parse kwargs and return an activity in Bq"""

    # TODO init with mass or #atoms instead of activity
    # TODO handle unit prefixes with or without pint

    if 'bq' in kwargs:
        return float(kwargs['bq'])
    elif 'uci' in kwargs:
        return float(kwargs['uci']) * BQ_TO_UCI
    else:
        raise IsotopeError('Missing arg for {}'.format(error_name))


class IsotopeQuantity(object):
    """An amount of an isotope."""

    def __init__(self, isotope, date=None, **kwargs):
        """Initialize.

        Must provide an isotope (Isotope instance), an activity, and a date.
        """

        self._init_isotope(isotope)
        self._init_date(date)
        self.ref_activity = activity_from_kwargs(
            error_name='isotope activity', **kwargs)

    def _init_isotope(self, isotope):
        """Initialize the isotope.

        Right now this just does one error check, but in the future maybe
        isotope could be a string and it generates the Isotope instance from
        cached data?
        """

        if not isinstance(isotope, Isotope):
            raise TypeError(
                'Initialize IsotopeQuantity with an Isotope instance')
        self.isotope = isotope

    def _init_date(self, date):
        """Initialize the reference date/time."""

        self.ref_date = utils.handle_datetime(
            date, error_name='IsotopeQuantity date')
        if self.ref_date is None:
            # assume a long-lived source in the current epoch
            self.ref_date = datetime.datetime.now()

    def bq_at(self, date):
        """Calculate the activity [Bq] at a given time"""

        t1 = utils.handle_datetime(date)
        if t1 is None:
            raise ValueError('Cannot calculate activity for time None')
        dt = (t1 - self.ref_date).total_seconds()
        return self.ref_activity * 2**(-dt / self.isotope.halflife)

    def uci_at(self, date):
        """Calculate the activity [uCi] at a given time"""

        return self.bq_at(date) / BQ_TO_UCI

    def time_when(self, **kwargs):
        """Calculate the date/time when the activity is a given value"""

        target = activity_from_kwargs(**kwargs)

        dt = -self.isotope.halflife * np.log2(target / self.ref_activity)
        return self.ref_date + datetime.timedelta(seconds=dt)
