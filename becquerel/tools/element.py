"""Element symbols, names, Z values, and masses."""

from __future__ import print_function


# Element data (Z, symbol, name, and relative atomic mass) are from Wikipedia:
# https://en.wikipedia.org/wiki/List_of_chemical_elements

_Z_SYMBOL_NAME_MASS = (
    (1, 'H', 'Hydrogen', 1.01),
    (2, 'He', 'Helium', 4.00),
    (3, 'Li', 'Lithium', 6.94),
    (4, 'Be', 'Beryllium', 9.01),
    (5, 'B', 'Boron', 10.81),
    (6, 'C', 'Carbon', 12.01),
    (7, 'N', 'Nitrogen', 14.01),
    (8, 'O', 'Oxygen', 16.00),
    (9, 'F', 'Fluorine', 19.00),
    (10, 'Ne', 'Neon', 20.18),
    (11, 'Na', 'Sodium', 22.99),
    (12, 'Mg', 'Magnesium', 24.30),
    (13, 'Al', 'Aluminum', 26.98),
    (14, 'Si', 'Silicon', 28.09),
    (15, 'P', 'Phosphorus', 30.97),
    (16, 'S', 'Sulfur', 32.06),
    (17, 'Cl', 'Chlorine', 35.45),
    (18, 'Ar', 'Argon', 39.95),
    (19, 'K', 'Potassium', 39.10),
    (20, 'Ca', 'Calcium', 40.08),
    (21, 'Sc', 'Scandium', 44.96),
    (22, 'Ti', 'Titanium', 47.87),
    (23, 'V', 'Vanadium', 50.94),
    (24, 'Cr', 'Chromium', 52.00),
    (25, 'Mn', 'Manganese', 54.94),
    (26, 'Fe', 'Iron', 55.84),
    (27, 'Co', 'Cobalt', 58.93),
    (28, 'Ni', 'Nickel', 58.69),
    (29, 'Cu', 'Copper', 63.55),
    (30, 'Zn', 'Zinc', 65.38),
    (31, 'Ga', 'Gallium', 69.72),
    (32, 'Ge', 'Germanium', 72.63),
    (33, 'As', 'Arsenic', 74.92),
    (34, 'Se', 'Selenium', 78.97),
    (35, 'Br', 'Bromine', 79.90),
    (36, 'Kr', 'Krypton', 83.80),
    (37, 'Rb', 'Rubidium', 85.47),
    (38, 'Sr', 'Strontium', 87.62),
    (39, 'Y', 'Yttrium', 88.91),
    (40, 'Zr', 'Zirconium', 91.22),
    (41, 'Nb', 'Niobium', 92.91),
    (42, 'Mo', 'Molybdenum', 95.95),
    (43, 'Tc', 'Technetium', 98),
    (44, 'Ru', 'Ruthenium', 101.07),
    (45, 'Rh', 'Rhodium', 102.91),
    (46, 'Pd', 'Palladium', 106.42),
    (47, 'Ag', 'Silver', 107.87),
    (48, 'Cd', 'Cadmium', 112.41),
    (49, 'In', 'Indium', 114.82),
    (50, 'Sn', 'Tin', 118.71),
    (51, 'Sb', 'Antimony', 121.76),
    (52, 'Te', 'Tellurium', 127.60),
    (53, 'I', 'Iodine', 126.90),
    (54, 'Xe', 'Xenon', 131.29),
    (55, 'Cs', 'Cesium', 132.91),
    (56, 'Ba', 'Barium', 137.33),
    (57, 'La', 'Lanthanum', 138.91),
    (58, 'Ce', 'Cerium', 140.12),
    (59, 'Pr', 'Praseodymium', 140.91),
    (60, 'Nd', 'Neodymium', 144.24),
    (61, 'Pm', 'Promethium', 145),
    (62, 'Sm', 'Samarium', 150.36),
    (63, 'Eu', 'Europium', 151.96),
    (64, 'Gd', 'Gadolinium', 157.25),
    (65, 'Tb', 'Terbium', 158.93),
    (66, 'Dy', 'Dysprosium', 162.50),
    (67, 'Ho', 'Holmium', 164.93),
    (68, 'Er', 'Erbium', 167.26),
    (69, 'Tm', 'Thulium', 168.93),
    (70, 'Yb', 'Ytterbium', 173.04),
    (71, 'Lu', 'Lutetium', 174.97),
    (72, 'Hf', 'Hafnium', 178.49),
    (73, 'Ta', 'Tantalum', 180.95),
    (74, 'W', 'Tungsten', 183.84),
    (75, 'Re', 'Rhenium', 186.21),
    (76, 'Os', 'Osmium', 190.23),
    (77, 'Ir', 'Iridium', 192.22),
    (78, 'Pt', 'Platinum', 195.08),
    (79, 'Au', 'Gold', 196.97),
    (80, 'Hg', 'Mercury', 200.59),
    (81, 'Tl', 'Thallium', 204.38),
    (82, 'Pb', 'Lead', 207.20),
    (83, 'Bi', 'Bismuth', 208.98),
    (84, 'Po', 'Polonium', 209),
    (85, 'At', 'Astatine', 210),
    (86, 'Rn', 'Radon', 222),
    (87, 'Fr', 'Francium', 223),
    (88, 'Ra', 'Radium', 226),
    (89, 'Ac', 'Actinium', 227),
    (90, 'Th', 'Thorium', 232.04),
    (91, 'Pa', 'Protactinium', 231.04),
    (92, 'U', 'Uranium', 238.03),
    (93, 'Np', 'Neptunium', 237),
    (94, 'Pu', 'Plutonium', 244),
    (95, 'Am', 'Americium', 243),
    (96, 'Cm', 'Curium', 247),
    (97, 'Bk', 'Berkelium', 247),
    (98, 'Cf', 'Californium', 251),
    (99, 'Es', 'Einsteinium', 252),
    (100, 'Fm', 'Fermium', 257),
    (101, 'Md', 'Mendelevium', 258),
    (102, 'No', 'Nobelium', 259),
    (103, 'Lr', 'Lawrencium', 266),
    (104, 'Rf', 'Rutherfordium', 267),
    (105, 'Db', 'Dubnium', 268),
    (106, 'Sg', 'Seaborgium', 269),
    (107, 'Bh', 'Bohrium', 270),
    (108, 'Hs', 'Hassium', 269),
    (109, 'Mt', 'Meitnerium', 278),
    (110, 'Ds', 'Darmstadtium', 281),
    (111, 'Rg', 'Roentgenium', 282),
    (112, 'Cn', 'Copernicium', 285),
    (113, 'Nh', 'Nihonium', 286),
    (114, 'Fl', 'Flerovium', 289),
    (115, 'Mc', 'Moscovium', 289),
    (116, 'Lv', 'Livermorium', 293),
    (117, 'Ts', 'Tennessine', 294),
    (118, 'Og', 'Oganesson', 294),
)


ZS = set([d[0] for d in _Z_SYMBOL_NAME_MASS])
SYMBOLS = set([d[1] for d in _Z_SYMBOL_NAME_MASS])
SYMBOLS_LOWER = set([d[1].lower() for d in _Z_SYMBOL_NAME_MASS])
SYMBOLS_UPPER = set([d[1].upper() for d in _Z_SYMBOL_NAME_MASS])
NAMES = set([d[2] for d in _Z_SYMBOL_NAME_MASS])
NAMES_LOWER = set([d[2].lower() for d in _Z_SYMBOL_NAME_MASS])
NAMES_UPPER = set([d[2].upper() for d in _Z_SYMBOL_NAME_MASS])

_SYMBOL_FROM_SYMBOL_LOWER = {d[1].lower(): d[1] for d in _Z_SYMBOL_NAME_MASS}
_SYMBOL_FROM_SYMBOL_UPPER = {d[1].upper(): d[1] for d in _Z_SYMBOL_NAME_MASS}

_NAME_FROM_NAME_LOWER = {d[2].lower(): d[2] for d in _Z_SYMBOL_NAME_MASS}
_NAME_FROM_NAME_UPPER = {d[2].upper(): d[2] for d in _Z_SYMBOL_NAME_MASS}

_Z_FROM_SYMBOL = {d[1]: d[0] for d in _Z_SYMBOL_NAME_MASS}
_Z_FROM_NAME = {d[2]: d[0] for d in _Z_SYMBOL_NAME_MASS}

_SYMBOL_FROM_Z = {d[0]: d[1] for d in _Z_SYMBOL_NAME_MASS}
_SYMBOL_FROM_NAME = {d[2]: d[1] for d in _Z_SYMBOL_NAME_MASS}

_NAME_FROM_Z = {d[0]: d[2] for d in _Z_SYMBOL_NAME_MASS}
_NAME_FROM_SYMBOL = {d[1]: d[2] for d in _Z_SYMBOL_NAME_MASS}

_MASS_FROM_Z = {d[0]: d[3] for d in _Z_SYMBOL_NAME_MASS}
_MASS_FROM_SYMBOL = {d[1]: d[3] for d in _Z_SYMBOL_NAME_MASS}


class ElementError(Exception):
    """Problem with element properties."""

    pass


class ElementZError(ElementError):
    """Bad element Z value."""

    pass


class ElementSymbolError(ElementError):
    """Bad element symbol."""

    pass


class ElementNameError(ElementError):
    """Bad element name."""

    pass


def validated_z(z):
    """Convert z into a valid element atomic number.

    Args:
      z: numeric or string type representing the atomic number Z.

    Returns:
      Integer z that is a valid atomic number.

    Raises:
      ElementZError: if z cannot be converted to integer or is out of range.
    """

    try:
        int(z)
    except ValueError:
        raise ElementZError('Element Z="{}" invalid'.format(z))
    if int(z) not in ZS:
        raise ElementZError('Element Z="{}" not found'.format(z))
    return int(z)


def validated_symbol(sym):
    """Convert symbol into a valid mixed-case element symbol.

    Args:
      sym: a string type

    Returns:
      A valid mixed-case element symbol that matches sym.

    Raises:
      ElementSymbolError: if sym does not match a valid symbol.
    """

    try:
        sym.lower()
    except AttributeError:
        raise ElementSymbolError('Element symbol "{}" invalid'.format(sym))
    if sym.lower() not in SYMBOLS_LOWER:
        raise ElementSymbolError('Element symbol "{}" not found'.format(sym))
    return _SYMBOL_FROM_SYMBOL_LOWER[sym.lower()]


def validated_name(nm):
    """Convert name into a valid mixed-case element name.

    Args:
      nm: a string type

    Returns:
      A valid mixed-case element name that matches nm.

    Raises:
      ElementNameError: if nm does not match a valid name.
    """

    try:
        nm.lower()
    except AttributeError:
        raise ElementNameError('Element name "{}" invalid'.format(nm))
    # special case: Alumin[i]um
    if nm.lower() == 'aluminium':
        nm = 'Aluminum'
    # special case: C[a]esium
    if nm.lower() == 'caesium':
        nm = 'Cesium'
    if nm.lower() not in NAMES_LOWER:
        raise ElementNameError('Element name "{}" not found'.format(nm))
    return _NAME_FROM_NAME_LOWER[nm.lower()]


def element_z(sym_or_name):
    """Convert element symbol or name into a valid element atomic number Z.

    Args:
      sym_or_name: string type representing an element symbol or name.

    Returns:
      Integer z that is a valid atomic number matching the symbol or name.

    Raises:
      ElementZError: if the symbol or name cannot be converted.
    """

    try:
        return _Z_FROM_SYMBOL[validated_symbol(sym_or_name)]
    except ElementSymbolError:
        pass
    try:
        return _Z_FROM_NAME[validated_name(sym_or_name)]
    except ElementNameError:
        raise ElementZError('Must supply either the element symbol or name')


def element_symbol(name_or_z):
    """Convert element name or Z into a valid mixed-case element symbol.

    Args:
      name_or_z: string or numeric type representing an element name or Z.

    Returns:
      A valid mixed-case element symbol that matches the name or Z.

    Raises:
      ElementSymbolError: if the symbol or name cannot be converted.
    """

    try:
        return _SYMBOL_FROM_Z[validated_z(name_or_z)]
    except ElementZError:
        pass
    try:
        return _SYMBOL_FROM_NAME[validated_name(name_or_z)]
    except ElementNameError:
        raise ElementSymbolError('Must supply either the Z or element name')


def element_name(sym_or_z):
    """Convert element symbol or Z into a valid mixed-case element name.

    Args:
      sym_or_z: string or numeric type representing an element symbol or Z.

    Returns:
      A valid mixed-case element name that matches the symbol or Z.

    Raises:
      ElementNameError: if the symbol or Z cannot be converted.
    """

    try:
        return _NAME_FROM_SYMBOL[validated_symbol(sym_or_z)]
    except ElementSymbolError:
        pass
    try:
        return _NAME_FROM_Z[validated_z(sym_or_z)]
    except ElementZError:
        raise ElementNameError('Must supply either the element symbol or Z')


class Element(object):
    """Basic properties (symbol, name, Z, and mass) of an element.

    Also provides string formatting:
    >>> elem = Element('Ge')
    >>> '{:%n(%s) Z=%z}'.format(elem)
    'Germanium(Ge) Z=32'

    Properties:
      symbol (read-only): the mixed-case element symbol (e.g., "Ge")
      name (read-only): the mixed-case element name (e.g., "Germanium")
      Z (read-only): an integer giving the atomic number (e.g., 32)
      atomic_mass (read-only): a float giving the atomic mass in amu
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, arg):
        """Instantiate by providing a name, symbol, or Z value.

        Args:
          arg: a numeric or string type giving a name, symbol, or Z value.

        Raises:
          ElementError: if Element could not be instantiated.
        """

        self.symbol = None
        self.name = None
        self.Z = None
        self.atomic_mass = None
        try:
            self._init_sym(arg)
        except ElementSymbolError:
            try:
                self._init_name(arg)
            except ElementNameError:
                try:
                    self._init_z(arg)
                except ElementZError:
                    raise ElementError(
                        'Could not instantiate Element: {}'.format(arg))
        self.atomic_mass = _MASS_FROM_SYMBOL[self.symbol]

    def _init_sym(self, arg):
        """Initialize with an element symbol."""
        self.symbol = validated_symbol(arg)
        self.name = element_name(self.symbol)
        self.Z = element_z(self.symbol)

    def _init_name(self, arg):
        """Initialize with an element name."""
        self.name = validated_name(arg)
        self.symbol = element_symbol(self.name)
        self.Z = element_z(self.name)

    def _init_z(self, arg):
        """Initialize with an element Z."""
        self.Z = validated_z(arg)
        self.symbol = element_symbol(self.Z)
        self.name = element_name(self.Z)

    def __str__(self):
        """Define behavior of str() on this class."""
        return '{}'.format(self)

    def __format__(self, formatstr):
        """Define behavior of string's format method.

        Format codes:
            '%s': element symbol
            '%n': element name
            '%z': element Z
        """

        str0 = str(formatstr)
        if len(str0) == 0:
            str0 = '%n(%s) Z=%z'
        str0 = str0.replace('%s', self.symbol)
        str0 = str0.replace('%n', self.name)
        str0 = str0.replace('%z', '{}'.format(self.Z))
        return str0

    def __eq__(self, other):
        """Define equality of two elements."""
        try:
            return (
                self.name == other.name and self.symbol == other.symbol and
                self.Z == other.Z)
        except:
            raise ElementError('Cannot determine equality')
