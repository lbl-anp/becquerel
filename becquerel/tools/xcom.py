"""Query photon cross section data from NIST XCOM database.

Provides the fetch_xcom_data function to query the NIST XCOM database,
as well as a few mixtures for common materials.

References:
  https://www.nist.gov/pml/xcom-photon-cross-sections-database
  http://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

"""

from __future__ import print_function
import requests
import pandas as pd
from . import element
from ..core.utils import isstring, Iterable

# Dry air relative weights taken from:
# http://www.engineeringtoolbox.com/air-composition-d_212.html
MIXTURE_AIR_DRY = [
    'N2 75.47',
    'O2 23.20',
    'Ar 0.933',
    'CO2 0.03',
]

# Seawater composition for 3.5% salinity taken from:
# https://en.wikipedia.org/wiki/Seawater#Compositional_differences_from_freshwater
MIXTURE_SEAWATER = [
    'O 85.84',
    'H 10.82',
    'Cl 1.94',
    'Na 1.08',
    'Mg 0.1292',
    'S 0.091',
    'Ca 0.04',
    'K 0.04',
    'Br 0.0067',
    'C 0.0028',
]

# typical Portland cement values taken from:
# http://matse1.matse.illinois.edu/concrete/prin.html
MIXTURE_PORTLAND_CEMENT = [
    'Ca3SiO5 50',
    'Ca2SiO4 25',
    'Ca3Al2O6 10',
    'Ca4Al2Fe2O10 10',
    'CaSO6H4 5',
]


# base URL for three scripts on NIST where data can be queried
_URL = 'https://physics.nist.gov/cgi-bin/Xcom/xcom3_'


# Dictionary of data that will be posted to URL
_DATA = {
    'NumAdd': '1',          # always '1'?
    'Energies': '',         # additional energies separated by ; (MeV)
    'WindowXmin': '0.001',  # lower limit of energy grid (MeV)
    'WindowXmax': '100',    # upper limit of energy grid (MeV)
    'Output': '',           # 'on' for standard energy grid
    'OutOpt': 'PIC',        # return cross sections in cm^2/g
    'ResizeFlag': 'on',     # seems to determine whether Xmin and Xmax are used
}

# abbreviated names for the table columns (these are used in the dataframe)
COLUMNS_SHORT = ['energy', 'coherent', 'incoherent', 'photoelec',
                 'pair_nuc', 'pair_elec', 'total_w_coh', 'total_wo_coh']

# medium length names for the table columns
COLUMNS_MEDIUM = {
    'energy': 'Energy',
    'coherent': 'Coherent',
    'incoherent': 'Incoherent',
    'photoelec': 'Photoelectric',
    'pair_nuc': 'Pair production (nuclear)',
    'pair_elec': 'Pair Production (electron)',
    'total_w_coh': 'Total with coherent',
    'total_wo_coh': 'Total without coherent'
}

# verbose names for the table columns
COLUMNS_LONG = {
    'energy': 'Photon Energy',
    'coherent': 'Coherent Scattering',
    'incoherent': 'Incoherent Scattering',
    'photoelec': 'Photoelectric Absorption',
    'pair_nuc': 'Pair Production in Nuclear Field',
    'pair_elec': 'Pair Production in Electron Field',
    'total_w_coh': 'Total Attenuation with Coherent Scattering',
    'total_wo_coh': 'Total Attenuation without Coherent Scattering'
}


class XCOMError(Exception):
    """General XCOM error."""

    pass


class XCOMInputError(XCOMError):
    """Error related to the user input to XCOMQuery."""

    pass


class XCOMRequestError(XCOMError):
    """Error related to communicating with XCOM or parsing the result."""

    pass


class _XCOMQuery(object):
    """Query photon cross section data from NIST XCOM database.

    After the data have been successfully queried, they are stored in a
    pandas DataFrame that is accessible through this class's methods.

    The DataFrame column names amd units are:
      'energy': Photon Energy [keV]
      'coherent': coherent scattering cross section [cm^2/g]
      'incoherent': incoherent scattering cross section [cm^2/g]
      'photoelec': photoelectric absorption cross section [cm^2/g]
      'pair_nuc': pair production cross section in nuclear field [cm^2/g]
      'pair_elec': pair production cross section in electron field [cm^2/g]
      'total_w_coh': total attenuation with coherent scattering [cm^2/g]
      'total_wo_coh': total attenuation without coherent scattering [cm^2/g]

    Methods:
      update: add or change the search criteria
      perform: perform the query and parse the results
      __getitem__: use [] to get the DataFrame column
      __len__: len() returns the length of the DataFrame
      __str__: str() returns the str method of the DataFrame
      __format__: format() uses the format method of the DataFrame
      keys: returns the DataFrame keys

    """

    def __init__(self, arg, **kwargs):
        """Initialize and perform an XCOM query.

        Args:
          arg: the atomic number, element symbol, compound string, or mixture
            (the type of argument will be inferred from its content).
          e_range_kev (optional): a length-2 iterable giving the lower and
            upper bounds for a standard grid of energies in keV. Limits must be
            between 1 keV and 1E8 keV, inclusive.
          energies_kev (optional): an iterable of specific energies in keV at
            which cross sections will be evaluated. Energies must be between
            1 keV and 1E8 keV, inclusive.
          perform (optional): set to False to prevent query from immediately
            being performed. [default: True]

        Raises:
          XCOMInputError: if bad search criteria are set or search criteria
            are incomplete.
          XCOMRequestError: if there is a problem with the URL request, or
            a problem parsing the data.

        """
        self._url = _URL
        self._req = None
        self._text = None
        self.df = None
        self._data = dict(_DATA)
        self._method = ''
        # determine which kind of argument 'arg' is (symbol, Z, compound, mix)
        kwargs.update(_XCOMQuery._argument_type(arg))
        self.update(**kwargs)
        if kwargs.get('perform', True):
            self.perform()

    def __len__(self):
        """Pass-through to use DataFrame len()."""
        if self.df is None:
            return 0
        elif len(self.df.keys()) == 0:
            return 0
        else:
            return len(self.df[self.df.keys()[0]])

    def keys(self):
        """Pass-through for DataFrame keys method."""
        return self.df.keys()

    def __getitem__(self, key):
        """Pass-through so that [] accesses the DataFrame."""
        return self.df[key]

    def __str__(self):
        """Pass-through to use DataFrame str method."""
        return str(self.df)

    def __format__(self, formatstr):
        """Pass-through to use DataFrame format method."""
        return self.df.__format__(formatstr)

    @staticmethod
    def _argument_type(arg):
        """Determine if argument is a symbol, Z, compound, or mixture."""
        if isstring(arg):
            if arg.isdigit():
                return {'z': arg}
            elif arg.lower() in [s.lower() for s in element.SYMBOLS]:
                return {'symbol': arg}
            else:
                return {'compound': arg}
        elif isinstance(arg, int):
            return {'z': arg}
        elif isinstance(arg, Iterable):
            return {'mixture': arg}
        raise XCOMInputError(
            'Cannot determine if argument {}'.format(arg) +
            ' is a symbol, Z, compound, or mixture')

    @staticmethod
    def _check_z(zstr):
        """Check whether the Z is valid. Raise XCOMInputError if not."""
        zint = int(zstr)
        if zint < 1 or zint > 100:
            raise XCOMInputError(
                'XCOM only supports Z from 1 to 100 (z={})'.format(zstr))

    @staticmethod
    def _check_compound(formula):
        """Check whether the compound is valid. Raise XCOMInputError if not."""
        if not formula.isalnum():
            raise XCOMInputError('Formula not valid: {}'.format(formula))

    @staticmethod
    def _check_mixture(formulae):
        """Check whether the mixture is valid. Raise XCOMInputError if not."""
        if not isinstance(formulae, Iterable):
            raise XCOMInputError(
                'Mixture formulae must be an iterable: {}'.format(formulae))
        for formula in formulae:
            try:
                compound, weight = formula.split()
            except AttributeError:
                raise XCOMInputError(
                    'Mixture formulae "{}" line "{}" must be a string'.format(
                        formulae, formula))
            except ValueError:
                raise XCOMInputError(
                    'Mixture formulae "{}" line "{}" must split into 2'.format(
                        formulae, formula))
            _XCOMQuery._check_compound(compound)
            try:
                float(weight)
            except (ValueError, TypeError):
                raise XCOMInputError(
                    'Mixture formulae "{}" has bad weight "{}"'.format(
                        formulae, weight))

    def update(self, **kwargs):
        """Update the search criteria.

        Before calling perform(), one of the following search criteria must
        be set here: z, symbol, compound, or mixture. A valid query will also
        require setting either e_range_kev and/or energies_kev.

        Args:
          symbol (optional): a string of the element symbol, e.g., 'Ge'.
          z (optional): an integer of the element atomic number, e.g., 32.
          compound (optional): a string of the chemical formula, e.g., 'H2O'.
          mixture (optional): a list of compounds and relative weights, e.g.,
            ['H2O 0.5', 'Ge 0.5']
          e_range_kev (optional): a length-2 iterable giving the lower and
            upper bounds for a standard grid of energies in keV. Limits must be
            between 1 keV and 1E8 keV, inclusive.
          energies_kev (optional): an iterable of specific energies in keV at
            which cross sections will be evaluated. Energies must be between
            1 keV and 1E8 keV, inclusive.

        Raises:
          XCOMInputError: if bad search criteria are set.

        """
        # check for valid keywords
        for kwarg in kwargs:
            if kwarg not in [
                    'symbol', 'z', 'compound', 'mixture',
                    'e_range_kev', 'energies_kev', 'perform']:
                raise XCOMInputError('Unknown keyword: "{}"'.format(kwarg))

        # determine the search method (element, compound, or mixture)
        if 'symbol' in kwargs:
            self._method = '1'
            sym = kwargs['symbol']
            self._data['ZSym'] = sym
        elif 'z' in kwargs:
            self._method = '1'
            znum = kwargs['z']
            _XCOMQuery._check_z(znum)
            self._data['ZNum'] = '{:d}'.format(int(znum))
        elif 'compound' in kwargs:
            # convert compound to mixture to avoid occasional problem
            # with XCOM compound queries (see issue #76)
            # self._method = '2'
            self._method = '3'
            formula = kwargs['compound']
            _XCOMQuery._check_compound(formula)
            formulae = [formula + ' 1']
            _XCOMQuery._check_mixture(formulae)
            formulae = '\r\n'.join(formulae)
            # self._data['Formula'] = formula
            self._data['Formulae'] = formulae
        elif 'mixture' in kwargs:
            self._method = '3'
            formulae = kwargs['mixture']
            _XCOMQuery._check_mixture(formulae)
            formulae = '\r\n'.join(formulae)
            self._data['Formulae'] = formulae

        # include standard grid of energies
        if 'e_range_kev' in kwargs:
            if not isinstance(kwargs['e_range_kev'], Iterable):
                raise XCOMInputError(
                    'XCOM e_range_kev must be iterable of length 2: {}'.format(
                        kwargs['e_range_kev']))
            if len(kwargs['e_range_kev']) != 2:
                raise XCOMInputError(
                    'XCOM e_range_kev must be iterable of length 2: {}'.format(
                        kwargs['e_range_kev']))
            if kwargs['e_range_kev'][0] < 1:
                raise XCOMInputError(
                    'XCOM e_range_kev[0] must be >= 1 keV: {}'.format(
                        kwargs['e_range_kev'][0]))
            if kwargs['e_range_kev'][1] > 1e8:
                raise XCOMInputError(
                    'XCOM e_range_kev[1] must be <= 1E8 keV: {}'.format(
                        kwargs['e_range_kev'][1]))
            if kwargs['e_range_kev'][0] >= kwargs['e_range_kev'][1]:
                raise XCOMInputError(
                    'XCOM e_range_kev[0] must be < e_range_kev[1]: {}'.format(
                        kwargs['e_range_kev']))
            self._data['WindowXmin'] = '{:.6f}'.format(
                kwargs['e_range_kev'][0] / 1000.)
            self._data['WindowXmax'] = '{:.6f}'.format(
                kwargs['e_range_kev'][1] / 1000.)
            self._data['Output'] = 'on'

        # additional energies
        if 'energies_kev' in kwargs:
            if not isinstance(kwargs['energies_kev'], Iterable):
                raise XCOMInputError(
                    'XCOM energies_kev must be an iterable: {}'.format(
                        kwargs['energies_kev']))
            for energy in kwargs['energies_kev']:
                if energy < 1 or energy > 1e8:
                    raise XCOMInputError(
                        'XCOM energy must be >= 1 and <= 1E8 keV: {}'.format(
                            energy))
            self._data['Energies'] = ';'.join([
                '{:.6f}'.format(erg / 1000.)
                for erg in kwargs['energies_kev']])

    def _request(self):
        """Request data table from the URL."""
        self._req = requests.post(self._url + self._method, data=self._data)
        if not self._req.ok or self._req.reason != 'OK' or \
                self._req.status_code != 200:
            raise XCOMRequestError(
                'XCOM Request failed: reason={}, status_code={}'.format(
                    self._req.reason, self._req.status_code))
        if 'Error' in self._req.text:
            raise XCOMRequestError(
                'XCOM returned an error:\n{}'.format(self._req.text))

    def _parse_text(self):
        """Parse table contained in the text into a dictionary."""
        self._text = str(self._req.text)
        if len(self._text) == 0:
            raise XCOMRequestError('XCOM returned no text')
        tables = pd.read_html(self._text, header=0, skiprows=[1, 2])
        if len(tables) != 1:
            raise XCOMRequestError('More than one HTML table found')
        self.df = tables[0]
        if len(self.df.keys()) != 1 + len(COLUMNS_SHORT):
            raise XCOMRequestError(
                'Found {} columns but expected {}'.format(
                    len(self.df.keys()), 1 + len(COLUMNS_SHORT)))
        # remove 'edge' column
        self.df = self.df[self.df.keys()[1:]]
        self.df.columns = COLUMNS_SHORT
        if len(self) == 0:
            raise XCOMRequestError('Parsed DataFrame is empty')

    def perform(self):
        """Perform the query.

        Before calling perform(), set the search criteria using update()
        or __init__().

        Raises:
          XCOMInputError: if search criteria are incomplete.
          XCOMRequestError: if there is a problem with the URL request, or
            a problem parsing the data.

        """
        if self._method not in ['1', '2', '3']:
            raise XCOMInputError(
                'XCOM search method not set. Need to call update() method.')
        if self._data['Energies'] == '' and self._data['Output'] == '':
            raise XCOMInputError('No energies_kev or e_range_kev requested.')
        # submit the query
        self._request()
        # package the output into a pandas DataFrame
        self._parse_text()
        # convert energy from MeV to keV
        self.df['energy'] *= 1000.


def fetch_xcom_data(arg, **kwargs):
    """Query photon cross section data from NIST XCOM database.

    Returns a pandas DataFrame containing the cross section data.

    The DataFrame column names amd units are:
      'energy': Photon Energy [keV]
      'coherent': coherent scattering cross section [cm^2/g]
      'incoherent': incoherent scattering cross section [cm^2/g]
      'photoelec': photoelectric absorption cross section [cm^2/g]
      'pair_nuc': pair production cross section in nuclear field [cm^2/g]
      'pair_elec': pair production cross section in electron field [cm^2/g]
      'total_w_coh': total attenuation with coherent scattering [cm^2/g]
      'total_wo_coh': total attenuation without coherent scattering [cm^2/g]

    Args:
      arg: the atomic number, element symbol, compound string, or mixture
        (the type of argument will be inferred from its content).
      e_range_kev (optional): a length-2 iterable giving the lower and
        upper bounds for a standard grid of energies in keV. Limits must be
        between 1 keV and 1E8 keV, inclusive.
      energies_kev (optional): an iterable of specific energies in keV at
        which cross sections will be evaluated. Energies must be between
        1 keV and 1E8 keV, inclusive.

    Raises:
      XCOMInputError: if bad search criteria are set or search criteria
        are incomplete.
      XCOMRequestError: if there is a problem with the URL request, or
        a problem parsing the data.

    """
    query = _XCOMQuery(arg, **kwargs)
    return query.df
