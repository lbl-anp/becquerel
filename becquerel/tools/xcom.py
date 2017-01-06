"""Query photon cross section data from NIST XCOM database.

Use the XCOMQuery class to query the database.

References:
  https://www.nist.gov/pml/xcom-photon-cross-sections-database
  http://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

"""

from __future__ import print_function
from collections import OrderedDict
import requests
import pandas as pd
from .units import units


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


# URL at NIST where data are posted
_URL = 'http://physics.nist.gov/cgi-bin/Xcom/data.pl'


# Dictionary of data that will be posted to URL
_DATA = {
    'Method': '',           # '1' = element, '2' = compound, '3' = mixture
    'NumAdd': '1',          # always '1'?
    'Energies': '',         # additional energies separated by ; (MeV)
    'WindowXmin': '0.001',  # lower limit of energy grid (MeV)
    'WindowXmax': '100',    # upper limit of energy grid (MeV)
    'Output': '',           # 'on' for standard energy grid
    'photoelectric': 'on',  # return photoelectric cross section
    'coherent': 'on',       # return coherent scattering cross section
    'incoherent': 'on',     # return incoherent scattering cross section
    'nuclear': 'on',        # return nuclear pair production cross section
    'electron': 'on',       # return electron pair production cross section
    'with': 'on',           # return total cross section with coherent scat.
    'without': 'on',        # return total cross section w/o coherent scat.
    'OutOpt': 'PIC',        # return cross sections in cm^2/g
    'character': 'bar',     # data delimiter
}

# abbreviated names for the table columns (these are used in the dataframe)
COLUMNS_SHORT = ['energy', 'C', 'I', 'PA', 'PPN', 'PPE', 'T+C', 'T-C']

# medium length names for the table columns
COLUMNS_MEDIUM = {
    'energy': 'Energy',
    'C': 'Coherent',
    'I': 'Incoherent',
    'PA': 'Photoelectric',
    'PPN': 'Pair production (nuclear)',
    'PPE': 'Pair Production (electron)',
    'T+C': 'Total with coherent',
    'T-C': 'Total without coherent'
}

# verbose names for the table columns
COLUMNS_LONG = {
    'energy': 'Photon Energy',
    'C': 'Coherent Scattering',
    'I': 'Incoherent Scattering',
    'PA': 'Photoelectric Absorption',
    'PPN': 'Pair Production in Nuclear Field',
    'PPE': 'Pair Production in Electron Field',
    'T+C': 'Total Attenuation with Coherent Scattering',
    'T-C': 'Total Attenuation without Coherent Scattering'
}


class XCOMError(Exception):
    """General XCOM error."""

    pass


class XCOMQuery(object):
    """Query photon cross section data from NIST XCOM database."""

    def __init__(self, **kwargs):
        """Initialize and perform an XCOM query. Return a DataFrame."""
        self._url = _URL
        self._data = None
        self._req = None
        self._text = None
        self._df = None
        self._set_data(**kwargs)
        self._perform()

    def __len__(self):
        """Length of any one of the data lists."""
        if self._df is None:
            return 0
        elif len(self._df.keys()) == 0:
            return 0
        else:
            return len(self._df[self._df.keys()[0]])

    def keys(self):
        """Return the data columns."""
        return self._df.keys()

    def __getitem__(self, key):
        """Return the column given by the key."""
        return self._df[key]

    def __str__(self):
        """Use str method for DataFrame."""
        return str(self._df)

    def __format__(self, formatstr):
        """Use format method for DataFrame."""
        return self._df.__format__(formatstr)

    @staticmethod
    def check_keywords(**kwargs):
        """Ensure only one of symbol, Z, compound, or mixture in kwargs."""
        kws = ['symbol', 'Z', 'compound', 'mixture']
        for kw1 in kws:
            for kw2 in kws:
                if kw1 != kw2:
                    if kw1 in kwargs and kw2 in kwargs:
                        raise XCOMError(
                            'Cannot provide both {} and {}'.format(kw1, kw2))

    @staticmethod
    def check_z(zstr):
        """Check whether the Z is valid. Raise XCOMError if not."""
        zint = int(zstr)
        if zint < 1 or zint > 100:
            raise XCOMError(
                'XCOM only supports Z from 1 to 100 (z={})'.format(zstr))

    @staticmethod
    def check_compound(formula):
        """Check whether the compound is valid. Raise XCOMError if not."""
        if not formula.isalnum():
            raise XCOMError('Formula not valid: {}'.format(formula))

    @staticmethod
    def check_mixture(formulae):
        """Check whether the mixture is valid. Raise XCOMError if not."""
        try:
            for formula in formulae:
                pass
        except:
            raise XCOMError(
                'Mixture formulae must be an iterable: {}'.format(
                    formulae))
        for formula in formulae:
            try:
                compound, weight = formula.split()
            except:
                raise XCOMError(
                    'Mixture formulae "{}" has bad line "{}"'.format(
                        formulae, formula))
            XCOMQuery.check_compound(compound)
            try:
                float(weight)
            except:
                raise XCOMError(
                    'Mixture formulae "{}" has bad weight "{}"'.format(
                        formulae, weight))

    def _set_data(self, **kwargs):
        """Construct query data."""
        self._data = dict(_DATA)

        # check the keywords
        XCOMQuery.check_keywords(**kwargs)

        # determine the method (element, compound, or mixture)
        if 'symbol' in kwargs:
            self._data['Method'] = '1'
            self._data['ZSym'] = kwargs['symbol']
        elif 'Z' in kwargs:
            self._data['Method'] = '1'
            znum = kwargs['Z']
            XCOMQuery.check_z(znum)
            self._data['ZNum'] = '{:d}'.format(int(znum))
        elif 'compound' in kwargs:
            self._data['Method'] = '2'
            formula = kwargs['compound']
            XCOMQuery.check_compound(formula)
            self._data['Formula'] = formula
        elif 'mixture' in kwargs:
            self._data['Method'] = '3'
            formulae = kwargs['mixture']
            XCOMQuery.check_mixture(formulae)
            formulae = '\r\n'.join(formulae)
            self._data['Formulae'] = formulae

        # include standard grid of energies
        if 'e_range' in kwargs:
            self._data['WindowXmin'] = '{:.6f}'.format(
                kwargs['e_range'][0] / 1000.)
            self._data['WindowXmax'] = '{:.6f}'.format(
                kwargs['e_range'][1] / 1000.)
            self._data['Output'] = 'on'

        # additional energies
        if 'energies' in kwargs:
            self._data['Energies'] = ';'.join(
                ['{:.6f}'.format(erg / 1000.) for erg in kwargs['energies']])

    def _request(self):
        """Request data table from the URL."""
        self._req = requests.post(self._url, data=self._data)
        if not self._req.ok or self._req.reason != 'OK' or \
                self._req.status_code != 200:
            raise XCOMError(
                'XCOM Request failed: reason={}, status_code={}'.format(
                    self._req.reason, self._req.status_code))
        if 'Error' in self._req.text:
            raise XCOMError(
                'XCOM returned an error:\n{}'.format(self._req.text))

    def _parse_text(self):
        """Parse table contained in the text into a dictionary."""
        self._text = str(self._req.text)
        if len(self._text) == 0:
            raise XCOMError('XCOM returned no text')
        lines = [line for line in self._text.split('\n')]
        table = OrderedDict()
        for key in COLUMNS_SHORT:
            table[key] = []
        for line in lines[3:]:
            tokens = line.split('|')
            if len(tokens) == 9:
                for column, token in zip(COLUMNS_SHORT, tokens):
                    table[column].append(float(token))
        self._df = pd.DataFrame(table)
        if len(self) == 0:
            raise XCOMError('Parsed DataFrame is empty')

    def _add_units(self):
        """Add units to column."""
        self._df['energy'] = [
            x * 1000. * units.keV for x in self._df['energy']]
        cm2_g = units.parse_expression('cm^2 / g')
        for key in self.keys():
            if key != 'energy':
                self._df[key] = [x * cm2_g for x in self._df[key]]

    def _perform(self):
        """Perform the query."""
        # submit the query
        self._request()
        # package the output into a pandas DataFrame
        self._parse_text()
        # add column units
        self._add_units()
