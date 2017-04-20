"""Query half-life and decay data from the National Nuclear Data Center.

References:
  http://www.nndc.bnl.gov
  http://www.nndc.bnl.gov/nudat2/indx_sigma.jsp
  http://www.nndc.bnl.gov/nudat2/indx_dec.jsp

"""

from __future__ import print_function
import requests
import pandas as pd
import uncertainties


WALLET_DECAY_MODE = {
    'ANY': 'ANY',
    'IT': 'IT',
    'B-': 'B-',
    'EC+B+': 'ECBP',
    'Double Beta': 'DB',
    'Neutron': 'N',
    'Proton': 'P',
    'Alpha': 'A',
    'Cluster': 'C',
    'SF': 'SF',
    'B-delayed n': 'DN',
    'B-delayed p': 'DP',
    'B-delayed a': 'DA',
    'B-delayed F': 'DF',
}


DECAYRAD_DECAY_MODE = {
    'ANY': 'ANY',
    'IT': 'IT',
    'B-': 'B-',
    'EC+B+': 'ECBP',
    'Neutron': 'N',
    'Proton': 'P',
    'Alpha': 'A',
    'SF': 'SF',
}


DECAYRAD_RADIATION_TYPE = {
    'ANY': 'ANY',
    'Gamma': 'G',
    'B-': 'BM',
    'B+': 'BP',
    'Electron': 'E',
    'Proton': 'P',
    'Alpha': 'A',
}


class NNDCError(Exception):
    """General NNDC request error."""

    pass


class NoDataFound(NNDCError):
    """No datasets were found within the specified search."""

    pass


def _parse_float_uncertainty(x, dx):
    """Parse a string and its uncertainty.

    Examples:
    >>> _parse_float_uncertainty('257.123', '0.005')
    257.123+/-0.005
    >>> _parse_float_uncertainty('8', '')
    8.0

    """
    if not isinstance(x, str):
        raise NNDCError('Value must be a string: {}'.format(x))
    if not isinstance(dx, str):
        raise NNDCError('Uncertainty must be a string: {}'.format(dx))
    # replace special characters to ignore
    if '%' in x:
        x = x.replace('%', '')
    if '+X' in x:
        x = x.replace('+X', '')
    if '+Y' in x:
        x = x.replace('+Y', '')
    # handle special ENSDF abbreviations, e.g.,
    # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/14/785/14785563.pdf
    # "One of the following expressions:
    #   LT, GT, LE, GE, AP, CA, SY
    # for less than, greater than, less than or equal to greater
    # than or equal to. approximately equal to, calculated, and
    # from systematics, respectively."
    if x == '':
        return None
    elif '****' in x or '<' in x or '>' in x or '~' in x or '?' in x:
        return None
    elif '****' in dx:
        return None
    elif dx in ['LT', 'GT', 'LE', 'GE', 'AP', 'CA', 'SY']:
        return None
    try:
        x2 = float(x)
    except ValueError:
        raise NNDCError(
            'Value cannot be parsed as float: "{}"'.format(x))
    if dx == '':
        return x2
    # handle multiple exponents with some uncertainties, e.g., "7E-4E-5"
    tokens = dx.split('E')
    if len(tokens) == 3:
        dx = 'E'.join(tokens[:2])
        factor = pow(10., int(tokens[2]))
    else:
        factor = 1.
    try:
        dx2 = float(dx) * factor
    except ValueError:
        raise NNDCError(
            'Uncertainty cannot be parsed as float: "{}"'.format(dx))
    return uncertainties.ufloat(x2, dx2)


class _NNDCQuery(object):
    """National Nuclear Data Center database query base class.

    Search criteria keywords:
        nuc     : (str) : the name of the isotope (e.g., 'Co-60')
        z, a, n : (int) : Z, A, N of the isotope
        z_range, etc. : (tuple of int) : range of Z, A, or N
        z_any, etc. : (bool) : whether any Z, A, or N is considered
        z_odd, etc. : (bool) : only odd Z, A, or N
        z_even, etc.: (bool) : only even Z, A, or N
        t_range : (tuple of float) : range of isotope half-lives
        elevel_range : (tuple of float) : range of nuc. energy level (MeV)

    """

    _URL = ''
    _DATA = {
        'spnuc': '',           # specify parent ('name', 'zan', or 'zanrange')
        'nuc': '',             # isotope name (use with 'name')
        'z': '',               # Z or element (use with 'zan')
        'zmin': '',            # Z min        (use with 'zanrange')
        'zmax': '',            # Z max        (use with 'zanrange')
        'a': '',               # A            (use with 'zan')
        'amin': '',            # A min        (use with 'zanrange')
        'amax': '',            # A max        (use with 'zanrange')
        'n': '',               # N            (use with 'zan')
        'nmin': '',            # N min        (use with 'zanrange')
        'nmax': '',            # N max        (use with 'zanrange')
        'evenz': '',           # 'any', 'even', or 'odd' Z (use with zanrange)
        'evena': '',           # 'any', 'even', or 'odd' A (use with zanrange')
        'evenn': '',           # 'any', 'even', or 'odd' N (use with zanrange)
        'tled': 'disabled',    # half-life condition on/off
        'tlmin': '0',          # half-life min
        'utlow': 'S',          # half-life min units ('S' = seconds)
        'tlmax': '3E17',       # half-life max
        'utupp': 'S',          # half-life max units ('ST' = stable, 'GY' = Gy)
        'notlim': 'disabled',  # half-life: no limit
        'dmed': 'disabled',    # decay mode condition on/off
        'dmn': 'ANY',          # decay mode: 'ANY' = any
        'eled': 'disabled',    # E(level) condition on/off
        'elmin': '0',          # E(level) min
        'elmax': '40',         # E(level) max
        'out': 'file',         # output to formatted file
        'unc': 'stdandard',    # standard style uncertainties
        'sub': 'Search',       # search for the data
    }
    _DUMMY_TEXT = ''

    def __init__(self, **kwargs):
        """Initialize query of NNDC data."""
        self._url = _NNDCQuery._URL
        self._data = dict(_NNDCQuery._DATA)
        self._text = self._DUMMY_TEXT
        self.df = pd.DataFrame()
        self.update(**kwargs)

    def __len__(self):
        """Length of any one of the data lists."""
        if self.df is None:
            return 0
        elif len(self.df.keys()) == 0:
            return 0
        else:
            return len(self.df[self.df.keys()[0]])

    def keys(self):
        """Return the data keys."""
        return self.df.keys()

    def __getitem__(self, key):
        """Return the list given by the key."""
        return self.df[key]

    def __setitem__(self, key, value):
        """Set the list given by the key."""
        self.df[key] = value

    def __str__(self):
        """Use str method for DataFrame."""
        return str(self.df)

    def __format__(self, formatstr):
        """Use format method for DataFrame."""
        return self.df.__format__(formatstr)

    @staticmethod
    def _request(url, data):
        """Request data table from the URL."""
        req = requests.post(url, data=data)
        if not req.ok or req.reason != 'OK' or req.status_code != 200:
            raise NNDCError('Request failed: ' + req.reason)
        for msg in [
                'Your search was unsuccessful',
                'No datasets were found within the specified search',
        ]:
            if msg in req.text:
                raise NoDataFound(msg)
        return req.text

    @staticmethod
    def _parse_headers(headers):
        """Parse table headers and ensure they are unique."""
        headers_new = []
        # reformat column headers if needed
        for j, hd in enumerate(headers):
            # rename so always have T1/2 (s)
            if hd == 'T1/2 (num)' or hd == 'T1/2 (seconds)':
                hd = 'T1/2 (s)'
            # for uncertainties, add previous column header to it
            if j > 0 and 'Unc' in hd:
                hd = headers[j - 1] + ' ' + hd
            if 'Unc' in hd and 'Unc.' not in hd:
                hd = hd.replace('Unc', 'Unc.')
            # expand abbreviated headers
            if 'Energy' in hd and 'Energy Level' not in hd:
                hd = hd.replace('Energy', 'Energy Level')
            if 'Par. Elevel' in hd:
                hd = hd.replace('Par. Elevel', 'Parent Energy Level')
            if 'Abund.' in hd:
                hd = hd.replace('Abund.', 'Abundance (%)')
            if 'Ene.' in hd:
                hd = hd.replace('Ene.', 'Energy')
            if 'Int.' in hd:
                hd = hd.replace('Int.', 'Intensity (%)')
            if 'Dec' in hd and 'Decay' not in hd:
                hd = hd.replace('Dec', 'Decay')
            if 'Rad' in hd and 'Radiation' not in hd:
                hd = hd.replace('Rad', 'Radiation')
            if 'EP' in hd:
                hd = hd.replace('EP', 'Endpoint')
            if 'Mass Exc' in hd and 'Mass Excess' not in hd:
                hd = hd.replace('Mass Exc', 'Mass Excess')
            headers_new.append(hd)
        if len(set(headers_new)) != len(headers_new):
            raise NNDCError(
                'Duplicate headers after parsing\n' +
                '    Original headers: "{}"\n'.format(headers) +
                '    Parsed headers:   "{}"'.format(headers_new))
        return headers_new

    @staticmethod
    def _parse_table(text):
        """Parse table contained in the text into a dictionary."""
        text = str(text)
        try:
            text = text.split('<pre>')[1]
            text = text.split('</pre>')[0]
            text = text.split('To save this output')[0]
            lines = text.split('\n')
        except:
            raise NNDCError('Unable to parse text:\n' + text)
        table = {}
        headers = None
        for line in lines:
            tokens = line.split('\t')
            tokens = [t.strip() for t in tokens]
            if len(tokens) <= 1:
                continue
            if headers is None:
                headers = tokens
                headers = _NNDCQuery._parse_headers(headers)
                for header in headers:
                    table[header] = []
            else:
                if len(tokens) != len(headers):
                    raise NNDCError(
                        'Too few data in table row\n' +
                        '    Headers: "{}"\n'.format(headers) +
                        '    Row:     "{}"'.format(tokens))
                for header, token in zip(headers, tokens):
                    table[header].append(token)
        return table

    _ALLOWED_KEYWORDS = [
        'perform', 'nuc', 'z', 'a', 'n',
        'z_range', 'a_range', 'n_range',
        'z_any', 'z_even', 'z_odd',
        'a_any', 'a_even', 'a_odd',
        'n_any', 'n_even', 'n_odd',
        'elevel_range', 't_range',
    ]

    def update(self, **kwargs):
        """Update the search criteria."""
        for kwarg in kwargs:
            if kwarg not in _NNDCQuery._ALLOWED_KEYWORDS:
                raise NNDCError('Unknown keyword: "{}"'.format(kwarg))
        if 'nuc' in kwargs:
            self._data['spnuc'] = 'name'
            self._data['nuc'] = kwargs['nuc']
        for x in ['z', 'a', 'n']:
            # handle Z, A, and N settings
            if x in kwargs:
                self._data['spnuc'] = 'zan'
                self._data[x.lower()] = '{}'.format(kwargs[x])
            # handle *_range, *_any, *_odd, *_even
            elif x + '_range' in kwargs:
                if len(kwargs[x + '_range']) != 2:
                    raise NNDCError(
                        '{}_range must have two elements: "{}"'.format(
                            x, kwargs[x + '_range']))
                self._data['spnuc'] = 'zanrange'
                self._data[x.lower() + 'min'] = '{}'.format(
                    kwargs[x + '_range'][0])
                self._data[x.lower() + 'max'] = '{}'.format(
                    kwargs[x + '_range'][1])
                if x + '_any' in kwargs:
                    self._data['even' + x.lower()] = 'any'
                elif x + '_even' in kwargs:
                    self._data['even' + x.lower()] = 'even'
                elif x + '_odd' in kwargs:
                    self._data['even' + x.lower()] = 'odd'
        # handle energy level condition
        if 'elevel_range' in kwargs:
            if len(kwargs['elevel_range']) != 2:
                raise NNDCError(
                    'elevel_range must have two elements: "{}"'.format(
                        kwargs['elevel_range']))
            self._data['eled'] = 'enabled'
            self._data['elmin'] = '{}'.format(kwargs['elevel_range'][0])
            self._data['elmax'] = '{}'.format(kwargs['elevel_range'][1])
        # handle half-life range condition
        if 't_range' in kwargs:
            if len(kwargs['t_range']) != 2:
                raise NNDCError(
                    't_range must have two elements: "{}"'.format(
                        kwargs['t_range']))
            self._data['tled'] = 'enabled'
            self._data['tlmin'] = '{}'.format(kwargs['t_range'][0])
            self._data['tlmax'] = '{}'.format(kwargs['t_range'][1])

    def perform(self):
        """Perform the query."""
        # check the conditions
        if self._data['spnuc'] == '':
            raise Exception('Parent nucleus conditions must be set')
        # submit the query
        try:
            self._text = _NNDCQuery._request(self._url, self._data)
        except NoDataFound:
            self._text = self._DUMMY_TEXT
        if len(self._text) == 0:
            raise NNDCError('NNDC returned no text')
        # package the output into a dictionary of arrays
        data = _NNDCQuery._parse_table(self._text)
        # create the DataFrame
        self.df = pd.DataFrame(data)
        # convert dimensionless integers to ints
        for col in ['A', 'Z', 'N', 'M']:
            if col in self.keys():
                self._convert_column(col, int)
        # combine uncertainty columns and add unit labels
        self._add_units_uncertainties()
        # add some more columns
        self._add_columns_energy_levels()
        # sort columns
        self._sort_columns()

    def _add_columns_energy_levels(self):
        """Add nuclear energy level 'M' and 'm' columns using energy levels."""
        if 'Energy Level (MeV)' not in self.df:
            return
        # add column of integer M giving the isomer level (0, 1, 2, ...)
        self.df['M'] = [0] * len(self)
        # add string m giving the isomer level name (e.g., '' or 'm' or 'm2')
        self.df['m'] = [''] * len(self)
        # loop over each isotope in the dataframe
        A_Z = [(a, z) for a, z in zip(self['A'], self['Z'])]
        A_Z = set(A_Z)
        for a, z in A_Z:
            isotope = (self['A'] == a) & (self['Z'] == z)
            e_levels = []
            e_levels_nominal = []
            for e_level in self['Energy Level (MeV)'][isotope]:
                if isinstance(e_level, uncertainties.core.Variable):
                    e_level_nominal = e_level.nominal_value
                else:
                    e_level_nominal = e_level
                if e_level_nominal not in e_levels_nominal:
                    e_levels.append(e_level)
                    e_levels_nominal.append(e_level_nominal)
            e_levels = sorted(e_levels)
            for M, e_level in enumerate(e_levels):
                isomer = isotope & \
                    (abs(self['Energy Level (MeV)'] - e_level) < 1e-10)
                self.df.loc[isomer, 'M'] = M
                if M > 0:
                    if len(e_levels) > 2:
                        self.df.loc[isomer, 'm'] = 'm{}'.format(M)
                    else:
                        self.df.loc[isomer, 'm'] = 'm'

    def _add_units_uncertainties(self):
        """Add units and uncertainties with some columns as applicable."""
        if 'Energy Level' in self.keys():
            self._convert_column('Energy Level', float)
            self.df.rename(
                columns={'Energy Level': 'Energy Level (MeV)'}, inplace=True)
            self.df['Energy Level (MeV)'] *= 1000.

        if 'Parent Energy Level' in self.keys():
            self._convert_column_uncertainty('Parent Energy Level')
            self.df.rename(
                columns={'Parent Energy Level': 'Parent Energy Level (MeV)'},
                inplace=True)

        if 'Mass Excess' in self.keys():
            self._convert_column_uncertainty('Mass Excess')
        self.df.rename(
            columns={'Mass Excess': 'Mass Excess (MeV)'}, inplace=True)

        self._convert_column('T1/2 (s)', float)

        if 'Abundance (%)' in self.keys():
            self._convert_column_uncertainty('Abundance (%)')

        if 'Branching (%)' in self.keys():
            self._convert_column(
                'Branching (%)',
                lambda x: _parse_float_uncertainty(x, ''))

        if 'Radiation Energy' in self.keys():
            self._convert_column_uncertainty('Radiation Energy')
            self.df.rename(
                columns={'Radiation Energy': 'Radiation Energy (keV)'},
                inplace=True)

        if 'Endpoint Energy' in self.keys():
            self._convert_column_uncertainty('Endpoint Energy')
            self.df.rename(
                columns={'Endpoint Energy': 'Endpoint Energy (keV)'},
                inplace=True)

        if 'Radiation Intensity (%)' in self.keys():
            self._convert_column_uncertainty('Radiation Intensity (%)')

        if 'Dose' in self.keys():
            self._convert_column_uncertainty('Dose')
            self.df.rename(
                columns={'Dose': 'Dose (MeV / Bq / s)'}, inplace=True)

    def _convert_column(self, col, function):
        """Convert column from string to another type."""
        col_new = []
        for x in self[col]:
            if x == '':
                col_new.append(None)
            else:
                col_new.append(function(x))
        self.df[col] = col_new

    def _convert_column_uncertainty(self, col):
        """Combine column and its uncertainty into one column."""
        col_new = []
        for x, dx in zip(self[col], self[col + ' Unc.']):
            x2 = _parse_float_uncertainty(x, dx)
            col_new.append(x2)
        self.df[col] = col_new
        del self.df[col + ' Unc.']

    def _sort_columns(self):
        """Sort columns."""
        preferred_order = [
            'Z', 'Element', 'A', 'm', 'M', 'N', 'JPi', 'T1/2',
            'Energy Level (MeV)', 'Decay Mode', 'Branching (%)',
            'Radiation', 'Radiation subtype',
            'Radiation Energy (keV)', 'Radiation Intensity (%)',
        ]
        new_cols = []
        for col in preferred_order:
            if col in self.keys():
                new_cols.append(col)
        for col in self.keys():
            if col not in new_cols:
                new_cols.append(col)
        self.df = self.df[new_cols]


class _NuclearWalletCardQuery(_NNDCQuery):
    """NNDC Nuclear Wallet Card data query.

    Nuclear Wallet Card Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat2/indx_sigma.jsp

    Help page: http://www.nndc.bnl.gov/nudat2/help/wchelp.jsp

      * Energy: Level energy in MeV.
      * JPi: Level spin and parity.
      * Mass Exc: Level Mass Excess in MeV.
      * T1/2 (txt): Level half-life in the format value+units+uncertainty.
      * T1/2 (seconds): value of the level half-life in seconds.
        Levels that are stable are assigned an "infinity" value.
      * Abund.: Natural abundance.
      * Dec Mode: Decay Mode name.
      * Branching (%): Percentual branching ratio for the corresponding
            decay mode.

    Search criteria keywords:
        nuc     : (str) : the name of the isotope (e.g., 'Co-60')
        z, a, n : (int) : Z, A, N of the isotope
        z_range, etc. : (tuple of int) : range of Z, A, or N
        z_any, etc. : (bool) : whether any Z, A, or N is considered
        z_odd, etc. : (bool) : only odd Z, A, or N
        z_even, etc.: (bool) : only even Z, A, or N
        t_range : (tuple of float) : range of isotope half-lives
        elevel_range : (tuple of float) : range of nuc. energy level (MeV)
        decay : (str) : isotope decay mode from WALLET_DECAY_MODE
        j :  (str) : nuclear spin
        parity : (str) : nuclear parity

    To prevent query from being immediately performed, instantiate with
    keyword perform=False.

    """

    _URL = 'http://www.nndc.bnl.gov/nudat2/sigma_searchi.jsp'
    _DATA = {
        'jled': 'disabled',    # J_pi(level) condition on/off
        'jlv': '',             # J
        'plv': 'ANY',          # parity
        'ord': 'zalt',         # order file by Z, A, E(level), T1/2
    }
    _DUMMY_TEXT = """
<html>
<body>

<pre>
A  	Element	Z  	N  	Energy  	JPi           	Mass Exc  	Unc  	T1/2 (txt)               	T1/2 (seconds)           	Abund.    	Unc     	Dec Mode	Branching (%)

 To save this output into a local file, clik on "File" in your browser menu and select "Save as"

</pre></body></html>
"""

    def __init__(self, **kwargs):
        """Initialize NNDC Nuclear Wallet Card search."""
        perform = kwargs.get('perform', True)
        kwargs['perform'] = False
        super(_NuclearWalletCardQuery, self).__init__(**kwargs)
        self._url = self._URL
        self._data.update(self._DATA)
        super(_NuclearWalletCardQuery, self).update(**kwargs)
        self.update(**kwargs)
        if perform:
            self.perform()

    _ALLOWED_KEYWORDS = _NNDCQuery._ALLOWED_KEYWORDS
    _ALLOWED_KEYWORDS.extend(['decay', 'j', 'parity'])

    def update(self, **kwargs):
        """Update the search criteria."""
        super(_NuclearWalletCardQuery, self).update(**kwargs)
        for kwarg in kwargs:
            if kwarg not in _NuclearWalletCardQuery._ALLOWED_KEYWORDS:
                raise NNDCError('Unknown keyword: "{}"'.format(kwarg))
        # handle decay mode
        if 'decay' in kwargs:
            if kwargs['decay'] not in WALLET_DECAY_MODE:
                raise NNDCError('Decay mode must be one of {}, not {}'.format(
                    WALLET_DECAY_MODE.keys(), kwargs['decay']))
            self._data['dmed'] = 'enabled'
            self._data['dmn'] = WALLET_DECAY_MODE[kwargs['decay']]
        # handle half-life range condition
        if 'j' in kwargs:
            self._data['jled'] = 'enabled'
            self._data['jlv'] = kwargs['j']
        if 'parity' in kwargs:
            self._data['jled'] = 'enabled'
            self._data['plv'] = kwargs['parity']


def fetch_wallet_card(**kwargs):
    """Perform NNDC Nuclear Wallet Card data query and return a DataFrame.

    Nuclear Wallet Card Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat2/indx_sigma.jsp

    Help page: http://www.nndc.bnl.gov/nudat2/help/wchelp.jsp

      * Energy: Level energy in MeV.
      * JPi: Level spin and parity.
      * Mass Exc: Level Mass Excess in MeV.
      * T1/2 (txt): Level half-life in the format value+units+uncertainty.
      * T1/2 (seconds): value of the level half-life in seconds.
        Levels that are stable are assigned an "infinity" value.
      * Abund.: Natural abundance.
      * Dec Mode: Decay Mode name.
      * Branching (%): Percentual branching ratio for the corresponding
            decay mode.

    Search criteria keywords:
        nuc     : (str) : the name of the isotope (e.g., 'Co-60')
        z, a, n : (int) : Z, A, N of the isotope
        z_range, etc. : (tuple of int) : range of Z, A, or N
        z_any, etc. : (bool) : whether any Z, A, or N is considered
        z_odd, etc. : (bool) : only odd Z, A, or N
        z_even, etc.: (bool) : only even Z, A, or N
        t_range : (tuple of float) : range of isotope half-lives
        elevel_range : (tuple of float) : range of nuc. energy level (MeV)
        decay : (str) : isotope decay mode from WALLET_DECAY_MODE
        j :  (str) : nuclear spin
        parity : (str) : nuclear parity

    """
    query = _NuclearWalletCardQuery(**kwargs)
    return query.df


class _DecayRadiationQuery(_NNDCQuery):
    """NNDC Decay Radiation data query.

    Decay Radiation Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat2/indx_dec.jsp

    Help page: http://www.nndc.bnl.gov/nudat2/help/dehelp.jsp

      * Radiation: Radiation type, i.e. G for gamma, E for electron.
      * Rad subtype: Further classification of the radiation type.
      * Rad Ene.: Radiation energy in keV.
      * EP Ene.: Beta-decay end point energy in keV.
      * Rad Int.: Radiation absolute intensity.
      * Dose: Radiation dose in MeV/Bq-s
      * Unc: Uncertainties

    Search criteria keywords:
        nuc     : (str) : the name of the isotope (e.g., 'Co-60')
        z, a, n : (int) : Z, A, N of the isotope
        z_range, etc. : (tuple of int) : range of Z, A, or N
        z_any, etc. : (bool) : whether any Z, A, or N is considered
        z_odd, etc. : (bool) : only odd Z, A, or N
        z_even, etc.: (bool) : only even Z, A, or N
        t_range : (tuple of float) : range of isotope half-lives
        elevel_range : (tuple of float) : range of nuc. energy level (MeV)
        decay : (str) : isotope decay mode from DECAYRAD_DECAY_MODE
        type :  (str) : radiation type from DECAYRAD_RADIATION_TYPE
        e_range : (tuple of float) : radiation energy range (keV)
        i_range : (tuple of float): intensity range (percent)

    To prevent query from being immediately performed, instantiate with
    keyword perform=False.

    """

    _URL = 'http://www.nndc.bnl.gov/nudat2/dec_searchi.jsp'
    _DATA = {
        'rted': 'enabled',     # radiation type condition on/off
        'rtn': 'ANY',          # radiation type: 'ANY' = any, 'G' = gamma
        'reed': 'disabled',    # radiation energy condition on/off
        'remin': '0',          # radiation energy min (keV)
        'remax': '10000',      # radiation energy max (keV)
        'ried': 'disabled',    # radiation intensity condition on/off
        'rimin': '0',          # radiation intensity min (%)
        'rimax': '100',        # radiation intensity max (%)
        'ord': 'zate',         # order file by Z, A, T1/2, E
    }
    _DUMMY_TEXT = """
<html>
<body>
<pre>
A  	Element	Z  	N  	Par. Elevel	Unc. 	JPi       	Dec Mode	T1/2 (txt)    	T1/2 (num)        	Daughter	Radiation	Rad subtype 	Rad Ene.  	Unc       	EP Ene.   	Unc       	Rad Int.  	Unc       	Dose        	Unc

</pre>
To save this output into a local File, clik on "File" in your browser menu and select "Save as"
</body></html>
"""

    def __init__(self, **kwargs):
        """Initialize NNDC Decay Radiation search."""
        perform = kwargs.get('perform', True)
        kwargs['perform'] = False
        super(_DecayRadiationQuery, self).__init__(**kwargs)
        self._url = self._URL
        self._data.update(self._DATA)
        super(_DecayRadiationQuery, self).update(**kwargs)
        self.update(**kwargs)
        if perform:
            self.perform()

    _ALLOWED_KEYWORDS = _NNDCQuery._ALLOWED_KEYWORDS
    _ALLOWED_KEYWORDS.extend(['decay', 'type', 'e_range', 'i_range'])

    def update(self, **kwargs):
        """Update the search criteria."""
        super(_DecayRadiationQuery, self).update(**kwargs)
        for kwarg in kwargs:
            if kwarg not in _DecayRadiationQuery._ALLOWED_KEYWORDS:
                raise NNDCError('Unknown keyword: "{}"'.format(kwarg))
        # handle decay mode
        if 'decay' in kwargs:
            if kwargs['decay'] not in DECAYRAD_DECAY_MODE:
                raise NNDCError('Decay mode must be one of {}, not {}'.format(
                    DECAYRAD_DECAY_MODE.keys(), kwargs['decay']))
            self._data['dmed'] = 'enabled'
            self._data['dmn'] = DECAYRAD_DECAY_MODE[kwargs['decay']]
        # handle radiation type
        if 'type' in kwargs:
            if kwargs['type'] not in DECAYRAD_RADIATION_TYPE:
                raise NNDCError(
                    'Radiation type must be one of {}, not {}'.format(
                        DECAYRAD_RADIATION_TYPE.keys(), kwargs['type']))
            self._data['rted'] = 'enabled'
            self._data['rtn'] = DECAYRAD_RADIATION_TYPE[kwargs['type']]
        # handle radiation energy range
        if 'e_range' in kwargs:
            if len(kwargs['e_range']) != 2:
                raise NNDCError(
                    'e_range must have two elements: "{}"'.format(
                        kwargs['e_range']))
            self._data['reed'] = 'enabled'
            self._data['remin'] = '{}'.format(kwargs['e_range'][0])
            self._data['remax'] = '{}'.format(kwargs['e_range'][1])
        # handle radiation intensity range
        if 'i_range' in kwargs:
            if len(kwargs['i_range']) != 2:
                raise NNDCError(
                    'i_range must have two elements: "{}"'.format(
                        kwargs['i_range']))
            self._data['ried'] = 'enabled'
            self._data['rimin'] = '{}'.format(kwargs['i_range'][0])
            self._data['rimax'] = '{}'.format(kwargs['i_range'][1])


def fetch_decay_radiation(**kwargs):
    """Perform NNDC Decay Radiation data query and return a DataFrame.

    Decay Radiation Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat2/indx_dec.jsp

    Help page: http://www.nndc.bnl.gov/nudat2/help/dehelp.jsp

      * Radiation: Radiation type, i.e. G for gamma, E for electron.
      * Rad subtype: Further classification of the radiation type.
      * Rad Ene.: Radiation energy in keV.
      * EP Ene.: Beta-decay end point energy in keV.
      * Rad Int.: Radiation absolute intensity.
      * Dose: Radiation dose in MeV/Bq-s
      * Unc: Uncertainties

    Search criteria keywords:
        nuc     : (str) : the name of the isotope (e.g., 'Co-60')
        z, a, n : (int) : Z, A, N of the isotope
        z_range, etc. : (tuple of int) : range of Z, A, or N
        z_any, etc. : (bool) : whether any Z, A, or N is considered
        z_odd, etc. : (bool) : only odd Z, A, or N
        z_even, etc.: (bool) : only even Z, A, or N
        t_range : (tuple of float) : range of isotope half-lives
        elevel_range : (tuple of float) : range of nuc. energy level (MeV)
        decay : (str) : isotope decay mode from DECAYRAD_DECAY_MODE
        type :  (str) : radiation type from DECAYRAD_RADIATION_TYPE
        e_range : (tuple of float) : radiation energy range (keV)
        i_range : (tuple of float): intensity range (percent)

    """
    query = _DecayRadiationQuery(**kwargs)
    return query.df
