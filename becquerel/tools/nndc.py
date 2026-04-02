"""Query half-life and decay data from the National Nuclear Data Center.

References:
  http://www.nndc.bnl.gov
  http://www.nndc.bnl.gov/nudat3/indx_sigma.jsp
  http://www.nndc.bnl.gov/nudat3/indx_dec.jsp

"""

import functools
import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import requests
import uncertainties

from ._nndc_dummy_text import (
    _DECAY_RADIATION_QUERY_DUMMY_TEXT,
    _NUCLEAR_WALLET_CARD_QUERY_DUMMY_TEXT,
)

PARITIES = ["+", "-", "any"]


DECAYRAD_DECAY_MODE = {
    "any": "ANY",
    "internal transition": "IT",
    "it": "IT",
    "beta-": "B-",
    "b-": "B-",
    "electron capture beta+": "ECBP",
    "ecbp": "ECBP",
    "ecb+": "ECBP",
    "ec+b+": "ECBP",
    "electron capture": "ECBP",
    "ec": "ECBP",
    "beta+": "ECBP",
    "b+": "ECBP",
    "neutron": "N",
    "n": "N",
    "proton": "P",
    "p": "P",
    "alpha": "A",
    "a": "A",
    "spontaneous fission": "SF",
    "sf": "SF",
}


WALLET_DECAY_MODE = dict(DECAYRAD_DECAY_MODE)
WALLET_DECAY_MODE.update(
    {
        "double beta": "DB",
        "bb": "DB",
        "cluster": "C",
        "c": "C",
        "beta-delayed neutron": "DN",
        "b-delayed n": "DN",
        "bdn": "DN",
        "beta-delayed proton": "DP",
        "b-delayed p": "DP",
        "bdp": "DP",
        "beta-delayed alpha": "DA",
        "b-delayed a": "DA",
        "bda": "DA",
        "beta-delayed fission": "DF",
        "b-delayed f": "DF",
        "bdf": "DF",
    }
)

_WALLET_SEARCH_URL = "https://www.nndc.bnl.gov/walletcards/StandardSearchServlet"
_NNDC_TIMEOUT_SECONDS = 60
_NNDC_MAX_RETRIES = 5
_WALLET_TOO_MANY_RESULTS = 3000
_WALLET_NUCLIDE_PATTERN = re.compile(
    r"^\s*(?:(?P<symbol>[A-Za-z]{1,3})-?(?P<mass>\d+)(?P<meta>m\d*)?|"
    r"(?P<mass2>\d+)-?(?P<symbol2>[A-Za-z]{1,3})(?P<meta2>m\d*)?)\s*$"
)
_WALLET_TIME_UNITS = {
    "fs": 1e-15,
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": 31557600.0,
    "ky": 31557600.0e3,
    "my": 31557600.0e6,
    "gy": 31557600.0e9,
}
_WALLET_DECAY_KEYS = {
    "A": frozenset({"a", "2a"}),
    "B-": frozenset({"B-"}),
    "ECBP": frozenset({"EC", "B+", "EC+B+"}),
    "IT": frozenset({"IT"}),
    "N": frozenset({"n", "2n", "3n", "4n"}),
    "P": frozenset({"p", "2p", "3p"}),
    "SF": frozenset({"f"}),
    "DB": frozenset({"2B-", "2EC"}),
    "DN": frozenset({"B-n", "B-2n", "B-3n", "B-4n", "B-5n", "B-6n", "B-7n"}),
    "DP": frozenset({"B+p", "B+2p", "B+3p", "ECp", "EC2p", "EC3p", "B-p"}),
    "DA": frozenset({"B-a", "B+a", "ECa", "ECap"}),
    "DF": frozenset({"B-f", "ECf", "EC+B+f"}),
}
_WALLET_NON_CLUSTER_DECAY_KEYS = frozenset(
    key for keys in _WALLET_DECAY_KEYS.values() for key in keys
).union({"IT", "EC+B+", "EC", "B+", "B-"})


DECAYRAD_RADIATION_TYPE = {
    "any": "ANY",
    "gamma": "G",
    "g": "G",
    "beta-": "BM",
    "b-": "BM",
    "beta+": "BP",
    "b+": "BP",
    "electron": "E",
    "e": "E",
    "proton": "P",
    "p": "P",
    "alpha": "A",
    "a": "A",
}


class NNDCError(Exception):
    """General NNDC request error."""


class NoDataFound(NNDCError):
    """No datasets were found within the specified search."""


class NNDCInputError(NNDCError):
    """Error related to the user input to an NNDC query."""


class NNDCRequestError(NNDCError):
    """Error related to communicating with NNDC or parsing the result."""


def _parse_headers(headers):
    """Parse table headers and ensure they are unique.

    Args:
      headers: a list of column header strings.

    Returns:
      a new list of strings where abbreviations have been expanded.

    Raises:
      NNDCRequestError: if there was a problem parsing the headers.

    """

    headers_new = []
    # reformat column headers if needed
    for j, hd in enumerate(headers):
        # rename so always have T1/2 (s)
        if hd in ("T1/2 (num)", "T1/2 (seconds)"):
            hd = "T1/2 (s)"
        # for uncertainties, add previous column header to it
        if j > 0 and "Unc" in hd:
            hd = headers[j - 1] + " " + hd
        if "Unc" in hd and "Unc." not in hd:
            hd = hd.replace("Unc", "Unc.")
        # expand abbreviated headers
        if "Energy" in hd and "Energy Level" not in hd:
            hd = hd.replace("Energy", "Energy Level")
        if "Par. Elevel" in hd:
            hd = hd.replace("Par. Elevel", "Parent Energy Level")
        if "Abund." in hd:
            hd = hd.replace("Abund.", "Abundance (%)")
        if "Ene." in hd:
            hd = hd.replace("Ene.", "Energy")
        if "Int." in hd:
            hd = hd.replace("Int.", "Intensity (%)")
        if "Dec" in hd and "Decay" not in hd:
            hd = hd.replace("Dec", "Decay")
        if "Rad" in hd and "Radiation" not in hd:
            hd = hd.replace("Rad", "Radiation")
        if "EP" in hd:
            hd = hd.replace("EP", "Endpoint")
        if "Mass Exc" in hd and "Mass Excess" not in hd:
            hd = hd.replace("Mass Exc", "Mass Excess")
        headers_new.append(hd)
    if len(set(headers_new)) != len(headers_new):
        raise NNDCRequestError(
            "Duplicate headers after parsing\n"
            f'    Original headers: "{headers}"\n'
            f'    Parsed headers:   "{headers_new}"'
        )
    return headers_new


def _parse_table(text):
    """Parse table contained in the text into a dictionary.

    Args:
      text: a string containing an HTML table from the NNDC request

    Returns:
      a dictionary of lists keyed by the column headers.

    Raises:
      NNDCRequestError: if unable to parse the table.

    """

    text = str(text)
    try:
        text = text.split("<pre>")[1]
        text = text.split("</pre>")[0]
        text = text.split("To save this output")[0]
        lines = text.split("\n")
    except Exception as exc:
        raise NNDCRequestError(f"Unable to parse text:\n{exc}\n{text}") from exc
    table = {}
    headers = None
    for line in lines:
        tokens = line.split("\t")
        tokens = [t.strip() for t in tokens]
        if len(tokens) <= 1:
            continue
        if headers is None:
            headers = tokens
            headers = _parse_headers(headers)
            for header in headers:
                table[header] = []
        else:
            if len(tokens) != len(headers):
                raise NNDCRequestError(
                    "Too few data in table row\n"
                    f'    Headers: "{headers}"\n'
                    f'    Row:     "{tokens}"'
                )
            for header, token in zip(headers, tokens):
                table[header].append(token)
    return table


def _parse_float_uncertainty(x, dx):
    """Parse a string and its uncertainty into a float or ufloat.

    Examples:
      >>> _parse_float_uncertainty("257.123", "0.005")
      257.123+/-0.005
      >>> _parse_float_uncertainty("8", "")
      8.0

    Args:
      x: a string representing the nominal value of the quantity.
      dx: a string representing the uncertainty of the quantity.

    Returns:
      a float (if dx == '') or a ufloat.

    Raises:
      NNDCRequestError: if values cannot be parsed.

    """

    if not isinstance(x, str):
        raise NNDCRequestError(f"Value must be a string: {x}")
    if not isinstance(dx, str):
        raise NNDCRequestError(f"Uncertainty must be a string: {dx}")
    # ignore percents
    if "%" in x:
        x = x.replace("%", "")
    # ignore unknown ground state levels (X, Y, Z, W)
    for sym in ["X", "Y", "Z", "W"]:
        if "+" + sym in x:
            x = x.replace("+" + sym, "")
        elif x == sym:
            x = "0"
    # handle special ENSDF abbreviations, e.g.,
    # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/14/785/14785563.pdf
    # "One of the following expressions:
    #   LT, GT, LE, GE, AP, CA, SY
    # for less than, greater than, less than or equal to greater
    # than or equal to. approximately equal to, calculated, and
    # from systematics, respectively."
    for sym in ["*", "<", ">", "=", "~", "?", "@", "&", "P", "N"]:
        while sym in x:
            x = x.replace(sym, "")
    # correct specific typos in the database
    if "E-11 0" in x:
        x = x.replace("E-11 0", "E-11")
    if "E-12 0" in x:
        x = x.replace("E-12 0", "E-12")
    if "0.0000 1" in x:
        x = x.replace("0.0000 1", "0.0000")
    if "2 .8E-7" in x:
        x = x.replace("2 .8E-7", "2.8E-7")
    if "8 .0E-E5" in x:
        x = x.replace("8 .0E-E5", "8.0E-5")
    # handle blank or missing data
    if x in ("", " "):
        return None
    if "****" in dx or dx in ["LT", "GT", "LE", "GE", "AP", "CA", "SY"]:
        dx = ""
    try:
        x2 = float(x)
    except ValueError as exc:
        raise NNDCRequestError(f'Value cannot be parsed as float: "{x}"') from exc
    if dx == "":
        return x2
    # handle multiple exponents with some uncertainties, e.g., "7E-4E-5"
    tokens = dx.split("E")
    if len(tokens) == 3:
        dx = "E".join(tokens[:2])
        factor = pow(10.0, int(tokens[2]))
    else:
        factor = 1.0
    try:
        dx2 = float(dx) * factor
    except ValueError as exc:
        raise NNDCRequestError(
            f'Uncertainty cannot be parsed as float: "{dx}"'
        ) from exc
    return uncertainties.ufloat(x2, dx2)


def _format_range(x_range):
    """Return two strings for the two range elements, blank if not finite.

    Args:
      x_range: an iterable of 2 range limits, which can be numbers
        or inf/NaN/None.

    Returns:
      an iterable of 2 strings.

    Raises:
      NNDCInputError: if x_range is not an iterable of length 2.

    """

    try:
        x1, x2 = x_range
    except (TypeError, ValueError) as exc:
        raise NNDCInputError(
            f'Range keyword arg must have two elements: "{x_range}"'
        ) from exc
    try:
        if np.isfinite(x1):
            x1 = f"{x1}"
        else:
            x1 = ""
    except TypeError:
        x1 = ""
    try:
        if np.isfinite(x2):
            x2 = f"{x2}"
        else:
            x2 = ""
    except TypeError:
        x2 = ""
    return x1, x2


def _wallet_format_nuclide(nuclide):
    """Convert legacy isotope strings like `Co-60` into NNDC wallet format."""
    nuclide = str(nuclide).strip()
    match = _WALLET_NUCLIDE_PATTERN.match(nuclide)
    if match is None:
        return nuclide
    groups = match.groupdict()
    symbol = groups["symbol"] or groups["symbol2"]
    mass = groups["mass"] or groups["mass2"]
    meta = groups["meta"] or groups["meta2"]
    if meta:
        raise NNDCRequestError(
            "Request failed: wallet card search does not support "
            f"metastable nuclides: {nuclide}"
        )
    symbol = symbol[0].upper() + symbol[1:].lower()
    return f"{mass}{symbol}"


def _wallet_int_or_none(value):
    """Convert a numeric wallet search parameter to int or None."""
    if value in ("", None):
        return None
    return int(float(value))


def _wallet_float_or_none(value):
    """Convert a numeric wallet value to float or None."""
    if value in ("", None):
        return None
    return float(value)


def _wallet_value(entry, scale=1.0):
    """Extract a scalar value from a wallet JSON quantity."""
    if not entry or entry.get("value") is None:
        return None
    return float(entry["value"]) / scale


def _wallet_half_life_seconds(row):
    """Return the half-life in seconds, or inf for stable isotopes."""
    if row.get("stable") is True:
        return np.inf
    half_life = row.get("halfLife")
    if not half_life or half_life.get("value") is None:
        return None
    unit = str(half_life.get("unit", "")).lower()
    factor = _WALLET_TIME_UNITS.get(unit)
    if factor is None:
        return None
    return float(half_life["value"]) * factor


def _wallet_level_energy_mev(row):
    """Return the level energy in MeV."""
    level_energy = row.get("levelEnergy")
    if not level_energy or level_energy.get("value") is None:
        return None
    if level_energy.get("isRelative") is True:
        return None
    unit = str(level_energy.get("unit", "")).lower()
    scale = 1000.0 if unit == "kev" else 1.0
    return float(level_energy["value"]) / scale


def _wallet_abundance(row):
    """Return a scalar abundance percentage if available."""
    abundance = _wallet_value(row.get("abundance"))
    if abundance is not None:
        return abundance
    abundance_range = row.get("abundanceRange")
    if not abundance_range:
        return None
    minimum = _wallet_float_or_none(abundance_range.get("minimum"))
    maximum = _wallet_float_or_none(abundance_range.get("maximum"))
    if minimum is None or maximum is None:
        return None
    return 0.5 * (minimum + maximum)


def _wallet_branching_ratios(row):
    """Return the branching ratio mapping for a wallet JSON row."""
    return row.get("branchingRatios") or {}


def _wallet_decay_mode_code(mode):
    """Translate a wallet JSON decay mode key into the legacy table code."""
    if mode in {"EC", "B+"}:
        return mode
    for legacy_mode, wallet_modes in _WALLET_DECAY_KEYS.items():
        if mode in wallet_modes:
            return legacy_mode
    return mode


def _wallet_branching_value(branch):
    """Extract a branching ratio value from a wallet decay-mode entry."""
    value = branch.get("value")
    if value is not None and float(value) != 0.0:
        return float(value)
    measurements = (branch.get("measurements") or {}).get("measuredValues") or []
    for measured in measurements:
        if measured.get("isIncluded") and measured.get("value") is not None:
            return float(measured["value"])
    if value is not None:
        return float(value)
    return None


def _wallet_decay_modes(row):
    """Return all decay modes and branching ratios for a wallet row."""
    branching = _wallet_branching_ratios(row)
    if not branching:
        return [(None, None)]
    return [
        (_wallet_decay_mode_code(mode), _wallet_branching_value(branch))
        for mode, branch in branching.items()
    ]


def _wallet_primary_branching(row):
    """Return the dominant decay mode and branching ratio for a wallet row."""
    branching = _wallet_branching_ratios(row)
    if not branching:
        return None, None
    mode, branch = max(
        branching.items(),
        key=lambda item: _wallet_branching_value(item[1]) or 0.0,
    )
    return _wallet_decay_mode_code(mode), _wallet_branching_value(branch)


def _wallet_matches_decay_mode(mode, row):
    """Return whether a wallet row matches a legacy decay mode code."""
    branching = _wallet_branching_ratios(row)
    if mode == "ANY":
        return True
    if mode == "C":
        return any(key not in _WALLET_NON_CLUSTER_DECAY_KEYS for key in branching)
    return any(key in branching for key in _WALLET_DECAY_KEYS.get(mode, frozenset()))


@functools.lru_cache(maxsize=256)
def _nndc_request_text(url, payload_items):
    """POST a classic NNDC form request with retry/caching."""
    payload = dict(payload_items)
    last_response = None
    headers = {"User-Agent": "Mozilla/5.0"}
    if url.startswith("https://www.nndc.bnl.gov/"):
        index_url = (
            url.replace("dec_searchi.jsp", "indx_dec.jsp")
            .replace("sigma_searchi.jsp", "indx_sigma.jsp")
            .replace("dec_search.jsp", "indx_dec.jsp")
            .replace("sigma_search.jsp", "indx_sigma.jsp")
        )
        headers.update(
            {
                "Referer": index_url,
                "Origin": "https://www.nndc.bnl.gov",
            }
        )
    else:
        index_url = None
    for attempt in range(_NNDC_MAX_RETRIES):
        with requests.Session() as session, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ResourceWarning)
            if index_url is not None:
                session.get(index_url, headers=headers, timeout=_NNDC_TIMEOUT_SECONDS)
            resp = session.post(
                url,
                data=payload,
                headers=headers,
                stream=False,
                timeout=_NNDC_TIMEOUT_SECONDS,
            )
        last_response = resp
        if resp.status_code != 429:
            break
        retry_after = resp.headers.get("Retry-After")
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = 0.5 * (2**attempt)
        time.sleep(delay)
    if not last_response.ok or last_response.status_code != 200:
        reason = last_response.reason or f"HTTP {last_response.status_code}"
        raise NNDCRequestError("Request failed: " + reason)
    return last_response.text


@functools.lru_cache(maxsize=128)
def _wallet_search(url, payload_json):
    """Fetch wallet card data from the current NNDC JSON endpoint."""
    with requests.Session() as session, warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        resp = session.post(
            url,
            data=payload_json,
            headers={"Content-Type": "application/json"},
            stream=False,
            timeout=_NNDC_TIMEOUT_SECONDS,
        )
    if not resp.ok or resp.status_code != 200:
        raise NNDCRequestError("Request failed: " + resp.reason)
    if not resp.text.strip():
        return []
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError as exc:
        raise NNDCRequestError("Unable to parse wallet card JSON response") from exc
    if not isinstance(data, list):
        raise NNDCRequestError("Unexpected wallet card response format")
    return data


class _NNDCQuery:
    """National Nuclear Data Center database query base class.

    Args:
      perform: a boolean dictating whether to immediately perform the query.
      nuc     : (str) : the name of the isotope (e.g., 'Co-60')
      z, a, n : (int) : Z, A, N of the isotope
      z_range, etc. : (tuple of int) : range of Z, A, or N
      z_any, etc. : (bool) : whether any Z, A, or N is considered
      z_odd, etc. : (bool) : only odd Z, A, or N
      z_even, etc.: (bool) : only even Z, A, or N
      t_range : (tuple of float) : range of isotope half-lives in seconds

    Raises:
      NNDCInputError: if there is a problem with the input.
      NNDCRequestError: if there was a problem with the data requested.

    """

    _URL = ""
    _DATA = {
        "spnuc": "",  # specify parent ('name', 'zan', or 'zanrange')
        "nuc": "",  # isotope name (use with 'name')
        "z": "",  # Z or element (use with 'zan')
        "zmin": "",  # Z min        (use with 'zanrange')
        "zmax": "",  # Z max        (use with 'zanrange')
        "a": "",  # A            (use with 'zan')
        "amin": "",  # A min        (use with 'zanrange')
        "amax": "",  # A max        (use with 'zanrange')
        "n": "",  # N            (use with 'zan')
        "nmin": "",  # N min        (use with 'zanrange')
        "nmax": "",  # N max        (use with 'zanrange')
        "evenz": "",  # 'any', 'even', or 'odd' Z (use with zanrange)
        "evena": "",  # 'any', 'even', or 'odd' A (use with zanrange')
        "evenn": "",  # 'any', 'even', or 'odd' N (use with zanrange)
        "tled": "disabled",  # half-life condition on/off
        "tlmin": "0",  # half-life min
        "utlow": "S",  # half-life min units ('S' = seconds)
        "tlmax": "3E17",  # half-life max
        "utupp": "S",  # half-life max units ('ST' = stable, 'GY' = Gy)
        "notlim": "disabled",  # half-life: no limit
        "dmed": "disabled",  # decay mode condition on/off
        "dmn": "ANY",  # decay mode: 'ANY' = any
        "out": "file",  # output to formatted file
        "unc": "stdandard",  # standard style uncertainties
        "sub": "Search",  # search for the data
    }
    _ALLOWED_KEYWORDS = [
        "perform",
        "nuc",
        "z",
        "a",
        "n",
        "z_range",
        "a_range",
        "n_range",
        "z_any",
        "z_even",
        "z_odd",
        "a_any",
        "a_even",
        "a_odd",
        "n_any",
        "n_even",
        "n_odd",
        "t_range",
    ]
    _DUMMY_TEXT = ""

    def __init__(self, **kwargs):
        """Initialize query of NNDC data."""
        perform = kwargs.get("perform", True)
        kwargs["perform"] = False
        self._data = dict(self._DATA)
        self._text = self._DUMMY_TEXT
        self.df = pd.DataFrame()
        self.update(**kwargs)
        if perform:
            self.perform()

    def __len__(self):
        """Length of any one of the data lists."""
        if self.df is None or len(self.df.keys()) == 0:
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

    def _request(self):
        """Request data table from the URL."""
        text = _nndc_request_text(self._URL, tuple(sorted(self._data.items())))
        for msg in [
            "Your search was unsuccessful",
            "Your search exceeded the maximum number of results",
            "There are too many results for your search",
        ]:
            if msg in text:
                raise NNDCRequestError("Request failed: " + msg)
        msg = "No datasets were found within the specified search"
        if msg in text:
            raise NoDataFound(msg)
        return text

    def update(self, **kwargs):
        """Update the search criteria."""
        for kwarg in kwargs:
            if kwarg not in self._ALLOWED_KEYWORDS:
                raise NNDCInputError(f'Unknown keyword: "{kwarg}"')
        if "nuc" in kwargs:
            self._data["spnuc"] = "name"
            self._data["nuc"] = kwargs["nuc"]
        for x in ["z", "a", "n"]:
            # handle Z, A, and N settings
            if x in kwargs:
                self._data["spnuc"] = "zanrange"
                self._data[x + "min"], self._data[x + "max"] = _format_range(
                    (
                        kwargs[x],
                        kwargs[x],
                    )
                )
            # handle *_range, *_any, *_odd, *_even
            elif x + "_range" in kwargs:
                self._data["spnuc"] = "zanrange"
                self._data[x + "min"], self._data[x + "max"] = _format_range(
                    kwargs[x + "_range"]
                )
                if self._data[x + "min"] == "":
                    self._data[x + "min"] = "0"
                if self._data[x + "max"] == "":
                    self._data[x + "max"] = "300"
            if x + "_any" in kwargs:
                self._data["even" + x] = "any"
            elif x + "_even" in kwargs:
                self._data["even" + x] = "even"
            elif x + "_odd" in kwargs:
                self._data["even" + x] = "odd"
        # handle half-life range condition
        if "t_range" in kwargs:
            self._data["tled"] = "enabled"
            self._data["tlmin"], self._data["tlmax"] = _format_range(kwargs["t_range"])

    def perform(self):
        """Perform the query."""
        # check the conditions
        if self._data["spnuc"] == "":
            self.update(z_range=(None, None))
        # submit the query
        try:
            self._text = self._request()
        except NoDataFound:
            self._text = self._DUMMY_TEXT
        if len(self._text) == 0:
            raise NNDCRequestError("NNDC returned no text")
        # package the output into a dictionary of arrays
        data = _parse_table(self._text)
        # create the DataFrame
        self.df = pd.DataFrame(data)
        # convert dimensionless integers to ints
        for col in ["A", "Z", "N", "M"]:
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
        if "Energy Level (MeV)" not in self.df:
            return
        # add column of integer M giving the isomer level (0, 1, 2, ...)
        self.df["M"] = [0] * len(self)
        # add string m giving the isomer level name (e.g., '' or 'm' or 'm2')
        self.df["m"] = [""] * len(self)
        # loop over each isotope in the dataframe
        A_Z = list(zip(self["A"], self["Z"]))
        A_Z = set(A_Z)
        for a, z in A_Z:
            isotope = (self["A"] == a) & (self["Z"] == z)
            e_levels = []
            e_levels_nominal = []
            for e_level in self["Energy Level (MeV)"][isotope]:
                if isinstance(e_level, uncertainties.core.Variable):
                    e_level_nominal = e_level.nominal_value
                else:
                    e_level_nominal = e_level
                if e_level_nominal not in e_levels_nominal:
                    e_levels.append(e_level)
                    e_levels_nominal.append(e_level_nominal)
            e_levels = sorted(e_levels)
            for M, e_level in enumerate(e_levels):
                isomer = isotope & (abs(self["Energy Level (MeV)"] - e_level) < 1e-10)
                self.df.loc[isomer, "M"] = M
                if M > 0:
                    if len(e_levels) > 2:
                        self.df.loc[isomer, "m"] = f"m{M}"
                    else:
                        self.df.loc[isomer, "m"] = "m"

    def _add_units_uncertainties(self):
        """Add units and uncertainties with some columns as applicable."""
        if "Energy Level" in self.keys():
            self._convert_column(
                "Energy Level", lambda x: _parse_float_uncertainty(x, "")
            )
            self.df = self.df.rename(columns={"Energy Level": "Energy Level (MeV)"})

        if "Parent Energy Level" in self.keys():
            self._convert_column_uncertainty("Parent Energy Level")
            self.df = self.df.rename(
                columns={"Parent Energy Level": "Energy Level (MeV)"}
            )
            self.df["Energy Level (MeV)"] *= 0.001

        if "Mass Excess" in self.keys():
            self._convert_column_uncertainty("Mass Excess")
        self.df = self.df.rename(columns={"Mass Excess": "Mass Excess (MeV)"})

        self._convert_column("T1/2 (s)", float)

        if "Abundance (%)" in self.keys():
            self._convert_column_uncertainty("Abundance (%)")

        if "Branching (%)" in self.keys():
            self._convert_column(
                "Branching (%)", lambda x: _parse_float_uncertainty(x, "")
            )

        if "Radiation Energy" in self.keys():
            self._convert_column_uncertainty("Radiation Energy")
            self.df = self.df.rename(
                columns={"Radiation Energy": "Radiation Energy (keV)"}
            )

        if "Endpoint Energy" in self.keys():
            self._convert_column_uncertainty("Endpoint Energy")
            self.df = self.df.rename(
                columns={"Endpoint Energy": "Endpoint Energy (keV)"}
            )

        if "Radiation Intensity (%)" in self.keys():
            self._convert_column_uncertainty("Radiation Intensity (%)")

        if "Dose" in self.keys():
            self._convert_column_uncertainty("Dose")
            self.df = self.df.rename(columns={"Dose": "Dose (MeV / Bq / s)"})

    def _convert_column(self, col, function):
        """Convert column from string to another type."""
        col_new = []
        for x in self[col]:
            if x == "":
                col_new.append(None)
            else:
                col_new.append(function(x))
        self.df[col] = col_new

    def _convert_column_uncertainty(self, col):
        """Combine column and its uncertainty into one column."""
        col_new = []
        for x, dx in zip(self[col], self[col + " Unc."]):
            x2 = _parse_float_uncertainty(x, dx)
            col_new.append(x2)
        self.df[col] = col_new
        del self.df[col + " Unc."]

    def _sort_columns(self):
        """Sort columns."""
        preferred_order = [
            "Z",
            "Element",
            "A",
            "m",
            "M",
            "N",
            "JPi",
            "T1/2",
            "Energy Level (MeV)",
            "Decay Mode",
            "Branching (%)",
            "Radiation",
            "Radiation subtype",
            "Radiation Energy (keV)",
            "Radiation Intensity (%)",
        ]
        new_cols = [col for col in preferred_order if col in self.keys()]
        new_cols += [col for col in self.keys() if col not in new_cols]
        self.df = self.df[new_cols]


class _NuclearWalletCardQuery(_NNDCQuery):
    """NNDC Nuclear Wallet Card data query.

    Nuclear Wallet Card Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat3/indx_sigma.jsp

    Help page: http://www.nndc.bnl.gov/nudat3/help/wchelp.jsp

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

    Args:
      perform: a boolean dictating whether to immediately perform the query.
      nuc     : (str) : the name of the isotope (e.g., 'Co-60')
      z, a, n : (int) : Z, A, N of the isotope
      z_range, etc. : (tuple of int) : range of Z, A, or N
      z_any, etc. : (bool) : whether any Z, A, or N is considered
      z_odd, etc. : (bool) : only odd Z, A, or N
      z_even, etc.: (bool) : only even Z, A, or N
      t_range : (tuple of float) : range of isotope half-lives in seconds
      elevel_range : (tuple of float) : range of nuc. energy level (MeV)
      decay : (str) : isotope decay mode from WALLET_DECAY_MODE
      j :  (str) : nuclear spin
      parity : (str) : nuclear parity

    Raises:
      NNDCInputError: if there is a problem with the input.
      NNDCRequestError: if there was a problem with the data requested.

    """

    _URL = _WALLET_SEARCH_URL
    _DATA = dict(_NNDCQuery._DATA)
    _DATA.update(
        {
            "eled": "disabled",  # E(level) condition on/off
            "elmin": "0",  # E(level) min
            "elmax": "40",  # E(level) max
            "jled": "disabled",  # J_pi(level) condition on/off
            "jlv": "",  # J
            "plv": "ANY",  # parity
            "ord": "zalt",  # order file by Z, A, E(level), T1/2
        }
    )
    _ALLOWED_KEYWORDS = list(_NNDCQuery._ALLOWED_KEYWORDS)
    _ALLOWED_KEYWORDS.extend(["elevel_range", "decay", "j", "parity"])
    _DUMMY_TEXT = _NUCLEAR_WALLET_CARD_QUERY_DUMMY_TEXT

    def update(self, **kwargs):
        """Update the search criteria."""
        super().update(**kwargs)
        # handle decay mode
        if "decay" in kwargs:
            if kwargs["decay"].lower() not in WALLET_DECAY_MODE:
                raise NNDCInputError(
                    "Decay mode must be one of {}, not {}".format(
                        WALLET_DECAY_MODE.keys(), kwargs["decay"].lower()
                    )
                )
            self._data["dmed"] = "enabled"
            self._data["dmn"] = WALLET_DECAY_MODE[kwargs["decay"].lower()]
        # handle energy level condition
        if "elevel_range" in kwargs:
            self._data["eled"] = "enabled"
            self._data["elmin"], self._data["elmax"] = _format_range(
                kwargs["elevel_range"]
            )
            if self._data["elmax"] == "":
                self._data["elmax"] = "1000000000"
        # handle spin and parity
        if "j" in kwargs:
            self._data["jled"] = "enabled"
            self._data["jlv"] = kwargs["j"]
        if "parity" in kwargs:
            if kwargs["parity"].lower() not in PARITIES:
                raise NNDCInputError(
                    "Parity must be one of {}, not {}".format(
                        PARITIES, kwargs["parity"].lower()
                    )
                )
            self._data["jled"] = "enabled"
            self._data["plv"] = kwargs["parity"].upper()

    def _build_wallet_payload(self):
        """Translate legacy wallet query parameters into the new NNDC JSON payload."""
        payload = {
            "nuclide": "",
            "element": None,
            "zMin": None,
            "zMax": None,
            "nMin": None,
            "nMax": None,
            "aMin": None,
            "aMax": None,
        }
        if self._data["spnuc"] == "name":
            payload["nuclide"] = _wallet_format_nuclide(self._data["nuc"])
            return payload
        payload["zMin"] = _wallet_int_or_none(self._data["zmin"])
        payload["zMax"] = _wallet_int_or_none(self._data["zmax"])
        payload["nMin"] = _wallet_int_or_none(self._data["nmin"])
        payload["nMax"] = _wallet_int_or_none(self._data["nmax"])
        payload["aMin"] = _wallet_int_or_none(self._data["amin"])
        payload["aMax"] = _wallet_int_or_none(self._data["amax"])
        return payload

    def _filter_wallet_results(self, results):
        """Apply legacy wallet filters that now exist only client-side on NNDC."""
        if self._data["jled"] == "enabled":
            if self._data["jlv"] != "":
                raise NNDCRequestError(
                    "Request failed: wallet card spin filtering is not supported"
                )
            if self._data["plv"] not in ("", "ANY"):
                raise NNDCRequestError(
                    "Request failed: wallet card parity filtering is not supported"
                )
        filtered = list(results)
        for field_code, json_field in [
            ("z", "atomicNumber"),
            ("a", "atomicMass"),
            ("n", "neutronNumber"),
        ]:
            parity = self._data["even" + field_code]
            if parity == "even":
                filtered = [row for row in filtered if row[json_field] % 2 == 0]
            elif parity == "odd":
                filtered = [row for row in filtered if row[json_field] % 2 == 1]
        if self._data["tled"] == "enabled":
            t_min = _wallet_float_or_none(self._data["tlmin"])
            t_max = _wallet_float_or_none(self._data["tlmax"])
            if self._data["notlim"] == "enabled":
                t_min = None
                t_max = None
            if t_min is not None or t_max is not None:
                if t_min is None:
                    t_min = 0.0
                if t_max is None:
                    t_max = np.inf
                filtered = [
                    row
                    for row in filtered
                    if (seconds := _wallet_half_life_seconds(row)) is not None
                    and t_min <= seconds <= t_max
                ]
        if self._data["dmed"] == "enabled":
            filtered = [
                row
                for row in filtered
                if _wallet_matches_decay_mode(self._data["dmn"], row)
            ]
        if self._data["eled"] == "enabled":
            e_min = _wallet_float_or_none(self._data["elmin"])
            e_max = _wallet_float_or_none(self._data["elmax"])
            if e_min is None:
                e_min = 0.0
            if e_max is None:
                e_max = np.inf
            filtered = [
                row
                for row in filtered
                if (energy := _wallet_level_energy_mev(row)) is not None
                and e_min <= energy <= e_max
            ]
        return filtered

    def _wallet_to_dataframe(self, results):
        """Convert wallet JSON rows into the legacy DataFrame schema."""
        records = []
        for row in results:
            level_index = int(row.get("levelIndex", 0))
            half_life_text = (
                "STABLE"
                if row.get("stable") is True
                else " ".join(row.get("halfLifeNDS") or [])
            )
            base_record = {
                "Z": int(row["atomicNumber"]),
                "Element": row.get("elementCode"),
                "A": int(row["atomicMass"]),
                "m": (
                    ""
                    if level_index == 0
                    else ("m" if level_index == 1 else f"m{level_index}")
                ),
                "M": level_index,
                "N": int(row["neutronNumber"]),
                "JPi": (row.get("spinParityRecord") or {}).get("evaluatorInput"),
                "T1/2": half_life_text,
                "T1/2 (txt)": half_life_text,
                "Energy Level (MeV)": _wallet_level_energy_mev(row),
                "Mass Excess (MeV)": _wallet_value(row.get("massExcess"), scale=1000.0),
                "T1/2 (s)": _wallet_half_life_seconds(row),
                "Abundance (%)": _wallet_abundance(row),
            }
            for decay_mode, branching_ratio in _wallet_decay_modes(row):
                records.append(
                    {
                        **base_record,
                        "Decay Mode": decay_mode,
                        "Branching (%)": branching_ratio,
                    }
                )
        self.df = pd.DataFrame(records)
        self._sort_columns()

    def perform(self):
        """Perform the wallet card query using the current NNDC JSON endpoint."""
        if self._data["spnuc"] == "":
            self.update(z_range=(None, None))
        payload = self._build_wallet_payload()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        results = _wallet_search(self._URL, payload_json)
        results = self._filter_wallet_results(results)
        if len(results) == 0:
            raise NNDCRequestError("Request failed: Your search was unsuccessful")
        if len(results) > _WALLET_TOO_MANY_RESULTS:
            raise NNDCRequestError(
                "Request failed: There are too many results for your search"
            )
        self._text = json.dumps(results)
        self._wallet_to_dataframe(results)


def fetch_wallet_card(**kwargs):
    """Perform NNDC Nuclear Wallet Card data query and return a DataFrame.

    Nuclear Wallet Card Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat3/indx_sigma.jsp

    Help page: http://www.nndc.bnl.gov/nudat3/help/wchelp.jsp

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

    Args:
      nuc     : (str) : the name of the isotope (e.g., 'Co-60')
      z, a, n : (int) : Z, A, N of the isotope
      z_range, etc. : (tuple of int) : range of Z, A, or N
      z_any, etc. : (bool) : whether any Z, A, or N is considered
      z_odd, etc. : (bool) : only odd Z, A, or N
      z_even, etc.: (bool) : only even Z, A, or N
      t_range : (tuple of float) : range of isotope half-lives in seconds
      elevel_range : (tuple of float) : range of nuc. energy level (MeV)
      decay : (str) : isotope decay mode from WALLET_DECAY_MODE
      j :  (str) : nuclear spin
      parity : (str) : nuclear parity

    Returns:
      pandas DataFrame with the requested data.

    Raises:
      NNDCInputError: if there is a problem with the input.
      NNDCRequestError: if there was a problem with the data requested.

    """

    query = _NuclearWalletCardQuery(**kwargs)
    return query.df


class _DecayRadiationQuery(_NNDCQuery):
    """NNDC Decay Radiation data query.

    Decay Radiation Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat3/indx_dec.jsp

    Help page: http://www.nndc.bnl.gov/nudat3/help/dehelp.jsp

      * Radiation: Radiation type, i.e. G for gamma, E for electron.
      * Rad subtype: Further classification of the radiation type.
      * Rad Ene.: Radiation energy in keV.
      * EP Ene.: Beta-decay end point energy in keV.
      * Rad Int.: Radiation absolute intensity.
      * Dose: Radiation dose in MeV/Bq-s
      * Unc: Uncertainties

    Args:
      nuc     : (str) : the name of the isotope (e.g., 'Co-60')
      z, a, n : (int) : Z, A, N of the isotope
      z_range, etc. : (tuple of int) : range of Z, A, or N
      z_any, etc. : (bool) : whether any Z, A, or N is considered
      z_odd, etc. : (bool) : only odd Z, A, or N
      z_even, etc.: (bool) : only even Z, A, or N
      t_range : (tuple of float) : range of isotope half-lives in seconds
      decay : (str) : isotope decay mode from DECAYRAD_DECAY_MODE
      elevel_range : (tuple of float) : range of parent energy level (MeV)
      type :  (str) : radiation type from DECAYRAD_RADIATION_TYPE
      e_range : (tuple of float) : radiation energy range (keV)
      i_range : (tuple of float): intensity range (percent)

    Raises:
      NNDCInputError: if there is a problem with the input.
      NNDCRequestError: if there was a problem with the data requested.

    """

    _URL = "https://www.nndc.bnl.gov/nudat3/dec_searchi.jsp"
    _DATA = dict(_NNDCQuery._DATA)
    _DATA.update(
        {
            "rted": "enabled",  # radiation type condition on/off
            "rtn": "ANY",  # radiation type: 'ANY' = any, 'G' = gamma
            "reed": "disabled",  # radiation energy condition on/off
            "remin": "0",  # radiation energy min (keV)
            "remax": "10000",  # radiation energy max (keV)
            "ried": "disabled",  # radiation intensity condition on/off
            "rimin": "0",  # radiation intensity min (%)
            "rimax": "100",  # radiation intensity max (%)
            "ord": "zate",  # order file by Z, A, T1/2, E
        }
    )
    _ALLOWED_KEYWORDS = list(_NNDCQuery._ALLOWED_KEYWORDS)
    _ALLOWED_KEYWORDS.extend(["elevel_range", "decay", "type", "e_range", "i_range"])
    _DUMMY_TEXT = _DECAY_RADIATION_QUERY_DUMMY_TEXT

    def update(self, **kwargs):
        """Update the search criteria."""
        super().update(**kwargs)
        # handle decay mode
        if "decay" in kwargs:
            if kwargs["decay"].lower() not in DECAYRAD_DECAY_MODE:
                raise NNDCInputError(
                    "Decay mode must be one of {}, not {}".format(
                        DECAYRAD_DECAY_MODE.keys(), kwargs["decay"].lower()
                    )
                )
            self._data["dmed"] = "enabled"
            self._data["dmn"] = DECAYRAD_DECAY_MODE[kwargs["decay"].lower()]
        # handle radiation type
        if "type" in kwargs:
            if kwargs["type"].lower() not in DECAYRAD_RADIATION_TYPE:
                raise NNDCInputError(
                    "Radiation type must be one of {}, not {}".format(
                        DECAYRAD_RADIATION_TYPE.keys(), kwargs["type"].lower()
                    )
                )
            self._data["rted"] = "enabled"
            self._data["rtn"] = DECAYRAD_RADIATION_TYPE[kwargs["type"].lower()]
        # handle energy level condition
        self.elevel_range = (0, 1e9)
        if "elevel_range" in kwargs:
            x = _format_range(kwargs["elevel_range"])
            try:
                x0 = float(x[0])
            except ValueError:
                x0 = 0.0
            try:
                x1 = float(x[1])
            except ValueError:
                x1 = 1e9
            self.elevel_range = (x0, x1)
        # handle radiation energy range
        if "e_range" in kwargs:
            self._data["reed"] = "enabled"
            self._data["remin"], self._data["remax"] = _format_range(kwargs["e_range"])
        # handle radiation intensity range
        if "i_range" in kwargs:
            self._data["ried"] = "enabled"
            self._data["rimin"], self._data["rimax"] = _format_range(kwargs["i_range"])


def fetch_decay_radiation(**kwargs):
    """Perform NNDC Decay Radiation data query and return a DataFrame.

    Decay Radiation Search can be performed at this URL:
        http://www.nndc.bnl.gov/nudat3/indx_dec.jsp

    Help page: http://www.nndc.bnl.gov/nudat3/help/dehelp.jsp

      * Radiation: Radiation type, i.e. G for gamma, E for electron.
      * Rad subtype: Further classification of the radiation type.
      * Rad Ene.: Radiation energy in keV.
      * EP Ene.: Beta-decay end point energy in keV.
      * Rad Int.: Radiation absolute intensity.
      * Dose: Radiation dose in MeV/Bq-s
      * Unc: Uncertainties

    Args:
      nuc     : (str) : the name of the isotope (e.g., 'Co-60')
      z, a, n : (int) : Z, A, N of the isotope
      z_range, etc. : (tuple of int) : range of Z, A, or N
      z_any, etc. : (bool) : whether any Z, A, or N is considered
      z_odd, etc. : (bool) : only odd Z, A, or N
      z_even, etc.: (bool) : only even Z, A, or N
      t_range : (tuple of float) : range of isotope half-lives in seconds
      elevel_range : (tuple of float) : range of parent energy level (MeV)
      decay : (str) : isotope decay mode from DECAYRAD_DECAY_MODE
      type :  (str) : radiation type from DECAYRAD_RADIATION_TYPE
      e_range : (tuple of float) : radiation energy range (keV)
      i_range : (tuple of float): intensity range (percent)

    Returns:
      pandas DataFrame with the requested data.

    Raises:
      NNDCInputError: if there is a problem with the input.
      NNDCRequestError: if there was a problem with the data requested.

    """

    query = _DecayRadiationQuery(**kwargs)
    # apply elevel_range filter (hack around the web API)
    elevel = query.df["Energy Level (MeV)"]
    keep = (elevel >= query.elevel_range[0]) & (elevel <= query.elevel_range[1])
    query.df = query.df[keep]
    return query.df
