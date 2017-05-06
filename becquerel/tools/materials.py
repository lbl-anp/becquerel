"""Query material data NIST X-Ray Mass Attenuation Coefficients website.

References:
  https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
  http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html
  http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

"""

from __future__ import print_function
from collections import Iterable
import requests
import pandas as pd
from six import string_types


_URL_TABLE1 = 'http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html'
_URL_TABLE2 = 'http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html'


ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

ELEMENT_Z_TO_SYMBOL = {
    z: symbol for z, symbol in enumerate(ELEMENT_SYMBOLS)}


class NISTMaterialsError(Exception):
    """General error for NIST materials data."""

    pass


class NISTMaterialsRequestError(NISTMaterialsError):
    """Error related to communicating with NIST or parsing the result."""

    pass


def _get_request(url):
    """Perform GET request.

    Args:
      url: URL (string) to get.

    Returns:
      requests object.

    Raises:
      NISTMaterialsRequestError: if there was a problem making the request.

    """

    req = requests.get(url)
    if not req.ok or req.reason != 'OK' or req.status_code != 200:
        raise NISTMaterialsRequestError(
            'NIST materials request failed: reason={}, status_code={}'.format(
                req.reason, req.status_code))
    return req


def fetch_element_data():
    """Retrieve data for the elements.

    Data are found in Table 1:
      http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html

    Returns:
      a pandas DataFrame of the material data.

    Raises:
      NISTMaterialsRequestError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE1)
    # remove extra columns in Hydrogen row and extra empty rows
    text = req.text
    text = text.replace('<TD ROWSPAN="92">&nbsp;</TD>', '')
    text = text.replace('TD>&nbsp;</TD>', '')
    text = text.replace('</TD></TR><TR>', '</TD></TR>')
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise NISTMaterialsRequestError(
            '1 HTML table expected, but found {}'.format(len(tables)))
    df = tables[0]
    if len(df) != 92:
        raise NISTMaterialsRequestError(
            '92 elements expected, but found {}'.format(len(df)))
    # set column names
    df.columns = ['Z', 'Symbol', 'Element', 'Z_over_A', 'I_eV', 'Density']
    return df


def convert_composition(comp):
    """Convert composition by Z into composition by symbol.

    Args:
      comp: a list of strings from the last column of Table 2, e.g.,
        ["1: 0.111898", "8: 0.888102"]

    Returns:
      A list of strings of composition by symbol, e.g.,
        ["H 0.111898", "O 0.888102"]

    Raises:
      NISTMaterialsRequestError: if there was a problem making the conversion.

    """
    comp_sym = []
    if not isinstance(comp, Iterable):
        raise NISTMaterialsRequestError(
            'Compound must be an iterable of strings: {}'.format(comp))
    for line in comp:
        if not isinstance(line, string_types):
            raise NISTMaterialsRequestError(
                'Line must be a string type: {} {}'.format(line, type(line)))
        try:
            z, weight = line.split(':')
        except ValueError:
            raise NISTMaterialsRequestError(
                'Unable to split compound line: {}'.format(line))
        try:
            z = int(z)
        except ValueError:
            raise NISTMaterialsRequestError(
                'Unable to convert Z {} to integer: {}'.format(z, line))
        if z not in ELEMENT_Z_TO_SYMBOL:
            raise NISTMaterialsRequestError(
                'Unable to convert Z {} to symbol: {}'.format(z, line))
        comp_sym.append(ELEMENT_Z_TO_SYMBOL[z] + ' ' + weight.strip())
    return comp_sym


def fetch_compound_data():
    """Retrieve data for the compounds.

    Data are found in Table 2:
      http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

    Returns:
      a pandas DataFrame of the material data.

    Raises:
      NISTMaterialsRequestError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE2)
    # remove extra columns and replace <BR> symbols in composition lists
    text = req.text
    text = text.replace('<TD ROWSPAN="2">&nbsp;</TD>', '')
    text = text.replace('<TD ROWSPAN="50"> &nbsp; </TD>', '')
    text = text.replace('<BR>', ';')
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise NISTMaterialsRequestError(
            '1 HTML table expected, but found {}'.format(len(tables)))
    df = tables[0]
    if len(df) != 48:
        raise NISTMaterialsRequestError(
            '48 compounds expected, but found {}'.format(len(df)))
    # set column names
    df.columns = ['Material', 'Z_over_A', 'I_eV', 'Density', 'Composition_Z']
    # clean up Z composition
    df['Composition_Z'] = [
        [line.strip() for line in comp.split(';')]
        for comp in df['Composition_Z']]
    # create a column of compositions by symbol (for use with fetch_xcom_data)
    df['Composition_symbol'] = [
        convert_composition(comp) for comp in df['Composition_Z']]
    return df
