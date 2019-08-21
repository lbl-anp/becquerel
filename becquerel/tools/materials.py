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
from .element import element_symbol
from ..core.utils import isstring

MAX_Z = 92
N_COMPOUNDS = 48


_URL_TABLE1 = 'http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html'
_URL_TABLE2 = 'http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html'


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
      A pandas DataFrame of the material data. Colums are 'Z', 'Symbol',
      'Element', 'Z_over_A', 'I_eV', and 'Density'.

    Raises:
      NISTMaterialsRequestError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE1)
    text = req.text
    # rename first two header columns
    text = text.replace(
        '<TH scope="col" COLSPAN="2"><I>Z</I></TH>',
        '<TH scope="col">Z</TH><TH scope="col">Symbol</TH>')
    # remove row that makes the table too wide
    text = text.replace('<TD COLSPAN="10"><HR SIZE="1" NOSHADE></TD>', '')
    # remove extra header columns
    text = text.replace('<TD COLSPAN="2">', '<TD>')
    text = text.replace('<TD COLSPAN="4">', '<TD>')
    text = text.replace('TD>&nbsp;</TD>', '')
    # remove extra columns in Hydrogen row
    text = text.replace('<TD ROWSPAN="92">&nbsp;</TD>', '')
    # remove open <TR> at the end of the table
    text = text.replace('</TD></TR><TR>', '</TD></TR>')
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise NISTMaterialsRequestError(
            '1 HTML table expected, but found {}'.format(len(tables)))
    df = tables[0]
    if len(df) != MAX_Z:
        raise NISTMaterialsRequestError(
            '{} elements expected, but found {}'.format(MAX_Z, len(df)))
    if len(df.columns) != 6:
        raise NISTMaterialsRequestError(
            '10 columns expected, but found {} ({})'.format(
                len(df.columns), df.columns))
    # rename columns
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
        if not isstring(line):
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
        if z < 1 or z > MAX_Z:
            raise NISTMaterialsRequestError(
                'Z {} out of range [1, {}]: {}'.format(z, line, MAX_Z))
        comp_sym.append(element_symbol(z) + ' ' + weight.strip())
    return comp_sym


def fetch_compound_data():
    """Retrieve data for the compounds.

    Data are found in Table 2:
      http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

    Returns:
      A pandas DataFrame of the material data. Columns are 'Material',
      'Z_over_A', 'I_eV', 'Density', 'Composition_Z', and 'Composition_symbol'.

    Raises:
      NISTMaterialsRequestError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE2)
    text = req.text
    # remove extra header columns
    text = text.replace('<TD ROWSPAN="2">&nbsp;</TD>', '')
    # remove extra rows in header row
    text = text.replace(' ROWSPAN="2"', '')
    # remove extra columns in third header row
    text = text.replace('<TD COLSPAN="9"><HR SIZE="1" NOSHADE></TD>', '')
    # remove extra columns in the first material row
    text = text.replace('<TD ROWSPAN="50"> &nbsp; </TD>', '')
    # replace <BR> symbols in composition lists with semicolons
    text = text.replace('<BR>', ';')
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise NISTMaterialsRequestError(
            '1 HTML table expected, but found {}'.format(len(tables)))
    df = tables[0]
    if len(df) != N_COMPOUNDS:
        raise NISTMaterialsRequestError(
            '{} compounds expected, but found {}'.format(N_COMPOUNDS, len(df)))
    if len(df.columns) != 5:
        raise NISTMaterialsRequestError(
            '5 columns expected, but found {} ({})'.format(
                len(df.columns), df.columns))
    # rename columns
    df.columns = ['Material', 'Z_over_A', 'I_eV', 'Density', 'Composition_Z']
    # clean up Z composition
    df['Composition_Z'] = [
        [line.strip() for line in comp.split(';')]
        for comp in df['Composition_Z']]
    # create a column of compositions by symbol (for use with fetch_xcom_data)
    df['Composition_symbol'] = [
        convert_composition(comp) for comp in df['Composition_Z']]
    return df
