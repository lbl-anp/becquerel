"""Query material data from NIST X-Ray Mass Attenuation Coefficients website.

References:
  https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
  http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html
  http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

"""

import requests
import pandas as pd
from collections.abc import Iterable
from .element import element_symbol
from .materials_error import MaterialsError

MAX_Z = 92
N_COMPOUNDS = 48


_URL_TABLE1 = "http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html"
_URL_TABLE2 = "http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html"


def _get_request(url):
    """Perform GET request.

    Args:
      url: URL (string) to get.

    Returns:
      requests object.

    Raises:
      MaterialsError: if there was a problem making the request.

    """

    req = requests.get(url)
    if not req.ok or req.reason != "OK" or req.status_code != 200:
        raise MaterialsError(
            "NIST materials request failed: reason={}, status_code={}".format(
                req.reason, req.status_code
            )
        )
    return req


def fetch_element_data():
    """Retrieve data for the elements.

    Data are found in Table 1:
      http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html

    Returns:
      A pandas DataFrame of the material data. Columns are 'Z', 'Symbol',
      'Element', 'Z_over_A', 'I_eV', and 'Density'.

    Raises:
      MaterialsError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE1)
    text = req.text
    # rename first two header columns
    text = text.replace(
        '<TH scope="col" COLSPAN="2"><I>Z</I></TH>',
        '<TH scope="col">Z</TH><TH scope="col">Symbol</TH>',
    )
    # remove row that makes the table too wide
    text = text.replace('<TD COLSPAN="10"><HR SIZE="1" NOSHADE></TD>', "")
    # remove extra header columns
    text = text.replace('<TD COLSPAN="2">', "<TD>")
    text = text.replace('<TD COLSPAN="4">', "<TD>")
    text = text.replace("TD>&nbsp;</TD>", "")
    # remove extra columns in Hydrogen row
    text = text.replace('<TD ROWSPAN="92">&nbsp;</TD>', "")
    # remove open <TR> at the end of the table
    text = text.replace("</TD></TR><TR>", "</TD></TR>")
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise MaterialsError(f"1 HTML table expected, but found {len(tables)}")
    df = tables[0]
    if len(df) != MAX_Z:
        raise MaterialsError(f"{MAX_Z} elements expected, but found {len(df)}")
    if len(df.columns) != 6:
        raise MaterialsError(
            f"10 columns expected, but found {len(df.columns)} ({df.columns})"
        )
    # rename columns
    df.columns = ["Z", "Symbol", "Element", "Z_over_A", "I_eV", "Density"]

    # add composition by Z
    df["Composition_Z"] = [[f"{z}: 1.000000"] for z in df["Z"].values]
    # add composition by symbol
    df["Composition_symbol"] = [
        convert_composition(comp) for comp in df["Composition_Z"]
    ]

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
      MaterialsError: if there was a problem making the conversion.

    """
    comp_sym = []
    if not isinstance(comp, Iterable):
        raise MaterialsError(f"Compound must be an iterable of strings: {comp}")
    for line in comp:
        if not isinstance(line, str):
            raise MaterialsError(f"Line must be a string type: {line} {type(line)}")
        try:
            z, weight = line.split(":")
        except ValueError:
            raise MaterialsError(f"Unable to split compound line: {line}")
        try:
            z = int(z)
        except ValueError:
            raise MaterialsError(f"Unable to convert Z {z} to integer: {line}")
        if z < 1 or z > MAX_Z:
            raise MaterialsError(f"Z {z} out of range [1, {line}]: {MAX_Z}")
        comp_sym.append(element_symbol(z) + " " + weight.strip())
    return comp_sym


def fetch_compound_data():
    """Retrieve data for the compounds.

    Data are found in Table 2:
      http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html

    Returns:
      A pandas DataFrame of the material data. Columns are 'Material',
      'Z_over_A', 'I_eV', 'Density', 'Composition_Z', and 'Composition_symbol'.

    Raises:
      MaterialsError: if there was a problem obtaining the data.

    """
    req = _get_request(_URL_TABLE2)
    text = req.text
    # remove extra header columns
    text = text.replace('<TD ROWSPAN="2">&nbsp;</TD>', "")
    # remove extra rows in header row
    text = text.replace(' ROWSPAN="2"', "")
    # remove extra columns in third header row
    text = text.replace('<TD COLSPAN="9"><HR SIZE="1" NOSHADE></TD>', "")
    # remove extra columns in the first material row
    text = text.replace('<TD ROWSPAN="50"> &nbsp; </TD>', "")
    # replace <BR> symbols in composition lists with semicolons
    text = text.replace("<BR>", ";")
    # read HTML table into pandas DataFrame
    tables = pd.read_html(text, header=0, skiprows=[1, 2])
    if len(tables) != 1:
        raise MaterialsError(f"1 HTML table expected, but found {len(tables)}")
    df = tables[0]
    if len(df) != N_COMPOUNDS:
        raise MaterialsError(f"{N_COMPOUNDS} compounds expected, but found {len(df)}")
    if len(df.columns) != 5:
        raise MaterialsError(
            f"5 columns expected, but found {len(df.columns)} ({df.columns})"
        )
    # rename columns
    df.columns = ["Material", "Z_over_A", "I_eV", "Density", "Composition_Z"]
    # clean up Z composition
    df["Composition_Z"] = [
        [line.strip() for line in comp.split(";")] for comp in df["Composition_Z"]
    ]
    # create a column of compositions by symbol (for use with fetch_xcom_data)
    df["Composition_symbol"] = [
        convert_composition(comp) for comp in df["Composition_Z"]
    ]
    return df
