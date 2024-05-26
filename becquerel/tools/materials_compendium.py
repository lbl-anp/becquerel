"""Read material densities and compositions from the PNNL Compendium.

The report is:

    Detwiler, Rebecca S., McConn, Ronald J., Grimes, Thomas F., Upton, Scott
    A., & Engel, Eric J. Compendium of Material Composition Data for Radiation
    Transport Modeling. United States. PNNL-15870 Revision 2.
    https://doi.org/10.2172/1782721

and it is available at:

    https://compendium.cwmd.pnnl.gov

"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .materials_error import MaterialsError, MaterialsWarning

FNAME = Path(__file__).parent / "MaterialsCompendium.json"


def json_elements_to_weight_fractions(elements):
    """Calculate element weight fractions from the Elements data."""
    results = []
    for element in elements:
        assert element["Element"].isalpha()
        line = f"{element['Element']} {element['WeightFraction_whole']:.6f}"
        results.append(line)
    return results


def json_elements_to_atom_fractions(elements):
    """Calculate element atomic number fractions from the Elements data."""
    results = []
    for element in elements:
        line = f"{element['Element']} {element['AtomFraction_whole']:.6f}"
        results.append(line)
    return results


def fetch_compendium_data():
    """Read material data from the Compendium."""
    # read the file
    if not FNAME.exists():
        warnings.warn(
            'Material data from the "Compendium of Material Composition Data for '
            'Radiation Transport Modeling" cannot be found. If these data are '
            "desired, please visit the following URL: "
            'https://compendium.cwmd.pnnl.gov and select "Download JSON". Then '
            f"move the resulting file to the following path: {FNAME}",
            MaterialsWarning,
        )
        data = []
    else:
        with FNAME.open() as f:
            data = json.load(f)

    # extract relevant data
    if isinstance(data, list):
        print("Pre-March 2022 JSON detected")
    elif isinstance(data, dict):
        print("Post-March 2022 JSON detected")
        if "siteVersion" not in data.keys() or "data" not in data.keys():
            raise MaterialsError(
                "Attempt to read Compendium JSON failed; "
                "dictionary must have keys 'siteVersion' "
                "and 'data' but have keys " + str(list(data.keys()))
            )
        print(f"Compendium data - site version: {data['siteVersion']}")
        data = data["data"]
    else:
        raise MaterialsError(
            "Attempt to read Compendium JSON failed; "
            "object must be a list or dict but is a " + str(type(data))
        )
    names = [datum["Name"] for datum in data]
    formulae = [datum["Formula"] if "Formula" in datum else "-" for datum in data]
    densities = [datum["Density"] for datum in data]
    weight_fracs = [
        json_elements_to_weight_fractions(datum["Elements"]) for datum in data
    ]

    # assemble data into a dataframe like the NIST data
    df = pd.DataFrame()
    df["Material"] = names
    df["Formula"] = formulae
    df["Density"] = np.array(densities, dtype=float)
    df["Composition_symbol"] = weight_fracs
    return df
