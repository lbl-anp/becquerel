"""Load material data for use in attenuation calculations with XCOM."""

import csv
import os
import warnings
import numpy as np
from .materials_error import MaterialsError, MaterialsWarning
from .materials_compendium import fetch_compendium_data
from .materials_nist import fetch_element_data, fetch_compound_data


FILENAME = os.path.join(os.path.split(__file__)[0], "materials.csv")


def _load_and_compile_materials():
    """Retrieve and merge all of the material sources into one dictionary.

    Returns
    -------
    materials
        Dictionary keyed by material names containing the material data.
    """
    # fetch the data sources
    data_elem = fetch_element_data()
    data_mat = fetch_compound_data()
    data_comp = fetch_compendium_data()

    # perform various checks on the Compendium data
    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        rho1 = data_comp["Density"].values[j]
        rho2 = None
        if name in data_elem["Element"].values:
            rho2 = data_elem["Density"][data_elem["Element"] == name].values[0]
        elif name in data_mat["Material"].values:
            rho2 = data_mat["Density"][data_mat["Material"] == name].values[0]
        if rho2:
            if not np.isclose(rho1, rho2, atol=2e-2):
                raise MaterialsError(
                    f"Material {name} densities do not match between different "
                    f"data sources:  {rho1:.6f}  {rho2:.6f}"
                )

    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        if name in data_mat["Material"].values:
            weight_fracs1 = data_comp["Composition_symbol"].values[j]
            weight_fracs2 = data_mat["Composition_symbol"][
                data_mat["Material"] == name
            ].values[0]
            if len(weight_fracs1) != len(weight_fracs2):
                raise MaterialsError(
                    f"Material {name} has different number of weight fractions "
                    f"in the different sources: {weight_fracs1}  {weight_fracs2}"
                )
            for k in range(len(weight_fracs1)):
                elem1, frac1 = weight_fracs1[k].split(" ")
                elem2, frac2 = weight_fracs2[k].split(" ")
                if elem1 != elem2:
                    raise MaterialsError(
                        f"Material {name} weight fraction elements do not match "
                        f"between different data sources:  {elem1}  {elem2}"
                    )
                frac1 = float(frac1)
                frac2 = float(frac2)
            if not np.isclose(frac1, frac2, atol=3e-4):
                raise MaterialsError(
                    f"Material {name} weight fractions do not match between "
                    f"different data sources:  {elem1}  {frac1:.6f}  {frac2:.6f}"
                )

    # make a dictionary of all the materials
    materials = {}
    for j in range(len(data_elem)):
        name = data_elem["Element"].values[j]
        formula = data_elem["Symbol"].values[j]
        density = data_elem["Density"].values[j]
        weight_fracs = data_elem["Composition_symbol"].values[j]
        materials[name] = {
            "formula": formula,
            "density": density,
            "weight_fractions": weight_fracs,
            "source": '"NIST (http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html)"',  # noqa: E501
        }
        #  add duplicate entry under element symbol for backwards compatibility
        materials[formula] = materials[name]

    for j in range(len(data_mat)):
        name = data_mat["Material"].values[j]
        formula = "-"
        density = data_mat["Density"].values[j]
        weight_fracs = data_mat["Composition_symbol"].values[j]
        materials[name] = {
            "formula": formula,
            "density": density,
            "weight_fractions": weight_fracs,
            "source": '"NIST (http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html)"',  # noqa: E501
        }

    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        formula = data_comp["Formula"].values[j]
        density = data_comp["Density"].values[j]
        weight_fracs = data_comp["Composition_symbol"].values[j]
        if name in materials:
            # replace material formula if compendium has one
            # otherwise do not overwrite the NIST data
            materials[name][formula] = formula
        else:
            materials[name] = {
                "formula": formula,
                "density": density,
                "weight_fractions": weight_fracs,
                "source": (
                    '"Detwiler, Rebecca S., McConn, Ronald J., Grimes, '
                    "Thomas F., Upton, Scott A., & Engel, Eric J. Compendium of "
                    "Material Composition Data for Radiation Transport Modeling. "
                    "United States. PNNL-15870 Revision 2., "
                    "https://doi.org/10.2172/1782721 "
                    '(https://compendium.cwmd.pnnl.gov)"'
                ),
            }

    return materials


def _write_materials_csv(materials):
    """Write material data to materials.csv.

    Parameters
    ----------
    materials : dict
        Dictionary of materials.
    """
    if os.path.exists(FILENAME):
        warnings.warn(
            f"Materials data CSV already exists at {FILENAME} and will be overwritten",
            MaterialsWarning,
        )
    mat_list = sorted(materials.keys())
    with open(FILENAME, "w") as f:
        print("%name,formula,density,weight fractions,source", file=f)
        for name in mat_list:
            line = ""
            data = materials[name]
            line = f"\"{name}\",\"{data['formula']}\",{data['density']:.6f},"
            line += ";".join(data["weight_fractions"])
            line += f",{data['source']}"
            print(line, file=f)


def _read_materials_csv():
    """Load material data from materials.csv.

    Returns
    -------
    materials
        Dictionary keyed by material names containing the material data.
    """
    if not os.path.exists(FILENAME):
        raise MaterialsError(f"Materials data CSV does not exist at {FILENAME}")
    materials = {}
    with open(FILENAME, "r") as f:
        lines = f.readlines()
        for tokens in csv.reader(
            lines,
            quotechar='"',
            delimiter=",",
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        ):
            if tokens[0].startswith("%"):
                continue
            name = tokens[0]
            formula = tokens[1]
            density = float(tokens[2])
            weight_fracs = tokens[3].split(";")
            source = tokens[4]
            materials[name] = {
                "formula": formula,
                "density": density,
                "weight_fractions": weight_fracs,
                "source": source,
            }
    return materials


def force_load_and_write_materials_csv():
    """Load all material data and write to CSV file.

    Returns
    -------
    materials
        Dictionary keyed by material names containing the material data.
    """
    materials = _load_and_compile_materials()
    _write_materials_csv(materials)
    return materials


def fetch_materials(force=False):
    """Fetch all available materials.

    On first ever function call, will check NIST website for data using
    the tools in materials_nist.py and will attempt to load the PNNL Compendium
    data using the tools in materials_compendium.py. The Compendium materials
    will only be available if the JSON data MaterialsCompendium.json are
    downloaded and placed in the same location as materials_compendium.py.

    Parameters
    ----------
    force : bool
        Whether to force the reloading and rewriting of the materials CSV.

    Returns
    -------
    materials
        Dictionary keyed by material names containing the material data.
    """
    if force or not os.path.exists(FILENAME):
        materials = force_load_and_write_materials_csv()
    materials = _read_materials_csv()
    return materials


def remove_materials_csv():
    """Remove materials.csv if it exists."""
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
