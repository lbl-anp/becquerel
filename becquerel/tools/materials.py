"""Load material data for use in attenuation calculations with XCOM."""

import csv
import os
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
    data_elem = fetch_element_data()
    print("")
    for col in [
        "Element",
        "Symbol",
        "Density",
        "Composition_Z",
        "Composition_symbol",
    ]:
        print(col, data_elem[col].values[:5])

    data_mat = fetch_compound_data()
    print("")
    for col in [
        "Material",
        "Density",
        "Composition_Z",
        "Composition_symbol",
    ]:
        print(col, data_mat[col].values[:5])

    data_comp = fetch_compendium_data()
    print("")
    for col in [
        "Material",
        "Formula",
        "Density",
        "Composition_symbol",
    ]:
        print(col, data_comp[col].values[:5])

    # perform various checks on the Compendium data
    print("")
    print("Check Density")
    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        rho1 = data_comp["Density"].values[j]
        rho2 = None
        if name in data_elem["Element"].values:
            rho2 = data_elem["Density"][data_elem["Element"] == name].values[0]
        elif name in data_mat["Material"].values:
            rho2 = data_mat["Density"][data_mat["Material"] == name].values[0]
        if rho2:
            print("")
            print("")
            print("-" * 90)
            print("")
            print(f"{name:<60s}  {rho1:.6f}")
            print(f"{name:<60s}  {rho1:.6f}    {rho2:.6f}")
            assert np.isclose(rho1, rho2, atol=2e-2)

    print("")
    print("Check weight compositions")
    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        if name in data_mat["Material"].values:
            weight_fracs1 = data_comp["Composition_symbol"].values[j]
            weight_fracs2 = data_mat["Composition_symbol"][
                data_mat["Material"] == name
            ].values[0]
            print("")
            print("")
            print("-" * 90)
            print("")
            print(name)
            print(data_comp["Formula"].values[j])
            print(weight_fracs1)
            print(weight_fracs2)
            assert len(weight_fracs1) == len(weight_fracs2)
            for k in range(len(weight_fracs1)):
                elem1, frac1 = weight_fracs1[k].split(" ")
                elem2, frac2 = weight_fracs2[k].split(" ")
                assert elem1 == elem2
                assert np.isclose(float(frac1), float(frac2), atol=3e-4)

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
            "source": '"NIST (http://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html)"',
        }

    for j in range(len(data_mat)):
        name = data_mat["Material"].values[j]
        formula = "-"
        density = data_mat["Density"].values[j]
        weight_fracs = data_mat["Composition_symbol"].values[j]
        materials[name] = {
            "formula": formula,
            "density": density,
            "weight_fractions": weight_fracs,
            "source": '"NIST (http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html)"',
        }

    for j in range(len(data_comp)):
        name = data_comp["Material"].values[j]
        formula = data_comp["Formula"].values[j]
        density = data_comp["Density"].values[j]
        weight_fracs = data_comp["Composition_symbol"].values[j]
        if name in materials:
            # replace material formula if McConn has one
            # otherwise do not overwrite the NIST data
            materials[name][formula] = formula
        else:
            materials[name] = {
                "formula": formula,
                "density": density,
                "weight_fractions": weight_fracs,
                "source": '"Detwiler, Rebecca S., McConn, Ronald J., Grimes, Thomas F., Upton, Scott A., & Engel, Eric J. Compendium of Material Composition Data for Radiation Transport Modeling. United States. PNNL-15870 Revision 2., https://doi.org/10.2172/1782721 (https://compendium.cwmd.pnnl.gov)"',
            }

    print(materials)
    return materials


def _write_materials_csv(materials):
    """Write material data to materials.csv.

    Parameters
    ----------
    materials : dict
        Dictionary of materials.
    """
    # write all materials to CSV file
    mat_list = sorted(materials.keys())
    print(mat_list)
    print(len(mat_list))

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
            print(tokens)
            if tokens[0].startswith("%"):
                continue
            name = tokens[0]
            formula = tokens[1]
            density = float(tokens[2])
            weight_fracs = tokens[3].split(";")
            source = tokens[4]
            print(name)

            materials[name] = {
                "formula": formula,
                "density": density,
                "weight_fractions": weight_fracs,
                "source": source,
            }
    return materials


def fetch_materials():
    """Fetch all available materials.

    On first ever function call, will check NIST website for data using
    the tools in materials_nist.py and will download and attempt to parse
    the McConn, et al. Compendium using the tools in materials_compendium.py.
    The Compendium materials will only be available if the optional dependency
    PyPDF2 package is installed.

    Returns
    -------
    materials
        Dictionary keyed by material names containing the material data.
    """
    if not os.path.exists(FILENAME):
        materials = _load_and_compile_materials()
        _write_materials_csv(materials)

    materials = _read_materials_csv()
    return materials
