"""Read material densities and compositions from the McConn et al. Compendium.

The report is:

    McConn, Gesh, Pagh, Rucker, & Williams, "Compendium of Material
    Composition Data for Radiation Transport Modeling", PIET-43741-TM-963,
    PNNL-15870 Rev. 1

and it is available at:

    https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev1.pdf

"""

import os
import urllib.request
import numpy as np
import pandas as pd
from .element import Element
from .isotope import Isotope

try:
    import PyPDF2
except ImportError:
    raise Exception("The PyPDF2 package is required to parse material data")

FNAME = "PNNL-15870Rev1.pdf"
URL = "https://www.pnnl.gov/main/publications/external/technical_reports/" + FNAME
FNAME_LOCAL = os.path.join(os.path.join(os.path.split(__file__)[0], FNAME))
FNAME_LOCAL = os.path.splitext(FNAME_LOCAL)[0]


def calc_z_over_a(atom_fractions):
    """Calculate Z/A given the fraction of each element.

    Parameters
    ----------
    atom_fractions : array_like
        A list of pairs, where each pair is an element and the fraction
        of that element's atoms in the material.

    Returns
    -------
    z_over_a : float
        The mean atomic number over atomic mass.
    """
    z_sum = 0
    a_sum = 0
    assert isinstance(atom_fractions, list)
    for sym, frac in atom_fractions:
        frac = float(frac)
        # handle special cases in the Compendium where "element" is an isotope
        if "-" in sym:
            elem = Isotope(sym)
            elem.atomic_mass = float(sym.split("-")[1])
        else:
            elem = Element(sym)
        z_sum += frac * elem.Z
        a_sum += frac * elem.atomic_mass
    return z_sum / a_sum


def fetch_compendium_data():
    """Read material data from the McConn et al. Compendium."""
    # download the PDF file
    if not os.path.exists(FNAME_LOCAL + ".pdf"):
        with urllib.request.urlopen(URL) as fin:
            data = fin.read()
            with open(FNAME_LOCAL + ".pdf", "wb") as fout:
                fout.write(data)

    # read text from the file
    if not os.path.exists(FNAME_LOCAL + ".txt"):
        reader = PyPDF2.PdfFileReader(FNAME_LOCAL + ".pdf")
        print(reader.documentInfo)
        text = ""
        print("Extracting text from PDF")
        for p in range(32, 370):
            text += reader.getPage(p).extractText()
        with open(FNAME_LOCAL + ".txt", "w") as f:
            print(text, file=f)

    # analyze text
    with open(FNAME_LOCAL + ".txt", "r") as f:
        lines = f.readlines()[6:]
    # text = "".join(lines)
    # print('\n' + '-' * 80 + '\n')
    # print(text[:10000])
    # print('\n' + '-' * 80 + '\n')
    # print(text[-10000:])
    # print('\n' + '-' * 80 + '\n')

    # remove excess header and footer text
    lines_final = []
    j = 0
    while j < len(lines):
        if "PIET" in lines[j]:
            # skip this line and everything up to "357" (end of footer)
            # print("")
            # print("Skipping:")
            # print(lines[j].strip())
            # print(lines[j + 1].strip())
            # print(lines[j + 2].strip())
            # print(lines[j + 3].strip())
            # print(lines[j + 4].strip())
            # print(lines[j + 5].strip())
            j += 5
            line = lines[j].replace("357", "")
            # print(line.strip())
            lines_final.append(line)
        else:
            lines_final.append(lines[j])
        j += 1
    lines = lines_final

    print("\n" + "-" * 80 + "\n")
    text = "".join(lines_final)
    print(text[:10000])

    # handle a special case where " 3 " appears twice for Acetylene
    for j in range(len(lines)):
        if lines[j].startswith("s 3 - 4."):
            lines[j] = lines[j][8:]

    # go line-by-line to find the material data
    names = []
    formulae = []
    densities = []
    weight_fracs = []
    atom_fracs = []
    j = 0
    while j < len(lines):
        # for j0 in range(j, j + 300):
        #     if j0 > len(lines) - 1:
        #         continue
        #     print("preview:", j0, "Formula" in lines[j0], lines[j0][:-1])

        # find the start
        while j < len(lines) and (
            "Formula" not in lines[j]
            or "Formula = H" in lines[j]
            or "Formula = C" in lines[j]
            or "Formula from" in lines[j]
            or "Formula and" in lines[j]
            or "Formula for" in lines[j]
        ):
            j += 1
        if j == len(lines):
            break

        # find the name
        mat_num = f" {len(names) + 1} "
        j0 = j
        # print("test 1:", j0, mat_num, lines[j0])
        while mat_num not in lines[j0]:
            j0 -= 1
            # print("test 1:", j0, mat_num, lines[j0][:-1])
            assert abs(j0 - j) < 10
        name = lines[j0].split(mat_num)[1][:-1]
        for j1 in range(j0 + 1, j):
            name += lines[j1][:-1]
        name += lines[j].split("Formula")[0][:-1]
        name = name.strip()
        print(name)
        names.append(name)
        # print(names)

        # find the formula
        formula = lines[j].split("Formula")[1][:-1]
        # print("test 2:", j, formula)
        j += 1
        while "Molecular" not in lines[j]:
            formula += lines[j][:-1]
            # print("test 2:", j, formula, lines[j][:-1])
            j += 1
        formula += lines[j].split("Molecular")[0].strip()
        # print("test 2:", j, formula, lines[j][:-1])
        formula = formula.split("=")[1]
        formula = formula.strip()
        print(formula)
        formulae.append(formula)
        # print(formulae)

        # find the density
        while "Density" not in lines[j]:
            # print("test 3:", j, lines[j][:-1])
            j += 1
        density = lines[j][:-1]
        while "Total" not in lines[j]:
            # print("test 3:", j, lines[j][:-1])
            j += 1
            density += lines[j][:-1]
        if "Total" in lines[j]:
            # print("test 3:", j, lines[j][:-1])
            density += lines[j][:-1]
        density = density.split("Density (g/cm3) =")[1].split("Total")[0].strip()
        print(density)
        densities.append(density)
        # print(densities)

        # find the column of weight fractions
        while "Density" not in lines[j]:
            # print("test 4:", j, lines[j][:-1])
            j += 1
        # print("test 4:", j, lines[j][:-1])
        j0 = j + 1
        j = j0
        while "Total" not in lines[j]:
            # print("test 5:", j, lines[j][:-1])
            j += 1
        # print("test 5:", j, lines[j][:-1])
        j1 = j - 1
        # for j in range(j0, j1 + 1):
        #     print("table:", j, lines[j][:-1])

        j = j0
        weight_rows = []
        atom_rows = []
        row = ""
        while j <= j1:
            row += lines[j][:-1]
            # print("row:", row)
            tokens = row.strip().split(" ")
            if len(tokens) == 6:
                print("row:", row)
                weight_rows.append([tokens[0], tokens[3]])
                atom_rows.append([tokens[0], tokens[4]])
                # print("rows:", weight_rows, atom_rows)
                row = ""
            j += 1
        print("rows:", weight_rows, atom_rows)
        weight_fracs.append(weight_rows)
        atom_fracs.append(atom_rows)

        # move to next one
        j += 1

    # assemble data into a dataframe like the NIST data
    df = pd.DataFrame()
    df["Material"] = names
    df["Formula"] = formulae
    df["Density"] = np.array(densities, dtype=float)
    df["Z_over_A"] = [calc_z_over_a(afs) for afs in atom_fracs]
    df["Composition_symbol"] = [
        [" ".join(tokens) for tokens in wfs] for wfs in weight_fracs
    ]
    return df


if __name__ == "__main__":
    data = fetch_compendium_data()
    print(data)
