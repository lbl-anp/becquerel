"""Test NIST material data queries."""

import json
import warnings
from pathlib import Path

import pytest
from utils import xcom_is_up

from becquerel.tools import (
    MaterialsError,
    MaterialsWarning,
    fetch_materials,
    materials,
    materials_compendium,
    remove_materials_csv,
)
from becquerel.tools.materials_nist import convert_composition


def _get_warning_messages(record):
    return [str(rec.message) for rec in record]


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
class TestConvertComposition:
    """Test convert_composition."""

    def test_success(self):
        """Test convert_composition works for a simple example."""
        results = convert_composition(["1: 0.111898", "8: 0.888102"])
        assert results[0] == "H 0.111898"
        assert results[1] == "O 0.888102"

    def test_not_iterable(self):
        """Test convert_composition exception for non-iterable argument."""
        with pytest.raises(MaterialsError):
            convert_composition(100)

    def test_not_string(self):
        """Test convert_composition exception for non-string argument."""
        with pytest.raises(MaterialsError):
            convert_composition([100, 100])

    def test_bad_line(self):
        """Test convert_composition exception for badly-formed line."""
        with pytest.raises(MaterialsError):
            convert_composition(["1: :0.111898", "8: 0.888102"])

    def test_bad_z(self):
        """Test convert_composition exception for bad Z value."""
        with pytest.raises(MaterialsError):
            convert_composition(["X: 0.111898", "8: 0.888102"])

    def test_z_out_of_range(self):
        """Test convert_composition exception for Z value out of range."""
        with pytest.raises(MaterialsError):
            convert_composition(["118: 0.111898", "8: 0.888102"])


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials():
    """Test fetch_materials."""
    fetch_materials()
    assert materials.FILENAME.exists()


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials_force():
    """Test fetch_materials with force=True."""
    assert materials.FILENAME.exists()
    with pytest.warns(MaterialsWarning) as record:
        fetch_materials(force=True)
    if not materials_compendium.FNAME.exists():
        assert len(record) == 2, (
            "Expected two MaterialsWarnings to be raised; "
            f"got {_get_warning_messages(record)}"
        )
    else:
        assert len(record) == 1, (
            "Expected one MaterialsWarning to be raised; "
            f"got {_get_warning_messages(record)}"
        )
    assert materials.FILENAME.exists()


def test_materials_dummy_csv():
    """Test fetch_materials with a dummy materials.csv file."""
    # point to and generate a dummy CSV file
    fname_orig = materials.FILENAME
    materials.FILENAME = Path(str(fname_orig)[:-4] + "_dummy.csv")
    if materials.FILENAME.exists():
        materials.FILENAME.unlink()
    with materials.FILENAME.open("w") as f:
        print("%name,formula,density,weight fractions,source", file=f)
        print('Dummy,-,1.0,"H 0.5;O 0.5","dummy entry"', file=f)
    fetch_materials()
    # remove the dummy file and point back to original
    materials.FILENAME.unlink()
    materials.FILENAME = fname_orig


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials_dummy_compendium_pre2022():
    """Test fetch_materials with a dummy Compendium JSON file.

    The dummy JSON file uses the format seen prior to March 2022.
    """
    # point to an generate a dummy JSON file
    fname_orig = materials_compendium.FNAME
    materials_compendium.FNAME = Path(str(fname_orig)[:-5] + "_dummy.json")
    data = [
        {
            "Density": 8.4e-5,
            "Elements": [
                {
                    "AtomFraction_whole": 1.0,
                    "Element": "H",
                    "WeightFraction_whole": 1.0,
                }
            ],
            "Formula": "H2",
            "Name": "Hydrogen",
        },
        {
            "Density": 1.16e-3,
            "Elements": [
                {
                    "AtomFraction_whole": 1.0,
                    "Element": "N",
                    "WeightFraction_whole": 1.0,
                }
            ],
            "Formula": "N2",
            "Name": "Nitrogen",
        },
    ]
    with materials_compendium.FNAME.open("w") as f:
        json.dump(data, f, indent=4)
    # Check that no warning is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials._load_and_compile_materials()
    # remove the dummy file and point back to original
    materials_compendium.FNAME.unlink()
    materials_compendium.FNAME = fname_orig


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials_dummy_compendium_2022():
    """Test fetch_materials with a dummy Compendium JSON file.

    The dummy JSON file uses the format first seen in March 2022.
    """
    # point to an generate a dummy JSON file
    fname_orig = materials_compendium.FNAME
    materials_compendium.FNAME = Path(str(fname_orig)[:-5] + "_dummy.json")
    data = {
        "siteVersion": "0.0.0",
        "data": [
            {
                "Density": 8.4e-5,
                "Elements": [
                    {
                        "AtomFraction_whole": 1.0,
                        "Element": "H",
                        "WeightFraction_whole": 1.0,
                    }
                ],
                "Formula": "H2",
                "Name": "Hydrogen",
            },
            {
                "Density": 1.16e-3,
                "Elements": [
                    {
                        "AtomFraction_whole": 1.0,
                        "Element": "N",
                        "WeightFraction_whole": 1.0,
                    }
                ],
                "Formula": "N2",
                "Name": "Nitrogen",
            },
        ],
    }
    with materials_compendium.FNAME.open("w") as f:
        json.dump(data, f, indent=4)
    # Check that no warning is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials._load_and_compile_materials()
    # remove siteVersion and make sure there is an error raised
    del data["siteVersion"]
    with materials_compendium.FNAME.open("w") as f:
        json.dump(data, f, indent=4)
    with pytest.raises(MaterialsError):
        materials._load_and_compile_materials()
    # remove the dummy file and point back to original
    materials_compendium.FNAME.unlink()
    materials_compendium.FNAME = fname_orig


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials_dummy_compendium_error():
    """Test fetch_materials with a dummy Compendium JSON file.

    The dummy JSON file returns something that is not a list or dict.
    """
    # point to an generate a dummy JSON file
    fname_orig = materials_compendium.FNAME
    materials_compendium.FNAME = Path(str(fname_orig)[:-5] + "_dummy.json")
    data = None
    with materials_compendium.FNAME.open("w") as f:
        json.dump(data, f, indent=4)
    with pytest.raises(MaterialsError):
        materials._load_and_compile_materials()
    # remove the dummy file and point back to original
    materials_compendium.FNAME.unlink()
    materials_compendium.FNAME = fname_orig


@pytest.mark.webtest
@pytest.mark.skipif(not xcom_is_up(), reason="XCOM is down.")
def test_materials_no_compendium():
    """Test fetch_materials with no Compendium JSON file."""
    # point to a dummy JSON file that does not exist
    fname_orig = materials_compendium.FNAME
    materials_compendium.FNAME = Path(str(fname_orig)[:-5] + "_dummy.json")
    if materials_compendium.FNAME.exists():
        materials_compendium.FNAME.unlink()
    with pytest.warns(MaterialsWarning) as record:
        materials_compendium.fetch_compendium_data()
    assert len(record) == 1, (
        "Expected one MaterialsWarning to be raised; "
        f"got {_get_warning_messages(record)}"
    )
    # point back to original file
    materials_compendium.FNAME = fname_orig


def test_remove_materials_csv():
    """Test remove_materials_csv."""
    # point to and generate a dummy CSV file
    fname_orig = materials.FILENAME
    materials.FILENAME = Path(str(fname_orig)[:-4] + "_dummy.csv")
    if materials.FILENAME.exists():
        materials.FILENAME.unlink()
    with materials.FILENAME.open("w") as f:
        print(file=f)
    remove_materials_csv()
    assert not materials.FILENAME.exists()
    # make sure remove works if the file does not exist
    remove_materials_csv()
    assert not materials.FILENAME.exists()
    # point back to original file
    materials.FILENAME = fname_orig
