"""Test NIST material data queries."""

from becquerel.tools import materials
import pytest


@pytest.mark.webtest
class TestConvertComposition:
    """Test convert_composition."""

    def test_success(self):
        """Test convert_composition works for a simple example............."""
        materials.convert_composition(["1: 0.111898", "8: 0.888102"])

    def test_not_iterable(self):
        """Test convert_composition exception for non-iterable argument...."""
        with pytest.raises(materials.NISTMaterialsRequestError):
            materials.convert_composition(100)

    def test_not_string(self):
        """Test convert_composition exception for non-string argument......"""
        with pytest.raises(materials.NISTMaterialsRequestError):
            materials.convert_composition([100, 100])

    def test_bad_line(self):
        """Test convert_composition exception for badly-formed line........"""
        with pytest.raises(materials.NISTMaterialsRequestError):
            materials.convert_composition(["1: :0.111898", "8: 0.888102"])

    def test_bad_z(self):
        """Test convert_composition exception for bad Z value.............."""
        with pytest.raises(materials.NISTMaterialsRequestError):
            materials.convert_composition(["X: 0.111898", "8: 0.888102"])

    def test_z_out_of_range(self):
        """Test convert_composition exception for Z value out of range....."""
        with pytest.raises(materials.NISTMaterialsRequestError):
            materials.convert_composition(["118: 0.111898", "8: 0.888102"])


@pytest.mark.webtest
def test_element_data():
    """Test fetch_element_data........................................."""
    materials.fetch_element_data()


@pytest.mark.webtest
def test_compound_data():
    """Test fetch_compound_data........................................"""
    materials.fetch_compound_data()
