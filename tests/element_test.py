"""Test Element class."""

from becquerel.tools import element
import pytest


class TestElementFunctions:
    """Test Element functions."""

    def test_validated_z_good(self):
        """Test validated_z................................................"""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            assert element.validated_z(z1) == z1

    def test_validated_z_exception(self):
        """Test validated_z(119) raises ElementZError......................"""
        with pytest.raises(element.ElementZError):
            element.validated_z(119)

    def test_validated_symbol_good(self):
        """Test validated_symbol..........................................."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            for sym2 in [sym1, sym1.lower(), sym1.upper()]:
                assert element.validated_symbol(sym2) == sym1

    def test_validated_symbol_exception(self):
        """Test validated_symbol('Xz') raises ElementSymbolError..........."""
        with pytest.raises(element.ElementSymbolError):
            element.validated_symbol("Xz")

    def test_validated_name_good(self):
        """Test validated_name............................................."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            for name2 in [name1, name1.lower(), name1.upper()]:
                assert element.validated_name(name2) == name1

    def test_validated_name_exception(self):
        """Test validated_name('Xzzzzz') raises ElementNameError..........."""
        with pytest.raises(element.ElementNameError):
            element.validated_name("Xzzzzz")

    def test_validated_name_aluminum(self):
        """Test validated_name('Aluminum') returns 'Aluminum'.............."""
        name1 = "Aluminum"
        for name2 in [name1, name1.lower(), name1.upper()]:
            assert element.validated_name(name2) == "Aluminum"

    def test_validated_name_aluminium(self):
        """Test validated_name('Aluminium') returns 'Aluminum'............."""
        name1 = "Aluminium"
        for name2 in [name1, name1.lower(), name1.upper()]:
            assert element.validated_name(name2) == "Aluminum"

    def test_validated_name_cesium(self):
        """Test validated_name('Cesium') returns 'Cesium'.................."""
        name1 = "Cesium"
        for name2 in [name1, name1.lower(), name1.upper()]:
            assert element.validated_name(name2) == "Cesium"

    def test_validated_name_caesium(self):
        """Test validated_name('Caesium') returns 'Cesium'................."""
        name1 = "Caesium"
        for name2 in [name1, name1.lower(), name1.upper()]:
            assert element.validated_name(name2) == "Cesium"

    def test_element_z(self):
        """Test element_z.................................................."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            for sym2 in [sym1, sym1.lower(), sym1.upper()]:
                assert element.element_z(sym2) == z1
            for name2 in [name1, name1.lower(), name1.upper()]:
                assert element.element_z(name2) == z1

    def test_element_z_exception(self):
        """Test element_z with bad input raises ElementZError.............."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            with pytest.raises(element.ElementZError):
                element.element_z(z1)

    def test_element_symbol(self):
        """Test element_symbol............................................."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            assert element.element_symbol(z1) == sym1
            for name2 in [name1, name1.lower(), name1.upper()]:
                assert element.element_symbol(name2) == sym1

    def test_element_symbol_exception(self):
        """Test element_symbol with bad input raises ElementSymbolError...."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            with pytest.raises(element.ElementSymbolError):
                element.element_symbol(sym1)

    def test_element_name(self):
        """Test element_name..............................................."""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            assert element.element_name(z1) == name1
            for sym2 in [sym1, sym1.lower(), sym1.upper()]:
                assert element.element_name(sym2) == name1

    def test_element_name_exception(self):
        """Test element_name with bad input raises ElementNameError........"""
        for z1, sym1, name1, mass1 in element._Z_SYMBOL_NAME_MASS:
            with pytest.raises(element.ElementNameError):
                element.element_name(name1)


@pytest.mark.parametrize(
    "z, sym, name",
    [
        (1, "H", "Hydrogen"),
        (2, "He", "Helium"),
        (13, "Al", "Aluminum"),
        (19, "K", "Potassium"),
        (32, "Ge", "Germanium"),
        (70, "Yb", "Ytterbium"),
        (92, "U", "Uranium"),
        (118, "Og", "Oganesson"),
    ],
)
def test_element(z, sym, name):
    """Run instantiation tests for various elements.

    Instantiate for element symbol and name, in mixed case, upper case,
    and lower case. Also by Z as both integer and string.
    """

    args = [name, name.lower(), name.upper()]
    args.extend([sym, sym.lower(), sym.upper()])
    args.extend([z, str(z)])
    print(args)
    for arg in args:
        print("")
        print("arg: ", arg)
        elem = element.Element(arg)
        print(elem)
        assert elem.Z == z
        assert elem.symbol == sym
        assert elem.name == name


class TestElementInitExceptions:
    """Test Element class throws exceptions."""

    def test_bad_arg_symbol(self):
        """Test Element init with a bad symbol raises ElementError........."""
        with pytest.raises(element.ElementError):
            element.Element("Xx")

    def test_bad_arg_name(self):
        """Test Element init with a bad name raises ElementError..........."""
        with pytest.raises(element.ElementError):
            element.Element("Xirconium")

    def test_bad_arg_z(self):
        """Test Element init with a bad Z raises ElementError.............."""
        with pytest.raises(element.ElementError):
            element.Element(0)


class TestElementsEqual:
    """Test Element class equality."""

    def test_h(self):
        """Test Element equality: H........................................"""
        assert element.Element("H") == element.Element(1)

    def test_og(self):
        """Test Element equality: Og......................................."""
        assert element.Element("Og") == element.Element(118)

    def test_bad(self):
        """Test Element equality: H != 0..................................."""
        with pytest.raises(element.ElementError):
            elem = element.Element("H")
            elem == 0


class TestElementStrFormat:
    """Test Element class string formatting."""

    def test_h(self):
        """Test Element string formatting: H..............................."""
        assert "{:%n (%s) %z}".format(element.Element("H")) == "Hydrogen (H) 1"

    def test_og(self):
        """Test Element string formatting: Og.............................."""
        assert "{:%n (%s) %z}".format(element.Element("Og")) == "Oganesson (Og) 118"
