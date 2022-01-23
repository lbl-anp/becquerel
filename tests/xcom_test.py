"""Test XCOM data queries."""

import numpy as np
from becquerel.tools import xcom
import pytest

XCOM_URL_ORIG = xcom._URL


# energies to query using energies keyword
ENERGIES_3 = [60.0, 662.0, 1460.0]


# standard grid energies for Germanium from 1 keV to 10 MeV
GE_GRID_ENERGIES = [
    1,
    1.103,
    1.217,
    1.217,
    1.232,
    1.248,
    1.248,
    1.328,
    1.414,
    1.414,
    1.500,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    11.100,
    11.100,
    15,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    400,
    500,
    600,
    800,
    1000,
    1022,
    1250,
    1500,
    2000,
    2044,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]


# standard grid energies for Germanium, plus 60, 662, and 1460
GE_GRID_ENERGIES_PLUS_3 = [
    1,
    1.103,
    1.217,
    1.217,
    1.232,
    1.248,
    1.248,
    1.328,
    1.414,
    1.414,
    1.500,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    11.100,
    11.100,
    15,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    400,
    500,
    600,
    662,
    800,
    1000,
    1022,
    1250,
    1460,
    1500,
    2000,
    2044,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]


@pytest.mark.webtest
class TestFetchXCOMData:
    """Test fetch_xcom_data function."""

    def test_sym_energy(self):
        """Test fetch_xcom_data with symbol and one energy."""
        energies = [1460.0]
        xd = xcom.fetch_xcom_data("Ge", energies_kev=energies)
        assert len(xd) == len(energies)
        assert np.allclose(xd.energy, energies)

    def test_sym_energies(self):
        """Test fetch_xcom_data with symbol and three energies."""
        xd = xcom.fetch_xcom_data("Ge", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_sym_upper(self):
        """Test fetch_xcom_data with uppercase symbol and three energies."""
        xd = xcom.fetch_xcom_data("GE", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_sym_lower(self):
        """Test fetch_xcom_data with lowercase symbol and three energies."""
        xd = xcom.fetch_xcom_data("ge", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_z_int(self):
        """Test fetch_xcom_data with z (integer) and three energies."""
        xd = xcom.fetch_xcom_data(32, energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_z_str(self):
        """Test fetch_xcom_data with z (string) and three energies."""
        xd = xcom.fetch_xcom_data("32", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_compound1(self):
        """Test fetch_xcom_data with compound (H2O) and three energies."""
        xd = xcom.fetch_xcom_data("H2O", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_compound2(self):
        """Test fetch_xcom_data with compound (NaCl) and three energies."""
        xd = xcom.fetch_xcom_data("NaCl", energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_mixture(self):
        """Test fetch_xcom_data with mixture and three energies."""
        xd = xcom.fetch_xcom_data(["H2O 0.9", "NaCl 0.1"], energies_kev=ENERGIES_3)
        assert len(xd) == len(ENERGIES_3)
        assert np.allclose(xd.energy, ENERGIES_3)

    def test_standard_grid(self):
        """Test fetch_xcom_data with standard grid."""
        xd = xcom.fetch_xcom_data("Ge", e_range_kev=[1.0, 10000.0])
        assert len(xd) == len(GE_GRID_ENERGIES)
        assert np.allclose(xd.energy, GE_GRID_ENERGIES)

    def test_standard_grid_energies(self):
        """Test fetch_xcom_data with three energies and standard grid."""
        xd = xcom.fetch_xcom_data(
            "Ge", energies_kev=ENERGIES_3, e_range_kev=[1.0, 10000.0]
        )
        assert len(xd) == len(GE_GRID_ENERGIES_PLUS_3)
        assert np.allclose(xd.energy, GE_GRID_ENERGIES_PLUS_3)

    def test_mixtures_predefined(self):
        """Test fetch_xcom_data for predefined mixtures."""
        mixtures = [key for key in dir(xcom) if key.startswith("MIXTURE")]
        for mixture in mixtures:
            xd = xcom.fetch_xcom_data(getattr(xcom, mixture), energies_kev=ENERGIES_3)
            assert len(xd) == len(ENERGIES_3)
            assert np.allclose(xd.energy, ENERGIES_3)

    def test_except_z_range(self):
        """Test fetch_xcom_data raises exception if z is out of range."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(130, energies_kev=ENERGIES_3)

    def test_except_bad_compound(self):
        """Test fetch_xcom_data raises except for bad compound formula."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("H2O++", energies_kev=ENERGIES_3)

    def test_except_bad_mixture1(self):
        """Test fetch_xcom_data raises except for badly formed mixture (1)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(["H2O 0.9", "NaCl"], energies_kev=ENERGIES_3)

    def test_except_bad_mixture2(self):
        """Test fetch_xcom_data raises except for badly formed mixture (2)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(["H2O 1 1", "NaCl 1"], energies_kev=ENERGIES_3)

    def test_except_bad_mixture3(self):
        """Test fetch_xcom_data raises except for badly formed mixture (3)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data({"H2O": "1", "NaCl": "1"}, energies_kev=ENERGIES_3)

    def test_except_bad_mixture4(self):
        """Test fetch_xcom_data raises except for badly formed mixture (4)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(["H2O 0.9", ["NaCl", "0.1"]], energies_kev=ENERGIES_3)

    def test_except_bad_mixture5(self):
        """Test fetch_xcom_data raises except for badly formed mixture (5)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(["H2O 0.9", "NaCl $0.1"], energies_kev=ENERGIES_3)

    def test_except_bad_arg(self):
        """Test fetch_xcom_data raises exception if given bad argument."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(None, energies_kev=ENERGIES_3)

    def test_except_no_energies(self):
        """Test fetch_xcom_data raises except if no energies are requested."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge")

    def test_except_status_404(self):
        """Test fetch_xcom_data raises exception if website not found."""
        xcom._URL = "http://httpbin.org/status/404"
        with pytest.raises(xcom.XCOMRequestError):
            xcom.fetch_xcom_data("Ge", energies_kev=ENERGIES_3)
        xcom._URL = XCOM_URL_ORIG

    def test_except_website_empty(self):
        """Test fetch_xcom_data raises except if data from website is empty."""
        xcom._URL = "http://httpbin.org/post"
        with pytest.raises(xcom.XCOMRequestError):
            xcom.fetch_xcom_data("Ge", energies_kev=ENERGIES_3)
        xcom._URL = XCOM_URL_ORIG

    def test_except_energies_kev_float(self):
        """Test fetch_xcom_data raises except if energies_kev not iterable."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", energies_kev=1460.0)

    def test_except_energies_kev_low(self):
        """Test fetch_xcom_data raises exception if energies_kev too low."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", energies_kev=[60.0, 662.0, 1460.0, 0.001])

    def test_except_energies_kev_high(self):
        """Test fetch_xcom_data raises exception if energies_kev too high."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", energies_kev=[60.0, 662.0, 1460.0, 1e9])

    def test_except_e_range_kev_float(self):
        """Test fetch_xcom_data raises except if e_range_kev not iterable."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=100.0)

    def test_except_e_range_kev_len(self):
        """Test fetch_xcom_data raises exception if len(e_range_kev) != 2."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=[1.0, 10000.0, 100000.0])

    def test_except_e_range_kev_0_range(self):
        """Test fetch_xcom_data exception if e_range_kev[0] out of range."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=[0.1, 10000.0])

    def test_except_e_range_kev_1_range(self):
        """Test fetch_xcom_data exception if e_range_kev[1] out of range."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=[0.1, 1e9])

    def test_except_e_range_kev_order(self):
        """Test fetch_xcom_data raises except if e_range_kev out of order."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=[1000.0, 1.0])

    def test_except_bad_kwarg(self):
        """Test fetch_xcom_data raises exception if bad keyword given."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data("Ge", e_range_kev=[1.0, 10000.0], bad_kw=None)
