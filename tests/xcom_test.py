"""Test XCOM data queries."""

from __future__ import print_function
from becquerel.tools import xcom
import pytest

# pylint: disable=protected-access,too-many-public-methods
XCOM_URL_ORIG = xcom._URL


@pytest.mark.webtest
class TestFetchXCOMData(object):
    """Test fetch_xcom_data function."""

    def test_01(self):
        """Test fetch_xcom_data with symbol and one energy................."""
        energies = [1460.]
        xd = xcom.fetch_xcom_data('Ge', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_02(self):
        """Test fetch_xcom_data with symbol and three energies............."""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data('Ge', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_03(self):
        """Test fetch_xcom_data with uppercase symbol and three energies..."""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data('GE', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_04(self):
        """Test fetch_xcom_data with lowercase symbol and three energies..."""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data('ge', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_05(self):
        """Test fetch_xcom_data with z (integer) and three energies........"""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data(32, energies_kev=energies)
        assert len(xd) == len(energies)

    def test_06(self):
        """Test fetch_xcom_data with z (string) and three energies........."""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data('32', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_07(self):
        """Test fetch_xcom_data with compound (H2O) and three energies....."""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data('H2O', energies_kev=energies)
        assert len(xd) == len(energies)

    def test_08(self):
        """Test fetch_xcom_data with mixture and three energies............"""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data(
            ['H2O 0.9', 'NaCl 0.1'], energies_kev=energies)
        assert len(xd) == len(energies)

    def test_09(self):
        """Test fetch_xcom_data with three energies and standard grid......"""
        energies = [60., 662., 1460.]
        xd = xcom.fetch_xcom_data(
            'Ge', energies_kev=energies, e_range_kev=[1., 10000.])
        assert len(xd) > len(energies)

    def test_10(self):
        """Test fetch_xcom_data for predefined mixtures...................."""
        energies = [60., 662., 1460.]
        mixtures = [key for key in dir(xcom) if key.startswith('MIXTURE')]
        for mixture in mixtures:
            xd = xcom.fetch_xcom_data(
                getattr(xcom, mixture), energies_kev=energies)
            assert len(xd) == len(energies)

    def test_11(self):
        """Test fetch_xcom_data raises exception if z is out of range......"""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(130, energies_kev=[60., 662., 1460.])

    def test_12(self):
        """Test fetch_xcom_data raises except for badly formed mixture (1)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(
                ['H2O 0.9', 'NaCl'], energies_kev=[60., 662., 1460.])

    def test_13(self):
        """Test fetch_xcom_data raises except for badly formed mixture (2)."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(
                ['H2O 1 1', 'NaCl 1'], energies_kev=[60., 662., 1460.])

    def test_14(self):
        """Test fetch_xcom_data raises exception if given bad argument....."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data(None, energies_kev=[60., 662., 1460.])

    def test_15(self):
        """Test fetch_xcom_data raises except if no energies are requested."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge')

    def test_16(self):
        """Test fetch_xcom_data raises exception if website not found......"""
        xcom._URL = 'http://httpbin.org/status/404'
        with pytest.raises(xcom.XCOMRequestError):
            xcom.fetch_xcom_data('Ge', energies_kev=[60., 662., 1460.])
        xcom._URL = XCOM_URL_ORIG

    def test_17(self):
        """Test fetch_xcom_data raises except if data from website is empty"""
        xcom._URL = 'http://httpbin.org/post'
        with pytest.raises(xcom.XCOMRequestError):
            xcom.fetch_xcom_data('Ge', energies_kev=[60., 662., 1460.])
        xcom._URL = XCOM_URL_ORIG

    def test_20(self):
        """Test fetch_xcom_data raises except if energies_kev not iterable."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', energies_kev=1460.)

    def test_21(self):
        """Test fetch_xcom_data raises exception if energies_kev too low..."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', energies_kev=[60., 662., 1460., 0.001])

    def test_22(self):
        """Test fetch_xcom_data raises exception if energies_kev too high.."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', energies_kev=[60., 662., 1460., 1e9])

    def test_23(self):
        """Test fetch_xcom_data raises except if e_range_kev not iterable.."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', e_range_kev=100.)

    def test_24(self):
        """Test fetch_xcom_data raises exception if len(e_range_kev) != 2.."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', e_range_kev=[1., 10000., 100000.])

    def test_25(self):
        """Test fetch_xcom_data raises except if e_range_kev[0] bad........"""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', e_range_kev=[0.1, 10000.])

    def test_26(self):
        """Test fetch_xcom_data raises except if e_range_kev[1] bad........"""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', e_range_kev=[0.1, 1e9])

    def test_27(self):
        """Test fetch_xcom_data raises except if e_range_kev out of order.."""
        with pytest.raises(xcom.XCOMInputError):
            xcom.fetch_xcom_data('Ge', e_range_kev=[1000., 1.])
