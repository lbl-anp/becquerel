"""Test XCOM data queries."""

from __future__ import print_function
import unittest
from becquerel.tools import xcom

# pylint: disable=W0212
XCOM_URL_ORIG = xcom._URL


class XCOMQueryTests(unittest.TestCase):
    """Test XCOM queries."""

    def test_01(self):
        """Test XCOMQuery with symbol and one energy......................."""
        energies = [1460.]
        xd = xcom.XCOMQuery('Ge', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_02(self):
        """Test XCOMQuery with symbol and three energies..................."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('Ge', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_03(self):
        """Test XCOMQuery with uppercase symbol and three energies........."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('GE', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_04(self):
        """Test XCOMQuery with lowercase symbol and three energies........."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('ge', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_05(self):
        """Test XCOMQuery with z (integer) and three energies.............."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery(32, energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_06(self):
        """Test XCOMQuery with z (string) and three energies..............."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('32', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_07(self):
        """Test XCOMQuery with chemical compound (H2O) and three energies.."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('H2O', energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_08(self):
        """Test XCOMQuery with mixture and three energies.................."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery(['H2O 0.9', 'NaCl 0.1'], energies=energies)
        self.assertTrue(len(xd) == len(energies))

    def test_09(self):
        """Test XCOMQuery with three energies and standard energy grid....."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('Ge', energies=energies, e_range=[1., 10000.])
        self.assertTrue(len(xd) > len(energies))

    def test_10(self):
        """Test XCOMQuery for predefined mixtures.........................."""
        energies = [60., 662., 1460.]
        mixtures = [key for key in dir(xcom) if key.startswith('MIXTURE')]
        for mixture in mixtures:
            xd = xcom.XCOMQuery(getattr(xcom, mixture), energies=energies)
            self.assertTrue(len(xd) == len(energies))

    def test_11(self):
        """Test XCOMQuery raises exception for unknown symbol.............."""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                'Xx', energies=[60., 662., 1460.], e_range=[1., 10000.])

    def test_12(self):
        """Test XCOMQuery raises exception if z is out of range............"""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(130, energies=[60., 662., 1460.])

    def test_13(self):
        """Test XCOMQuery raises exception for badly formed mixture........"""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(['H2O 0.9', 'NaCl'], energies=[60., 662., 1460.])

    def test_14(self):
        """Test XCOMQuery raises exception if no energies are requested...."""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery('Ge')

    def test_15(self):
        """Test XCOMQuery raises exception if website not found............"""
        xcom._URL = 'http://httpbin.org/status/404'
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery('Ge', energies=[60., 662., 1460.])
        xcom._URL = XCOM_URL_ORIG

    def test_16(self):
        """Test XCOMQuery raises exception if data from website is empty..."""
        xcom._URL = 'http://httpbin.org/post'
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery('Ge', energies=[60., 662., 1460.])
        xcom._URL = XCOM_URL_ORIG

    def test_17(self):
        """Test XCOMQuery instantiated with perform=False.................."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('Ge', energies=energies, perform=False)
        xd.perform()
        self.assertTrue(len(xd) == len(energies))

    def test_18(self):
        """Test XCOMQuery instantiated with perform=False, update called..."""
        energies = [60., 662., 1460.]
        xd = xcom.XCOMQuery('Ge', perform=False)
        xd.update(energies=energies)
        xd.perform()
        self.assertTrue(len(xd) == len(energies))


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
