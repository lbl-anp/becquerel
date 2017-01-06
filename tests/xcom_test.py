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
        xd = xcom.XCOMQuery(energies=[1460.], symbol='Ge')
        self.assertTrue(len(xd) == 1)

    def test_02(self):
        """Test XCOMQuery with symbol and three energies..................."""
        xd = xcom.XCOMQuery(energies=[60., 662., 1460.], symbol='Ge')
        self.assertTrue(len(xd) == 3)

    def test_03(self):
        """Test XCOMQuery with Z (integer) and three energies.............."""
        xd = xcom.XCOMQuery(energies=[60., 662., 1460.], Z=32)
        self.assertTrue(len(xd) == 3)

    def test_04(self):
        """Test XCOMQuery with Z (string) and three energies..............."""
        xd = xcom.XCOMQuery(energies=[60., 662., 1460.], Z='32')
        self.assertTrue(len(xd) == 3)

    def test_05(self):
        """Test XCOMQuery with elemental compound (Ge) and three energies.."""
        xd = xcom.XCOMQuery(energies=[60., 662., 1460.], compound='Ge')
        self.assertTrue(len(xd) == 3)

    def test_06(self):
        """Test XCOMQuery with chemical compound (H2O) and three energies.."""
        xd = xcom.XCOMQuery(energies=[60., 662., 1460.], compound='H2O')
        self.assertTrue(len(xd) == 3)

    def test_07(self):
        """Test XCOMQuery with mixture and three energies.................."""
        xd = xcom.XCOMQuery(
            energies=[60., 662., 1460.], mixture=['H2O 0.9', 'NaCl 0.1'])
        self.assertTrue(len(xd) == 3)

    def test_08(self):
        """Test XCOMQuery with three energies and standard energy grid....."""
        xd = xcom.XCOMQuery(
            energies=[60., 662., 1460.], e_range=[1., 10000.], symbol='Ge')
        self.assertTrue(len(xd) > 3)

    def test_09(self):
        """Test XCOMQuery for predefined mixtures.........................."""
        mixtures = [key for key in dir(xcom) if key.startswith('MIXTURE')]
        for mixture in mixtures:
            xd = xcom.XCOMQuery(
                energies=[60., 662., 1460.], mixture=getattr(xcom, mixture))
            self.assertTrue(len(xd) == 3)

    def test_10(self):
        """Test XCOMQuery raises exception for unknown symbol.............."""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                energies=[60., 662., 1460.], e_range=[1., 10000.], symbol='Xx')

    def test_11(self):
        """Test XCOMQuery raises exception if both symbol and Z given......"""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                energies=[60., 662., 1460.], symbol='Ge', Z=32)

    def test_12(self):
        """Test XCOMQuery raises exception if symbol and compound given...."""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                energies=[60., 662., 1460.], symbol='Ge', compound='Ge')

    def test_13(self):
        """Test XCOMQuery raises exception if Z is out of range............"""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                energies=[60., 662., 1460.], Z=130)

    def test_14(self):
        """Test XCOMQuery raises exception for badly formed mixture........"""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(
                energies=[60., 662., 1460.], mixture=['H2O 0.9', 'NaCl'])

    def test_15(self):
        """Test XCOMQuery raises exception if no energies are requested...."""
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(symbol='Ge')

    def test_16(self):
        """Test XCOMQuery raises exception if website not found............"""
        xcom._URL = 'http://httpbin.org/status/404'
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(energies=[60., 662., 1460.], symbol='Ge')
        xcom._URL = XCOM_URL_ORIG

    def test_17(self):
        """Test XCOMQuery raises exception if data from website is empty..."""
        xcom._URL = 'http://httpbin.org/post'
        with self.assertRaises(xcom.XCOMError):
            xcom.XCOMQuery(energies=[60., 662., 1460.], symbol='Ge')
        xcom._URL = XCOM_URL_ORIG


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
