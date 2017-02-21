"""Test becquerel's Spectrum."""

from __future__ import print_function
import unittest
import numpy as np

import becquerel as bq

from parsers_test import SAMPLES

TEST_DATA_LENGTH = 256
TEST_COUNTS = 4
TEST_GAIN = 8.23
TEST_EDGES_KEV = np.arange(TEST_DATA_LENGTH + 1) * TEST_GAIN


class SpectrumFromFileTests(unittest.TestCase):
    """Test Spectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            bq.core.Spectrum.from_file(filename)

    def test_spe(self):
        """Test Spectrum.from_file for SPE file........................."""
        self.run_from_file('.spe')

    def test_spc(self):
        """Test Spectrum.from_file for SPC file........................."""
        self.run_from_file('.spc')

    def test_cnf(self):
        """Test Spectrum.from_file for CNF file........................."""
        self.run_from_file('.cnf')


class SpectrumConstructorTests(unittest.TestCase):
    """Test Spectrum.__init__()."""

    def test_uncal(self):
        """Test simple uncalibrated construction."""

        spec = get_test_uncal_spectrum()
        self.assertEqual(len(spec.data), TEST_DATA_LENGTH)
        self.assertFalse(spec.is_calibrated)

    def test_cal(self):
        """Test simple calibrated construction."""

        spec = get_test_cal_spectrum()
        self.assertEqual(len(spec.data), TEST_DATA_LENGTH)
        self.assertEqual(len(spec.bin_edges_kev), TEST_DATA_LENGTH + 1)
        self.assertEqual(len(spec.energies_kev), TEST_DATA_LENGTH)
        self.assertTrue(spec.is_calibrated)

    def test_init_exceptions(self):
        """Test errors on initialization."""

        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum([])
        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum(
                get_test_data(), bin_edges_kev=TEST_EDGES_KEV[:-1])

        bad_edges = TEST_EDGES_KEV.copy()
        bad_edges[12] = bad_edges[9]
        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum(get_test_data(), bin_edges_kev=bad_edges)

    def test_uncalibrated_exception(self):
        """Test UncalibratedError."""

        spec = get_test_uncal_spectrum()
        with self.assertRaises(bq.core.UncalibratedError):
            spec.energies_kev


class SpectrumAddSubtractTests(unittest.TestCase):
    """Test addition and subtraction of spectra"""

    def test_uncal_add_sub(self):
        """Test basic addition/subtraction of uncalibrated spectra"""

        spec1 = get_test_uncal_spectrum()
        spec2 = get_test_uncal_spectrum()
        tot = spec1 + spec2
        self.assertTrue(np.all(tot.data == spec1.data + spec2.data))
        diff = spec1 - spec2
        self.assertTrue(np.all(diff.data == spec1.data - spec2.data))

    def test_cal_uncal_add_sub(self):
        """Test basic addition of a calibrated with an uncalibrated spectrum.

        NOTE: not implemented yet - so check that it errors.
        """

        spec1 = get_test_uncal_spectrum()
        spec2 = get_test_cal_spectrum()
        with self.assertRaises(NotImplementedError):
            spec1 + spec2
        with self.assertRaises(NotImplementedError):
            spec1 - spec2

    def test_cal_add_sub(self):
        """Test basic addition of calibrated spectra.

        NOTE: not implemented yet - so check that it errors.
        """

        spec1 = get_test_cal_spectrum()
        spec2 = get_test_cal_spectrum()
        with self.assertRaises(NotImplementedError):
            spec1 + spec2
        with self.assertRaises(NotImplementedError):
            spec1 - spec2

    def test_add_sub_type_error(self):
        """Check that adding/subtracting a non-Spectrum gives a TypeError."""

        spec1 = get_test_uncal_spectrum()
        with self.assertRaises(TypeError):
            spec1 + 5
        with self.assertRaises(TypeError):
            spec1 - 5
        with self.assertRaises(TypeError):
            spec1 + 'asdf'
        with self.assertRaises(TypeError):
            spec1 - 'asdf'
        with self.assertRaises(TypeError):
            spec1 + get_test_data()
        with self.assertRaises(TypeError):
            spec1 - get_test_data()

    def test_add_sub_wrong_length(self):
        """
        Adding/subtracting spectra of different lengths gives a SpectrumError.
        """

        spec1 = get_test_uncal_spectrum()
        spec2 = bq.core.Spectrum(get_test_data(length=TEST_DATA_LENGTH * 2))
        with self.assertRaises(bq.core.SpectrumError):
            spec1 + spec2
        with self.assertRaises(bq.core.SpectrumError):
            spec1 - spec2


def get_test_data(length=TEST_DATA_LENGTH, expectation_val=TEST_COUNTS):
    """Build a vector of random counts."""
    return np.random.poisson(lam=expectation_val, size=length).astype(np.int)


def get_test_uncal_spectrum():
    uncal = bq.core.Spectrum(get_test_data())
    return uncal


def get_test_cal_spectrum():
    cal = bq.core.Spectrum(get_test_data(), bin_edges_kev=TEST_EDGES_KEV)
    return cal


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
