"""Test becquerel's Spectrum."""

from __future__ import print_function
import unittest
import numpy as np

import becquerel as bq

from parsers_test import SAMPLES

TEST_DATA_LENGTH = 256
TEST_COUNTS = 4
TEST_GAIN = 8.23


class SpectrumFromFileTests(unittest.TestCase):
    """Test Spectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            spec = bq.core.Spectrum.from_file(filename)

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

    energy_edges_kev = np.arange(TEST_DATA_LENGTH + 1) * TEST_GAIN

    def test_uncal(self):
        """Test simple uncalibrated construction."""

        spec = bq.core.Spectrum(get_test_data())
        self.assertEqual(len(spec.data), TEST_DATA_LENGTH)
        self.assertFalse(spec.is_calibrated)

    def test_cal(self):
        """Test simple calibrated construction."""

        spec = bq.core.Spectrum(
            get_test_data(), bin_edges_kev=self.energy_edges_kev)
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
                get_test_data(), bin_edges_kev=self.energy_edges_kev[:-1])

        bad_edges = self.energy_edges_kev.copy()
        bad_edges[12] = bad_edges[9]
        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum(get_test_data(), bin_edges_kev=bad_edges)

    def test_uncalibrated_exception(self):
        """Test UncalibratedError."""

        spec = bq.core.Spectrum(get_test_data())
        with self.assertRaises(bq.core.UncalibratedError):
            spec.energies_kev


def get_test_data(length=TEST_DATA_LENGTH, expectation_val=TEST_COUNTS):
    """Build a vector of random counts."""
    return np.random.poisson(lam=expectation_val, size=length).astype(np.int)


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
