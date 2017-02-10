"""Test becquerel's Spectrum."""

from __future__ import print_function
import unittest
import numpy as np

import becquerel as bq

from parsers_test import SAMPLES


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

    test_length = 256
    data = np.random.randint(0, 1e4, test_length, int)
    test_gain = 8.23
    energy_edges = np.arange(test_length + 1) * test_gain

    def test_uncal(self):
        """Test simple uncalibrated construction."""

        spec = bq.core.Spectrum(self.data)
        self.assertEqual(len(spec.data), self.test_length)
        self.assertFalse(spec.is_calibrated)

    def test_cal(self):
        """Test simple calibrated construction."""

        spec = bq.core.Spectrum(self.data, bin_edges_kev=self.energy_edges)
        self.assertEqual(len(spec.data), self.test_length)
        self.assertEqual(len(spec.bin_edges_kev), self.test_length + 1)
        self.assertEqual(len(spec.energies_kev), self.test_length)
        self.assertTrue(spec.is_calibrated)

    def test_init_exceptions(self):
        """Test errors on initialization."""

        with self.assertRaises(bq.core.SpectrumError):
            spec = bq.core.Spectrum([])
        with self.assertRaises(bq.core.SpectrumError):
            spec = bq.core.Spectrum(self.data, bin_edges_kev=self.energy_edges[:-1])

    def test_uncalibrated_exception(self):
        """Test UncalibratedError."""

        spec = bq.core.Spectrum(self.data)
        with self.assertRaises(bq.core.UncalibratedError):
            _ = spec.energies_kev


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
