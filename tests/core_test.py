"""Test becquerel's RawSpectrum, CalSpectrum, EnergyCal."""

from __future__ import print_function
import unittest
import numpy as np
import becquerel as bq

from parsers_test import SAMPLES


class RawSpectrumFromFileTests(unittest.TestCase):
    """Test RawSpectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            spec = bq.core.RawSpectrum.from_file(filename)

    def test_spe(self):
        """Test RawSpectrum.from_file for SPE file........................."""
        self.run_from_file('.spe')

    def test_spc(self):
        """Test RawSpectrum.from_file for SPC file........................."""
        self.run_from_file('.spc')

    def test_cnf(self):
        """Test RawSpectrum.from_file for CNF file........................."""
        self.run_from_file('.cnf')


class CalSpectrumFromFileTests(unittest.TestCase):
    """Test CalSpectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            spec = bq.core.CalSpectrum.from_file(filename)

    def test_spe(self):
        """Test CalSpectrum.from_file for SPE file........................."""
        self.run_from_file('.spe')

    def test_spc(self):
        """Test CalSpectrum.from_file for SPC file........................."""
        self.run_from_file('.spc')

    def test_cnf(self):
        """Test CalSpectrum.from_file for CNF file........................."""
        self.run_from_file('.cnf')


class CalSpectrumFromRawTests(unittest.TestCase):
    """Test CalSpectrum.from_raw() class method."""

    def run_from_file(self, extension):
        """Run the test of from_raw() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            raw = bq.core.RawSpectrum.from_file(filename)
            ecal = bq.core.EnergyCal.from_file_obj(raw.infileobject)
            spec = bq.core.CalSpectrum.from_raw(raw, ecal)


class EnergyCalBasicTests(unittest.TestCase):
    """Test EnergyCal coefficient initialization and channel_to_energy()."""

    def test_linear(self):
        """Test linear coefficients."""
        offset = 1.1
        slope = 0.233
        ch = np.array([[0, 1], [2, 4]])
        etest = offset + ch * slope

        ecal = bq.core.EnergyCal([offset, slope])
        self.assertTrue(np.all(ecal.channel_to_energy(ch) == etest))

    def test_quadratic(self):
        """Test quadratic coefficients."""
        offset = 1.1
        slope = 0.233
        quad = 0.004
        ch = np.array([[0, 1], [2, 4]])
        etest = offset + ch * slope + ch**2 * quad

        ecal = bq.core.EnergyCal([offset, slope, quad])
        self.assertTrue(np.all(ecal.channel_to_energy(ch) == etest))

    def test_coeff_error(self):
        """Test error on bad number of coefficients."""
        with self.assertRaises(bq.core.energycal.EnergyCalError):
            ecal = bq.core.EnergyCal([1.1])
        with self.assertRaises(bq.core.energycal.EnergyCalError):
            ecal = bq.core.EnergyCal([1.1, 2.2, 3.3, 4.4, 5.5])


class EnergyCalFromFileObjTests(unittest.TestCase):
    """Test EnergyCal.from_file_obj() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file_obj() for files with given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            raw = bq.core.RawSpectrum.from_file(filename)
            ecal = bq.core.EnergyCal.from_file_obj(raw.infileobject)

    def test_spe(self):
        """Test EnergyCal.from_file_obj for SPE file......................."""
        self.run_from_file('.spe')

    def test_spc(self):
        """Test EnergyCal.from_file_obj for SPC file......................."""
        self.run_from_file('.spc')

    def test_cnf(self):
        """Test EnergyCal.from_file_obj for CNF file......................."""
        self.run_from_file('.cnf')


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
