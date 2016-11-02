"""Test becquerel's RawSpectrum and CalSpectrum."""

from __future__ import print_function
import unittest
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


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
