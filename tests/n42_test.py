"""Test N42 parsing code."""

# pylint: disable=R0201

from __future__ import print_function
import os
import unittest
import requests
import matplotlib.pyplot as plt
from becquerel.parsers import n42_file as n42


def is_close(f1, f2, ppm=1.):
    """True if f1 and f2 are within the given parts per million."""
    return abs((f1 - f2) / f2) < ppm / 1.e6


class ParseDurationTests(unittest.TestCase):
    """Test parse_duration()."""

    def test_1(self):
        """Test parse_duration('PT0.9S')..................................."""
        answer = n42.parse_duration('PT0.9S')
        self.assertTrue(is_close(answer, 0.9))

    def test_2(self):
        """Test parse_duration('PT1.0S')..................................."""
        answer = n42.parse_duration('PT1.0S')
        self.assertTrue(is_close(answer, 1.))


DATA_UNCOMPRESSED_STR = '  0 0 0 3 4 5 2 1 0 0 0 0 0 5   '
DATA_UNCOMPRESSED_STR_LINES = '  0 0 0 3 4 5 2 1\n  0 0 0 0 0\n 5   '
DATA_COMPRESSED_STR = '  0 3 3 4 5 2 1 0 5 5   '
DATA_COMPRESSED_STR_LINES = '  0 3 3 \n4 5 2 \n  1 0 5 5   '
DATA_UNCOMPRESSED = [0, 0, 0, 3, 4, 5, 2, 1, 0, 0, 0, 0, 0, 5]
DATA_COMPRESSED = [0, 3, 3, 4, 5, 2, 1, 0, 5, 5]


class ParseChannelDataTests(unittest.TestCase):
    """Test parse_channel_data()."""

    def test_uncompressed_one_line(self):
        """Test parse_channel_data with one line..........................."""
        answer = n42.parse_channel_data(DATA_UNCOMPRESSED_STR)
        self.assertTrue(answer == DATA_UNCOMPRESSED)

    def test_uncompressed_multiple_lines(self):
        """Test parse_channel_data with multiple lines....................."""
        answer = n42.parse_channel_data(DATA_UNCOMPRESSED_STR_LINES)
        self.assertTrue(answer == DATA_UNCOMPRESSED)

    def test_compressed_one_line(self):
        """Test parse_channel_data with one line (compressed).............."""
        answer = n42.parse_channel_data(
            DATA_COMPRESSED_STR, compression='CountedZeroes')
        self.assertTrue(answer == DATA_UNCOMPRESSED)

    def test_compressed_multiple_lines(self):
        """Test parse_channel_data with multiple lines (compressed)........"""
        answer = n42.parse_channel_data(
            DATA_COMPRESSED_STR_LINES, compression='CountedZeroes')
        self.assertTrue(answer == DATA_UNCOMPRESSED)


class CompressChannelDataTests(unittest.TestCase):
    """Test compress_channel_data()."""

    def test_compress(self):
        """Test compress_channel_data......................................"""
        answer = n42.compress_channel_data(DATA_UNCOMPRESSED)
        self.assertTrue(answer == DATA_COMPRESSED)


class N42SampleTests(object):
    """Read N42 sample files and run tests on them (base class)."""

    def download_sample(self, filename):
        """Download a sample N42 file."""
        req = requests.get(n42.NIST_N42_URL + filename)
        xml_text = req.text.encode('ascii').decode('ascii')
        return xml_text

    def read_sample(self, filename):
        """Read a sample N42 file. Download if necessary."""
        print('')
        print(filename)
        sample_dir = os.path.join(os.path.dirname(__file__), 'samples')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        local_filename = os.path.join(sample_dir, filename)
        if not os.path.exists(local_filename):
            xml_text = self.download_sample(filename)
            with open(local_filename, 'w') as f:
                print(xml_text, file=f)
        with open(local_filename, 'r') as f:
            xml_text = f.read()
        xml_text = xml_text.encode('ascii').decode('ascii')
        return xml_text


class N42ParseSampleTests(N42SampleTests, unittest.TestCase):
    """Parse N42 sample files."""

    def parse_sample(self, filename):
        """Read and parse a sample N42 file."""
        xml_text = self.read_sample(filename)
        self.assertTrue(n42.valid_xml(xml_text))
        n42.parse_n42(xml_text)
        plt.show()

    def test_annexB(self):
        """Parse sample N42: AnnexB.n42...................................."""
        self.parse_sample('AnnexB.n42')

    def test_annexB_alternate(self):
        """Parse sample N42: AnnexB_alternate_energy_calibration.n42......."""
        self.parse_sample('AnnexB_alternate_energy_calibration.n42')

    def test_annexC(self):
        """Parse sample N42: AnnexC.n42...................................."""
        self.parse_sample('AnnexC.n42')

    def test_annexE(self):
        """Parse sample N42: AnnexE.n42...................................."""
        self.parse_sample('AnnexE.n42')

    def test_annexG(self):
        """Parse sample N42: AnnexG.n42...................................."""
        self.parse_sample('AnnexG.n42')

    def test_annexI(self):
        """Parse sample N42: AnnexI.n42...................................."""
        self.parse_sample('AnnexI.n42')


class N42ValidationTests(N42SampleTests, unittest.TestCase):
    """Test validation of N42 sample files."""

    def run_validation(self, filename):
        """Test validation of the file."""
        xml_text = self.read_sample(filename)
        self.assertTrue(n42.valid_xml(xml_text))

    def test_annexB(self):
        """Validate sample N42: AnnexB.n42................................."""
        self.run_validation('AnnexB.n42')

    def test_annexB_alternate(self):
        """Validate sample N42: AnnexB_alternate_energy_calibration.n42...."""
        self.run_validation('AnnexB_alternate_energy_calibration.n42')

    def test_annexC(self):
        """Validate sample N42: AnnexC.n42................................."""
        self.run_validation('AnnexC.n42')

    def test_annexE(self):
        """Validate sample N42: AnnexE.n42................................."""
        self.run_validation('AnnexE.n42')

    def test_annexG(self):
        """Validate sample N42: AnnexG.n42................................."""
        self.run_validation('AnnexG.n42')

    def test_annexI(self):
        """Validate sample N42: AnnexI.n42................................."""
        self.run_validation('AnnexI.n42')


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
