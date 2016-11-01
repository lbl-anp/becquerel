"""Test becquerel spectrum file parsers."""

from __future__ import print_function
import glob
import os
import unittest
import matplotlib.pyplot as plt
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), 'samples')


class SpectrumFileTests(unittest.TestCase):
    """Test spectrum file parsers."""

    def run_parser(self, cls, extension, write=False):
        """Run the test for the given class and file extension."""
        at_least_one_read = False
        plt.figure()
        plt.title('Testing ' + cls.__name__)
        filenames = glob.glob(os.path.join(SAMPLES_PATH, '*.*'))
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            path, fname = os.path.split(fname)
            if ext.lower() != extension.lower():
                continue
            print('')
            print(filename)
            spec = cls(filename)
            spec.read()
            spec.apply_calibration()
            at_least_one_read = True
            print(spec)
            plt.semilogy(
                spec.energies,
                spec.data / spec.energy_bin_widths / spec.livetime,
                label=fname)
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts/keV/sec')
            plt.xlim(0, 2800)
            if write:
                writename = os.path.join('.', fname + '_copy' + ext)
                spec.write(writename)
                os.remove(writename)
        plt.legend(prop={'size': 8})
        self.assertTrue(at_least_one_read)
        plt.show()

    def test_spe(self):
        """Test parsers.SpeFile............................................"""
        self.run_parser(bq.parsers.SpeFile, '.spe', write=True)

    def test_spc(self):
        """Test parsers.SpcFile............................................"""
        self.run_parser(bq.parsers.SpcFile, '.spc', write=False)

    def test_cnf(self):
        """Test parsers.CnfFile............................................"""
        self.run_parser(bq.parsers.CnfFile, '.cnf', write=False)


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
