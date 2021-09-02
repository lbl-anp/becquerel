"""Test becquerel spectrum file parsers."""

import glob
import os
import pytest
import matplotlib.pyplot as plt
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "samples")
SAMPLES = {}
for extension in [".spe", ".spc", ".cnf", ".hdf5"]:
    filenames = glob.glob(os.path.join(SAMPLES_PATH + "*", "*.*"))
    filenames_filtered = []
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        if ext.lower() == extension:
            filenames_filtered.append(filename)
    SAMPLES[extension] = filenames_filtered


class TestParsers(object):
    """Test spectrum file parsers."""

    def run_parser(self, read_fn, extension):
        """Run the test for the given class and file extension."""
        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            path, fname = os.path.split(fname)
            print("")
            print(filename)
            data, cal = read_fn(filename)
            print(data, cal)

    def test_spe(self):
        """Test parsers.spe.read."""
        self.run_parser(bq.parsers.spe.read, ".spe")

    def test_spc(self):
        """Test parsers.spc.read."""
        self.run_parser(bq.parsers.spc.read, ".spc")

    def test_cnf(self):
        """Test parsers.cnf.read."""
        self.run_parser(bq.parsers.cnf.read, ".cnf")

    def test_h5(self):
        """Test parsers.h5.read."""
        self.run_parser(bq.parsers.h5.read, ".hdf5")


@pytest.mark.plottest
class TestParsersSpectrumPlot(object):
    """Test spectrum file parsers and plot the spectra."""

    def run_parser(self, read_fn, extension, write=False):
        """Run the test for the given class and file extension."""
        try:
            plt.figure()
        except Exception:
            # TclError on CI bc no display. skip the test
            return
        plt.title("Testing " + extension)
        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            path, fname = os.path.split(fname)
            print("")
            print(filename)
            data, cal = read_fn(filename)
            spec = bq.Spectrum(**data)
            if cal is not None:
                spec.apply_calibration(cal)
            print(spec)
            plt.semilogy(
                spec.energies,
                spec.data / spec.energy_bin_widths / spec.livetime,
                label=fname,
            )
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts/keV/sec")
            plt.xlim(0, 2800)
            if write:
                writename = os.path.join(".", fname + "_copy" + ext)
                spec.write(writename)
                os.remove(writename)
        plt.legend(prop={"size": 8})
        plt.show()

    def test_spe(self):
        """Test parsers.spe.read."""
        self.run_parser(bq.parsers.spe.read, ".spe")

    def test_spc(self):
        """Test parsers.spc.read."""
        self.run_parser(bq.parsers.spc.read, ".spc")

    def test_cnf(self):
        """Test parsers.cnf.read."""
        self.run_parser(bq.parsers.cnf.read, ".cnf")

    def test_h5(self):
        """Test parsers.h5.read."""
        self.run_parser(bq.parsers.h5.read, ".hdf5")
