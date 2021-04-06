"""Test becquerel spectrum file parsers."""

import glob
import os
import pytest
import matplotlib.pyplot as plt
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "samples")
SAMPLES = {}
for extension in [".spe", ".spc", ".cnf"]:
    filenames = glob.glob(os.path.join(SAMPLES_PATH + "*", "*.*"))
    filenames_filtered = []
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        if ext.lower() == extension:
            filenames_filtered.append(filename)
    SAMPLES[extension] = filenames_filtered


class TestSpectrumFile:
    """Test spectrum file parsers."""

    def run_parser(self, cls, extension, write=False):
        """Run the test for the given class and file extension."""
        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            path, fname = os.path.split(fname)
            print("")
            print(filename)
            spec = cls(filename)
            print(spec)
            if write:
                writename = os.path.join(".", fname + "_copy" + ext)
                spec.write(writename)
                os.remove(writename)

    def test_spe(self):
        """Test parsers.SpeFile............................................"""
        with pytest.warns(bq.parsers.SpectrumFileParsingWarning):
            self.run_parser(bq.parsers.SpeFile, ".spe", write=True)

    def test_spc(self):
        """Test parsers.SpcFile............................................"""
        self.run_parser(bq.parsers.SpcFile, ".spc", write=False)

    def test_cnf(self):
        """Test parsers.CnfFile............................................"""
        self.run_parser(bq.parsers.CnfFile, ".cnf", write=False)


@pytest.mark.plottest
class TestSpectrumFilePlot:
    """Test spectrum file parsers and plot the spectra."""

    def run_parser(self, cls, extension, write=False):
        """Run the test for the given class and file extension."""
        try:
            plt.figure()
        except Exception:
            # TclError on CI bc no display. skip the test
            return
        plt.title("Testing " + cls.__name__)
        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            path, fname = os.path.split(fname)
            print("")
            print(filename)
            spec = cls(filename)
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
        """Test parsers.SpeFile............................................"""
        with pytest.warns(bq.parsers.SpectrumFileParsingWarning):
            self.run_parser(bq.parsers.SpeFile, ".spe", write=True)

    def test_spc(self):
        """Test parsers.SpcFile............................................"""
        self.run_parser(bq.parsers.SpcFile, ".spc", write=False)

    def test_cnf(self):
        """Test parsers.CnfFile............................................"""
        self.run_parser(bq.parsers.CnfFile, ".cnf", write=False)
