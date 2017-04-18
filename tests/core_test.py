"""Test becquerel's Spectrum."""

from __future__ import print_function
import pytest
import numpy as np
import matplotlib.pyplot as plt

import becquerel as bq

from parsers_test import SAMPLES

TEST_DATA_LENGTH = 256
TEST_COUNTS = 4
TEST_GAIN = 8.23
TEST_EDGES_KEV = np.arange(TEST_DATA_LENGTH + 1) * TEST_GAIN


@pytest.fixture
def spec_data():
    """Build a vector of random counts."""

    floatdata = np.random.poisson(lam=TEST_COUNTS, size=TEST_DATA_LENGTH)
    return floatdata.astype(np.int)


@pytest.fixture
def uncal_spec(spec_data):
    """Generate an uncalibrated spectrum."""

    return bq.core.Spectrum(spec_data)


@pytest.fixture
def uncal_spec_2(spec_data):
    """Generate an uncalibrated spectrum (2nd instance)."""

    return bq.core.Spectrum(spec_data)


@pytest.fixture
def uncal_spec_long(spec_data):
    """Generate an uncalibrated spectrum, of longer length."""

    floatdata = np.random.poisson(lam=TEST_COUNTS, size=TEST_DATA_LENGTH * 2)
    uncal = bq.core.Spectrum(floatdata.astype(np.int))
    return uncal


@pytest.fixture
def cal_spec(spec_data):
    """Generate a calibrated spectrum."""

    return bq.core.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


@pytest.fixture
def cal_spec_2(spec_data):
    """Generate a calibrated spectrum (2nd instance)."""

    return bq.core.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


class TestSpectrumFromFile(object):
    """Test Spectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
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


class TestSpectrumConstructor(object):
    """Test Spectrum.__init__()."""

    def test_uncal(self, uncal_spec):
        """Test simple uncalibrated construction."""

        assert len(uncal_spec.data) == TEST_DATA_LENGTH
        assert not uncal_spec.is_calibrated

    def test_cal(self, cal_spec):
        """Test simple calibrated construction."""

        assert len(cal_spec.data) == TEST_DATA_LENGTH
        assert len(cal_spec.bin_edges_kev) == TEST_DATA_LENGTH + 1
        assert len(cal_spec.energies_kev) == TEST_DATA_LENGTH
        assert cal_spec.is_calibrated

    def test_init_exceptions(self, spec_data):
        """Test errors on initialization."""

        with pytest.raises(bq.core.SpectrumError):
            bq.core.Spectrum([])
        with pytest.raises(bq.core.SpectrumError):
            bq.core.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV[:-1])

        bad_edges = TEST_EDGES_KEV.copy()
        bad_edges[12] = bad_edges[9]
        with pytest.raises(bq.core.SpectrumError):
            bq.core.Spectrum(spec_data, bin_edges_kev=bad_edges)

    def test_uncalibrated_exception(self, uncal_spec):
        """Test UncalibratedError."""

        with pytest.raises(bq.core.UncalibratedError):
            uncal_spec.energies_kev


class TestSpectrumAddSubtract(object):
    """Test addition and subtraction of spectra"""

    def test_uncal_add_sub(self, uncal_spec, uncal_spec_2):
        """Test basic addition/subtraction of uncalibrated spectra"""

        tot = uncal_spec + uncal_spec_2
        assert np.all(tot.data == uncal_spec.data + uncal_spec_2.data)
        diff = uncal_spec - uncal_spec_2
        assert np.all(diff.data == uncal_spec.data - uncal_spec_2.data)

    def test_cal_uncal_add_sub(self, uncal_spec, cal_spec):
        """Test basic addition of a calibrated with an uncalibrated spectrum.

        NOTE: not implemented yet - so check that it errors.
        """

        with pytest.raises(NotImplementedError):
            uncal_spec + cal_spec
        with pytest.raises(NotImplementedError):
            uncal_spec - cal_spec

    def test_cal_add_sub(self, cal_spec, cal_spec_2):
        """Test basic addition of calibrated spectra.

        NOTE: not implemented yet - so check that it errors.
        """

        with pytest.raises(NotImplementedError):
            cal_spec + cal_spec_2
        with pytest.raises(NotImplementedError):
            cal_spec - cal_spec_2

    def test_add_sub_type_error(self, uncal_spec, spec_data):
        """Check that adding/subtracting a non-Spectrum gives a TypeError."""

        with pytest.raises(TypeError):
            uncal_spec + 5
        with pytest.raises(TypeError):
            uncal_spec - 5
        with pytest.raises(TypeError):
            uncal_spec + 'asdf'
        with pytest.raises(TypeError):
            uncal_spec - 'asdf'
        with pytest.raises(TypeError):
            uncal_spec + spec_data
        with pytest.raises(TypeError):
            uncal_spec - spec_data

    def test_add_sub_wrong_length(self, uncal_spec, uncal_spec_long):
        """
        Adding/subtracting spectra of different lengths gives a SpectrumError.
        """

        with pytest.raises(bq.core.SpectrumError):
            uncal_spec + uncal_spec_long
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec - uncal_spec_long


class TestSpectrumMultiplyDivide(object):
    """Test multiplication and division of spectra"""

    def test_uncal_mul_div(self, uncal_spec):
        """
        Basic multiplication/division of uncalibrated spectrum by a scalar.
        """

        doubled = uncal_spec * 2
        assert np.all(doubled.data == 2 * uncal_spec.data)
        halved = uncal_spec / 2
        assert np.all(halved.data == uncal_spec.data / 2.0)

    def test_cal_mul_div(self, cal_spec):
        """Basic multiplication/division of calibrated spectrum by a scalar."""

        doubled = cal_spec * 2
        assert np.all(doubled.data == 2 * cal_spec.data)
        halved = cal_spec / 2
        assert np.all(halved.data == cal_spec.data / 2.0)
        halved_again = cal_spec * 0.5
        assert np.all(halved_again.data == cal_spec.data * 0.5)

    def test_mul_div_type_error(self, uncal_spec, spec_data):
        """Multiplication/division with a non-scalar gives a TypeError."""

        with pytest.raises(TypeError):
            uncal_spec * uncal_spec
        with pytest.raises(TypeError):
            uncal_spec / uncal_spec
        with pytest.raises(TypeError):
            uncal_spec * 'asdf'
        with pytest.raises(TypeError):
            uncal_spec / 'asdf'
        with pytest.raises(TypeError):
            uncal_spec * spec_data
        with pytest.raises(TypeError):
            uncal_spec / spec_data

    def test_mul_div_bad_factor(self, uncal_spec):
        """Multiplication/division with zero/inf/nan gives a SpectrumError."""

        with pytest.raises(bq.core.SpectrumError):
            uncal_spec * 0
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec / 0
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec * np.inf
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec / np.inf
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec * np.nan
        with pytest.raises(bq.core.SpectrumError):
            uncal_spec / np.nan


@pytest.fixture(params=[
    np.linspace(0, 3000, 3001),
    np.linspace(0, 3000, 2230),
    np.linspace(0, 3000, 7777)],
                ids=[
    "1 keV bins",
    "slightly larger bins",
    "slightly smaller bins"])
def old_edges(request):
    return request.param


@pytest.fixture(params=[
    np.linspace(0, 3000, 3001),
    np.linspace(0, 3000, 39000),
    np.linspace(0, 3000, 17),
    np.linspace(-6, 3002, 2222),
    np.linspace(-0.3, 3000, 256)],
                ids=[
    "1 keV bins",
    "small bins",
    "large bins",
    "medium bins larger range",
    "large bins slightly larger range"])
def new_edges(request):
    return request.param


@pytest.fixture(params=[1, 50, 12555],
                ids=["sparse counts", "medium counts", "high counts"])
def lam(request):
    return request.param


class TestRebin(object):
    """Tests for core.rebin()"""

    def test_rebin_counts(self, lam, old_edges, new_edges):
        """Check total counts in spectrum data before and after rebin"""

        old_counts = np.random.poisson(lam=lam, size=len(old_edges) - 1)
        new_counts = bq.core.rebin(old_counts, old_edges, new_edges)
        assert np.isclose(old_counts.sum(), new_counts.sum())

    @pytest.mark.plottest
    def test_uncal_spectrum_counts(self, uncal_spec):
        """Plot the old and new spectrum bins as a sanity check"""

        old_edges = np.concatenate([
            uncal_spec.channels.astype('float') - 0.5,
            np.array([uncal_spec.channels[-1] + 0.5])])
        new_edges = old_edges + 0.3
        new_data = bq.core.rebin(uncal_spec.data, old_edges, new_edges)
        plt.figure()
        plt.plot(*bq.core.bin_edges_and_heights_to_steps(old_edges,
                                                         uncal_spec.data),
                 color='dodgerblue', label='original')
        plt.plot(*bq.core.bin_edges_and_heights_to_steps(new_edges,
                                                         new_data),
                 color='firebrick', label='rebinned')
        plt.show()
