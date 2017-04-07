"""Test becquerel's Spectrum."""

from __future__ import print_function
import pytest
import numpy as np
from uncertainties import ufloat, UFloat, unumpy

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
            spec = bq.core.Spectrum.from_file(filename)
            assert spec.livetime is not None

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

    def test_livetime(self, spec_data):
        """Test manual livetime input."""

        lt = 86400
        spec = bq.core.Spectrum(spec_data, livetime=lt)
        assert spec.livetime == lt

        lt = 300.6
        spec = bq.core.Spectrum(spec_data, livetime=lt)
        assert spec.livetime == lt


class TestUncertainties(object):
    """Test uncertainties functionality in Spectrum"""

    def test_construct_float_int(self, spec_data):
        """Construct spectrum with non-UFloats (float and int)."""

        spec = bq.core.Spectrum(spec_data)
        assert isinstance(spec.data[0], UFloat)
        spec = bq.core.Spectrum(spec_data.astype(float))
        assert isinstance(spec.data[0], UFloat)

    def test_construct_ufloat(self, spec_data):
        """Construct spectrum with UFloats"""

        udata = unumpy.uarray(spec_data, np.ones_like(spec_data))
        spec = bq.core.Spectrum(udata)
        assert isinstance(spec.data[0], UFloat)
        assert spec.data[0].std_dev == 1

    def test_construct_float_int_uncs(self, spec_data):
        """Construct spectrum with non-UFloats and specify uncs."""

        uncs = np.ones_like(spec_data)
        spec = bq.core.Spectrum(spec_data, uncs=uncs)
        assert isinstance(spec.data[0], UFloat)
        uncs2 = np.array([c.std_dev for c in spec.data])
        assert np.allclose(uncs2, 1)

    def test_construct_errors(self, spec_data):
        """Construct spectrum with UFloats plus uncs and get an error."""

        uncs = np.ones_like(spec_data)
        udata = unumpy.uarray(spec_data, uncs)
        with pytest.raises(bq.core.SpectrumError):
            bq.core.Spectrum(udata, uncs=uncs)

        udata[0] = 1
        with pytest.raises(bq.core.SpectrumError):
            bq.core.Spectrum(udata)

    def test_properties(self, spec_data):
        """Test data_vals and data_uncs."""

        spec = bq.core.Spectrum(spec_data)
        assert isinstance(spec.data[0], UFloat)
        assert np.allclose(spec.data_vals, spec_data)
        expected_uncs = np.sqrt(spec_data)
        expected_uncs[expected_uncs == 0] = 1
        assert np.allclose(spec.data_uncs, expected_uncs)

        uncs = spec_data
        udata = unumpy.uarray(spec_data, uncs)
        spec = bq.core.Spectrum(udata)
        assert np.allclose(spec.data_vals, spec_data)
        assert np.allclose(spec.data_uncs, uncs)

        uncs = np.ones_like(spec_data)
        spec = bq.core.Spectrum(spec_data, uncs=uncs)
        assert np.allclose(spec.data_uncs, uncs)


class TestSpectrumAddSubtract(object):
    """Test addition and subtraction of spectra"""

    def test_uncal_add_sub(self, uncal_spec, uncal_spec_2):
        """Test basic addition/subtraction of uncalibrated spectra"""

        tot = uncal_spec + uncal_spec_2
        uncs = np.sqrt(uncal_spec.data_uncs**2 + uncal_spec_2.data_uncs**2)
        assert np.all(
            tot.data == uncal_spec.data + uncal_spec_2.data)
        assert np.all(
            tot.data_vals == uncal_spec.data_vals + uncal_spec_2.data_vals)
        assert np.allclose(tot.data_uncs, uncs)
        diff = uncal_spec - uncal_spec_2
        assert np.all(
            diff.data == uncal_spec.data - uncal_spec_2.data)
        assert np.all(
            diff.data_vals == uncal_spec.data_vals - uncal_spec_2.data_vals)
        assert np.allclose(tot.data_uncs, uncs)

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

    def test_norm_subtract(self, uncal_spec, uncal_spec_2):
        """Test Spectrum.norm_subtract method (set livetime manually)"""

        livetime1 = 300.
        livetime2 = 600.
        uncal_spec.livetime = livetime1
        uncal_spec_2.livetime = livetime2
        spec3 = uncal_spec.norm_subtract(uncal_spec_2)
        np.testing.assert_allclose(
            spec3.data_vals, uncal_spec.data_vals -
            (livetime1 / livetime2) * uncal_spec_2.data_vals)


class TestSpectrumMultiplyDivide(object):
    """Test multiplication and division of spectra"""

    def test_uncal_mul_div(self, uncal_spec):
        """
        Basic multiplication/division of uncalibrated spectrum by a scalar.
        """

        doubled = uncal_spec * 2
        assert np.all(doubled.data == 2 * uncal_spec.data)
        assert np.all(doubled.data_vals == 2 * uncal_spec.data_vals)
        assert np.all(doubled.data_uncs == 2 * uncal_spec.data_uncs)
        halved = uncal_spec / 2
        assert np.all(halved.data == uncal_spec.data / 2.0)
        assert np.allclose(halved.data_vals, uncal_spec.data_vals / 2.0)
        assert np.allclose(halved.data_uncs, uncal_spec.data_uncs / 2.0)

    def test_cal_mul_div(self, cal_spec):
        """Basic multiplication/division of calibrated spectrum by a scalar."""

        doubled = cal_spec * 2
        assert np.all(doubled.data == 2 * cal_spec.data)
        assert np.all(doubled.data_vals == 2 * cal_spec.data_vals)
        assert np.all(doubled.data_uncs == 2 * cal_spec.data_uncs)
        halved = cal_spec / 2
        assert np.all(halved.data == cal_spec.data / 2.0)
        assert np.allclose(halved.data_vals, cal_spec.data_vals / 2.0)
        assert np.allclose(halved.data_uncs, cal_spec.data_uncs / 2.0)
        halved_again = cal_spec * 0.5
        assert np.all(halved_again.data == cal_spec.data * 0.5)
        assert np.allclose(halved.data_vals, cal_spec.data_vals * 0.5)
        assert np.allclose(halved.data_uncs, cal_spec.data_uncs * 0.5)

    def test_uncal_mul_div_uncertainties(self, uncal_spec):
        """
        Multiplication/division of uncal spectrum by a scalar with uncertainty.
        """

        factor = ufloat(2, 0.1)
        doubled = uncal_spec * factor
        assert np.all(doubled.data_vals == 2 * uncal_spec.data_vals)
        assert np.all(doubled.data_uncs >= 2 * uncal_spec.data_uncs)
        halved = uncal_spec / factor
        assert np.allclose(halved.data_vals, uncal_spec.data_vals / 2.0)
        assert np.all(halved.data_uncs >= uncal_spec.data_uncs / 2.0)

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
