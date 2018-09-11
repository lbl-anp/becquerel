"""Test becquerel's Spectrum."""

from __future__ import print_function
import pytest
import datetime
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

    return bq.Spectrum(spec_data)


@pytest.fixture
def uncal_spec_2(spec_data):
    """Generate an uncalibrated spectrum (2nd instance)."""

    return bq.Spectrum(spec_data)


@pytest.fixture
def uncal_spec_cps(spec_data):
    """Generate an uncalibrated spectrum with cps data."""

    return bq.Spectrum(cps=spec_data)


@pytest.fixture
def uncal_spec_long(spec_data):
    """Generate an uncalibrated spectrum, of longer length."""

    floatdata = np.random.poisson(lam=TEST_COUNTS, size=TEST_DATA_LENGTH * 2)
    uncal = bq.Spectrum(floatdata.astype(np.int))
    return uncal


@pytest.fixture
def cal_spec(spec_data):
    """Generate a calibrated spectrum."""

    return bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


@pytest.fixture
def cal_spec_2(spec_data):
    """Generate a calibrated spectrum (2nd instance)."""

    return bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


class TestSpectrumFromFile(object):
    """Test Spectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""

        filenames = SAMPLES.get(extension, [])
        assert len(filenames) >= 1
        for filename in filenames:
            spec = bq.Spectrum.from_file(filename)
            assert spec.livetime is not None

    def test_spe(self):
        """Test Spectrum.from_file for SPE file........................."""
        with pytest.warns(UserWarning):
            self.run_from_file('.spe')

    def test_spc(self):
        """Test Spectrum.from_file for SPC file........................."""

        self.run_from_file('.spc')

    def test_cnf(self):
        """Test Spectrum.from_file for CNF file........................."""

        self.run_from_file('.cnf')

    def test_error(self):
        """Test _get_file_object() raises error for bad file type"""

        with pytest.raises(NotImplementedError):
            bq.Spectrum.from_file('foo.bar')


# ----------------------------------------------
#         Test Spectrum.__init__()
# ----------------------------------------------

def test_uncal(uncal_spec):
    """Test simple uncalibrated construction."""

    assert len(uncal_spec.counts) == TEST_DATA_LENGTH
    assert not uncal_spec.is_calibrated


def test_uncal_cps(uncal_spec_cps):
    """Test simple uncalibrated construction w CPS. More CPS tests later"""

    assert len(uncal_spec_cps.cps) == TEST_DATA_LENGTH
    assert not uncal_spec_cps.is_calibrated


def test_cal(cal_spec):
    """Test simple calibrated construction."""

    assert len(cal_spec.counts) == TEST_DATA_LENGTH
    assert len(cal_spec.bin_edges_kev) == TEST_DATA_LENGTH + 1
    assert len(cal_spec.energies_kev) == TEST_DATA_LENGTH
    assert cal_spec.is_calibrated


def test_init_exceptions(spec_data):
    """Test errors on initialization."""

    with pytest.raises(bq.SpectrumError):
        bq.Spectrum([])
    with pytest.raises(bq.SpectrumError):
        bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV[:-1])
    with pytest.raises(bq.SpectrumError):
        bq.Spectrum(spec_data, cps=spec_data)
    with pytest.raises(bq.SpectrumError):
        bq.Spectrum(bin_edges_kev=TEST_EDGES_KEV)

    bad_edges = TEST_EDGES_KEV.copy()
    bad_edges[12] = bad_edges[9]
    with pytest.raises(ValueError):
        bq.Spectrum(spec_data, bin_edges_kev=bad_edges)


def test_uncalibrated_exception(uncal_spec):
    """Test UncalibratedError."""

    with pytest.raises(bq.UncalibratedError):
        uncal_spec.energies_kev


# ----------------------------------------------
#      Test Spectrum livetime properties
# ----------------------------------------------

@pytest.fixture(params=[86400, 300.6, 0.88])
def livetime(request):
    return request.param


def test_livetime_arg(spec_data, livetime):
    """Test manual livetime input."""

    spec = bq.Spectrum(spec_data, livetime=livetime)
    assert spec.livetime == livetime


def test_livetime_arg_cps(spec_data, livetime):
    """Test manual livetime input with CPS."""

    cps = spec_data / float(livetime)
    spec = bq.Spectrum(cps=cps, livetime=livetime)
    assert spec.livetime == livetime


def test_no_livetime(spec_data):
    """Test livetime property when not specified."""

    spec = bq.Spectrum(spec_data)
    assert spec.livetime is None

    cps_spec = bq.Spectrum(cps=spec_data / 300.6)
    assert np.isnan(cps_spec.livetime)


# ----------------------------------------------
#     Test start_time, stop_time, realtime
# ----------------------------------------------

@pytest.mark.parametrize('start, stop', [
    (datetime.datetime(2017, 1, 1, 17, 0, 3),
     datetime.datetime(2017, 1, 1, 18, 0, 3)),
    ('2017-01-19 17:21:00', '2017-01-20 14:19:32'),
    (datetime.datetime(2017, 1, 1, 0, 30, 0, 385), '2017-01-01 12:44:22')
])
@pytest.mark.parametrize('rt', [3600, 2345.6])
def test_acqtime_construction(spec_data, start, stop, rt):
    """Test construction with 2 out of 3 of start, stop, and realtime."""

    bq.Spectrum(spec_data, start_time=start, stop_time=stop)
    bq.Spectrum(spec_data, start_time=start, realtime=rt)
    bq.Spectrum(spec_data, realtime=rt, stop_time=stop)


@pytest.mark.parametrize('start, stop, rt, expected_err', [
    ('2017-01-19 17:21:00', '2017-01-20 17:21:00', 86400, bq.SpectrumError),
    ('2017-01-19 17:21:00', '2017-01-18 17:21:00', None, ValueError),
])
def test_bad_acqtime_construction(spec_data, start, stop, rt, expected_err):
    """Test bad construction of a spectrum with start, stop, or realtimes."""

    with pytest.raises(expected_err):
        bq.Spectrum(spec_data, start_time=start, stop_time=stop, realtime=rt)


def test_bad_realtime_livetime(spec_data):
    """Test error of livetime > realtime."""

    with pytest.raises(ValueError):
        bq.Spectrum(spec_data, livetime=300, realtime=290)


# ----------------------------------------------
#         Test uncertainties in Spectrum
# ----------------------------------------------

def test_construct_float_int(spec_data):
    """Construct spectrum with non-UFloats (float and int)."""

    spec = bq.Spectrum(spec_data)
    assert isinstance(spec.counts[0], UFloat)
    spec = bq.Spectrum(spec_data.astype(float))
    assert isinstance(spec.counts[0], UFloat)


def test_construct_ufloat(spec_data):
    """Construct spectrum with UFloats"""

    ucounts = unumpy.uarray(spec_data, np.ones_like(spec_data))
    spec = bq.Spectrum(ucounts)
    assert isinstance(spec.counts[0], UFloat)
    assert spec.counts[0].std_dev == 1


def test_construct_float_int_uncs(spec_data):
    """Construct spectrum with non-UFloats and specify uncs."""

    uncs = np.ones_like(spec_data)
    spec = bq.Spectrum(spec_data, uncs=uncs)
    assert isinstance(spec.counts[0], UFloat)
    uncs2 = np.array([c.std_dev for c in spec.counts])
    assert np.allclose(uncs2, 1)


def test_construct_errors(spec_data):
    """Construct spectrum with UFloats plus uncs and get an error."""

    uncs = np.ones_like(spec_data)
    ucounts = unumpy.uarray(spec_data, uncs)
    with pytest.raises(bq.core.utils.UncertaintiesError):
        bq.Spectrum(ucounts, uncs=uncs)

    ucounts[0] = 1
    with pytest.raises(bq.core.utils.UncertaintiesError):
        bq.Spectrum(ucounts)


def test_properties(spec_data):
    """Test counts_vals and counts_uncs."""

    spec = bq.Spectrum(spec_data)
    assert isinstance(spec.counts[0], UFloat)
    assert np.allclose(spec.counts_vals, spec_data)
    expected_uncs = np.sqrt(spec_data)
    expected_uncs[expected_uncs == 0] = 1
    assert np.allclose(spec.counts_uncs, expected_uncs)

    uncs = spec_data
    ucounts = unumpy.uarray(spec_data, uncs)
    spec = bq.Spectrum(ucounts)
    assert np.allclose(spec.counts_vals, spec_data)
    assert np.allclose(spec.counts_uncs, uncs)

    uncs = np.ones_like(spec_data)
    spec = bq.Spectrum(spec_data, uncs=uncs)
    assert np.allclose(spec.counts_uncs, uncs)


# ----------------------------------------------
#         Test Spectrum.bin_widths
# ----------------------------------------------

def test_bin_widths(cal_spec):
    """Test Spectrum.bin_widths"""

    cal_spec.bin_widths
    assert len(cal_spec.bin_widths) == len(cal_spec.counts)
    assert np.allclose(cal_spec.bin_widths, TEST_GAIN)


def test_bin_width_error(uncal_spec):
    """Test Spectrum.bin_widths error"""

    with pytest.raises(bq.UncalibratedError):
        uncal_spec.bin_widths


# ----------------------------------------------
#         Test Spectrum CPS and CPS/keV
# ----------------------------------------------

@pytest.mark.parametrize('construction_kwargs', [
    {'livetime': 300.0},
    {'livetime': 300.0,
     'bin_edges_kev': TEST_EDGES_KEV},
])
def test_cps(spec_data, construction_kwargs):
    """Test cps property and uncertainties on uncal and cal spectrum."""

    spec = bq.Spectrum(spec_data, **construction_kwargs)
    spec.cps
    spec.cps_vals
    spec.cps_uncs
    assert np.all(spec.counts_vals == spec_data)
    assert np.allclose(spec.cps_vals, spec_data / spec.livetime)
    assert np.allclose(spec.cps_uncs, spec.counts_uncs / spec.livetime)


def test_cpskev(spec_data, livetime):
    """Test cpskev property and uncertainties"""

    spec = bq.Spectrum(spec_data, livetime=livetime,
                       bin_edges_kev=TEST_EDGES_KEV)
    spec.cpskev
    spec.cpskev_vals
    spec.cpskev_uncs
    assert np.allclose(
        spec.cpskev_vals, spec_data / spec.bin_widths / float(livetime))
    assert np.allclose(
        spec.cpskev_uncs, spec.counts_uncs / spec.bin_widths / float(livetime))


def test_cps_cpsspec(spec_data, livetime):
    """Test cps property of CPS-style spectrum."""

    spec = bq.Spectrum(cps=spec_data / float(livetime))
    spec.cps
    assert np.all(spec.cps_vals == spec_data / float(livetime))
    assert np.all(np.isnan(spec.cps_uncs))
    assert spec.counts is None
    assert spec.counts_vals is None
    assert spec.counts_uncs is None


def test_cps_errors(uncal_spec):
    """Test errors in CPS."""

    with pytest.raises(bq.SpectrumError):
        uncal_spec.cps


def test_cpskev_errors(spec_data):
    """Test errors in CPS/keV."""

    spec = bq.Spectrum(spec_data, livetime=300.9)
    with pytest.raises(bq.UncalibratedError):
        spec.cpskev


# ----------------------------------------------
#   Test addition and subtraction of spectra
# ----------------------------------------------

def get_spectrum(t, lt=None):
    """Get spectrum to use in parameterized tests.

    Pytest Note:
      one might think you could do:
        @pytest.mark.parametrize('spec1, spec2', [
            (uncal_spec, uncal_spec),
            (cal_spec, cal_spec)
        ])
        def test_add(spec1, spec2):
          ...

      but you can't put fixtures inside parametrize().
    """

    if t == 'uncal':
        spec = uncal_spec(spec_data())
    elif t == 'cal':
        spec = cal_spec(spec_data())
    elif t == 'cal_new':
        edges = np.arange(TEST_DATA_LENGTH + 1) * 0.67
        spec = bq.Spectrum(spec_data(), bin_edges_kev=edges)
    elif t == 'uncal_long':
        spec = uncal_spec_long(spec_data())
    elif t == 'uncal_cps':
        spec = uncal_spec_cps(spec_data())
    elif t == 'data':
        spec = spec_data()
    else:
        return t
    try:
        spec.livetime = lt
    except AttributeError:
        pass
    return spec


@pytest.mark.parametrize('lt1, lt2', [
    (300, 600),
    (12.6, 0.88),
    (300, 12.6),
    (12.6, None),
    (None, None)])
@pytest.mark.parametrize('type1, type2', [
    ('uncal', 'uncal'),
    ('cal', 'cal')])
def test_add(type1, type2, lt1, lt2):
    """Test addition of spectra"""

    spec1, spec2 = (get_spectrum(type1, lt=lt1),
                    get_spectrum(type2, lt=lt2))

    tot = spec1 + spec2
    assert np.all(tot.counts == spec1.counts + spec2.counts)
    assert np.all(tot.counts_vals ==
                  spec1.counts_vals + spec2.counts_vals)
    if lt1 and lt2:
        assert tot.livetime == lt1 + lt2
    else:
        assert tot.livetime is None


@pytest.mark.parametrize('type1, type2, expected_error', [
    ('uncal', 'cal', bq.SpectrumError),
    ('uncal', 'uncal_long', bq.SpectrumError),
    ('uncal', 5, TypeError),
    (5, 'cal', TypeError),
    ('cal', 'asdf', TypeError),
    ('asdf', 'uncal', TypeError),
    ('uncal', 'data', TypeError),
    ('cal', 'cal_new', NotImplementedError)
])
def test_add_sub_errors(type1, type2, expected_error):
    """Test addition and subtraction that causes errors"""

    spec1, spec2 = get_spectrum(type1), get_spectrum(type2)
    with pytest.raises(expected_error):
        spec1 + spec2
    with pytest.raises(expected_error):
        spec1 - spec2


@pytest.mark.parametrize('type1, type2', [
    ('uncal', 'uncal'),
    ('cal', 'cal')])
def test_add_uncs(type1, type2):
    """Test uncertainties on addition of uncal spectra"""

    spec1, spec2 = get_spectrum(type1), get_spectrum(type2)

    tot = spec1 + spec2
    uncs = np.sqrt(spec1.counts_uncs**2 + spec2.counts_uncs**2)
    assert np.allclose(tot.counts_uncs, uncs)


@pytest.mark.parametrize('type1, type2, lt1, lt2', [
    ('uncal_cps', 'uncal_cps', 300, 12.6),
    ('uncal_cps', 'uncal_cps', None, 12.6),
    ('uncal_cps', 'uncal_cps', None, None),
    ('uncal_cps', 'uncal', None, 300)])
def test_add_sub_cps(type1, type2, lt1, lt2):
    """Test addition and subtraction of CPS spectra"""

    spec1, spec2 = (get_spectrum(type1, lt=lt1),
                    get_spectrum(type2, lt=lt2))

    tot = spec1 + spec2
    assert tot.counts is None
    assert np.all(tot.cps_vals == spec1.cps_vals + spec2.cps_vals)
    assert np.isnan(tot.livetime)

    diff = spec1 - spec2
    assert diff.counts is None
    assert np.allclose(diff.cps_vals, spec1.cps_vals - spec2.cps_vals)


@pytest.mark.parametrize('lt1, lt2', [
    (300, 600),
    (12.6, 0.88),
    (300, 12.6)])
@pytest.mark.parametrize('type1, type2', [
    ('uncal', 'uncal'),
    ('cal', 'cal')])
def test_subtract(type1, type2, lt1, lt2):
    """Test Spectrum subtraction"""

    spec1, spec2 = (get_spectrum(type1, lt=lt1),
                    get_spectrum(type2, lt=lt2))
    diff = spec1 - spec2
    assert diff.counts is None
    assert np.allclose(diff.cps_vals, spec1.cps_vals - spec2.cps_vals)
    assert np.all(diff.cps_uncs > spec1.cps_uncs)
    assert np.all(diff.cps_uncs > spec2.cps_uncs)


@pytest.mark.parametrize('type1, type2, lt1, lt2', [
    ('uncal', 'uncal', 300, None),
    ('cal', 'cal', None, None)])
def test_subtract_errors(type1, type2, lt1, lt2):
    """Test errors in Spectrum subtract with no livetime"""

    spec1, spec2 = (get_spectrum(type1, lt=lt1),
                    get_spectrum(type2, lt=lt2))
    with pytest.raises(bq.SpectrumError):
        spec1 - spec2


# ----------------------------------------------
#  Test multiplication and division of spectra
# ----------------------------------------------

@pytest.mark.parametrize('factor', [0.88, 1, 2, 43.6])
@pytest.mark.parametrize('spectype', ['uncal', 'cal'])
def test_basic_mul_div(spectype, factor):
    """
    Basic multiplication/division of uncalibrated spectrum by a scalar.
    """

    spec = get_spectrum(spectype)

    mult = spec * factor
    assert np.allclose(mult.counts_vals, factor * spec.counts_vals)
    assert np.allclose(mult.counts_uncs, factor * spec.counts_uncs)
    assert mult.livetime is None
    div = spec / factor
    assert np.allclose(div.counts_vals, spec.counts_vals / factor)
    assert np.allclose(div.counts_uncs, spec.counts_uncs / factor)
    assert div.livetime is None


@pytest.mark.parametrize('factor', [0.88, 1, 2, 43.6])
def test_cps_mul_div(uncal_spec_cps, factor):
    """Multiplication/division of a CPS spectrum."""

    mult = uncal_spec_cps * factor
    assert np.allclose(mult.cps_vals, factor * uncal_spec_cps.cps_vals)
    assert np.isnan(mult.livetime)
    div = uncal_spec_cps / factor
    assert np.allclose(div.cps_vals, uncal_spec_cps.cps_vals / factor)
    assert np.isnan(div.livetime)


@pytest.mark.parametrize('factor', [
    ufloat(0.88, 0.01),
    ufloat(1, 0.1),
    ufloat(43, 1)])
@pytest.mark.parametrize('spectype', ['uncal', 'cal'])
def test_uncal_mul_div_uncertainties(spectype, factor):
    """
    Multiplication/division of uncal spectrum by a scalar with uncertainty.
    """

    spec = get_spectrum(spectype)

    mult = spec * factor
    assert np.allclose(
        mult.counts_vals, factor.nominal_value * spec.counts_vals)
    assert np.all(
        (mult.counts_uncs > factor.nominal_value * spec.counts_uncs) |
        (spec.counts_vals == 0))
    assert mult.livetime is None
    div = spec / factor
    assert np.allclose(
        div.counts_vals, spec.counts_vals / factor.nominal_value)
    assert np.all(
        (div.counts_uncs > spec.counts_uncs / factor.nominal_value) |
        (spec.counts_vals == 0))
    assert div.livetime is None


@pytest.mark.parametrize('type1, type2, error', [
    ('uncal', 'uncal', TypeError),
    ('uncal', 'asdf', TypeError),
    ('uncal', 'data', TypeError),
    ('uncal', 0, ValueError),
    ('uncal', np.inf, ValueError),
    ('uncal', np.nan, ValueError),
    ('uncal', ufloat(0, 1), ValueError),
    ('uncal', ufloat(np.inf, np.nan), ValueError)])
def test_mul_div_errors(type1, type2, error):
    """Multiplication/division errors."""

    spec, bad_factor = get_spectrum(type1), get_spectrum(type2)

    with pytest.raises(error):
        spec * bad_factor
    with pytest.raises(error):
        spec / bad_factor


# ----------------------------------------------
#         Test Spectrum.calibrate_like
# ----------------------------------------------

def test_calibrate_like(uncal_spec, cal_spec):
    """Test calibrate_like with an uncalibrated spectrum."""

    uncal_spec.calibrate_like(cal_spec)
    assert uncal_spec.is_calibrated
    assert np.all(uncal_spec.bin_edges_kev == cal_spec.bin_edges_kev)


def test_recalibrate_like(cal_spec):
    """Test calibrate_like with an already calibrated spectrum."""

    cal_new = get_spectrum('cal_new')
    edges1 = cal_spec.bin_edges_kev
    cal_spec.calibrate_like(cal_new)
    assert cal_spec.is_calibrated
    assert np.all(cal_spec.bin_edges_kev == cal_new.bin_edges_kev)
    assert cal_spec.bin_edges_kev[-1] != edges1[-1]


def test_calibrate_like_error(uncal_spec, uncal_spec_2):
    """Test that calibrate_like raises an error if arg is uncalibrated"""

    with pytest.raises(bq.UncalibratedError):
        uncal_spec.calibrate_like(uncal_spec_2)


def test_calibrate_like_copy(uncal_spec, cal_spec):
    """Test that calibrate_like makes a copy of the bin edges"""

    uncal_spec.calibrate_like(cal_spec)
    assert uncal_spec.bin_edges_kev is not cal_spec.bin_edges_kev
    cal_spec.rm_calibration()
    assert uncal_spec.is_calibrated


# ----------------------------------------------
#         Test Spectrum.combine_bins
# ----------------------------------------------

@pytest.mark.parametrize('spectype', ['uncal', 'cal', 'uncal_cps'])
def test_combine_bins(spectype):
    """Test combine_bins with no padding."""

    spec = get_spectrum(spectype)

    f = 8
    combined = spec.combine_bins(f)

    assert len(combined) == TEST_DATA_LENGTH / f
    if spec.counts is not None:
        assert combined.counts_vals[0] == np.sum(spec.counts_vals[:f])
        assert np.sum(combined.counts_vals) == np.sum(spec.counts_vals)
    else:
        assert combined.cps_vals[0] == np.sum(spec.cps_vals[:f])
        assert np.sum(combined.cps_vals) == np.sum(spec.cps_vals)


@pytest.mark.parametrize('spectype', ['uncal', 'cal', 'uncal_cps'])
def test_combine_bins_padding(spectype):
    """Test combine_bins with padding (an uneven factor)."""

    spec = get_spectrum(spectype)

    f = 10
    combined = spec.combine_bins(f)
    assert len(combined) == np.ceil(float(TEST_DATA_LENGTH) / f)
    if spec.counts is not None:
        assert combined.counts_vals[0] == np.sum(spec.counts_vals[:f])
        assert np.sum(combined.counts_vals) == np.sum(spec.counts_vals)
    else:
        assert combined.cps_vals[0] == np.sum(spec.cps_vals[:f])
        assert np.sum(combined.cps_vals) == np.sum(spec.cps_vals)


# calibration methods tested in energycal_test.py

# ----------------------------------------------
#         Test Spectrum.downsample
# ----------------------------------------------

@pytest.fixture
def many_counts_data():
    floatdata = np.random.poisson(lam=1000, size=TEST_DATA_LENGTH)
    return floatdata.astype(int)


@pytest.mark.parametrize('spec, f', [
    (uncal_spec(many_counts_data()), 2),
    (cal_spec(many_counts_data()), 2),
    (uncal_spec(many_counts_data()), 1.5),
    (cal_spec(many_counts_data()), 1.5),
    (uncal_spec(many_counts_data()), 999.99),
    (cal_spec(many_counts_data()), 999.99)
])
def test_downsample(spec, f):
    """Test Spectrum.downsample on uncalibrated and calibrated spectra"""

    s1 = np.sum(spec.counts_vals)
    spec2 = spec.downsample(f)
    s2 = np.sum(spec2.counts_vals)
    r = float(s2) / s1
    five_sigma = 5 * np.sqrt(s1 / f) / (s1 / f)

    assert np.isclose(r, 1.0 / f, atol=five_sigma)


def test_no_downsample(cal_spec):
    """Test that downsample(1) doesn't do anything"""

    s1 = np.sum(cal_spec.counts_vals)
    spec2 = cal_spec.downsample(1.0)
    s2 = np.sum(spec2.counts_vals)
    assert s1 == s2


def test_zero_downsample(cal_spec):
    """Test that downsample(very large number) gives 0"""

    spec2 = cal_spec.downsample(10**10)
    s2 = np.sum(spec2.counts_vals)
    assert s2 == 0


def test_downsample_handle_livetime(cal_spec):
    """Test handle_livetime behavior"""

    f = 2
    test_livetime = 300.0
    cal_spec.livetime = test_livetime

    spec2 = cal_spec.downsample(f)
    assert spec2.livetime is None

    spec3 = cal_spec.downsample(f, handle_livetime='preserve')
    assert spec3.livetime == cal_spec.livetime

    spec4 = cal_spec.downsample(f, handle_livetime='reduce')
    assert spec4.livetime == cal_spec.livetime / f


def test_downsample_error(cal_spec):
    """Test that downsample(<1) raises ValueError"""

    with pytest.raises(ValueError):
        cal_spec.downsample(0.5)


def test_downsample_cps_error(uncal_spec_cps):
    """Test that downsampling a CPS spectrum gives a SpectrumError"""

    with pytest.raises(bq.SpectrumError):
        uncal_spec_cps.downsample(12)


def test_downsample_handle_livetime_error(uncal_spec):
    """Test bad value of handle_livetime"""

    with pytest.raises(ValueError):
        uncal_spec.downsample(5, handle_livetime='asdf')


# ----------------------------------------------
#         Test Spectrum.__len__
# ----------------------------------------------


@pytest.fixture(params=[1, 8, 256, 16384])
def length(request):
    return request.param


def test_len(length):
    """Test len(spectrum)"""

    floatdata = np.random.poisson(lam=TEST_COUNTS, size=length)
    spec = bq.Spectrum(floatdata.astype(int))
    assert len(spec) == length


def test_len_cps(length, livetime):
    """Test len(spectrum) for a CPS-based spectrum"""

    floatdata = np.random.poisson(lam=TEST_COUNTS, size=length)
    spec = bq.Spectrum(cps=floatdata / livetime)
    assert len(spec) == length


# ----------------------------------------------
#         Test Spectrum.copy
# ----------------------------------------------

def test_copy_uncal(uncal_spec):
    """Test copy method on uncal spectrum"""

    uncal2 = uncal_spec.copy()
    assert np.all(uncal2.counts_vals == uncal_spec.counts_vals)
    assert np.all(uncal2.counts_uncs == uncal_spec.counts_uncs)
    assert uncal2 is not uncal_spec
    assert uncal2.counts is not uncal_spec.counts
    assert uncal2.counts[0] is not uncal_spec.counts[0]


def test_copy_cal(cal_spec):
    """Test copy method on cal spectrum"""

    cal2 = cal_spec.copy()
    assert np.all(cal2.counts_vals == cal_spec.counts_vals)
    assert np.all(cal2.counts_uncs == cal_spec.counts_uncs)
    assert np.all(cal2.bin_edges_kev == cal_spec.bin_edges_kev)
    assert cal2 is not cal_spec
    assert cal2.counts is not cal_spec.counts
    assert cal2.counts[0] is not cal_spec.counts[0]
    assert cal2.bin_edges_kev is not cal_spec.bin_edges_kev
