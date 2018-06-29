"""Test energycal.py"""

from __future__ import print_function
import pytest
import numpy as np

import becquerel as bq

TEST_DATA_LENGTH = 256
TEST_COUNTS = 4
TEST_GAIN = 8.23
TEST_EDGES_KEV = np.arange(TEST_DATA_LENGTH + 1) * TEST_GAIN


@pytest.fixture(params=[0.37, 3.7, 1, 2])
def slope(request):
    return request.param


@pytest.fixture(params=[-4, 0, 1.1])
def offset(request):
    return request.param


@pytest.fixture(params=[
    (32, 67, 115),
    [32, 67, 115],
    np.array((32, 67, 115)),
    [31.7, 67.2, 115]
])
def chlist(request):
    return request.param


@pytest.fixture(params=[
    (661.66, 1460.83, 2614.5),
    [661.66, 1460.83, 2614.5],
    np.array((661.66, 1460.83, 2614.5)),
    (662, 1461, 2615)
])
def kevlist(request):
    return request.param


@pytest.fixture(params=[
    32, -5, 67.83, (34, 35), np.arange(113, 7678.3, 55.5)])
def channels(request):
    return request.param


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
def cal_spec(spec_data):
    """Generate a calibrated spectrum."""

    return bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)

@pytest.fixture
def linear_regression(x, y):
    """Perform a linear regression manually"""

    x = np.array(x)
    y = np.array(y)
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    b = (n*np.sum(x*y)-sx*sy)/(n*np.sum(x**2)-sx*sx)
    a = 1.0/n*(sy-b*sx)
    return [a, b]

# ----------------------------------------------------
#        Construction tests
# ----------------------------------------------------

def test_construction_empty():
    """Test empty construction"""

    bq.LinearEnergyCal()


def test_construction_chkevlist(chlist, kevlist):
    """Test construction (not fitting) from chlist, kevlist"""

    cal = bq.LinearEnergyCal.from_points(
        chlist=chlist, kevlist=kevlist)
    assert len(cal.channels) == len(chlist)
    assert len(cal.energies) == len(chlist)
    assert len(cal.calpoints) == len(chlist)
    assert len(cal.calpoints[0]) == 2


def test_construction_coefficients(slope, offset):
    """Test construction from coefficients"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.coeffs['b'] == slope
    assert cal.coeffs['c'] == offset


def test_construction_bad_coefficients(slope, offset):
    """Test construction with bad coefficients"""

    coeffs = {'a': offset, 'b': slope}
    with pytest.raises(bq.EnergyCalError):
        bq.LinearEnergyCal.from_coeffs(coeffs)


def test_construction_bad_points(chlist, kevlist):
    """Test from_points with bad input"""

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(
            chlist=chlist, kevlist=kevlist[:-1])
    excinfo.match('Channels and energies must be same length')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(chlist=32, kevlist=661.7)
    excinfo.match('Inputs should be vector iterables, not scalars')


# ----------------------------------------------------
#        other EnergyCal method tests
# ----------------------------------------------------

def test_methods_add_calpoint():
    """Test add_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.add_calpoint(32, 661.7)
    cal.add_calpoint(67, 1460.83)
    cal.add_calpoint(35, 661.7)
    assert len(cal.calpoints) == 2
    assert len(cal.channels) == 2
    assert len(cal.energies) == 2


def test_methods_new_calpoint():
    """Test new_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.new_calpoint(67, 1460.83)
    with pytest.raises(bq.EnergyCalError):
        cal.new_calpoint(35, 661.7)
    assert len(cal.calpoints) == 2
    assert len(cal.channels) == 2
    assert len(cal.energies) == 2


def test_methods_rm_calpoint():
    """Test rm_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.rm_calpoint(661.7)
    cal.rm_calpoint(1460.83)
    assert len(cal.calpoints) == 0
    assert len(cal.channels) == 0
    assert len(cal.energies) == 0


def test_methods_ch2kev(slope, offset, channels):
    """Test ch2kev (both scalar and array)"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)

    if np.isscalar(channels):
        assert cal.ch2kev(channels) == slope * channels + offset
    else:
        assert np.all(
            cal.ch2kev(channels) == slope * np.array(channels) + offset)


def test_methods_kev2ch(slope, offset, channels):
    """Test kev2ch"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)

    if np.isscalar(channels):
        assert np.isclose(cal.kev2ch(cal.ch2kev(channels)), channels)
    else:
        assert np.all(np.isclose(cal.kev2ch(cal.ch2kev(channels)), channels))

# update_fit is in LinearEnergyCal section


# ----------------------------------------------------
#        LinearEnergyCal tests
# ----------------------------------------------------

def test_linear_construction_coefficients(slope, offset):
    """Test alternate construction coefficient names"""

    coeffs = {'p0': offset, 'p1': slope}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset

    coeffs = {'offset': offset, 'slope': slope}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset

    coeffs = {'b': offset, 'm': slope}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset


def test_linear_fitting_with_fit(chlist, kevlist):
    """Test fitting with calling update_fit function"""

    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)
    cal.update_fit()
    assert(np.allclose(linear_regression(chlist, kevlist), [cal.offset, cal.slope]))


def test_linear_fitting_without_fit(chlist, kevlist):
    """Test linear fitting without calling update_fit function"""

    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)
    assert(np.allclose(linear_regression(chlist, kevlist), [cal.offset, cal.slope]))


def test_linear_fitting_with_origin(chlist, kevlist):
    """Test linear fitting with including origin"""

    c = np.append(0, chlist)
    k = np.append(0, kevlist)
    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist, include_origin=True)
    assert(np.allclose(linear_regression(c, k), [cal.offset, cal.slope]))


def test_linear_bad_fitting():
    """Test fitting - too few calpoints (EnergyCalBase.update_fit())"""

    chlist = (67, 133)
    kevlist = (661.7, 1460.83)
    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)
    cal.update_fit()

    cal = bq.LinearEnergyCal()
    with pytest.raises(bq.EnergyCalError):
        cal.update_fit()

    chlist, kevlist = (67,), (661.7,)
    with pytest.raises(bq.EnergyCalError):
        cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)

# ----------------------------------------------------
#        Spectrum calibration methods tests
# ----------------------------------------------------

def test_apply_calibration(uncal_spec, chlist, kevlist):
    """Apply calibration on an uncalibrated spectrum"""

    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)
    uncal_spec.apply_calibration(cal)
    assert uncal_spec.is_calibrated
    assert np.allclose(
        uncal_spec.energies_kev, cal.ch2kev(uncal_spec.channels))


def test_apply_calibration_recal(cal_spec, chlist, kevlist):
    """Apply calibration over an existing calibration"""

    cal = bq.LinearEnergyCal.from_points(chlist=chlist, kevlist=kevlist)
    old_bin_edges = cal_spec.bin_edges_kev
    cal_spec.apply_calibration(cal)
    assert not np.any(old_bin_edges == cal_spec.bin_edges_kev)


def test_rm_calibration(cal_spec):
    """Remove calibration from a calibrated spectrum"""

    assert cal_spec.is_calibrated
    cal_spec.rm_calibration()
    assert not cal_spec.is_calibrated


def test_rm_calibration_error(uncal_spec):
    """Test that rm_calibration does not error on an uncalibrated spectrum"""

    assert not uncal_spec.is_calibrated
    uncal_spec.rm_calibration()
    assert not uncal_spec.is_calibrated
