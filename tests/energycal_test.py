"""Test energycal.py"""

from __future__ import print_function
import pytest
import numpy as np
from uncertainties import unumpy, ufloat

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


@pytest.fixture
def pairlist():
    lst = (
        (32, 661.66),
        (67, 1460.83),
        (115, 2614.5))
    return lst


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


# ----------------------------------------------------
#        Construction tests
# ----------------------------------------------------

def test_construction_empty():
    """Test empty construction"""

    bq.core.LinearEnergyCal()


def test_construction_chkevlist(chlist, kevlist):
    """Test construction (not fitting) from chlist, kevlist"""

    cal = bq.core.LinearEnergyCal.from_points(
        chlist=chlist, kevlist=kevlist)
    assert len(cal.channels) == len(chlist)
    assert len(cal.energies) == len(chlist)
    assert len(cal.calpoints) == len(chlist)
    assert len(cal.calpoints[0]) == 2
    assert np.allclose(np.sort(cal.ch_vals), np.array(chlist))
    assert np.isnan(cal.ch_uncs).all()


def test_construction_chlist_unc(chlist, kevlist):
    """Test construction (not fitting) from chlist, kevlist with uncertainties
    """

    ch_ufloats = unumpy.uarray(chlist, 5)
    cal = bq.core.LinearEnergyCal.from_points(
        chlist=ch_ufloats, kevlist=kevlist)
    assert len(cal.channels) == len(chlist)
    assert len(cal.energies) == len(chlist)
    assert len(cal.calpoints) == len(chlist)
    assert len(cal.calpoints[0]) == 2
    assert np.all(np.sort(cal.ch_vals) == np.array(chlist))
    assert np.all(cal.ch_uncs == 5)

    ch_unc = np.ones_like(chlist)
    cal = bq.core.LinearEnergyCal.from_points(
        chlist=chlist, kevlist=kevlist, ch_uncs=ch_unc)
    assert len(cal.channels) == len(chlist)
    assert len(cal.energies) == len(chlist)
    assert len(cal.calpoints) == len(chlist)
    assert len(cal.calpoints[0]) == 2
    assert np.all(np.sort(cal.ch_vals) == np.array(chlist))
    assert np.all(cal.ch_uncs == 1)

    ch_ufloats[-1] = 42
    with pytest.raises(bq.core.UncertaintiesError):
        cal = bq.core.LinearEnergyCal.from_points(
            chlist=ch_ufloats, kevlist=kevlist)


def test_construction_pairlist(pairlist):
    """Test construction (not fitting) from pairlist"""

    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    assert len(cal.channels) == len(pairlist)
    assert len(cal.energies) == len(pairlist)
    assert len(cal.calpoints) == len(pairlist)
    assert len(cal.calpoints[0]) == 2
    assert np.isnan(cal.ch_uncs).all()


def test_construction_coefficients(slope, offset):
    """Test construction from coefficients"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.coeffs['b'] == slope
    assert cal.coeffs['c'] == offset


def test_construction_bad_coefficients(slope, offset):
    """Test construction with bad coefficients"""

    coeffs = {'a': offset, 'b': slope}
    with pytest.raises(bq.core.EnergyCalError):
        bq.core.LinearEnergyCal.from_coeffs(coeffs)


def test_construction_bad_points(chlist, kevlist, pairlist):
    """Test from_points with bad input"""

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(chlist=chlist, pairlist=pairlist)
    excinfo.match('Redundant calibration inputs')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(chlist=chlist)
    excinfo.match('Require both chlist and kevlist')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(
            chlist=chlist, kevlist=kevlist[:-1])
    excinfo.match('Channels and energies must be same length')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points()
    excinfo.match('Calibration points are required')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(chlist=[], kevlist=[])
    excinfo.match('Calibration points are required')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(chlist=32, kevlist=661.7)
    excinfo.match('Inputs should be iterables, not scalars')

    with pytest.raises(bq.core.BadInput) as excinfo:
        bq.core.LinearEnergyCal.from_points(pairlist=(32, 661.7))
    excinfo.match('Inputs should be iterables, not scalars')


# ----------------------------------------------------
#        other EnergyCal method tests
# ----------------------------------------------------

def test_methods_add_calpoint():
    """Test add_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.add_calpoint(32, 661.7)
    cal.add_calpoint(67, 1460.83)
    cal.add_calpoint(35, 661.7)
    assert len(cal.calpoints) == 2
    assert len(cal.channels) == 2
    assert len(cal.energies) == 2
    assert np.isnan(cal.ch_uncs).all()


def test_methods_add_calpoint_unc():
    """Test add_calpoint with uncertainty"""

    cal = bq.core.LinearEnergyCal()
    cal.add_calpoint(ufloat(32, 3), 661.7)
    cal.add_calpoint(67, 1460.83, ch_unc=4.5)
    assert len(cal.calpoints) == 2
    assert len(cal.channels) == 2
    assert len(cal.energies) == 2
    assert not np.isnan(cal.ch_uncs).any()


def test_methods_new_calpoint():
    """Test new_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.new_calpoint(67, 1460.83)
    with pytest.raises(bq.core.EnergyCalError):
        cal.new_calpoint(35, 661.7)
    assert len(cal.calpoints) == 2
    assert len(cal.channels) == 2
    assert len(cal.energies) == 2


def test_methods_rm_calpoint():
    """Test rm_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.rm_calpoint(661.7)
    cal.rm_calpoint(1460.83)
    assert len(cal.calpoints) == 0
    assert len(cal.channels) == 0
    assert len(cal.energies) == 0


def test_methods_ch2kev(slope, offset, channels):
    """Test ch2kev (both scalar and array)"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)

    if np.isscalar(channels):
        assert cal.ch2kev(channels) == slope * channels + offset
    else:
        assert np.all(
            cal.ch2kev(channels) == slope * np.array(channels) + offset)


def test_methods_kev2ch(slope, offset, channels):
    """Test kev2ch"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)

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
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset

    coeffs = {'offset': offset, 'slope': slope}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset

    coeffs = {'b': offset, 'm': slope}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.slope == slope
    assert cal.offset == offset


def test_linear_fitting(pairlist):
    """Test fitting"""

    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    cal.update_fit()


def test_linear_bad_fitting():
    """
    Test fitting - too few calpoints (EnergyCalBase.update_fit())
    """

    pairlist = ((67, 661.7), (133, 1460.83))
    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    cal.update_fit()

    cal = bq.core.LinearEnergyCal()
    with pytest.raises(bq.core.EnergyCalError):
        cal.update_fit()

    pairlist = ((67, 661.7),)
    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    with pytest.raises(bq.core.EnergyCalError):
        cal.update_fit()
