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


@pytest.fixture
def pairlist():
    lst = (
        (32, 661.66),
        (67, 1460.83),
        (115, 2614.5))
    return lst


@pytest.fixture
def chkevlists(pairlist):
    chlist, kevlist = zip(*pairlist)
    return chlist, kevlist


@pytest.fixture(params=[
    32, -5, 67.83, (34, 35), np.arange(113, 7678.3, 55.5)])
def channels(request):
    return request.param


# ----------------------------------------------------
#        Construction tests
# ----------------------------------------------------

def test_construction_01():
    """Test empty construction"""

    bq.LinearEnergyCal()


def test_construction_02(chkevlists):
    """Test construction (not fitting) from chlist, kevlist"""

    chlist, kevlist = chkevlists
    cal = bq.LinearEnergyCal.from_points(
        chlist=chlist, kevlist=kevlist)
    assert len(cal.channels) == 3
    assert len(cal.energies) == 3
    assert len(cal.calpoints) == 3
    assert len(cal.calpoints[0]) == 2


def test_construction_03(pairlist):
    """Test construction (not fitting) from pairlist"""

    cal = bq.LinearEnergyCal.from_points(pairlist=pairlist)
    assert len(cal.channels) == len(pairlist)
    assert len(cal.energies) == len(pairlist)
    assert len(cal.calpoints) == len(pairlist)
    assert len(cal.calpoints[0]) == 2


def test_construction_04(slope, offset):
    """Test construction from coefficients"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.coeffs['b'] == slope
    assert cal.coeffs['c'] == offset


def test_construction_05(slope, offset):
    """Test construction with bad coefficients"""

    coeffs = {'a': offset, 'b': slope}
    with pytest.raises(bq.EnergyCalError):
        bq.LinearEnergyCal.from_coeffs(coeffs)


def test_construction_06(chkevlists, pairlist):
    """Test from_points with bad input"""

    chlist, kevlist = chkevlists
    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(chlist=chlist, pairlist=pairlist)
    excinfo.match('Redundant calibration inputs')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(chlist=chlist)
    excinfo.match('Require both chlist and kevlist')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(
            chlist=chlist, kevlist=kevlist[:-1])
    excinfo.match('Channels and energies must be same length')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points()
    excinfo.match('Calibration points are required')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(chlist=[], kevlist=[])
    excinfo.match('Calibration points are required')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(chlist=32, kevlist=661.7)
    excinfo.match('Inputs should be iterables, not scalars')

    with pytest.raises(bq.BadInput) as excinfo:
        bq.LinearEnergyCal.from_points(pairlist=(32, 661.7))
    excinfo.match('Inputs should be iterables, not scalars')


# ----------------------------------------------------
#        other EnergyCal method tests
# ----------------------------------------------------

def test_methods_01():
    """Test add_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.add_calpoint(32, 661.7)
    cal.add_calpoint(67, 1460.83)
    cal.add_calpoint(35, 661.7)


def test_methods_02():
    """Test new_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.new_calpoint(67, 1460.83)
    with pytest.raises(bq.EnergyCalError):
        cal.new_calpoint(35, 661.7)


def test_methods_03():
    """Test rm_calpoint"""

    cal = bq.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.rm_calpoint(661.7)
    cal.rm_calpoint(1460.83)


def test_methods_04(slope, offset, channels):
    """Test ch2kev (both scalar and array)"""

    coeffs = {'b': slope, 'c': offset}
    cal = bq.LinearEnergyCal.from_coeffs(coeffs)

    if np.isscalar(channels):
        assert cal.ch2kev(channels) == slope * channels + offset
    else:
        assert np.all(
            cal.ch2kev(channels) == slope * np.array(channels) + offset)


def test_methods_05(slope, offset, channels):
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

def test_linear_01(slope, offset):
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


def test_linear_02(pairlist):
    """Test fitting"""

    cal = bq.LinearEnergyCal.from_points(pairlist=pairlist)
    cal.update_fit()


def test_linear_03():
    """
    Test fitting - too few calpoints (EnergyCalBase.update_fit())
    """

    pairlist = ((67, 661.7), (133, 1460.83))
    cal = bq.LinearEnergyCal.from_points(pairlist=pairlist)
    cal.update_fit()

    cal = bq.LinearEnergyCal()
    with pytest.raises(bq.EnergyCalError):
        cal.update_fit()

    pairlist = ((67, 661.7),)
    cal = bq.LinearEnergyCal.from_points(pairlist=pairlist)
    with pytest.raises(bq.EnergyCalError):
        cal.update_fit()
