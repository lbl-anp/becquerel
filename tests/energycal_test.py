"""Test energycal.py"""

from __future__ import print_function
import pytest
import numpy as np

import becquerel as bq

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


# ----------------------------------------------------
#        Construction tests
# ----------------------------------------------------

# TODO: parameterize inputs with pytest

def test_construction_01():
    """Test empty construction"""

    bq.core.LinearEnergyCal()


def test_construction_02():
    """Test construction (not fitting) from chlist, kevlist"""

    chlist = [67, 133, 241]
    kevlist = [661.7, 1460.83, 2614.5]
    cal = bq.core.LinearEnergyCal.from_points(
        chlist=chlist, kevlist=kevlist)
    assert len(cal.channels) == 3
    assert len(cal.energies) == 3
    assert len(cal.calpoints) == 3
    assert len(cal.calpoints[0]) == 2


def test_construction_03():
    """Test construction (not fitting) from pairlist"""

    pairlist = ((67, 661.7), (133, 1460.83), (241, 2614.5))
    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    assert len(cal.channels) == 3
    assert len(cal.energies) == 3
    assert len(cal.calpoints) == 3
    assert len(cal.calpoints[0]) == 2


def test_construction_04():
    """Test construction from coefficients"""

    slope = 0.37
    offset = -4
    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)
    assert cal.coeffs['b'] == slope
    assert cal.coeffs['c'] == offset


def test_construction_05():
    """Test construction with bad coefficients"""

    slope = 0.37
    offset = -4
    coeffs = {'a': offset, 'b': slope}
    with pytest.raises(bq.core.EnergyCalError):
        bq.core.LinearEnergyCal.from_coeffs(coeffs)


def test_construction_06():
    """Test from_points with bad input"""

    chlist = [67, 133, 241]
    kevlist = [661.7, 1460.83, 2614.5]
    pairlist = ((67, 661.7), (133, 1460.83), (241, 2614.5))
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

def test_methods_01():
    """Test add_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.add_calpoint(32, 661.7)
    cal.add_calpoint(67, 1460.83)
    cal.add_calpoint(35, 661.7)


def test_methods_02():
    """Test new_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.new_calpoint(67, 1460.83)
    with pytest.raises(bq.core.EnergyCalError):
        cal.new_calpoint(35, 661.7)


def test_methods_03():
    """Test update_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    with pytest.raises(bq.core.EnergyCalError):
        cal.update_calpoint(67, 1460.83)
    cal.update_calpoint(35, 661.7)


def test_methods_04():
    """Test rm_calpoint"""

    cal = bq.core.LinearEnergyCal()
    cal.new_calpoint(32, 661.7)
    cal.rm_calpoint(661.7)
    cal.rm_calpoint(1460.83)


def test_methods_05():
    """Test ch2kev (both scalar and array)"""

    slope = 0.37
    offset = -4
    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)

    ch = 37
    assert cal.ch2kev(ch) == slope * ch + offset
    ch = [37, 99]
    assert cal.ch2kev(ch)[0] == slope * ch[0] + offset
    assert cal.ch2kev(ch)[1] == slope * ch[1] + offset


def test_methods_06():
    """Test kev2ch"""

    slope = 0.37
    offset = -4
    coeffs = {'b': slope, 'c': offset}
    cal = bq.core.LinearEnergyCal.from_coeffs(coeffs)

    ch = 37
    assert np.isclose(cal.kev2ch(cal.ch2kev(ch)), ch)
    ch = [37, 99]
    assert np.isclose(cal.kev2ch(cal.ch2kev(ch))[0], ch[0])
    assert np.isclose(cal.kev2ch(cal.ch2kev(ch))[1], ch[1])

# update_fit is in LinearEnergyCal section


# ----------------------------------------------------
#        LinearEnergyCal tests
# ----------------------------------------------------

def test_linear_01():
    """Test alternate construction coefficient names"""

    slope = 0.37
    offset = -4
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


def test_linear_02():
    """Test fitting"""

    pairlist = ((67, 661.7), (133, 1460.83), (241, 2614.5))
    cal = bq.core.LinearEnergyCal.from_points(pairlist=pairlist)
    cal.update_fit()


def test_linear_03():
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
