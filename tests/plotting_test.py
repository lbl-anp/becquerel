"""Test core.plotting"""

from __future__ import print_function
import pytest
import numpy as np

import becquerel as bq

pytestmark = pytest.mark.plottest

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
def uncal_spec_cps(spec_data):
    """Generate an uncalibrated spectrum with cps data."""

    return bq.Spectrum(cps=spec_data)


@pytest.fixture
def cal_spec(spec_data):
    """Generate a calibrated spectrum."""

    return bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


@pytest.fixture(params=['uncal', 'cal'])
def counts_spec(request):
    if request.param == 'uncal':
        return uncal_spec(spec_data())
    elif request.param == 'cal':
        return cal_spec(spec_data())


# ----------------------------------------------
#                Very basic tests
# ----------------------------------------------

def test_plot_uncal_counts(uncal_spec):
    """Plot an uncalibrated spectrum"""

    bq.plot_spectrum(uncal_spec)


def test_plot_title(uncal_spec):
    """Check the title"""

    bq.plot_spectrum(uncal_spec, title='0-1. This is a custom title')


# ----------------------------------------------
#             Check each counts_mode
# ----------------------------------------------

def test_counts_mode(counts_spec):
    """Test counts_mode='counts'"""

    bq.plot_spectrum(counts_spec, counts_mode='counts',
                     title='0-2. Counts mode')


def test_counts_mode(counts_spec):
    """Test counts_mode='cps'"""

    counts_spec.livetime = 300.0
    bq.plot_spectrum(counts_spec, counts_mode='cps',
                     title='0-3. CPS mode')


def test_counts_mode(cal_spec):
    """Test counts_mode='cpskev'"""

    cal_spec.livetime = 300.0
    bq.plot_spectrum(cal_spec, counts_mode='cpskev',
                     title='0-4. CPS/keV mode')


# ----------------------------------------------
#                Check axes labels
# ----------------------------------------------

def test_plot_uncal_counts_labels(uncal_spec):
    """Default xlabel, ylabel for uncal counts spec"""

    bq.plot_spectrum(
        uncal_spec, title='1-1. Check xlabel is Channels, ylabel is Counts')


def test_plot_cal_counts_labels(cal_spec):
    """Default xlabel, ylabel for cal counts spec"""

    bq.plot_spectrum(
        cal_spec, title='1-2. Check xlabel is keV, ylabel is Counts')


def test_plot_countrate_label(uncal_spec_cps):
    """Default ylabel for cps spec"""

    bq.plot_spectrum(uncal_spec_cps, title='1-3. Check ylabel is Countrate')


def test_plot_countrate_density_label(cal_spec):
    """Default ylabel for cpskev spec"""

    cal_spec.livetime = 300.0
    bq.plot_spectrum(cal_spec, counts_mode='cpskev',
                     title='1-4. Check ylabel is Countrate (/keV)')


def test_custom_labels(counts_spec):
    """Custom xlabel, ylabel"""

    bq.plot_spectrum(
        counts_spec, xlabel='Custom x label', ylabel='Custom y label',
        title='1-5. Check custom xlabel and ylabel')


# ----------------------------------------------
#                Check y scale
# ----------------------------------------------

def test_yscale(counts_spec):
    """Default yscale"""

    bq.plot_spectrum(counts_spec, title='2-1. Check default yscale (log)')


def test_yscale_linear(counts_spec):
    """Linear yscale"""

    bq.plot_spectrum(counts_spec, yscale='linear',
                     title='2-2. Check yscale is linear')


def test_yscale_log(counts_spec):
    """Log yscale"""

    bq.plot_spectrum(counts_spec, yscale='log',
                     title='2-3. Check yscale is log')


def test_yscale_symlog(counts_spec):
    """Symlog yscale"""

    bq.plot_spectrum(counts_spec, yscale='symlog',
                     title='2-4. Check yscale is symlog')


# ----------------------------------------------
#                Check x scale
# ----------------------------------------------

def test_xscale_default(counts_spec):
    """Default xscale (linear)"""

    bq.plot_spectrum(counts_spec, title='3-1. Check default xscale (linear)')


def test_xscale_linear(counts_spec):
    """Linear xscale"""

    bq.plot_spectrum(counts_spec, xscale='linear',
                     title='3-2. Check xscale is linear')


def test_xscale_log(counts_spec):
    """Log xscale"""

    bq.plot_spectrum(counts_spec, xscale='log',
                     title='3-2. Check xscale is log')
