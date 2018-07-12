"""Test core.plotting"""

from __future__ import print_function
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import becquerel as bq
from becquerel import SpectrumPlotter as sp

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

    return bq.Spectrum(cps=spec_data, livetime=300)


@pytest.fixture
def cal_spec(spec_data):
    """Generate a calibrated spectrum."""

    return bq.Spectrum(spec_data, bin_edges_kev=TEST_EDGES_KEV)


@pytest.fixture
def cal_spec_cps(spec_data):
    """Generate a calibrated spectrum with cps data."""

    return bq.Spectrum(
        cps=spec_data, bin_edges_kev=TEST_EDGES_KEV, livetime=300)


@pytest.fixture(params=[1, 10, 100])
def y_counts_spec(request):
    floatdata = np.random.poisson(lam=request.param, size=TEST_DATA_LENGTH)
    return bq.Spectrum(floatdata.astype(np.int))


#@pytest.fixture
#def neg_spec(spec_data):
#    """Generate an uncalibrated spectrum."""
#
#    return bq.Spectrum(-spec_data)

# ----------------------------------------------
#                Very basic tests
# ----------------------------------------------


def test_plot_uncal_counts(uncal_spec):
    """Plot an uncalibrated spectrum"""

    uncal_spec.plot()
    assert plt.gca().get_xlabel() == "Channel"
    assert plt.gca().get_ylabel() == "Counts"
    assert plt.gca().get_title() == ""
    plt.close("all")


def test_plot_uncal_counts(cal_spec):
    """Plot an calibrated spectrum"""

    cal_spec.fill_between()
    assert plt.gca().get_xlabel() == "Energy [keV]"
    assert plt.gca().get_ylabel() == "Counts"
    assert plt.gca().get_title() == ""
    #assert np.allclose(plt.gca().get_lines()[0].get_xdata(), TEST_EDGES_KEV)
    plt.close("all")


# ----------------------------------------------
#             Check each mode
# ----------------------------------------------


def test_channel_mode(uncal_spec):
    """Test xmode='channel'"""

    uncal_spec.plot(xmode='channel')
    assert plt.gca().get_xlabel() == "Channel"
    plt.close("all")


def test_energy_mode(cal_spec):
    """Test xmode='energy'"""

    cal_spec.plot(xmode='energy')
    assert plt.gca().get_xlabel() == "Energy [keV]"
    plt.close("all")


def test_counts_mode(cal_spec):
    """Test ymode='counts'"""

    cal_spec.plot(ymode='counts')
    assert plt.gca().get_ylabel() == "Counts"
    plt.close("all")


def test_cps_mode(cal_spec_cps):
    """Test ymode='cps'"""

    cal_spec_cps.plot(ymode='cps')
    assert plt.gca().get_ylabel() == "Countrate [1/s]"
    plt.close("all")


def test_cpskev_mode(cal_spec_cps):
    """Test ymode='cpskev'"""

    cal_spec_cps.plot(ymode='cpskev')
    assert plt.gca().get_ylabel() == "Countrate [1/s/keV]"
    plt.close("all")


# ----------------------------------------------
#                Check axes labels
# ----------------------------------------------


def test_plot_uncal_counts_labels(uncal_spec):
    """Default xlabel, ylabel for uncal counts spec"""

    uncal_spec.plot()
    assert plt.gca().get_xlabel() == "Channel"
    assert plt.gca().get_ylabel() == "Counts"
    plt.close("all")


def test_plot_cal_counts_labels(cal_spec):
    """Default xlabel, ylabel for cal counts spec"""

    cal_spec.plot()
    assert plt.gca().get_xlabel() == "Energy [keV]"
    assert plt.gca().get_ylabel() == "Counts"
    plt.close("all")


def test_plot_countrate_label(uncal_spec_cps):
    """Default ylabel for uncal cps spec"""

    uncal_spec_cps.plot()
    assert plt.gca().get_xlabel() == "Channel"
    assert plt.gca().get_ylabel() == "Countrate [1/s]"
    plt.close("all")


def test_plot_countrate_density_label(cal_spec_cps):
    """Default ylabel for cal cpskev spec"""

    cal_spec_cps.plot(ymode='cpskev')
    assert plt.gca().get_xlabel() == "Energy [keV]"
    assert plt.gca().get_ylabel() == "Countrate [1/s/keV]"
    plt.close("all")


def test_custom_labels(cal_spec):
    """Custom xlabel, ylabel"""

    xlab = "Custom x label"
    ylab = "Custom y label"
    cal_spec.plot(xlabel=xlab, ylabel=ylab)
    assert plt.gca().get_xlabel() == xlab
    assert plt.gca().get_ylabel() == ylab
    plt.close("all")


# ----------------------------------------------
#                Check title
# ----------------------------------------------


def test_plot_default_title(uncal_spec):
    """Default title of a inprogram spec"""

    uncal_spec.plot()
    assert plt.gca().get_title() == ""
    plt.close("all")


def test_plot_default_file_title(uncal_spec):
    """Default title of a file spec"""

    fname = "/path/to/custom/file"
    uncal_spec.infilename = fname
    uncal_spec.plot()
    assert plt.gca().get_title() == fname
    plt.close("all")


def test_plot_default_custom_title(uncal_spec):
    """Default title of a file spec"""

    title = "My custom title"
    uncal_spec.plot(title=title)
    assert plt.gca().get_title() == title
    plt.close("all")


# ----------------------------------------------
#                Check y scale
# ----------------------------------------------


def test_yscale(y_counts_spec):
    """Default yscale"""

    y_counts_spec.plot()
    assert plt.gca().get_yscale() == "linear"


def test_yscale_linear(y_counts_spec):
    """Linear yscale"""

    y_counts_spec.plot(yscale='linear')
    assert plt.gca().get_yscale() == "linear"
    plt.close("all")


def test_yscale_log(y_counts_spec):
    """Log yscale"""

    y_counts_spec.plot(yscale='log')
    assert plt.gca().get_yscale() == "log"


def test_yscale_symlog(y_counts_spec):
    """Symlog yscale"""

    y_counts_spec.plot(yscale='symlog')
    assert plt.gca().get_yscale() == "symlog"
    plt.close("all")


def test_yscale_linear_scale(y_counts_spec):
    """Linear yscale with default scale"""

    y_counts_spec.plot(yscale='linear', ylim="default")
    assert plt.gca().get_yscale() == "linear"
    plt.close("all")


def test_yscale_log_scale(y_counts_spec):
    """Log yscale with default scale"""

    y_counts_spec.plot(yscale='log', ylim="default")
    assert plt.gca().get_yscale() == "log"
    plt.close("all")


def test_yscale_symlog_scale(y_counts_spec):
    """Symlog yscale with default scale"""

    y_counts_spec.plot(yscale='symlog', ylim="default")
    assert plt.gca().get_yscale() == "symlog"
    plt.close("all")


# ----------------------------------------------
#                Check x and y limits
# ----------------------------------------------


def test_xlim(uncal_spec):
    """Custom x limits"""

    xlim = (50, 100)
    uncal_spec.plot(xlim=xlim)
    assert plt.gca().get_xlim() == xlim
    plt.close("all")


def test_ylim(uncal_spec):
    """Custom y limits"""

    ylim = (0.8, 11)
    uncal_spec.plot(ylim=ylim)
    assert plt.gca().get_ylim() == ylim
    plt.close("all")


# ----------------------------------------------
#                Axes
# ----------------------------------------------


def test_axes(uncal_spec):
    """Custom y limits"""

    _, ax = plt.subplots()
    uncal_spec.plot(ax=ax)
    assert plt.gca() == ax
    plt.close("all")


# ----------------------------------------------
#                Kwargs
# ----------------------------------------------


def test_kwargs(uncal_spec):
    """Test kwargs, color and fmt in this case"""

    uncal_spec.plot('--', color='#ffffff')
    assert plt.gca().get_lines()[0].get_linestyle() == '--'
    assert plt.gca().get_lines()[0].get_color() == '#ffffff'
    plt.close("all")


def test_kwargs_SpectrumPlotter(uncal_spec):
    """Test kwargs, color and fmt in this case"""

    sp(uncal_spec).plot('--', color='#ffffff')
    assert plt.gca().get_lines()[0].get_linestyle() == '--'
    assert plt.gca().get_lines()[0].get_color() == '#ffffff'
    plt.close("all")


# ----------------------------------------------
#                multi line plot
# ----------------------------------------------


def test_multi(uncal_spec, cal_spec):
    """Test multiple plots"""

    _, ax = plt.subplots()
    uncal_spec.plot(ax=ax)
    cal_spec.plot(ax=ax)

    assert len(plt.gca().get_lines()) == 2
    plt.close("all")


# ----------------------------------------------
#                check errors
# ----------------------------------------------


def test_error_positional_parameters(cal_spec):
    """Test errors when to many positional parameters are provided"""

    with pytest.raises(bq.PlottingError):
        cal_spec.plot('x', 'x')
    with pytest.raises(bq.PlottingError):
        sp(cal_spec).plot('x', 'x')


def test_uncal_as_cal(uncal_spec):
    """Test errors for calibrated reqested for an uncalibrated spectrum"""

    with pytest.raises(bq.PlottingError):
        uncal_spec.plot(xmode='energy')


def test_unknown_xmode(uncal_spec):
    """Test errors for unknown xmode"""

    with pytest.raises(bq.PlottingError):
        uncal_spec.plot(xmode='unknown')


def test_unknown_ymode(uncal_spec):
    """Test errors for unknown ymode"""

    with pytest.raises(bq.PlottingError):
        uncal_spec.plot(ymode='unknown')


def test_cps_without_cps(uncal_spec):
    """Test errors for cps requested but not provided, causes a SpectrumError"""

    with pytest.raises(bq.SpectrumError):
        uncal_spec.plot(ymode='cps')


def test_cnts_without_cnts(uncal_spec_cps):
    """Test errors for cnts requested but not provided"""

    with pytest.raises(bq.PlottingError):
        uncal_spec_cps.plot(ymode='cnts')


#def test_call_ylim_default_without_any_input(neg_spec):
#    """Test errors for cnts requested but not provided"""
#
#    with pytest.raises(bq.PlottingError):
#        neg_spec.plot(yscale='log', ylim='default')


def test_wrong_xlim(cal_spec):
    """Test errors wrong xlim structure"""

    with pytest.raises(bq.PlottingError):
        cal_spec.plot(xlim=0)


def test_wrong_ylim(cal_spec):
    """Test errors for wrong ylim structure"""

    with pytest.raises(bq.PlottingError):
        cal_spec.plot(ylim=0)


# ----------------------------------------------
#                check getters
# ----------------------------------------------


def test_get_xmode(cal_spec_cps):
    """Test get xmode function"""

    tsp = sp(cal_spec_cps)
    assert tsp.xmode == 'energy'


def test_get_ymode(cal_spec_cps):
    """Test get ymode function"""

    tsp = sp(cal_spec_cps)
    assert tsp.ymode == 'cps'


def test_get_xlabel(cal_spec_cps):
    """Test get xlabel function"""

    tsp = sp(cal_spec_cps)
    assert tsp.xlabel == 'Energy [keV]'


def test_get_ylabel(cal_spec_cps):
    """Test get ylabel function"""

    tsp = sp(cal_spec_cps)
    assert tsp.ylabel == 'Countrate [1/s]'


def test_get_xlim(cal_spec_cps):
    """Test get xlim function"""

    tsp = sp(cal_spec_cps, xlim='default')
    assert tsp.xlim == (cal_spec_cps.bin_edges_kev[0],
                        cal_spec_cps.bin_edges_kev[-1])


def test_get_linthreshy(cal_spec_cps):
    """Test get linthreshy function"""

    tsp = sp(cal_spec_cps, linthreshy=1)
    assert tsp.linthreshy == 1


# ----------------------------------------------
#                check error modes
# ----------------------------------------------


def test_error_modes_counts(uncal_spec):
    """Test error mode for counts mode"""

    tsp = sp(uncal_spec, ymode='counts')
    assert np.allclose(tsp.yerror, uncal_spec.counts_uncs)


def test_error_modes_cps(cal_spec):
    """Test error mode for cps mode"""

    cal_spec.livetime = 200
    tsp = sp(cal_spec, ymode='cps')
    assert np.allclose(tsp.yerror, cal_spec.cps_uncs)


def test_error_modes_cpskev(cal_spec):
    """Test error mode for cps mode"""

    cal_spec.livetime = 200
    tsp = sp(cal_spec, ymode='cpskev')
    assert np.allclose(tsp.yerror, cal_spec.cpskev_uncs)


# ----------------------------------------------
#                check error plots
# ----------------------------------------------


def test_errornone(uncal_spec):
    """Test error mode none"""

    ax = uncal_spec.plot(ymode='counts', emode='none')

    colls = 0
    polys = 0
    lines = 0
    for i in ax.get_children():
        if type(i) is matplotlib.collections.LineCollection:
            colls = colls + 1
        if type(i) is matplotlib.collections.PolyCollection:
            polys = polys + 1

        if type(i) is matplotlib.lines.Line2D:
            lines = lines + 1
    assert colls == 0
    assert polys == 0
    assert lines == 1
    plt.close("all")


def test_errorbars(uncal_spec):
    """Test error bar plot"""

    ax = uncal_spec.plot(ymode='counts', emode='bars')

    colls = 0
    lines = 0
    for i in ax.get_children():
        if type(i) is matplotlib.collections.LineCollection:
            colls = colls + 1
        if type(i) is matplotlib.lines.Line2D:
            lines = lines + 1
    assert colls == 1
    assert lines >= 1
    plt.close("all")


def test_errorband(uncal_spec):
    """Test error band mode"""

    ax = uncal_spec.plot(ymode='counts', emode='band')

    colls = 0
    lines = 0
    for i in ax.get_children():
        if type(i) is matplotlib.collections.PolyCollection:
            colls = colls + 1
        if type(i) is matplotlib.lines.Line2D:
            lines = lines + 1
    assert colls == 1
    assert lines == 1
    plt.close("all")


def test_unknown_emode(uncal_spec):
    """Test error mode unknown"""
    with pytest.raises(bq.SpectrumError):
        ax = uncal_spec.plot(ymode='counts', emode='unknown')
