"""Test PeakFinder and AutoCalibrator classes."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pytest
import becquerel as bq


# read in spectra
filename1 = os.path.join(os.path.dirname(bq.__file__), "../tests/samples/sim_spec.csv")
filename2 = os.path.join(
    os.path.dirname(bq.__file__), "../tests/samples/Mendocino_07-10-13_Acq-10-10-13.Spe"
)
filename3 = os.path.join(
    os.path.dirname(bq.__file__), "../tests/samples/nai_detector.csv"
)
filename4 = os.path.join(os.path.dirname(bq.__file__), "../tests/samples/SGM102432.csv")

counts = []
with open(filename1) as f:
    for line in f:
        tokens = line.strip().split(",")
        if len(tokens) == 2:
            counts.append(float(tokens[1]))
spec1 = bq.Spectrum(counts=counts)

spec2 = bq.Spectrum.from_file(filename2)

counts = []
with open(filename3) as f:
    for line in f:
        tokens = line.strip().split(",")
        if len(tokens) == 2:
            counts.append(float(tokens[1]))
spec3 = bq.Spectrum(counts=counts)

counts = []
with open(filename4) as f:
    for line in f:
        tokens = line.strip().split(",")
        if len(tokens) == 2:
            counts.append(float(tokens[1]))
spec4 = bq.Spectrum(counts=counts)


REQUIRED = [609.32, 1460.82, 2614.3]
OPTIONAL = [238.63, 338.32, 351.93, 911.20, 1120.294, 1620.50, 1764.49, 2118.514]


def test_peakfilter_base():
    """Test that the PeakFilter base class cannot be used."""
    pf = bq.PeakFilter(700, 20, fwhm_at_0=15)
    pf.fwhm(200)
    with pytest.raises(NotImplementedError):
        pf.kernel_matrix(np.arange(1000))


def test_peakfilter_exceptions():
    """Test PeakFilter base class exceptions."""
    with pytest.raises(bq.PeakFilterError):
        bq.PeakFilter(-700, 20, fwhm_at_0=15)
    with pytest.raises(bq.PeakFilterError):
        bq.PeakFilter(700, -20, fwhm_at_0=15)
    with pytest.raises(bq.PeakFilterError):
        bq.PeakFilter(700, 20, fwhm_at_0=-15)


@pytest.mark.parametrize("cls", [bq.GaussianPeakFilter])
def test_peakfilter(cls):
    """Test basic functionality of PeakFilter."""
    pf = cls(700, 20, fwhm_at_0=15)
    pf.fwhm(200)
    pf.kernel_matrix(np.arange(1000))
    pf.convolve(np.arange(1000), np.ones(999))


@pytest.mark.plottest
@pytest.mark.parametrize("cls", [bq.GaussianPeakFilter])
def test_peakfilter_plot_matrix(cls):
    """Test PeakFilter.plot_matrix."""
    pf = cls(200, 20, fwhm_at_0=10)
    plt.figure()
    pf.plot_matrix(300)
    plt.show()


def test_peakfinder():
    """Test basic functionality of PeakFinder."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    finder.find_peak(500, min_snr=3.0)
    assert np.isclose(finder.channels[0], 485.5)
    finder.reset()
    finder.find_peaks()
    assert len(finder.channels) == 9
    finder.reset()
    finder.find_peaks(min_snr=0.5, xmin=50, xmax=1000, max_num=10)
    assert len(finder.channels) == 10


def test_peakfinder_exceptions():
    """Test PeakFinder exceptions."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    # init
    with pytest.raises(bq.PeakFinderError):
        finder = bq.PeakFinder(spec1, kernel, min_sep=-10)
    with pytest.raises(bq.PeakFinderError):
        finder = bq.PeakFinder(None, kernel)
    with pytest.raises(bq.PeakFinderError):
        finder = bq.PeakFinder(spec1, None)
    finder = bq.PeakFinder(spec1, kernel)
    # sort_by
    with pytest.raises(bq.PeakFinderError):
        finder.sort_by([1, 2, 3])
    # add_peak
    with pytest.raises(bq.PeakFinderError):
        finder.add_peak(-10)
    with pytest.raises(bq.PeakFinderError):
        finder.add_peak(5)  # below threshold so no peak here
    # find_peak
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(-1)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(3000)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, frac_range=(-0.1, 1.1))
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, frac_range=(1.1, 1.2))
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, frac_range=(0.8, 0.9))
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, frac_range=(1.2, 0.8))
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, min_snr=-3)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, min_snr=10)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peak(700, min_snr=100)
    # find_peaks
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(xmin=-1)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(xmin=3000)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(xmax=-1)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(xmax=3000)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(xmin=700, xmax=600)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(min_snr=-3)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(min_snr=50)
    with pytest.raises(bq.PeakFinderError):
        finder.find_peaks(max_num=0)


@pytest.mark.plottest
def test_peakfinder_plot():
    """Test PeakFinder.plot."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    # finder.find_peaks(min_snr=2, xmin=50)
    finder.find_peaks(min_snr=0.5, xmin=50, xmax=1000, max_num=10)

    plt.figure()
    finder.plot(linecolor="k")
    plt.show()


def test_fit_function_exceptions():
    """Test exceptions in the fit functions."""
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.fit_gain([0], [], [2])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.fit_gain([0], [1], [])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.fom_gain([0], [], [2])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.fom_gain([0], [1], [])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.find_best_gain([0, 1, 2], [2, 4], [300, 400, 500])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.find_best_gain([0], [2], [300, 400, 500])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.find_best_gain([0, 1, 2], [2, 4, 6], [300])
    with pytest.raises(bq.AutoCalibratorError):
        bq.core.autocal.find_best_gain([0, 1], [2, 4], [300, 400, 500])


def test_autocal_spec1():
    """Test basic functionality of AutoCalibrator."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    cal = bq.AutoCalibrator(finder)
    finder.find_peaks(min_snr=1, xmin=50)
    assert len(cal.peakfinder.channels) == 10
    cal.fit(
        REQUIRED,
        optional=OPTIONAL,
        gain_range=[2.5, 4.0],
        de_max=25.0,
        verbose=True,
    )
    assert len(cal.fit_channels) == 7
    assert np.isclose(cal.gain, 3.01, rtol=1e-2)


def test_autocal_one_line():
    """Test AutoCalibrator with one line."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    cal = bq.AutoCalibrator(finder)
    finder.find_peaks(min_snr=8, xmin=50)
    assert len(cal.peakfinder.channels) == 1
    cal.fit(
        [1460.82],
        gain_range=[2.5, 4.0],
        de_max=20.0,
    )
    assert len(cal.fit_channels) == 1
    assert np.isclose(cal.gain, 3.01, rtol=1e-2)


def test_autocal_exceptions():
    """Test AutoCalibrator exceptions."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    # init error
    with pytest.raises(bq.AutoCalibratorError):
        cal = bq.AutoCalibrator(None)
    cal = bq.AutoCalibrator(finder)
    # only one peak found, but multiple energies required
    finder.find_peaks(min_snr=10, xmin=50)
    assert len(cal.peakfinder.channels) == 1
    with pytest.raises(bq.AutoCalibratorError):
        cal.fit(
            REQUIRED,
            gain_range=[2.5, 4.0],
            de_max=20.0,
        )
    # multiple peaks found, but only one energy given
    finder.reset()
    finder.find_peaks(min_snr=1, xmin=50)
    assert len(cal.peakfinder.channels) == 10
    with pytest.raises(bq.AutoCalibratorError):
        cal.fit(
            [1460.82],
            gain_range=[2.5, 4.0],
            de_max=20.0,
        )
    # more energies required than peaks found
    finder.reset()
    finder.find_peaks(min_snr=4, xmin=50)
    assert len(cal.peakfinder.channels) == 4
    with pytest.raises(bq.AutoCalibratorError):
        cal.fit(
            [238.63, 351.93, 609.32, 1460.82, 2614.3],
            gain_range=[2.5, 4.0],
            de_max=20.0,
        )


def test_autocal_no_fit():
    """Test AutoCalibrator with no valid fit found."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    cal = bq.AutoCalibrator(finder)
    finder.find_peaks(min_snr=2, xmin=50)
    assert len(cal.peakfinder.channels) == 8
    with pytest.raises(bq.AutoCalibratorError):
        cal.fit(
            REQUIRED,
            optional=OPTIONAL,
            gain_range=[1, 2.0],
            de_max=20.0,
        )


@pytest.mark.plottest
def test_autocal_plot():
    """Test AutoCalibrator.plot."""
    kernel = bq.GaussianPeakFilter(500, 50, fwhm_at_0=10)
    finder = bq.PeakFinder(spec1, kernel)
    cal = bq.AutoCalibrator(finder)
    finder.find_peaks(min_snr=1, xmin=50)
    assert len(cal.peakfinder.channels) == 10
    cal.fit(
        REQUIRED,
        optional=OPTIONAL,
        gain_range=[2.5, 4.0],
        de_max=20.0,
    )
    plt.figure()
    cal.plot()
    plt.show()


def test_autocal_spec2():
    """Test AutoCalibrator on spectrum 2."""
    kernel = bq.GaussianPeakFilter(3700, 10, 5)
    finder = bq.PeakFinder(spec2, kernel)
    cal = bq.AutoCalibrator(finder)
    cal.peakfinder.find_peaks(min_snr=15, xmin=1000)
    assert len(cal.peakfinder.channels) == 12
    cal.fit(
        REQUIRED,
        optional=OPTIONAL,
        gain_range=[0.1, 0.6],
        de_max=5.0,
    )
    assert len(cal.fit_channels) == 6
    assert np.isclose(cal.gain, 0.3785, rtol=1e-2)


def test_autocal_spec3():
    """Test AutoCalibrator on spectrum 3."""
    kernel = bq.GaussianPeakFilter(700, 50, 10)
    finder = bq.PeakFinder(spec3, kernel)
    cal = bq.AutoCalibrator(finder)
    cal.peakfinder.find_peaks(min_snr=3, xmin=100)
    assert len(cal.peakfinder.channels) == 8
    # this fit succeeds but misidentifies the lines
    cal.fit(
        [609.32, 1460.82],
        optional=[],
        gain_range=[0.1, 5.0],
        de_max=50.0,
    )
    assert len(cal.fit_channels) == 2
    assert np.isclose(cal.gain, 2.59, rtol=1e-2)
    # this fit correctly identifies the lines
    cal.fit(
        [609.32, 1460.82],
        optional=OPTIONAL,
        gain_range=[0.1, 5.0],
        de_max=50.0,
    )
    assert len(cal.fit_channels) == 7
    assert np.isclose(cal.gain, 2.02, rtol=1e-2)


def test_autocal_spec4():
    """Test AutoCalibrator on spectrum 4."""
    kernel = bq.GaussianPeakFilter(2400, 120, 30)
    finder = bq.PeakFinder(spec4, kernel)
    cal = bq.AutoCalibrator(finder)
    cal.peakfinder.find_peaks(min_snr=3, xmin=100)
    assert len(cal.peakfinder.channels) == 7
    cal.fit(
        [356.0129, 661.657, 1460.82],
        optional=[911.20, 1120.294, 1620.50, 1764.49, 2118.514, 2614.3],
        gain_range=[0.3, 0.7],
        de_max=100.0,
    )
    assert len(cal.fit_channels) == 4
    assert np.isclose(cal.gain, 0.6133, rtol=1e-2)
