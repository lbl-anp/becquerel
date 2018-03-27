"""Automatic spectral peak search using convolutions."""

from __future__ import print_function
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from .spectrum import Spectrum
from .energycal import LinearEnergyCal


MAJOR_BACKGROUND_LINES = [
    1460.82,
    2614.51,
    1620.50,
    911.20,
    338.32,
    238.63,
    2204.06,
    1764.49,
    1377.67,
    1238.12,
    1120.29,
    768.36,
    609.32,
    351.93,
    295.22,
    242.00,
    186.21,
]


class SpectralPeakFilter(object):
    """An energy-dependent kernel that can be convolved with a spectrum.

    To detect lines, a kernel should have a positive component in the center
    and negative wings to subtract the continuum, e.g., a boxcar:

    +2|     ┌───┐     |
      |     │   │     |
     0|─┬───┼───┼───┬─|
    -1| └───┘   └───┘ |

    The positive part is meant to detect the peak, and the negative part to
    sample the continuum under the peak.

    The kernel should sum to 0.

    The width of the kernel scales proportionally to the square root
    of the channels, with a minimum value set by 'fwhm_at_0'.

    """

    def __init__(self, ref_channel, ref_fwhm, fwhm_at_0=1.):
        """Initialize with a reference line position and FWHM in channels."""
        assert ref_channel > 0
        assert ref_fwhm > 0
        self.ref_channel = float(ref_channel)
        self.ref_fwhm = float(ref_fwhm)
        self.fwhm_at_0 = float(fwhm_at_0)

    def fwhm(self, channel):
        """Calculate the expected FWHM at the given channel."""
        # f(x)^2 = f0^2 + k x^2
        # f1^2 = f0^2 + k x1^2
        # k = (f1^2 - f0^2) / x1^2
        # f(x)^2 = f0^2 + (f1^2 - f0^2) (x/x1)^2
        f0 = self.fwhm_at_0
        f1 = self.ref_fwhm
        x1 = self.ref_channel
        fwhm_sqr = f0**2 + (f1**2 - f0**2) * (channel / x1)**2
        return np.sqrt(fwhm_sqr)

    def kernel(self, channel, n_channels):
        """Generate the kernel for the given channel."""
        raise NotImplementedError

    def kernel_matrix(self, n_channels):
        """Build a matrix of the kernel evaluated at each channel."""
        kern = np.zeros((n_channels, n_channels))
        for j in range(n_channels):
            kern[:, j] = self.kernel(j, n_channels)
        kern_pos = +1 * kern.clip(0, np.inf)
        kern_neg = -1 * kern.clip(-np.inf, 0)
        # normalize negative part to be equal to the positive part
        kern_neg *= kern_pos.sum(axis=0) / kern_neg.sum(axis=0)
        return kern_pos - kern_neg

    def plot_matrix(self, n_channels):
        """Plot the matrix of kernels evaluated across the channels."""
        kern_mat = self.kernel_matrix(n_channels)
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)

        plt.imshow(
            kern_mat.T[::-1, :], cmap=plt.get_cmap('bwr'),
            vmin=kern_min, vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels])
        plt.colorbar()
        plt.xlabel('Input channel')
        plt.ylabel('Output channel')
        plt.gca().set_aspect('equal')

    def convolve(self, data):
        """Convolve this kernel with the data."""
        kern_mat = self.kernel_matrix(len(data))
        kern_mat_pos = +1 * kern_mat.clip(0, np.inf)
        kern_mat_neg = -1 * kern_mat.clip(-np.inf, 0)
        peak_plus_bkg = np.dot(kern_mat_pos, data)
        bkg = np.dot(kern_mat_neg, data)
        signal = np.dot(kern_mat, data)
        noise = np.sqrt(np.dot(kern_mat**2, data))
        snr = np.zeros_like(signal)
        snr[noise > 0] = signal[noise > 0] / noise[noise > 0]
        return peak_plus_bkg, bkg, signal, noise, snr


class BoxcarPeakFilter(SpectralPeakFilter):
    """A spectral kernel that is a boxcar with negative wings.

    The kernel is proportional to this form:

    +2|     ┌───┐     |
      |     │   │     |
     0|─┬───┼───┼───┬─|
    -1| └───┘   └───┘ |

    """

    def kernel(self, channel, n_channels):
        """Generate the kernel for the given channel."""
        n_center = int(np.ceil(self.fwhm(channel)))
        if n_center % 2 == 0:
            n_center += 1
        kernel0 = -0.5 * np.ones(n_center)
        kernel0 = np.append(kernel0, np.ones(n_center))
        kernel0 = np.append(kernel0, -0.5 * np.ones(n_center))
        n_side = len(kernel0) // 2
        kernel = np.zeros(2 * n_side + n_channels)
        kernel[channel:channel + len(kernel0)] = kernel0[:]
        kernel = kernel[n_side:-n_side]
        assert len(kernel) == n_channels
        positive = kernel > 0
        negative = kernel < 0
        kernel[positive] /= sum(kernel[positive])
        kernel[negative] /= -1 * sum(kernel[negative])
        return kernel


def _g0(x, mean, sigma):
    """Gaussian function."""
    z = (x - mean) / sigma
    return np.exp(-z**2 / 2.)


def _g1(x, mean, sigma):
    """First derivative of a gaussian."""
    z = (x - mean)
    return -1 * z * _g0(x, mean, sigma)


class GaussianPeakFilter(SpectralPeakFilter):
    """A spectral kernel that is the second derivative of a Gaussian.

    g0(x, u, s) = 1 / (sqrt(2 pi) s) * exp(-(x-u)^2/(2 s^2))
    g1(x, u, s) = -(x - u) / (s^2) * g0
    g2(x, u, s) = (x - u)^2 / (s^4) * g0 - (1/s^2) * g0
                = g0(x,u,s) ((x-u)^2/s^2 - 1) / s^2

    integral(-g2, x=x0..x1)
        = (-g1) for x=x0..x1
        = g1(x0) - g1(x1)

    """

    def kernel(self, channel, n_channels):
        """Generate the kernel for the given channel."""
        fwhm = self.fwhm(channel)
        sigma = fwhm / 2.355
        edges = np.arange(n_channels + 1)
        g1_x0 = _g1(edges[:-1], channel + 0.5, sigma)
        g1_x1 = _g1(edges[1:], channel + 0.5, sigma)
        kernel = g1_x0 - g1_x1
        return kernel


def find_peak(data, kernel, channel, f=0.2, min_snr=3.):
    """Find the highest SNR peak within (1-f)*channel and (1+f)*channel."""
    assert isinstance(kernel, SpectralPeakFilter)
    assert 0 < f < 1
    assert min_snr > 0.
    channels = np.arange(len(data))
    _, _, _, _, snr = kernel.convolve(data)
    chan_range = ((1 - f) * channel <= channels) & \
        (channels <= (1 + f) * channel)
    max_snr = snr[chan_range].max()
    if max_snr < min_snr:
        return None
    ind_max = np.where((snr == max_snr) & chan_range)[0][0]
    return ind_max


def find_peaks(data, kernel, min_snr=3., min_chan=0, max_chan=np.inf):
    """Find the highest SNR peaks in the data."""
    assert isinstance(kernel, SpectralPeakFilter)
    assert min_snr > 0.
    channels = np.arange(len(data))
    _, _, _, _, snr = kernel.convolve(data)
    # find regions above min_snr
    if snr.max() < min_snr:
        raise Exception(
            'SNR threshold is {:.3f} but maximum SNR is {:.3f}'.format(
                min_snr, snr.max()))
    if snr[0] >= min_snr:
        snr[0] = 0.
    if snr[-1] >= min_snr:
        snr[-1] = 0.
    left_edges = np.where((snr[:-1] < min_snr) & (snr[1:] >= min_snr))[0] + 1
    right_edges = np.where((snr[:-1] >= min_snr) & (snr[1:] < min_snr))[0]
    assert len(left_edges) == len(right_edges)
    # find maximum within each region
    peak_channels = []
    peak_snrs = []
    for left, right in zip(left_edges, right_edges):
        in_region = (left <= channels) & (channels <= right)
        max_snr = snr[in_region].max()
        chan_max = channels[(snr == max_snr) & in_region][0]
        if min_chan <= chan_max <= max_chan:
            peak_channels.append(chan_max)
            peak_snrs.append(max_snr)
    return np.array(peak_channels), np.array(peak_snrs)


def fit_gain(channels, snrs, energies):
    """Calculate the mean gain to relate channels to energies.

    Assume that the peak channel error scales like fwhm/snr,
    which is proportional to sqrt(x)/snr.

    Minimize:
        F = sum(snr^2 * (x * g - y)^2 / x)
        dF/dg = sum(2 * snr^2 * x * (x * g - y) / x)
              = 2 * sum(snr^2 * (x * g - y))
              = 2 * (g * sum(snr^2 * x) - sum(snr^2 * y))
              = 0
        g = sum(snr^2 * y) / sum(snr^2 * x)

    """
    assert len(channels) == len(energies)
    assert len(channels) == len(snrs)
    x = np.asarray(channels)
    s = np.asarray(snrs)
    y = np.asarray(energies)
    S2X = (s**2 * x).sum()
    S2Y = (s**2 * y).sum()
    gain = S2Y / S2X
    return gain


def fom_gain(channels, snrs, energies):
    """Calculate a leave-one-out cross-validation figure of merit.

    Assume that the peak channel error scales like fwhm/snr,
    which is proportional to sqrt(x)/snr. Then the figure of merit should be:

    F = sum(snr^2 * (x * g - y)^2 / x)

    s2x = sum(snr^2 x)
    s2y = sum(snr^2 y)

    Least-square gain when excluding channel j:
    g_j = sum_(i!=j)(snr_i^2 y_i) / sum_(i!=j)(snr_i^2 x_i)
        = (s2y - snr_j^2 y_j)/(s2x - snr_j^2 x_j)

    Figure of merit when excluding channel j:
    F_j = snr_j^2 (g_j x_j - y_j)^2 / x_j

    Leave-one-out cross validation FOM:
    F_LOO = sum_j(F_j) / N
    = sum_j(snr_j^2 (g_j x_j - y_j)^2 / x_j) / N
    = sum_j(snr_j^2 (s2y x_j - s2x y_j)^2 / (s2x - snr_j^2 x_j)^2 / x_j) / N

    """
    assert len(channels) == len(energies)
    assert len(channels) == len(snrs)
    x = np.asarray(channels)
    s = np.asarray(snrs)
    y = np.asarray(energies)
    N = len(channels)
    S2X = (s**2 * x).sum()
    S2Y = (s**2 * y).sum()
    squared_errs = s**2 * (S2Y * x - S2X * y)**2 / (S2X - s**2 * x)**2 / x
    return squared_errs.sum() / N / (N - 1)


def find_best_gain(
        channels, snrs, required_energies, optional=(),
        gain_range=(1e-3, 1e3), de_max=10.):
    """Find the gain that gives the best match of peaks to energies."""
    assert len(channels) == len(snrs)
    assert len(channels) >= 2
    assert len(required_energies) >= 2
    assert len(channels) >= len(required_energies)
    channels = np.array(channels)
    snrs = np.array(snrs)
    n_req = len(required_energies)
    # make sure the required and optional sets do not overlap
    optional = sorted(list(set(optional) - set(required_energies)))
    n_opt = len(optional)
    n_set = n_req + n_opt
    best_fom = None
    best_gain = None
    best_chans = None
    best_snrs = None
    best_ergs = None
    while n_set >= n_req:
        # cycle through energy combinations
        for comb_erg in combinations(optional, n_set - n_req):
            comb_erg = np.array(comb_erg)
            comb_erg = np.append(required_energies, comb_erg)
            comb_erg.sort()
            # use gain_range to reduce possible channel matches
            min_chan = (comb_erg / gain_range[1]).min()
            max_chan = (comb_erg / gain_range[0]).max()
            # cycle through channel combinations
            chan_inds = np.arange(len(channels))
            chan_inds = chan_inds[
                (min_chan <= channels) & (channels <= max_chan)]
            for chan_indices in combinations(chan_inds, n_set):
                chan_indices = np.array(chan_indices, dtype=int)
                comb_chan = np.array(channels)[chan_indices]
                comb_snr = np.array(snrs)[chan_indices]
                ind = np.argsort(comb_chan)
                comb_chan = comb_chan[ind]
                comb_snr = comb_snr[ind]
                # calculate gain
                gain = fit_gain(comb_chan, comb_snr, comb_erg)
                if gain < gain_range[0] or gain > gain_range[1]:
                    continue
                # calculate predicted energies
                pred_erg = gain * comb_chan
                de = pred_erg - comb_erg
                if (abs(de) > de_max).any():
                    continue
                # calculate figure of merit
                fom = fom_gain(comb_chan, comb_snr, comb_erg)
                # s0 = 'FOM: {:15.9f}'.format(fom)
                # s0 += '  gain: {:6.3f}'.format(gain)
                # s0 += '  ergs: {:50s}'.format(str(comb_erg))
                # # s0 += '  de: {:50s}'.format(str(de))
                # s0 += '  chans: {:40s}'.format(str(comb_chan))
                # print(s0)
                if best_fom is None:
                    best_fom = fom + 1.
                if fom < best_fom:
                    best_fom = fom
                    best_gain = gain
                    best_chans = comb_chan
                    best_snrs = comb_snr
                    best_ergs = comb_erg
                    # s0 = 'FOM: {:15.9f}'.format(best_fom)
                    # s0 += '  gain: {:6.3f}'.format(best_gain)
                    # s0 += '  ergs: {:50s}'.format(str(best_ergs))
                    # # s0 += '  de: {:50s}'.format(str(de))
                    # s0 += '  chans: {:40s}'.format(str(best_chans))
                    # print(s0)
        n_set -= 1
    if best_gain is None:
        return None
    else:
        return {
            'gain': best_gain,
            'channels': best_chans,
            'snrs': best_snrs,
            'energies': best_ergs,
        }


class AutoCalibrator(object):
    """Automatically calibrate a spectrum by convolving it with a filter."""

    def __init__(self, spectrum, kernel):
        """Initialize the calibration with a spectrum and kernel."""
        assert isinstance(spectrum, Spectrum)
        assert isinstance(kernel, SpectralPeakFilter)
        self.spectrum = spectrum
        self.kernel = kernel
        self.snr = np.zeros(len(self.spectrum))
        self.channels = np.arange(len(self.spectrum))
        peak_plus_bkg, bkg, signal, noise, snr = \
            self.kernel.convolve(self.spectrum.counts_vals)
        self.peak_plus_bkg = peak_plus_bkg
        self.bkg = bkg
        self.signal = signal
        self.noise = noise
        self.snr = snr.clip(0)
        self.min_snr = 3.0
        self.max_num = 20
        self.min_chan = 0
        self.max_chan = np.inf
        # peak-finding results
        self.peak_channels = []
        self.peak_snrs = []
        # fit results
        self.gain = None
        self.cal = None
        self.success = False
        self.fit_channels = []
        self.fit_snrs = []
        self.fit_energies = []

    def plot(self, facecolor='red', linecolor=None, alpha=0.5, peaks=True):
        """Plot the peak signal-to-noise ratios calculated using the kernel."""
        if facecolor is not None:
            plt.fill_between(
                self.spectrum.channels, self.snr, 0,
                color=facecolor, alpha=alpha)
        if linecolor is not None:
            plt.plot(self.spectrum.channels, self.snr, '-', color=linecolor)
        if peaks:
            for chan, snr in zip(self.peak_channels, self.peak_snrs):
                plt.plot([chan] * 2, [0, snr], 'b-')
                plt.plot(chan, snr, 'bo')
            for chan, snr in zip(self.fit_channels, self.fit_snrs):
                plt.plot([chan] * 2, [0, snr], 'g-')
                plt.plot(chan, snr, 'go')
        plt.xlim(0, len(self.spectrum))
        plt.ylim(0)
        plt.xlabel('Channels')
        plt.ylabel('SNR')

    def find_peak(self, channel, frac_range=[0.8, 1.2], min_snr=None):
        """Find the highest SNR peak within f0*channel and f1*channel."""
        assert 0 <= channel <= self.channels.max()
        assert 0 < frac_range[0] < 1
        assert 1 < frac_range[1]
        assert frac_range[0] < frac_range[1]
        if min_snr is not None:
            self.min_snr = min_snr
        assert self.min_snr > 0.
        assert self.min_snr <= self.snr.max()
        chan_range = \
            (frac_range[0] * channel <= self.channels) & \
            (self.channels <= frac_range[1] * channel)
        peak_snr = self.snr[chan_range].max()
        if peak_snr < self.min_snr:
            return None
        peak_chan = np.where((self.snr == peak_snr) & chan_range)[0][0]
        if peak_chan not in self.peak_channels:
            self.peak_channels.append(peak_chan)
            self.peak_snrs.append(peak_snr)
        return peak_chan, peak_snr

    def find_peaks(
            self, min_chan=None, max_chan=None, min_snr=None, max_num=None):
        """Find the highest SNR peaks in the data."""
        if min_chan is not None:
            self.min_chan = min_chan
        if max_chan is not None:
            self.max_chan = max_chan
        assert 0 <= self.min_chan
        assert self.min_chan < self.max_chan
        if min_snr is not None:
            self.min_snr = min_snr
        assert self.min_snr > 0.
        assert self.min_snr <= self.snr.max()
        if max_num is not None:
            self.max_num = max_num
        assert self.max_num >= 2
        # find regions above min_snr
        if self.snr.max() < self.min_snr:
            raise Exception(
                'SNR threshold is {:.3f} but maximum SNR is {:.3f}'.format(
                    self.min_snr, self.snr.max()))
        if self.snr[0] >= self.min_snr:
            self.snr[0] = 0.
        if self.snr[-1] >= self.min_snr:
            self.snr[-1] = 0.
        left_edges = np.where(
            (self.snr[:-1] < self.min_snr) &
            (self.snr[1:] >= self.min_snr))[0] + 1
        right_edges = np.where(
            (self.snr[:-1] >= self.min_snr) &
            (self.snr[1:] < self.min_snr))[0]
        assert len(left_edges) == len(right_edges)
        # find maximum within each region
        self.peak_channels = []
        self.peak_snrs = []
        for left, right in zip(left_edges, right_edges):
            in_region = (left <= self.channels) & (self.channels <= right)
            peak_snr = self.snr[in_region].max()
            peak_chan = self.channels[(self.snr == peak_snr) & in_region][0]
            if self.min_chan <= peak_chan <= self.max_chan:
                self.peak_channels.append(peak_chan)
                self.peak_snrs.append(peak_snr)
        self.peak_channels = np.array(self.peak_channels)
        self.peak_snrs = np.array(self.peak_snrs)
        # reduce number of channels to a maximum number max_n
        i = np.argsort(self.peak_snrs)
        self.peak_channels = self.peak_channels[i][::-1]
        self.peak_snrs = self.peak_snrs[i][::-1]
        self.peak_channels = self.peak_channels[-self.max_num:]
        self.peak_snrs = self.peak_snrs[-self.max_num:]
        # sort by channel
        i = np.argsort(self.peak_channels)
        self.peak_channels = self.peak_channels[i]
        self.peak_snrs = self.peak_snrs[i]

    def fit(self, required_energies, optional=(),
            gain_range=(1e-3, 1e3), de_max=10.):
        """Find the gain that gives the best match of peaks to energies."""
        assert len(self.peak_channels) >= 2
        assert len(required_energies) >= 2
        assert len(self.peak_channels) >= len(required_energies)
        fit = find_best_gain(
            self.peak_channels, self.peak_snrs, required_energies,
            optional=optional, gain_range=gain_range, de_max=de_max)
        if fit is None:
            self.fit_energies = []
            self.fit_snrs = []
            self.fit_energies = []
            self.gain = None
            self.cal = None
            self.success = False
        else:
            self.fit_channels = fit['channels']
            self.fit_snrs = fit['snrs']
            self.fit_energies = fit['energies']
            self.gain = fit['gain']
            self.cal = LinearEnergyCal.from_coeffs(
                {'offset': 0, 'slope': self.gain})
            self.success = True
