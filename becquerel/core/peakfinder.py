# -*- coding: utf-8 -*-

"""Spectral peak search using convolutions."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from .spectrum import Spectrum


class PeakFilterError(Exception):
    """Base class for errors in PeakFilter."""

    pass


class PeakFilter(object):
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
        if ref_channel <= 0:
            raise PeakFilterError('Reference channel must be positive')
        if ref_fwhm <= 0:
            raise PeakFilterError('Reference FWHM must be positive')
        if fwhm_at_0 < 0:
            raise PeakFilterError('FWHM at 0 must be non-negative')
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


class BoxcarPeakFilter(PeakFilter):
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
        positive = kernel > 0
        negative = kernel < 0
        kernel[positive] /= sum(kernel[positive])
        kernel[negative] /= -1 * sum(kernel[negative])
        return kernel


def _gaussian0(x, mean, sigma):
    """Gaussian function."""
    z = (x - mean) / sigma
    return np.exp(-z**2 / 2.)


def _gaussian1(x, mean, sigma):
    """First derivative of a gaussian."""
    z = (x - mean)
    return -1 * z * _gaussian0(x, mean, sigma)


class GaussianPeakFilter(PeakFilter):
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
        g1_x0 = _gaussian1(edges[:-1], channel + 0.5, sigma)
        g1_x1 = _gaussian1(edges[1:], channel + 0.5, sigma)
        kernel = g1_x0 - g1_x1
        return kernel


class PeakFinderError(Exception):
    """Base class for errors in PeakFinder."""

    pass


class PeakFinder(object):
    """Find peaks in a spectrum after convolving it with a kernel."""

    def __init__(self, spectrum, kernel, min_sep=5, fwhm_tol=(0.5, 1.5)):
        """Initialize with a spectrum and kernel."""
        if min_sep <= 0:
            raise PeakFinderError(
                'Minimum channel separation must be positive')
        self.min_sep = min_sep
        self.fwhm_tol = tuple(fwhm_tol)
        self.spectrum = None
        self.kernel = None
        self.snr = []
        self._peak_plus_bkg = []
        self._bkg = []
        self._signal = []
        self._noise = []
        self.channels = []
        self.snrs = []
        self.fwhms = []
        self.integrals = []
        self.backgrounds = []
        self.calculate(spectrum, kernel)

    def reset(self):
        """Restore peak finder to pristine starting condition."""
        self.channels = []
        self.snrs = []
        self.fwhms = []
        self.integrals = []
        self.backgrounds = []

    def sort_by(self, arr):
        """Sort peaks by the provided array."""
        if len(arr) != len(self.channels):
            raise PeakFinderError(
                'Sorting array has length {} but must have length {}'.format(
                    len(arr), len(self.channels)))
        self.channels = np.array(self.channels)
        self.snrs = np.array(self.snrs)
        self.fwhms = np.array(self.fwhms)
        self.integrals = np.array(self.integrals)
        self.backgrounds = np.array(self.backgrounds)
        i = np.argsort(arr)
        self.channels = list(self.channels[i])
        self.snrs = list(self.snrs[i])
        self.fwhms = list(self.fwhms[i])
        self.integrals = list(self.integrals[i])
        self.backgrounds = list(self.backgrounds[i])

    def calculate(self, spectrum, kernel):
        """Calculate the convolution of the spectrum with the kernel."""
        if not isinstance(spectrum, Spectrum):
            raise PeakFinderError(
                'Argument must be a Spectrum, not {}'.format(type(spectrum)))
        if not isinstance(kernel, PeakFilter):
            raise PeakFinderError(
                'Argument must be a PeakFilter, not {}'.format(type(kernel)))
        self.spectrum = spectrum
        self.kernel = kernel
        self.snr = np.zeros(len(self.spectrum))
        # calculate the convolution
        peak_plus_bkg, bkg, signal, noise, snr = \
            self.kernel.convolve(self.spectrum.counts_vals)
        self._peak_plus_bkg = peak_plus_bkg
        self._bkg = bkg
        self._signal = signal
        self._noise = noise
        self.snr = snr.clip(0)
        self.reset()

    def add_peak(self, chan):
        """Add a peak at the channel to list if it is not already there."""
        chan_min = self.spectrum.channels.min()
        chan_max = self.spectrum.channels.max()
        if chan < chan_min or chan > chan_max:
            raise PeakFinderError(
                'Channel {} is outside of range {}-{}'.format(
                    chan, chan_min, chan_max))
        new_channel = True
        for chan2 in self.channels:
            if abs(chan - chan2) <= self.min_sep:
                new_channel = False
        if new_channel:
            # estimate FWHM using the second derivative
            # snr(chan) = snr(chan0) - 0.5 d2snr/dchan2(chan0) (chan-chan0)^2
            # 0.5 = 1 - 0.5 d2snr/dchan2 (fwhm/2)^2 / snr0
            # 1 = d2snr/dchan2 (fwhm/2)^2 / snr0
            # fwhm = 2 sqrt(snr0 / d2snr/dchan2)
            fwhm0 = self.kernel.fwhm(chan)
            h = int(max(1, 0.2 * fwhm0))
            d2 = (1 * self.snr[chan - h]
                  - 2 * self.snr[chan]
                  + 1 * self.snr[chan + h]) / h**2
            if d2 >= 0:
                raise PeakFinderError(
                    'Second derivative must be negative at peak')
            d2 *= -1
            fwhm = 2 * np.sqrt(self.snr[chan] / d2)
            self.fwhms.append(fwhm)
            # add the peak if it has a similar FWHM to the kernel's FWHM
            if self.fwhm_tol[0] * fwhm0 <= fwhm <= self.fwhm_tol[1] * fwhm0:
                self.channels.append(chan)
                self.snrs.append(self.snr[chan])
                self.fwhms.append(fwhm)
                self.integrals.append(self._signal[chan])
                self.backgrounds.append(self._bkg[chan])
        # sort the peaks by channel
        self.sort_by(self.channels)

    def plot(self, facecolor='red', linecolor='red', alpha=0.5, peaks=True):
        """Plot the peak signal-to-noise ratios calculated using the kernel."""
        if facecolor is not None:
            plt.fill_between(
                self.spectrum.channels, self.snr, 0,
                color=facecolor, alpha=alpha)
        if linecolor is not None:
            plt.plot(self.spectrum.channels, self.snr, '-', color=linecolor)
        if peaks:
            for chan, snr, fwhm in zip(self.channels, self.snrs, self.fwhms):
                plt.plot([chan] * 2, [0, snr], 'b-', lw=1.5)
                plt.plot(chan, snr, 'bo')
                plt.plot(
                    [chan - fwhm / 2, chan + fwhm / 2], [snr / 2] * 2,
                    'b-', lw=1.5)
        plt.xlim(0, len(self.spectrum))
        plt.ylim(0)
        plt.xlabel('Channels')
        plt.ylabel('SNR')

    def find_peak(self, channel, frac_range=(0.8, 1.2), min_snr=2):
        """Find the highest SNR peak within f0*channel and f1*channel."""
        chan_min = self.spectrum.channels.min()
        chan_max = self.spectrum.channels.max()
        if channel < chan_min or channel > chan_max:
            raise PeakFinderError(
                'Channel {} is outside of range {}-{}'.format(
                    channel, chan_min, chan_max))
        if frac_range[0] < 0 or frac_range[0] > 1 or frac_range[1] < 1 or \
                frac_range[0] > frac_range[1]:
            raise PeakFinderError(
                'Fractional range {}-{} is invalid'.format(*frac_range))
        if min_snr < 0:
            raise PeakFinderError(
                'Minimum SNR {:.3f} must be > 0'.format(min_snr))
        if self.snr.max() < min_snr:
            raise PeakFinderError(
                'SNR threshold is {:.3f} but maximum SNR is {:.3f}'.format(
                    min_snr, self.snr.max()))
        chan0 = frac_range[0] * channel
        chan1 = frac_range[1] * channel
        chan_range = \
            (chan0 <= self.spectrum.channels) & \
            (self.spectrum.channels <= chan1)
        peak_snr = self.snr[chan_range].max()
        if peak_snr < min_snr:
            raise PeakFinderError(
                'No peak found in range {}-{} with SNR > {}'.format(
                    chan0, chan1, min_snr))
        peak_chan = np.where((self.snr == peak_snr) & chan_range)[0][0]
        self.add_peak(peak_chan)
        return peak_chan

    def find_peaks(
            self, min_chan=None, max_chan=None, min_snr=2, max_num=40):
        """Find the highest SNR peaks in the data."""
        if min_chan is None:
            min_chan = self.spectrum.channels.min()
        if max_chan is None:
            max_chan = self.spectrum.channels.max()
        if min_chan < self.spectrum.channels.min() or \
                min_chan > self.spectrum.channels.max() or \
                max_chan > self.spectrum.channels.max() or \
                max_chan < self.spectrum.channels.min() or \
                min_chan > max_chan:
            raise PeakFinderError(
                'Channel range {}-{} is invalid'.format(min_chan, max_chan))
        if min_snr < 0:
            raise PeakFinderError(
                'Minimum SNR {:.3f} must be > 0'.format(min_snr))
        if self.snr.max() < min_snr:
            raise PeakFinderError(
                'SNR threshold is {:.3f} but maximum SNR is {:.3f}'.format(
                    min_snr, self.snr.max()))
        max_num = int(max_num)
        if max_num < 1:
            raise PeakFinderError(
                'Must keep at least 1 peak, not {}'.format(max_num))
        # calculate the first derivative and second derivatives of the SNR
        d1 = (self.snr[2:] - self.snr[:-2]) / 2
        d1 = np.append(0, d1)
        d1 = np.append(d1, 0)
        d2 = self.snr[2:] - 2 * self.snr[1:-1] + self.snr[:-2]
        d2 = np.append(0, d2)
        d2 = np.append(d2, 0)
        # find maxima
        peak = (d1[2:] < 0) & (d1[:-2] > 0) & (d2[1:-1] < 0)
        peak = np.append(False, peak)
        peak = np.append(peak, False)
        # select peaks using SNR and channel criteria
        peak &= (min_snr <= self.snr)
        peak &= (min_chan <= self.spectrum.channels)
        peak &= (self.spectrum.channels <= max_chan)
        for chan in self.spectrum.channels[peak]:
            self.add_peak(chan)
        # reduce number of channels to a maximum number max_n of highest SNR
        self.sort_by(np.array(self.snrs))
        self.channels = self.channels[-max_num:]
        self.snrs = self.snrs[-max_num:]
        self.fwhms = self.fwhms[-max_num:]
        self.integrals = self.integrals[-max_num:]
        self.backgrounds = self.backgrounds[-max_num:]
        # sort by channel
        self.sort_by(self.channels)
