# -*- coding: utf-8 -*-

"""Spectral peak search using convolutions."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import warnings
from .spectrum import Spectrum


class PeakFilterError(Exception):
    """Base class for errors in PeakFilter."""

    pass


class PeakFilter(object):
    """An energy-dependent kernel that can be convolved with a spectrum.

    To detect lines, a kernel should have a positive component in the center
    and negative wings to subtract the continuum, e.g., a Gaussian or a boxcar:

    +2|     ┌───┐     |
      |     │   │     |
     0|─┬───┼───┼───┬─|
    -1| └───┘   └───┘ |

    The positive part is meant to detect the peak, and the negative part to
    sample the continuum under the peak.

    The kernel should sum to 0.

    The width of the kernel scales proportionally to the square root
    of the x values (which could be energy, ADC channels, fC of charge
    collected, etc.), with a minimum value set by 'fwhm_at_0'.

    """

    def __init__(self, ref_x, ref_fwhm, fwhm_at_0=1.):
        """Initialize with a reference line position and FWHM in x-values."""
        if ref_x <= 0:
            raise PeakFilterError('Reference x must be positive')
        if ref_fwhm <= 0:
            raise PeakFilterError('Reference FWHM must be positive')
        if fwhm_at_0 < 0:
            raise PeakFilterError('FWHM at 0 must be non-negative')
        self.ref_x = float(ref_x)
        self.ref_fwhm = float(ref_fwhm)
        self.fwhm_at_0 = float(fwhm_at_0)

    def fwhm(self, x):
        """Calculate the expected FWHM at the given x value."""
        # f(x)^2 = f0^2 + k x^2
        # f1^2 = f0^2 + k x1^2
        # k = (f1^2 - f0^2) / x1^2
        # f(x)^2 = f0^2 + (f1^2 - f0^2) (x/x1)^2
        f0 = self.fwhm_at_0
        f1 = self.ref_fwhm
        x1 = self.ref_x
        fwhm_sqr = f0**2 + (f1**2 - f0**2) * (x / x1)**2
        return np.sqrt(fwhm_sqr)

    def kernel(self, x, edges):
        """Generate the kernel for the given x value."""
        raise NotImplementedError

    def kernel_matrix(self, edges):
        """Build a matrix of the kernel evaluated at each x value."""
        n_channels = len(edges) - 1
        kern = np.zeros((n_channels, n_channels))
        for i, x in enumerate(edges[:-1]):
            kern[:, i] = self.kernel(x, edges)
        kern_pos = +1 * kern.clip(0, np.inf)
        kern_neg = -1 * kern.clip(-np.inf, 0)
        # normalize negative part to be equal to the positive part
        kern_neg *= kern_pos.sum(axis=0) / kern_neg.sum(axis=0)
        return kern_pos - kern_neg

    def plot_matrix(self, edges):
        """Plot the matrix of kernels evaluated across the x values."""
        n_channels = len(edges) - 1
        kern_mat = self.kernel_matrix(edges)
        kern_min = kern_mat.min()
        kern_max = kern_mat.max()
        kern_min = min(kern_min, -1 * kern_max)
        kern_max = max(kern_max, -1 * kern_min)

        plt.imshow(
            kern_mat.T[::-1, :], cmap=plt.get_cmap('bwr'),
            vmin=kern_min, vmax=kern_max,
            extent=[n_channels, 0, 0, n_channels])
        plt.colorbar()
        plt.xlabel('Input x')
        plt.ylabel('Output x')
        plt.gca().set_aspect('equal')

    def convolve(self, edges, data):
        """Convolve this kernel with the data."""
        kern_mat = self.kernel_matrix(edges)
        kern_mat_pos = +1 * kern_mat.clip(0, np.inf)
        kern_mat_neg = -1 * kern_mat.clip(-np.inf, 0)
        peak_plus_bkg = np.dot(kern_mat_pos, data)
        bkg = np.dot(kern_mat_neg, data)
        signal = np.dot(kern_mat, data)
        noise = np.sqrt(np.dot(kern_mat**2, data))
        snr = np.zeros_like(signal)
        snr[noise > 0] = signal[noise > 0] / noise[noise > 0]
        return peak_plus_bkg, bkg, signal, noise, snr


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

    def kernel(self, x, edges):
        """Generate the kernel for the given x value."""
        fwhm = self.fwhm(x)
        sigma = fwhm / 2.355
        g1_x0 = _gaussian1(edges[:-1], x, sigma)
        g1_x1 = _gaussian1(edges[1:], x, sigma)
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
                'Minimum x separation must be positive')
        self.min_sep = min_sep
        self.fwhm_tol = tuple(fwhm_tol)
        self.spectrum = None
        self.kernel = None
        self.snr = []
        self._peak_plus_bkg = []
        self._bkg = []
        self._signal = []
        self._noise = []
        self.centroids = []
        self.snrs = []
        self.fwhms = []
        self.integrals = []
        self.backgrounds = []
        self.calculate(spectrum, kernel)

    @property
    def channels(self):
        warnings.warn('channels is deprecated and will be removed in a future '
                      'release. Use centroids instead.', DeprecationWarning)
        return self.centroids

    def reset(self):
        """Restore peak finder to pristine starting condition."""
        self.centroids = []
        self.snrs = []
        self.fwhms = []
        self.integrals = []
        self.backgrounds = []

    def sort_by(self, arr):
        """Sort peaks by the provided array."""
        if len(arr) != len(self.centroids):
            raise PeakFinderError(
                'Sorting array has length {} but must have length {}'.format(
                    len(arr), len(self.centroids)))
        self.centroids = np.array(self.centroids)
        self.snrs = np.array(self.snrs)
        self.fwhms = np.array(self.fwhms)
        self.integrals = np.array(self.integrals)
        self.backgrounds = np.array(self.backgrounds)
        i = np.argsort(arr)
        self.centroids = list(self.centroids[i])
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

        bin_edges = self.spectrum.bin_edges_raw

        # calculate the convolution
        peak_plus_bkg, bkg, signal, noise, snr = \
            self.kernel.convolve(bin_edges, self.spectrum.counts_vals)
        self._peak_plus_bkg = peak_plus_bkg
        self._bkg = bkg
        self._signal = signal
        self._noise = noise
        self.snr = snr.clip(0)
        self.reset()

    def add_peak(self, xpeak):
        """Add a peak at xpeak to list if it is not already there."""
        bin_edges = self.spectrum.bin_edges_raw

        xmin = bin_edges.min()
        xmax = bin_edges.max()

        if xpeak < xmin or xpeak > xmax:
            raise PeakFinderError(
                'Peak x {} is outside of range {}-{}'.format(
                    xpeak, xmin, xmax))
        is_new_x = True
        for cent in self.centroids:
            if abs(xpeak - cent) <= self.min_sep:
                is_new_x = False
        if is_new_x:
            # estimate FWHM using the second derivative
            # snr(x) = snr(x0) - 0.5 d2snr/dx2(x0) (x-x0)^2
            # 0.5 = 1 - 0.5 d2snr/dx2 (fwhm/2)^2 / snr0
            # 1 = d2snr/dx2 (fwhm/2)^2 / snr0
            # fwhm = 2 sqrt(snr0 / d2snr/dx2)
            xbin = self.spectrum.find_bin_index(xpeak, use_kev=False)
            fwhm0 = self.kernel.fwhm(xpeak)
            bw = self.spectrum.bin_widths_raw[0]
            h = int(max(1, 0.2 * fwhm0 / bw))
            d2 = (1 * self.snr[xbin - h]
                  - 2 * self.snr[xbin]
                  + 1 * self.snr[xbin + h]) / h**2 / bw**2
            if d2 >= 0:
                raise PeakFinderError(
                    'Second derivative must be negative at peak')
            d2 *= -1
            fwhm = 2 * np.sqrt(self.snr[xbin] / d2)
            self.fwhms.append(fwhm)
            # add the peak if it has a similar FWHM to the kernel's FWHM
            if self.fwhm_tol[0] * fwhm0 <= fwhm <= self.fwhm_tol[1] * fwhm0:
                self.centroids.append(xpeak)
                self.snrs.append(self.snr[xbin])
                self.fwhms.append(fwhm)
                self.integrals.append(self._signal[xbin])
                self.backgrounds.append(self._bkg[xbin])
        # sort the peaks by centroid
        self.sort_by(self.centroids)

    def plot(self, facecolor='red', linecolor='red', alpha=0.5, peaks=True):
        """Plot the peak signal-to-noise ratios calculated using the kernel."""
        bin_edges = self.spectrum.bin_edges_raw

        if facecolor is not None:
            plt.fill_between(
                bin_edges[:-1], self.snr, 0,
                color=facecolor, alpha=alpha)
        if linecolor is not None:
            plt.plot(bin_edges[:-1], self.snr, '-', color=linecolor)
        if peaks:
            for cent, snr, fwhm in zip(self.centroids, self.snrs, self.fwhms):
                plt.plot([cent] * 2, [0, snr], 'b-', lw=1.5)
                plt.plot(cent, snr, 'bo')
                plt.plot(
                    [cent - fwhm / 2, cent + fwhm / 2], [snr / 2] * 2,
                    'b-', lw=1.5)
        plt.xlim(0, bin_edges.max())
        plt.ylim(0)
        plt.xlabel('x')
        plt.ylabel('SNR')

    def find_peak(self, xpeak, frac_range=(0.8, 1.2), min_snr=2):
        """Find the highest SNR peak within f0*xpeak and f1*xpeak."""
        bin_edges = self.spectrum.bin_edges_raw
        bin_centers = self.spectrum.bin_centers_raw
        xmin = bin_edges[0]
        xmax = bin_edges[-1]

        if xpeak < xmin or xpeak > xmax:
            raise PeakFinderError(
                'Guess xpeak {} is outside of range {}-{}'.format(
                    xpeak, xmin, xmax))
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
        x0 = frac_range[0] * xpeak
        x1 = frac_range[1] * xpeak
        x_range = (x0 <= bin_edges[:-1]) & (bin_edges[:-1] <= x1)
        peak_snr = self.snr[x_range].max()
        if peak_snr < min_snr:
            raise PeakFinderError(
                'No peak found in range {}-{} with SNR > {}'.format(
                    x0, x1, min_snr))

        peak_index = np.where((self.snr == peak_snr) & x_range)[0][0]
        peak_x = bin_centers[peak_index]
        self.add_peak(peak_x)
        return peak_x

    def find_peaks(self, xmin=None, xmax=None, min_snr=2, max_num=40):
        """Find the highest SNR peaks in the data."""
        bin_edges = self.spectrum.bin_edges_raw
        bin_centers = self.spectrum.bin_centers_raw

        if xmin is None:
            xmin = bin_edges.min()
        if xmax is None:
            xmax = bin_edges.max()
        if xmin < bin_edges.min() or \
                xmin > bin_edges.max() or \
                xmax > bin_edges.max() or \
                xmax < bin_edges.min() or \
                xmin > xmax:
            raise PeakFinderError(
                'x-axis range {}-{} is invalid'.format(xmin, xmax))
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
        # select peaks using SNR and centroid criteria
        peak &= (min_snr <= self.snr)
        peak &= (xmin <= bin_edges[:-1])
        peak &= (bin_edges[:-1] <= xmax)
        for x in bin_centers[peak]:
            self.add_peak(x)
        # reduce number of centroids to a maximum number max_n of highest SNR
        self.sort_by(np.array(self.snrs))
        self.centroids = self.centroids[-max_num:]
        self.snrs = self.snrs[-max_num:]
        self.fwhms = self.fwhms[-max_num:]
        self.integrals = self.integrals[-max_num:]
        self.backgrounds = self.backgrounds[-max_num:]
        # sort by centroid
        self.sort_by(self.centroids)
