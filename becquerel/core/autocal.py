"""Automatic calibration by associating peaks with energies."""

from __future__ import print_function
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from .peakfinder import PeakFinder
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


class AutoCalibratorError(Exception):
    """Base class for errors in AutoCalibrator."""

    pass


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
    if len(channels) != len(energies):
        raise AutoCalibratorError(
            'Number of channels ({}) must equal # of energies ({})'.format(
                len(channels), len(energies)))
    if len(channels) != len(snrs):
        raise AutoCalibratorError(
            'Number of channels ({}) must equal # of SNRs ({})'.format(
                len(channels), len(snrs)))
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
    if len(channels) != len(energies):
        raise AutoCalibratorError(
            'Number of channels ({}) must equal # of energies ({})'.format(
                len(channels), len(energies)))
    if len(channels) != len(snrs):
        raise AutoCalibratorError(
            'Number of channels ({}) must equal # of SNRs ({})'.format(
                len(channels), len(snrs)))
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
        gain_range=(1e-3, 1e3), de_max=10., verbose=False):
    """Find the gain that gives the best match of peaks to energies."""
    if len(channels) != len(snrs):
        raise AutoCalibratorError(
            'Number of channels ({}) must equal # of SNRs ({})'.format(
                len(channels), len(snrs)))
    if len(channels) < 2:
        raise AutoCalibratorError(
            'Number of channels ({}) must be at least 2'.format(len(channels)))
    if len(required_energies) < 2:
        raise AutoCalibratorError(
            'Number of required energies ({}) must be at least 2'.format(
                len(required_energies)))
    if len(channels) < len(required_energies):
        raise AutoCalibratorError(
            'Number of channels ({}) must be >= required energies ({})'.format(
                len(channels), len(required_energies)))

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
        if verbose:
            print('Searching groups of {}'.format(n_set))
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
                if verbose:
                    print('v')
                    s0 = 'Valid calibration found:\n'
                    s0 += 'FOM: {:15.9f}'.format(fom)
                    s0 += '  gain: {:6.3f}'.format(gain)
                    s0 += '  ergs: {:50s}'.format(str(comb_erg))
                    s0 += '  de: {:50s}'.format(str(de))
                    s0 += '  chans: {:40s}'.format(str(comb_chan))
                    print(s0)
                if best_fom is None:
                    best_fom = fom + 1.
                if fom < best_fom:
                    best_fom = fom
                    best_gain = gain
                    best_chans = comb_chan
                    best_snrs = comb_snr
                    best_ergs = comb_erg
                    if verbose:
                        s0 = 'Best calibration so far:\n'
                        s0 += 'FOM: {:15.9f}'.format(best_fom)
                        s0 += '  gain: {:6.3f}'.format(best_gain)
                        s0 += '  ergs: {:50s}'.format(str(best_ergs))
                        s0 += '  de: {:50s}'.format(str(de))
                        s0 += '  chans: {:40s}'.format(str(best_chans))
                        print(s0)
        n_set -= 1
    if best_gain is None:
        return None
    else:
        print('found best gain: %f keV/channel' % best_gain)
        return {
            'gain': best_gain,
            'channels': best_chans,
            'snrs': best_snrs,
            'energies': best_ergs,
        }


class AutoCalibrator(object):
    """Automatically calibrate a spectrum by convolving it with a filter.

    A note on nomenclature: for historic reasons, 'channels' is used in
    autocal.py for generic uncalibrated x-axis values. A 'channel' is no longer
    necessarily an integer channel number (i.e., bin) from a multi-channel
    analyzer, but could for instance be a float-type fC of charge collected.
    """

    def __init__(self, peakfinder):
        """Initialize the calibration with a spectrum and kernel."""
        self.set_peaks(peakfinder)
        self.gain = None
        self.cal = None
        self.fit_channels = []
        self.fit_snrs = []
        self.fit_energies = []
        self.reset()

    def reset(self):
        """Reset all of the members."""
        self.gain = None
        self.cal = None
        self.fit_channels = []
        self.fit_snrs = []
        self.fit_energies = []

    def set_peaks(self, peakfinder):
        """Use the peaks found by the PeakFinder."""
        if not isinstance(peakfinder, PeakFinder):
            raise AutoCalibratorError(
                'Argument must be a PeakFinder, not {}'.format(
                    type(peakfinder)))
        self.peakfinder = peakfinder

    def plot(self, **kwargs):
        """Plot the peaks found and the peaks used to fit."""
        self.peakfinder.plot(peaks=True, **kwargs)
        for chan, snr in zip(self.fit_channels, self.fit_snrs):
            plt.plot([chan] * 2, [0, snr], 'g-', lw=2)
            plt.plot(chan, snr, 'go')

    def fit(self, required_energies, optional=(),
            gain_range=(1e-4, 1e3), de_max=10., verbose=False):
        """Find the gain that gives the best match of peaks to energies."""
        if len(self.peakfinder.centroids) == 1 and \
                len(required_energies) == 1:
            # special case: only one line identified
            self.fit_channels = list(self.peakfinder.centroids)
            self.fit_snrs = list(self.peakfinder.snrs)
            self.fit_energies = list(required_energies)
            gain = required_energies[0] / self.peakfinder.centroids[0]
            self.gain = gain
            self.cal = LinearEnergyCal.from_coeffs(
                {'offset': 0, 'slope': self.gain})
            return
        # handle the usual case: multiple lines to match
        if len(self.peakfinder.centroids) < 2:
            raise AutoCalibratorError(
                'Need more than {} peaks to fit'.format(
                    len(self.peakfinder.centroids)))
        if len(required_energies) < 2:
            raise AutoCalibratorError(
                'Need more than {} energies to fit'.format(
                    len(required_energies)))
        if len(self.peakfinder.centroids) < len(required_energies):
            raise AutoCalibratorError(
                'Require {} energies but only {} peaks are available'.format(
                    len(required_energies), len(self.peakfinder.centroids)))
        fit = find_best_gain(
            self.peakfinder.centroids, self.peakfinder.snrs,
            required_energies, optional=optional, gain_range=gain_range,
            de_max=de_max, verbose=verbose)
        if fit is None:
            self.fit_channels = []
            self.fit_snrs = []
            self.fit_energies = []
            self.gain = None
            self.cal = None
            raise AutoCalibratorError('No valid fit was found')
        else:
            self.fit_channels = fit['channels']
            self.fit_snrs = fit['snrs']
            self.fit_energies = fit['energies']
            self.gain = fit['gain']
            self.cal = LinearEnergyCal.from_coeffs(
                {'offset': 0, 'slope': self.gain})
