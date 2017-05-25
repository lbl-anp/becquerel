"""Tools for plotting spectra."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat, unumpy


class PlottingError(Exception):
    """General exception for plotting.py"""

    pass


def plot_spectrum(spec, fmtstring=None, **kwargs):
    """Plot a Spectrum object."""

    plotter = SpectrumPlotter(spec, fmtstring=fmtstring, **kwargs)
    plotter.plot()
    return plotter.axes


def bin_edges_and_heights_to_steps(bin_edges, heights):
    """A robust alternative to matplotlib's drawstyle='steps-*'"""

    assert len(bin_edges) == len(heights) + 1
    x = np.zeros(len(bin_edges) * 2)
    y = np.zeros_like(x)
    x[::2] = bin_edges.astype(float)
    x[1::2] = bin_edges.astype(float)
    y[1:-1:2] = heights.astype(float)
    y[2:-1:2] = heights.astype(float)
    return x, y


class SpectrumPlotter(object):
    """Class for handling spectrum plotting."""

    def __init__(self, spec, **kwargs):
        """
        Args:
          spec: Spectrum instance to plot
        """

        self.spec = spec
        self._handle_axes(**kwargs)
        self._handle_title(**kwargs)
        self._handle_fmt(**kwargs)
        self._handle_counts_mode(**kwargs)
        self._handle_axes_scales(**kwargs)
        self._handle_axes_labels(**kwargs)
        self._handle_axes_data()
        self._handle_axes_limits(**kwargs)

    def _handle_fmt(self, fmtstring=None, **kwargs):
        """Define fmtstring"""

        if fmtstring is None:
            self.fmtstring = ''
        else:
            self.fmtstring = fmtstring

    def _handle_axes(self, axes=None, **kwargs):
        """Define axes, new_axes"""

        if axes is None:
            self.axes = None
            self.new_axes = True
        else:
            self.axes = axes
            self.new_axes = False

    def _handle_title(self, title=None, **kwargs):
        """Define title"""

        if title is None:
            self.title = 'Becquerel Spectrum'
        else:
            self.title = title

    def _handle_counts_mode(self, counts_mode=None, **kwargs):
        """Define counts_mode"""

        if counts_mode is not None:
            if counts_mode.lower() not in ('counts', 'cps', 'cpskev'):
                raise ValueError('Bad counts_mode: {}'.format(counts_mode))
            self.counts_mode = counts_mode.lower()
        elif self.spec.counts is not None:
            self.counts_mode = 'counts'
        else:
            self.counts_mode = 'cps'

    def _handle_axes_scales(self, yscale=None, **kwargs):
        """Define xscale and yscale. Requires ydata."""

        self.xscale = 'linear'

        if yscale is not None:
            if yscale.lower() not in ('linear', 'log', 'symlog'):
                raise ValueError('Bad yscale: {}'.format(yscale))
            self.yscale = yscale.lower()
        else:
            self.yscale = 'symlog'

    def _handle_axes_labels(self, xlabel=None, ylabel=None, **kwargs):
        """Define xlabel and ylabel. Requires counts_mode.
        """

        if xlabel is not None:
            self.xlabel = xlabel
        elif self.spec.is_calibrated:
            self.xlabel = 'Energy [keV]'
        else:
            self.xlabel = 'Energy [channels]'

        if ylabel is not None:
            self.ylabel = ylabel
        elif self.counts_mode == 'counts':
            self.ylabel = 'Counts'
        elif self.counts_mode == 'cps':
            self.ylabel = 'Countrate [1/s]'
        elif self.counts_mode == 'cpskev':
            self.ylabel = 'Countrate [1/s/keV]'

    def _handle_axes_data(self):
        """Define xedges and ydata. Requires counts_mode.
        """

        if self.spec.is_calibrated:
            self.xedges = self.spec.bin_edges_kev
        else:
            self.xedges = self.get_channel_edges()

        if self.counts_mode == 'counts':
            if self.spec.counts is None:
                raise PlottingError('Cannot plot counts of CPS-only spectrum')
            self.ydata = self.spec.counts_vals
        elif self.counts_mode == 'cps':
            self.ydata = self.spec.cps_vals
        elif self.counts_mode == 'cpskev':
            self.ydata = self.spec.cpskev_vals

    def get_channel_edges(self):
        """Get a vector of xedges for uncalibrated channels.
        """

        return np.arange(-0.5, len(self.spec) - 0.4)

    def _handle_axes_limits(self, xlim=None, ylim=None, **kwargs):
        """Define xlim, ylim, linthreshy.
        Requires xscale, yscale, xedges, ydata
        """

        if xlim is not None:
            if len(xlim) != 2:
                raise PlottingError('xlim should be length 2: {}'.format(xlim))
            self.xlim = xlim
        else:
            self.xlim = (np.min(self.xedges), np.max(self.xedges))

        if ylim is not None:
            if len(ylim) != 2:
                raise PlottingError('ylim should be length 2: {}'.format(ylim))
            self.ylim = ylim
        else:
            # y min
            data_min = np.min(self.ydata)
            if self.yscale == 'linear':
                ymin = 0
            elif self.yscale == 'log' and data_min < 0:
                raise PlottingError('Cannot plot negative values on a log ' +
                                    'scale; use symlog scale')
            elif self.yscale == 'symlog' and data_min > 0:
                ymin = 0
            elif self.yscale == 'log':
                # dynamic ymin, positive values
                ceil10 = 10**(np.ceil(np.log10(data_min)))
                sig_fig = np.floor(data_min / ceil10)
                ymin = sig_fig * ceil10 / 10   # 0.1 for min of 1
            elif self.yscale == 'symlog':
                # dynamic ymin, negative values
                ceil10 = 10**(np.ceil(np.log10(data_min)))
                sig_fig = np.floor(data_min / ceil10)
                if sig_fig < 3:
                    ymin = ceil10 * 3
                else:
                    ymin = ceil10 * 10

            # y max
            data_max = np.max(self.ydata)
            floor10 = 10**(np.floor(np.log10(data_max)))
            sig_fig = np.ceil(data_max / floor10)
            if self.yscale == 'linear':
                ymax = floor10 * (sig_fig + 1)
            elif sig_fig < 3:
                ymax = floor10 * 3
            else:
                ymax = floor10 * 10

            self.ylim = (ymin, ymax)

        if self.yscale == 'symlog':
            min_ind = np.argmin(np.abs(self.ydata[self.ydata != 0]))
            min_delta_y = np.min(np.abs(self.ydata - self.ydata[min_ind]))
            # floor10_abs = 10**(np.floor(np.log10(min_delta_y)))
            # sig_fig_abs = np.ceil(min_delta_y / floor10_abs)

            self.linthreshy = np.abs(min_delta_y)

    def plot(self):
        """Create actual plot."""

        if self.new_axes:
            self.axes = plt.axes()
        else:
            plt.axes(self.axes)

        plt.plot(self.xdata, unumpy.nominal_values(self.ydata), self.fmtstring,
                 axes=self.axes, drawstyle='steps-post',
                 **self.kwargs)
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_yscale(self.yscale, **self.yscale_kwargs)
        self.axes.set_ylim(self.ylim)

        plt.show()
