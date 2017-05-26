"""Tools for plotting spectra."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy


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
        self._handle_kwargs(**kwargs)

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

        if 'eval_over' in kwargs:
            self.eval_time = kwargs['eval_over']
            self.counts_mode = 'eval_over'
        elif counts_mode is not None:
            if counts_mode.lower() not in ('counts', 'cps', 'cpskev'):
                raise ValueError('Bad counts_mode: {}'.format(counts_mode))
            self.counts_mode = counts_mode.lower()
        elif self.spec.counts is not None:
            self.counts_mode = 'counts'
        else:
            self.counts_mode = 'cps'

    def _handle_axes_scales(self, yscale=None, **kwargs):
        """Define xscale and yscale."""

        self.xscale = 'linear'

        if yscale is not None:
            if yscale.lower() not in ('linear', 'log', 'symlog'):
                raise ValueError('Bad yscale: {}'.format(yscale))
            self.yscale = yscale.lower()
        else:
            self.yscale = 'symlog'
        self.default_linscaley = 1.5

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
        elif self.counts_mode == 'eval_over':
            self.ydata = self.spec.counts_vals_over(self.eval_time)

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

        min_ind = np.argmin(np.abs(self.ydata[self.ydata != 0]))
        delta_y = np.abs(self.ydata - self.ydata[min_ind])
        min_delta_y = np.min(delta_y[delta_y > 0])

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
            elif self.yscale == 'symlog' and data_min >= 0:
                ymin = 0
            else:
                ymin = dynamic_min(data_min, min_delta_y)

            data_max = np.max(self.ydata)
            ymax = dynamic_max(data_max, self.yscale)

            self.ylim = (ymin, ymax)

        if self.yscale == 'symlog':
            self.linthreshy = min_delta_y

    def _handle_kwargs(self, **kwargs):
        """Get kwargs for plt.plot and plt.yscale."""

        if 'plot_kwargs' in kwargs:
            self.plot_kwargs = kwargs['plot_kwargs']
        else:
            self.plot_kwargs = {}
        if 'label' in kwargs:
            self.plot_kwargs['label'] = kwargs['label']

        if 'yscale_kwargs' in kwargs:
            self.yscale_kwargs = kwargs['yscale_kwargs']
        else:
            self.yscale_kwargs = {}
        if self.yscale == 'symlog' and 'linscaley' not in self.yscale_kwargs:
            self.yscale_kwargs['linscaley'] = self.default_linscaley

    def plot(self):
        """Create actual plot."""

        if self.new_axes:
            self.axes = plt.axes()
        else:
            plt.axes(self.axes)

        self.xcorners, self.ycorners = bin_edges_and_heights_to_steps(
            self.xedges, unumpy.nominal_values(self.ydata))

        plt.plot(self.xcorners, self.ycorners, self.fmtstring,
                 axes=self.axes, **self.plot_kwargs)
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_yscale(self.yscale, **self.yscale_kwargs)
        self.axes.set_ylim(self.ylim)

        plt.show()


def dynamic_min(data_min, min_delta_y):
    """Get an axes lower limit (for y) based on data value.

    The lower limit is the next power of 10, or 3 * power of 10, below the min.

    Args:
      data_min: the minimum of the data (could be integers or floats)
      min_delta_y: the minimum step in y
    """

    if data_min > 0:
        ceil10 = 10**(np.ceil(np.log10(data_min)))
        sig_fig = np.floor(10 * data_min / ceil10)
        if sig_fig <= 3:
            ymin = ceil10 / 10
        else:
            ymin = ceil10 / 10 * 3
    elif data_min == 0:
        ymin = min_delta_y / 10.0
    else:
        # negative
        floor10 = 10**(np.floor(np.log10(-data_min)))
        sig_fig = np.floor(-data_min / floor10)
        if sig_fig < 3:
            ymin = -floor10 * 3
        else:
            ymin = -floor10 * 10

    return ymin


def dynamic_max(data_max, yscale):
    """Get an axes upper limit (for y) based on data value.

    The upper limit is the next power of 10, or 3 * power of 10, above the max.
    (For linear, the next N * power of 10.)

    Args:
      data_max: the maximum of the data (could be integers or floats)
    """

    floor10 = 10**(np.floor(np.log10(data_max)))
    sig_fig = np.ceil(data_max / floor10)
    if yscale == 'linear':
        sig_fig = np.floor(data_max / floor10)
        ymax = floor10 * (sig_fig + 1)
    elif sig_fig < 3:
        ymax = floor10 * 3
    else:
        ymax = floor10 * 10

    return np.maximum(ymax, 0)
