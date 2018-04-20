"""Tools for plotting spectra."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy


class PlottingError(Exception):
    """General exception for plotting.py"""

    pass


class SpectrumPlotter(object):
    """Class for handling spectrum plotting."""

    def __init__(self, spec, *fmt, x_mode = None, 
                 y_mode = None, ax = None, yscale = None, title = None, **kwargs):
        """
        Args:
          spec:   Spectrum instance to plot
          fmt:    matplotlib like plot format string
          x_mode: define what is plotted on x axis ('energy' or 'channel')
          y_mode: define what is plotted on y axis ('counts', 'cps', 'cpskev' 
                  or 'eval_over')
          ax:     matplotlib axes object, if not provided one is created with the
                  argument provided in 'figsize'
          yscale: matplotlib scale: 'linear', 'log', 'logit', 'symlog'
          title:  costum plot title
          kwargs: arguments that are directly passed to matplotlib plot command
        """

        self.spec   = spec
        if len(fmt) == 0:
            self.fmt  = ''
        elif len(fmt) == 1:
            self.fmt    = fmt[0]
        else:
          raise PlottingError('Wrong number of positional parameters')
        
        self.ax     = ax
        self.yscale = yscale
        self.title  = title
        self.kwargs = kwargs

        if ax is None:
            if 'figsize' in kwargs:
                _, self.ax = plt.subplots(figsize=kwargs.pop('figsize'))
            else:
                _, self.ax = plt.subplots()
        else:
            if 'figsize' in kwargs:
                raise PlottingError('It is not possible to provide ax and figsize at the same time')
            self.ax = ax

        self.x_mode = None
        self.y_mode = None
        self._handle_data_modes(x_mode, y_mode, **kwargs)
        self._handle_data_and_labels()


    def _handle_data_modes(self, x_mode=None, y_mode=None, **kwargs):
        """
        Define x and y data mode, handles all data errors
        
        Args:
          x_mode: energy, channel
          y_mode: counts, cps, cpskev, eval_over
        """

        mode = None
        if x_mode is None:
            if self.spec.is_calibrated:
                mode = 'energy'
            else:
                mode = 'channel'
        else:
            if x_mode.lower() in ('kev', 'energy'):
                if not self.spec.is_calibrated:
                    raise PlottingError('Spectrum is not calibrated, however x axis was requested as energy')
                mode = 'energy'
            elif x_mode.lower() in ('channel', 'channels'):
                mode = 'channel'

        if mode is None:
            raise PlottingError('Unknown x data mode: {}'.format(x_mode))
        elif self.x_mode is not None and self.x_mode is not mode:
            raise PlottingError('Spectrum x data mode does not fit predefined SpectrumPlotter mode')
        self.x_mode = mode

        mode = None
        if 'eval_over' in kwargs:
            self.eval_time = kwargs.pop('eval_over')
            mode = 'eval_over'
            self.kwargs = kwargs
        elif y_mode is None:
            if self.spec.counts is not None:
                mode = 'counts'
            elif self.spec.cps is not None:
                mode = 'cps'
            elif self.spec.cpskev is not None:
                mode = 'cpskev'
            else:
                raise PlottingError('Cannot evaluate y data from spectrum')
        elif y_mode.lower() in ('count', 'counts', 'cnt', 'cnts'):
            if self.spec.counts is None:
                raise PlottingError('Spectrum has counts not defined')
            mode = 'counts'
        elif y_mode.lower() == 'cps':
            if self.spec.cps is None:
                raise PlottingError('Spectrum has cps not defined')
            mode = 'cps'
        elif y_mode.lower() == 'cpskev':
            if self.spec.cps is None:
                raise PlottingError('Spectrum has cps not defined')
            mode = 'cpskev'

        if mode is None:
            raise PlottingError('Unknown y data mode: {}'.format(y_mode))
        elif self.y_mode is not None and self.y_mode is not mode:
            raise PlottingError('Spectrum y data mode does not fit predefined SpectrumPlotter mode')
        self.y_mode = mode


    def _handle_data_and_labels(self):
        """Define xedges/xlabel and ydata/ylabel. Assumes x_mode and y_mode have been properly set."""

        if self.x_mode == 'energy':
            self.xedges = self.spec.bin_edges_kev
            self.xlabel = 'Energy [keV]'
        elif self.x_mode == 'channel':
            self.xedges = self.get_channel_edges()
            self.xlabel = 'Channel'

        if self.y_mode == 'counts':
            self.ydata = self.spec.counts_vals
            self.ylabel = 'Counts'
        elif self.y_mode == 'cps':
            self.ydata = self.spec.cps_vals
            self.ylabel = 'Countrate [1/s]'
        elif self.y_mode == 'cpskev':
            self.ydata = self.spec.cpskev_vals
            self.ylabel = 'Countrate [1/s/keV]'
        elif self.y_mode == 'eval_over':
            self.ydata = self.spec.counts_vals_over(self.eval_time)
            self.ylabel = 'Countrate [1/s]'


    def _prepare_plot(self):
        """Prepare for the plotting."""

        self.xcorners, self.ycorners = self.bin_edges_and_heights_to_steps(
            self.xedges, unumpy.nominal_values(self.ydata))
        if not self.ax.get_xlabel():
            self.ax.set_xlabel(self.xlabel)
        if not self.ax.get_ylabel():
            self.ax.set_ylabel(self.ylabel)
        if self.yscale is not None:
            self.ax.set_yscale(self.yscale)
        if self.title is not None:
            self.ax.set_title(self.title)
        elif self.spec.infilename is not None:
            self.ax.set_title(self.infilename)


    def plot(self):
        """Create actual plot with matplotlib's plot method"""

        self._prepare_plot()
        self.ax.plot(self.xcorners, self.ycorners, self.fmt, **self.kwargs)
        return self.ax


    def fill_between(self):
        """Create actual plot with matplotlib's fill_between method"""

        self._prepare_plot()
        self.ax.fill_between(self.xcorners, self.ycorners, **self.kwargs)
        return self.ax


    def get_channel_edges(self):
        """Get a vector of xedges for uncalibrated channels."""

        n_edges = len(self.spec.channels) + 1
        return np.linspace(-0.5, self.spec.channels[-1] + 0.5, num=n_edges)


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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
