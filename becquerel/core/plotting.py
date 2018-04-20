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

        if 'figsize' in kwargs:
            self._figsize = kwargs.pop('figsize')
        else:
            self._figsize = None
        if 'eval_over' in kwargs:
            self._eval_time = kwargs.pop('eval_over')
        else:
            self._eval_time = None

        self.yscale = yscale
        self.ax     = ax
        self.title  = title
        self.kwargs = kwargs

        self.x_mode = x_mode
        self.y_mode = y_mode

    @property
    def x_mode(self):
        """
        Returns the current x axis plotting mode
        """
        return self._x_mode


    @x_mode.setter
    def x_mode(self, mode):
        """
        Define x data mode, handles all data errors, requires spec
        
        Args:
          mode: energy (or kev), channel (or channels, chn, chns)
        """

        if mode is None:
            if self.spec.is_calibrated:
                self._x_mode = 'energy'
            else:
                self._x_mode = 'channel'
        else:
            if mode.lower() in ('kev', 'energy'):
                if not self.spec.is_calibrated:
                    raise PlottingError('Spectrum is not calibrated, however x axis was requested as energy')
                self._x_mode = 'energy'
            elif mode.lower() in ('channel', 'channels', 'chn', 'chns'):
                self._x_mode = 'channel'
            else: 
                raise PlottingError('Unknown x data mode: {}'.format(mode))

        if self._x_mode == 'energy':
            self._xedges = self.spec.bin_edges_kev
            self._xlabel = 'Energy [keV]'
        elif self._x_mode == 'channel':
            self._xedges = self.get_channel_edges(self.spec.channels)
            self._xlabel = 'Channel'


    @property
    def y_mode(self):
        """
        Returns the current y axis plotting mode
        """
        return self._y_mode


    @y_mode.setter
    def y_mode(self, mode):
        """
        Define y data mode, handles all data errors, requires spec
        
        Args:
          mode: counts, cps, cpskev, eval_over
        """

        if self._eval_time is not None:
            self._y_mode = 'eval_over'
        elif mode is None:
            if self.spec.counts is not None:
                self._y_mode = 'counts'
            elif self.spec.cps is not None:
                self._y_mode = 'cps'
            elif self.spec.cpskev is not None:
                self._y_mode = 'cpskev'
            else:
                raise PlottingError('Cannot evaluate y data from spectrum')
        elif mode.lower() in ('count', 'counts', 'cnt', 'cnts'):
            if self.spec.counts is None:
                raise PlottingError('Spectrum has counts not defined')
            self._y_mode = 'counts'
        elif mode.lower() == 'cps':
            if self.spec.cps is None:
                raise PlottingError('Spectrum has cps not defined')
            self._y_mode = 'cps'
        elif mode.lower() == 'cpskev':
            if self.spec.cps is None:
                raise PlottingError('Spectrum has cps not defined')
            self._y_mode = 'cpskev'
        else:
            raise PlottingError('Unknown y data mode: {}'.format(mode))

        if self._y_mode == 'counts':
            self._ydata = self.spec.counts_vals
            self._ylabel = 'Counts'
        elif self._y_mode == 'cps':
            self._ydata = self.spec.cps_vals
            self._ylabel = 'Countrate [1/s]'
        elif self._y_mode == 'cpskev':
            self._ydata = self.spec.cpskev_vals
            self._ylabel = 'Countrate [1/s/keV]'
        elif self._y_mode == 'eval_over':
            self._ydata = self.spec.counts_vals_over(self._eval_time)
            self._ylabel = 'Countrate [1/s]'


    @property
    def ax(self):
        """
        Returns the current matplotlib axes object used for plotting.
        If no axes object is defined yet it will create one.
        """

        if self._ax is None:
            if self._figsize is None:
                _, self._ax = plt.subplots()
            else:
                _, self._ax = plt.subplots(self._figsize)
        return self._ax


    @ax.setter
    def ax(self, ax):
        """
        Defines the current matplotlib axes object used for plotting.
        Is affected by the figsize member variable, if set
        
        Args:
          ax: Axes to be set
        """

        self._ax = ax
        if ax is not None and self.yscale is None:
            self.yscale = ax.get_yscale()

    @property
    def xlabel(self):
        """
        Returns the current x label
        """
        return self._xlabel


    @property
    def ylabel(self):
        """
        Returns the current y label
        """
        return self._ylabel


    def get_corners(self):
        """
        Creates a stepped version of the current spectrum data
        
        Return:
          xcorner, ycorner: x and y values that can be used directly in
                            matplotlib's plotting function
        """

        return self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata))


    def _prepare_plot(self, **kwargs):
        """Prepare for the plotting."""

        self.kwargs.update(**kwargs)
        if not self.ax.get_xlabel():
            self.ax.set_xlabel(self._xlabel)
        if not self.ax.get_ylabel():
            self.ax.set_ylabel(self._ylabel)
        if self.yscale is not None:
            self.ax.set_yscale(self.yscale)
        if self.title is not None:
            self.ax.set_title(self.title)
        elif self.spec.infilename is not None:
            self.ax.set_title(self.spec.infilename)
        return self.get_corners()


    def plot(self, **kwargs):
        """
        Create actual plot with matplotlib's plot method.
        
        Args:
          kwargs: Any matplotlib plot() keyword argument, overwrites
                  previously defined keywords
        """

        xcorners, ycorners = self._prepare_plot(**kwargs)
        self.ax.plot(xcorners, ycorners, self.fmt, **self.kwargs)
        return self.ax


    def fill_between(self, **kwargs):
        """
        Create actual plot with matplotlib's fill_between method.
        
        Args:
          kwargs: Any matplotlib fill_between() keyword argument, overwrites
                  previously defined keywords
        """
        
        xcorners, ycorners = self._prepare_plot(**kwargs)
        self.ax.fill_between(xcorners, ycorners, **self.kwargs)
        return self.ax

    @staticmethod
    def get_channel_edges(channels):
        """Get a vector of xedges for uncalibrated channels."""

        n_edges = len(channels) + 1
        return np.linspace(-0.5, channels[-1] + 0.5, num=n_edges)


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

    @property
    def xlim(self):
        """Returns the xlim, requires xedges"""

        return (np.min(self._xedges), np.max(self._xedges))


    @property
    def ylim(self):
        """Returns ylim, requires yscale, ydata"""

        if self.yscale is None:
            raise PlottingError('No y scale and no axes defined, requires at least one of them')

        min_ind = np.argmin(np.abs(self._ydata[self._ydata != 0]))
        delta_y = np.abs(self._ydata - self._ydata[min_ind])
        min_delta_y = np.min(delta_y[delta_y > 0])

        data_min = np.min(self._ydata)
        if self.yscale == 'linear':
            ymin = 0
        elif self.yscale == 'log' and data_min < 0:
            raise PlottingError('Cannot plot negative values on a log ' +
                                'scale; use symlog scale')
        elif self.yscale == 'symlog' and data_min >= 0:
            ymin = 0
        else:
            ymin = self.dynamic_min(data_min, min_delta_y)

        data_max = np.max(self._ydata)
        ymax = self.dynamic_max(data_max, self.yscale)

        return ymin, ymax


    @property
    def linthreshy(self):
        """Returns linthreshy, requires ydata"""

        min_ind = np.argmin(np.abs(self._ydata[self._ydata != 0]))
        delta_y = np.abs(self._ydata - self._ydata[min_ind])
        return np.min(delta_y[delta_y > 0])
