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

    def __init__(self, spec, *fmt, **kwargs):
        """
        Args:
          spec:   Spectrum instance to plot
          fmt:    matplotlib like plot format string
          xmode:  define what is plotted on x axis ('energy' or 'channel')
          ymode:  define what is plotted on y axis ('counts', 'cps', 'cpskev')
          xlim:   set x axes limits, if set to 'default' use special scales
          ylim:   set y axes limits, if set to 'default' use special scales
          ax:     matplotlib axes object, if not provided one is created using
          yscale: matplotlib scale: 'linear', 'log', 'logit', 'symlog'
          title:  costum plot title, default is filename if available
          xlabel: costum xlabel value
          ylabel: costum ylabel value
          kwargs: arguments that are directly passed to matplotlib's plot command.
                  In addition it is possible to pass linthreshy if ylim='default'
                  and ymode='symlog'
        """

        self._xedges = None
        self._ydata = None
        self._xmode = None
        self._ymode = None
        self._xlabel = None
        self._ylabel = None
        self._ax = None
        self._xlim = None
        self._ylim = None
        self._linthreshy = None

        xmode = None
        ymode = None
        xlim = None
        ylim = None
        ax = None
        yscale = None
        title = None
        xlabel = None
        ylabel = None

        if 'xmode' in kwargs:
            xmode = kwargs.pop('xmode')
        if 'ymode' in kwargs:
            ymode = kwargs.pop('ymode')
        if 'xlim' in kwargs:
            xlim = kwargs.pop('xlim')
        if 'ylim' in kwargs:
            ylim = kwargs.pop('ylim')
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        if 'yscale' in kwargs:
            yscale = kwargs.pop('yscale')
        if 'title' in kwargs:
            title = kwargs.pop('title')
        if 'xlabel' in kwargs:
            xlabel = kwargs.pop('xlabel')
        if 'ylabel' in kwargs:
            ylabel = kwargs.pop('ylabel')

        self.spec = spec

        if hasattr(fmt, '__len__') and len(fmt) in [0, 1]:
            self.fmt = fmt
        else:
            raise PlottingError("Wrong number of positional arguments")

        if 'linthreshy' in kwargs:
            self._linthreshy = kwargs.pop("linthreshy")

        self.xlim = xlim
        self.ylim = ylim

        self.yscale = yscale
        self.ax = ax
        self.title = title
        self.kwargs = kwargs

        self.xmode = xmode
        self.ymode = ymode

        self.xlabel = xlabel
        self.ylabel = ylabel


    @property
    def xmode(self):
        """
        Returns the current x axis plotting mode
        """

        return self._xmode


    @xmode.setter
    def xmode(self, mode):
        """
        Define x data mode, handles all data errors, requires spec.
        Defines also xedges and xlabel.

        Args:
          mode: energy (or kev), channel (or channels, chn, chns)
        """

        # First, set the _xmode
        if mode is None:
            if self.spec.is_calibrated:
                self._xmode = 'energy'
            else:
                self._xmode = 'channel'
        else:
            if mode.lower() in ('kev', 'energy'):
                if not self.spec.is_calibrated:
                    raise PlottingError('Spectrum is not calibrated, however'
                                        ' x axis was requested as energy')
                self._xmode = 'energy'
            elif mode.lower() in ('channel', 'channels', 'chn', 'chns'):
                self._xmode = 'channel'
            else:
                raise PlottingError('Unknown x data mode: {}'.format(mode))

        # Then, set the _xedges and _xlabel based on the _xmode
        xedges, xlabel = self.spec.parse_xmode(self._xmode)
        self._xedges = xedges
        possible_labels = ['Energy [keV]', 'Channel']
        if self._xlabel in possible_labels or self._xlabel is None:
            # Only reset _xlabel if it's an old result from parse_xmode or None
            self._xlabel = xlabel


    @property
    def ymode(self):
        """
        Returns the current y axis plotting mode.
        """
        return self._ymode


    @ymode.setter
    def ymode(self, mode):
        """
        Define y data mode, handles all data errors, requires spec.
        Defines also ydata and ylabel. Does not check if cps is defined.
        If it is not defined it will trow a SpectrumError.

        Args:
          mode: counts, cps, cpskev
        """

        # First, set the _ymode
        if mode is None:
            if self.spec._counts is not None:
                self._ymode = 'counts'
            else:
                self._ymode = 'cps'
        elif mode.lower() in ('count', 'counts', 'cnt', 'cnts'):
            if self.spec._counts is None:
                raise PlottingError('Spectrum has counts not defined')
            self._ymode = 'counts'
        elif mode.lower() == 'cps':
            self._ymode = 'cps'
        elif mode.lower() == 'cpskev':
            self._ymode = 'cpskev'
        else:
            raise PlottingError('Unknown y data mode: {}'.format(mode))

        # Then, set the _ydata and _ylabel based on the _ymode
        ydata, _, ylabel = self.spec.parse_ymode(self._ymode)
        self._ydata = ydata
        possible_labels = ['Counts', 'Countrate [1/s]', 'Countrate [1/s/keV]']
        if self._ylabel in possible_labels or self._ylabel is None:
            # Only reset _ylabel if it's an old result from parse_ymode or None
            self._ylabel = ylabel


    @property
    def ax(self):
        """
        Returns the current matplotlib axes object used for plotting.
        If no axes object is defined yet it will create one.
        """

        if self._ax is None:
            _, self._ax = plt.subplots()
        return self._ax


    @ax.setter
    def ax(self, ax):
        """
        Defines the current matplotlib axes object used for plotting.

        Args:
          ax: Axes to be set
        """

        self._ax = ax

    @property
    def xlabel(self):
        """
        Returns the current x label
        """
        return self._xlabel


    @xlabel.setter
    def xlabel(self, label):
        """
        Sets the xlabel to a costum value.
        """

        if label is not None:
            self._xlabel = label


    @property
    def ylabel(self):
        """
        Returns the current y label
        """

        return self._ylabel


    @ylabel.setter
    def ylabel(self, label):
        """
        Sets the ylabel to a costum value.
        """

        if label is not None:
            self._ylabel = label

    @property
    def yerror(self):
        """
        Returns array of statistical errors for current mode
        FIXME: now redundant?
        """

        if self._ymode == 'counts':
            return self.spec.counts_uncs
        elif self._ymode == 'cps':
            return self.spec.cps_uncs
        elif self._ymode == 'cpskev':
            return self.spec.cpskev_uncs


    def get_corners(self):
        """
        Creates a stepped version of the current spectrum data.

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
        if self._xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self._ylim is not None:
            self.ax.set_ylim(self.ylim)
            if self.yscale == 'symlog' and self._ylim == 'default':
                self.ax.set_yscale(self.yscale, linthreshy=self.linthreshy)
        return self.get_corners()


    def plot(self, *fmt, **kwargs):
        """
        Create actual plot with matplotlib's plot method.

        Args:
          fmt:    Matplotlib plot like format string
          kwargs: Any matplotlib plot() keyword argument, overwrites
                  previously defined keywords
        """

        if hasattr(fmt, '__len__') and len(fmt) > 0:
            self.fmt = fmt

        if not hasattr(self.fmt, '__len__') or not len(self.fmt) in [0, 1]:
            raise PlottingError("Wrong number of positional argument")

        xcorners, ycorners = self._prepare_plot(**kwargs)
        self.ax.plot(xcorners, ycorners, *self.fmt, **self.kwargs)
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


    def errorbar(self, **kwargs):
        """
        Create errorbars with matplotlib's plot errorbar method.

        Args:
          kwargs: Any matplotlib errorbar() keyword argument, overwrites
                  previously defined keywords
        """

        self._prepare_plot(**kwargs)
        xdata = (self._xedges[0:-1]+self._xedges[1:])*0.5

        if 'fmt' in self.kwargs:
            self.fmt = (self.kwargs.pop('fmt'))

        if hasattr(self.fmt, '__len__') and len(self.fmt) == 0:
            self.fmt = (',',)

        if not hasattr(self.fmt, '__len__') or len(self.fmt) != 1:
            raise PlottingError("Wrong number of argument for fmt")

        self.ax.errorbar(xdata, self._ydata, yerr=self.yerror, fmt=self.fmt[0], **self.kwargs)


    def errorband(self, **kwargs):
        """
        Create an errorband with matplotlib's plot fill_between method.

        Args:
          kwargs: Any matplotlib fill_between() keyword argument, overwrites
                  previously defined keywords
        """

        self._prepare_plot(**kwargs)

        alpha = 0.5
        if 'alpha' in self.kwargs:
            alpha = self.kwargs.pop("alpha")

        xcorners, ycorlow = self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata-self.yerror))
        _, ycorhig = self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata+self.yerror))
        self.ax.fill_between(xcorners, ycorlow, ycorhig, alpha=alpha, **self.kwargs)
        return self.ax


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

        The upper limit is the next power of 10, or 3 * power of 10, above
        the max (For linear, the next N * power of 10.).

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
        """Returns the xlim, requires xedges."""

        if self._xlim is None or self._xlim == 'default':
            return np.min(self._xedges), np.max(self._xedges)

        return self._xlim

    @xlim.setter
    def xlim(self, limits):
        """Sets xlim."""

        if limits is not None and limits != 'default' and \
        (not hasattr(limits, '__len__') or len(limits) != 2):
            raise PlottingError('xlim should be length 2: {}'.format(limits))
        self._xlim = limits


    @property
    def ylim(self):
        """Returns ylim, requires yscale, ydata."""

        if self._ylim is None or self._ylim == 'default':
            yscale = self.yscale
            if yscale is None:
                yscale = self.ax.get_yscale()

            min_ind = np.argmin(np.abs(self._ydata[self._ydata != 0]))
            delta_y = np.abs(self._ydata - self._ydata[min_ind])
            min_delta_y = np.min(delta_y[delta_y > 0])

            data_min = np.min(self._ydata)
            if yscale == 'linear':
                ymin = 0
            elif yscale == 'log' and data_min < 0:
                raise PlottingError('Cannot plot negative values on a log ' +
                                    'scale; use symlog scale')
            elif yscale == 'symlog' and data_min >= 0:
                ymin = 0
            else:
                ymin = self.dynamic_min(data_min, min_delta_y)

            data_max = np.max(self._ydata)
            ymax = self.dynamic_max(data_max, yscale)
            return ymin, ymax

        return self._ylim


    @ylim.setter
    def ylim(self, limits):
        """Sets ylim."""

        if limits is not None and limits != 'default' and \
        (not hasattr(limits, '__len__') or len(limits) != 2):
            raise PlottingError('ylim should be length 2: {}'.format(limits))
        self._ylim = limits


    @property
    def linthreshy(self):
        """Returns linthreshy, requires ydata."""

        if self._linthreshy is not None:
            return self._linthreshy
        min_ind = np.argmin(np.abs(self._ydata[self._ydata != 0]))
        delta_y = np.abs(self._ydata - self._ydata[min_ind])
        return np.min(delta_y[delta_y > 0])
