"""Tools for plotting spectra."""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat, unumpy


def plot_spectrum(spec, fmtstring=None, **kwargs):
    """Plot a Spectrum object."""

    plotter = SpectrumPlotter(spec, fmtstring=fmtstring, **kwargs)
    plotter.plot()
    return plotter.axes


class SpectrumPlotter(object):
    """Class for handling spectrum plotting."""

    def __init__(self, spec, fmtstring=None, **kwargs):
        """
        Args:
          spec: Spectrum instance to plot
        """

        self.spec = spec
        self._handle_input(fmtstring, **kwargs)

    def _handle_input(
            self, fmtstring,
            axes=None,
            counts_mode=None,
            title=None,
            xlabel=None,
            ylabel=None,
            xscale=None,
            xlim=None,
            yscale=None,
            ylim=None,
            **kwargs):
        """
        Args:
          counts_mode: 'counts', 'cps', 'cpskev', ...
          title: string of text for plot title
          xlabel: string of text for xlabel
          ylabel: string of text for ylabel

        """

        if fmtstring is None:
            self.fmtstring = ''
        else:
            self.fmtstring = fmtstring

        if axes is None:
            self.axes = None
            self.new_axes = True
        else:
            self.axes = axes
            self.new_axes = False

        if counts_mode is None:
            self.counts_mode = 'counts'
        else:
            self.counts_mode = counts_mode.lower()

        if title is None:
            self.title = 'Becquerel Spectrum'
        else:
            self.title = title

        if self.spec.is_calibrated:
            self.xdata = self.spec.bin_edges_kev
            if xlabel is None:
                self.xlabel = 'Energy [keV]'
        else:
            # TODO depends on convention (issues #35, #36)
            self.xdata = np.arange(-0.5, len(self.spec.data) - 0.4)
            if xlabel is None:
                self.xlabel = 'Energy [channels]'
        if xlabel is not None:
            self.xlabel = xlabel

        if ylabel is None:
            if self.counts_mode == 'counts':
                self.ylabel = 'Counts'
                ydata = self.spec.data
            elif self.counts_mode == 'cps':
                self.ylabel = 'Countrate [1/s]'
                ydata = self.spec.data / self.spec.livetime
            elif self.counts_mode == 'cpskev':
                self.ylabel = 'Countrate [1/s/keV]'
                ydata = self.spec.data / self.spec.livetime / binwidths
        else:
            self.ylabel = ylabel
        self.ydata = np.append(ydata, ufloat(0, 1))

        if xscale is None:
            self.xscale = 'linear'
        else:
            self.xscale = xscale

        if xlim is None:
            if self.spec.is_calibrated:
                self.xlim = (0.0, self.spec.bin_edges_kev[-1])
            else:
                self.xlim = (0.0, len(self.spec.data))
        else:
            self.xlim = xlim

        if yscale is None:
            self.yscale = 'log'
        else:
            self.yscale = yscale

        if ylim is None and self.yscale == 'linear':
            # TODO round up to the nearest round number or so
            self.ylim = (0.0, np.max(self.ydata).nominal_value)
        elif ylim is None and self.yscale == 'log':
            # TODO
            self.ylim = (0.1, np.max(self.ydata).nominal_value)
        else:
            self.ylim = ylim

        self.kwargs = kwargs

    def plot(self):
        """Create actual plot."""

        print('Plotting')

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
        self.axes.set_yscale(self.yscale)
        self.axes.set_ylim(self.ylim)

        plt.show()
