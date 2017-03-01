
from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod


class EnergyCalBase(object):
    """Abstract base class for an energy calibration."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def channel_to_energy(self, channel):
        """Convert channel(s) to energy(ies)."""
        pass


class FittedEnergyCalBase(EnergyCalBase):
    """
    Abstract base class for an energy calibration based on spectral features.
    """

    @abstractmethod
    def fit(self):
        """Produce new calibration curve based on current points."""
        pass


class FixedLinearCal(EnergyCalBase):
    """Linear energy calibration, fixed by coefficients, not adaptable."""

    def __init__(self, offset_kev, slope_kev_ch):
        """
        E[keV] = offset_kev + slope_kev_ch * channel

        Args:
          slope_kev_ch: the slope of the calibration, in keV/channel.
          offset_kev: the offset of the calibration, in keV.
        """

        self.offset_kev = float(offset_kev)
        self.slope_kev_ch = float(slope_kev_ch)

    def channel_to_energy(self, channel):
        energy_kev = self.offset_kev + self.slope_kev_ch * np.array(channel)
        return energy_kev


class FixedQuadraticCal(EnergyCalBase):

    def __init__(self, offset_kev, slope_kev_ch, quad_kev_ch2):
        """
        E[keV] = offset_kev + slope_kev_ch * channel + quad_kev_ch2 * channel^2
        """

        self.offset_kev = float(offset_kev)
        self.slope_kev_ch = float(slope_kev_ch)
        self.quad = float(quad_kev_ch2)

    def channel_to_energy(self, channel):
        ch_array = np.array(channel)
        energy_kev = (self.offset_kev +
                      self.slope_kev_ch * ch_array +
                      self.quad * ch_array**2)
        return energy_kev


class FitLinearCal(FittedEnergyCalBase, FixedLinearCal):

    def __init__(self, peaks_list):
        """
        Args:
          peaks_list: an iterable containing instances of Features from
            module peaks.py
        """

        self._peaks_list = peaks_list
        self.fit()

    def fit(self):
        """Produce new calibration curve based on current points."""

        ch = [pk.energy_ch for pk in self._peaks_list]
        kev = [pk.assigned_energy_kev for pk in self._peaks_list]
        slope_kev_ch, offset_kev = np.polyfit(ch, kev, 1)

        self.offset_kev = offset_kev
        self.slope_kev_ch = slope_kev_ch

    def add_peak(self, peak, refit=True):
        """Add a peak to the calibration.

        Args:
          peak: a Feature object representing the peak.
          refit: if True, the calibration curve is automatically updated.
            [default: True]
        """

        if not isinstance(peak, peaks.FeatureBase):
            raise TypeError('peak should be subclassed from Feature')

        self._peaks_list.append(peak)
        if refit:
            self.fit()
