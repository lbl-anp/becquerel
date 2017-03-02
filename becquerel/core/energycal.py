
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


class FitEnergyCalBase(EnergyCalBase):
    """
    Abstract base class for an energy calibration based on spectral features.
    """

    def __init__(self, peaks_list):
        self._peaks_list = peaks_list
        self._peaks_dict = {}
        for i, pk in enumerate(peaks_list):
            self._peaks_dict[id(pk)] = i
        self._ch_list = [pk.energy_ch for pk in peaks_list]
        self._kev_list = [pk.assigned_energy_kev for pk in peaks_list]

    @abstractmethod
    def fit(self):
        """Produce new calibration curve based on current points."""
        pass

    def add_peak(self, peak, refit=True):
        """Add a peak to the calibration.

        Args:
          peak: a Feature object representing the peak.
          refit: if True, the calibration curve is automatically updated.
            [default: True]
        """

        if not isinstance(peak, peaks.FeatureBase):
            raise TypeError('peak should be subclassed from Feature')
        if peak in self._peaks_list:
            return None

        self._peaks_dict[id(peak)] = len(self._peaks_list)
        self._peaks_list.append(peak)
        if refit:
            self.fit()

    def rm_peak(self, peak_or_energy, refit=True):
        """Remove a peak from the calibration.

        Args:
          peak_or_energy: either a Feature object, or a value in keV
            representing the assigned energy value of the peak to be removed.
          refit: if True, the calibration curve is automatically updated.
            [default: True]
        """

        if isinstance(peak_or_energy, peaks.FeatureBase):
            peak = peak_or_energy
            self._peaks_list.remove(peak)
            del self._ch_list[self._peaks_dict[id(peak)]]
            del self._kev_list[self._peaks_dict[id(peak)]]
            self._peaks_dict.remove(id(peak))
        else:
            # ... restructure without using dict

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


class FitLinearCal(FitEnergyCalBase, FixedLinearCal):

    def __init__(self, peaks_list):
        """
        Args:
          peaks_list: an iterable containing instances of Features from
            module peaks.py
        """

        super(FitLinearCal, self).__init__(peaks_list)
        self.fit()

    def fit(self):
        """Produce new calibration curve based on current points."""

        slope_kev_ch, offset_kev = np.polyfit(
            self._ch_list, self._kev_list, 1)

        self.offset_kev = offset_kev
        self.slope_kev_ch = slope_kev_ch
