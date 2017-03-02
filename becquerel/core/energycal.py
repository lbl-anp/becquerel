
from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod


class EnergyCalBase(object):
    """Abstract base class for an energy calibration."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def ch2kev(self, channel):
        """Convert channel(s) to energy(ies)."""
        pass


class FitEnergyCalBase(EnergyCalBase):
    """
    Abstract base class for an energy calibration based on spectral features.
    """

    def __init__(self, peaks_list, **kwargs):
        self._peaks_list = peaks_list
        super().__init__(**kwargs)

    @abstractmethod
    def fit(self):
        """Produce new calibration curve based on current points."""
        pass

    @property
    def ch_list(self):
        return [pk.energy_ch for pk in self._peaks_list]

    @property
    def kev_list(self):
        return [pk.cal_energy_kev for pk in self._peaks_list]

    def add_peak(self, peak, refit=True):
        """Add a peak to the calibration.

        Args:
          peak: a Feature object representing the peak.
          refit: if True, the calibration curve is automatically updated.
            [default: True]
        """

        if not isinstance(peak, peaks.FeatureBase):
            raise TypeError('peak should be subclassed from FeatureBase')
        if peak in self._peaks_list:
            return None
        else:
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
        else:
            energy_kev = peak_or_energy
            ind = self.kev_list.index(energy_kev)
            del self._peaks_list[ind]
        if refit:
            self.fit()


class SimplePolyCal(EnergyCalBase):
    """Polynomial energy calibration, by coefficients, not spectral features.
    """

    def __init__(self, coeffs=None, wait=False, **kwargs):
        """
        E[keV] = sum_i( energy^i * coeffs[i] )

        Args:
          coeffs: the coefficients of the polynomial, in increasing order of
            terms.
        """

        if not wait:
            self._coeffs = np.array(coeffs, dtype=np.float)
        super().__init__(**kwargs)

    @property
    def coeffs(self):
        return self._coeffs

    def ch2kev(self, channel):
        ch_array = np.array(channel, dtype=np.float)
        energy_kev = np.zeros_like(ch_array)
        for i, coeff in enumerate(self.coeffs):
            energy_kev += coeff * ch_array**i
        return energy_kev


class FitPolyCal(FitEnergyCalBase, SimplePolyCal):

    def __init__(self, order=2, **kwargs):
        """
        Args:
          peaks_list: an iterable containing instances of Features from
            module peaks.py
          order: an integer indicating the polynomial order. [Default: 2]
        """

        super().__init__(wait=True, **kwargs)
        self._order = 2
        self.fit()

    @property
    def order(self):
        return self._order

    def fit(self):
        """Produce new calibration curve based on current points."""

        self._coeffs = np.polyfit(self.ch_list, self.kev_list, self.order)
