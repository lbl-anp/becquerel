"""Abstract and concrete classes for energy calibration."""

from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod
from . import peaks
from builtins import super


class EnergyCalBase(object):
    """Abstract base class for an energy calibration."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def ch2kev(self, channel):
        """Convert channel(s) to energy(s).

        Args:
          channel: a number or array of numbers representing channel number

        Returns:
          A float, if channel is a scalar.
          Otherwise, a np.array of floats of the same shape as input.
          Either way, it represents the energy(s) in keV.
        """
        pass


class FitEnergyCalBase(EnergyCalBase):
    """
    Abstract base class for an energy calibration based on spectral features.

    Abstract methods:
      fit: produce new calibration curve based on current calibration points

    Properties:
      ch_list (read-only): list of the channel values of the calibration points
      kev_list (read-only): list of the energies of the calibration points

    Methods:
      add_peak: add a calibration point
      rm_peak: remove a calibration point
    """

    __metaclass__ = ABCMeta

    def __init__(self, peaks_list, **kwargs):
        """Assign the peaks_list property.

        Args:
          peaks_list: an iterable of objects derived from FeatureBase
        """

        self._peaks_list = list(peaks_list)
        super().__init__(**kwargs)

    @abstractmethod
    def fit(self):
        """Produce new calibration curve based on current points."""

        pass

    @property
    def ch_list(self):
        """A list of the channel values of the calibration points."""

        return [pk.energy_ch for pk in self._peaks_list]

    @property
    def kev_list(self):
        """A list of the calibration energies [keV] of the calibration points.
        """

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

      cal = SimplePolyCal(coeffs=[4.7, 0.374])
      cal.ch2kev(32)
      spec.calibrate(cal)

    Properties:
      coeffs (read-only): array of polynomial coefficients

    Methods:
      ch2kev: convert channel(s) to energy(s)
    """

    def __init__(self, coeffs=None, wait=False, **kwargs):
        """
        E[keV] = sum_i( energy^i * coeffs[i] )

        Args:
          coeffs: the coefficients of the polynomial, in increasing order of
            terms.
          wait: a bool. If True, do not assign coeffs property yet.
            (Useful for subclasses.) [Default: False]
        """

        if not wait:
            self._coeffs = np.array(coeffs, dtype=float)
        super().__init__(**kwargs)

    @property
    def coeffs(self):
        """Array of the polynomial coefficients, from 0th-order to highest."""

        return self._coeffs

    def ch2kev(self, channel):
        """Convert channel(s) to energy(s).

        Args:
          channel: a number or array of numbers representing channel number

        Returns:
          A float, if channel is a scalar.
          Otherwise, a np.array of floats of the same shape as input.
          Either way, it represents the energy(s) in keV.
        """

        if np.isscalar(channel):
            ch_array = float(channel)
            energy_kev = 0.
        else:
            ch_array = np.array(channel, dtype=float)
            energy_kev = np.zeros_like(ch_array)
        for i, coeff in enumerate(self.coeffs):
            energy_kev += coeff * ch_array**i
        return energy_kev


class FitPolyCal(FitEnergyCalBase, SimplePolyCal):
    """
    Polynomial energy calibration, from a list of spectral features (peaks).

      pks = [ArbitraryEnergyPoint(32, 661.66), ...]
      cal = FitPolyCal(pks, order=1)
      cal.coeffs
      cal.add_peak(ArbitraryEnergyPoint(...))
      spec.calibrate(cal)

    Properties:
      ch_list (read-only): list of the channel values of the calibration points
      kev_list (read-only): list of the energies of the calibration points
      order (read-only): the order of the polynomial
      coeffs (read-only): array of polynomial coefficients

    Methods:
      ch2kev: convert channel(s) to energy(s)
      add_peak: add a calibration point
      rm_peak: remove a calibration point
    """

    def __init__(self, order=2, **kwargs):
        """
        Args:
          peaks_list: an iterable containing instances of Features from
            module peaks.py
          order: an integer indicating the polynomial order. [Default: 2]
        """

        super().__init__(wait=True, **kwargs)
        self._order = int(order)
        self.fit()

    @property
    def order(self):
        """An integer indicating the polynomial order."""

        return self._order

    def fit(self):
        """Produce a new calibration curve based on current calibration points.
        """

        self._coeffs = np.polyfit(
            self.ch_list, self.kev_list, self.order)[::-1]
