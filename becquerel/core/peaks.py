from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class FeatureBase(object):
    """Abstract base class for a spectral feature."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def energy_ch(self):
        """The (measured) characteristic energy of the feature, in channels.
        """
        pass

    @property
    def assigned_energy_kev(self):
        """Energy assigned by user, e.g. for calibration."""
        return self._assigned_energy_kev

    @assigned_energy_kev.setter
    def assigned_energy_kev(self, energy_kev):
        self._assigned_energy_kev = energy_kev


class PeakBase(FeatureBase):
    """Abstract base class for a peak feature."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def FWHM_kev(self):
        """The FWHM of the peak."""
        pass

    def area_c(self):
        """The peak area."""
        pass


class ArbitraryCalPoint(FeatureBase):
    """An arbitrary calibration point."""

    def __init__(self, ch, kev):
        """
        Args:
          ch: the channel value
          kev: the keV value to assign
        """

        self._ch = ch
        self._assigned_energy_kev = kev

    def energy_ch(self):
        return self._ch


class GrossROIPeak(PeakBase):
    """A simplistic gross-area peak feature. For demonstration only."""

    def __init__(self, spec, ROI_bounds_ch):
        """
        Args:
          spec: a Spectrum object.
          ROI_bounds_ch: an iterable of length 2 indicating the left and right
            sides of the ROI, in channels.
        """

        if len(ROI_bounds_ch) != 2:
            raise ValueError('ROI bounds should be an iterable of length 2; ' +
                             'got length {}'.format(len(ROI_bounds_ch)))

        self._spec = spec
        self._left_ch = ROI_bounds_ch[0]
        self._right_ch = ROI_bounds_ch[1]
        self._gross_area_c = integrate(spec, *ROI_bounds_ch)
        self._centroid_ch = self._measure_centroid()
        self._FWHM_ch = self._measure_FWHM()

    def _measure_centroid(self):
        """Calculate energy centroid of region. (don't really do this)"""
        weighted_terms = np.sum(
            [ch * self._spec.data[ch]
             for ch in range(self._left_ch, self._right_ch + 1)])
        centroid = float(weighted_terms) / self._gross_area_c
        return centroid

    def _measure_FWHM(self):
        """Not really a FWHM, don't actually do this."""
        return self._right_ch - self._left_ch

    @property
    def energy_ch(self):
        return self._centroid_ch

    @property
    def energy_kev(self):
        # TODO there needs to be some shortcut function for channel to energy,
        #   accessible directly in the spectrum class, I think.
        # so this form can later be simplified.
        centroid_kev = self._spec.cal.channel_to_energy(self.energy_ch)
        return centroid_kev

    @property
    def area_c(self):
        return self._gross_area_c

    @property
    def FWHM_ch(self):
        return self._FWHM_ch

    @property
    def FWHM_kev(self):
        width_kev = self._spec.cal.channel_to_energy(self.FWHM_ch)
        return width_kev


def integrate(spec, ROI_left_ch, ROI_right_ch):
    """Integrate a spectrum from one channel to another.

    Args:
      spec: a Spectrum object to integrate on
      ROI_left_ch: channel to integrate from
      ROI_right_ch: channel to integrate to

    Returns:
      a float of counts in the spectrum between ROI_left_ch and ROI_right_ch
    """

    left_ch = int(np.round(ROI_left_ch))
    right_ch = int(np.round(ROI_right_ch))

    integral = np.sum(spec.data[left_ch:right_ch + 1])  # end-bin inclusive

    return integral
