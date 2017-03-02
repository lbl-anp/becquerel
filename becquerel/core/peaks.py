from __future__ import print_function

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from builtins import super


class FeatureBase(object):
    """Abstract base class for any feature."""

    __metaclass__ = ABCMeta


class SpectralFeature(FeatureBase):
    """Abstract base class for a feature associated with a spectrum."""

    __metaclass__ = ABCMeta

    def __init__(self, spec):
        self._spec = spec


class EnergyFeature(FeatureBase):
    """Abstract base class for an energy feature."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def energy_ch(self):
        """The (measured) characteristic energy of the feature, in channels.
        """
        pass

    @energy_ch.setter
    def energy_ch(self):
        pass

    @property
    def cal_energy_kev(self):
        """Energy assigned by user, e.g. for calibration."""
        return self._cal_energy_kev

    @cal_energy_kev.setter
    def cal_energy_kev(self, energy_kev):
        self._cal_energy_kev = energy_kev


class AreaFeature(FeatureBase):
    """Abstract base class for an area (counts) feature."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def area_c(self):
        """The (measured) area of the feature, in counts."""
        pass

    @property
    def cal_area(self):
        """Area assigned by user, e.g. activity for efficiency calibration."""
        return self._cal_area

    @cal_area.setter
    def cal_area(self, area):
        self._cal_area = area


class ArbitraryCalPoint(EnergyFeature):
    """An arbitrary calibration point."""

    def __init__(self, ch, kev):
        """
        Args:
          ch: the channel value
          kev: the keV value to assign
        """

        self.energy_ch = ch
        self.cal_energy_kev = kev

    def energy_ch(self):
        return self.energy_ch


class GrossROIPeak(SpectralFeature, EnergyFeature, AreaFeature):
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

        super().__init__(self, spec)
        self._left_ch = ROI_bounds_ch[0]
        self._right_ch = ROI_bounds_ch[1]
        self._gross_area_c = self._spec.integrate(*ROI_bounds_ch)
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

    @energy_ch.setter
    def energy_ch(self, ch):
        self._centroid_ch = ch

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
