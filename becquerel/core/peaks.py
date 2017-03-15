from __future__ import print_function

import numpy as np
from abc import ABCMeta
from builtins import super


class FeatureBase(object):
    """Abstract base class for any feature."""

    __metaclass__ = ABCMeta


class SpectralFeature(FeatureBase):
    """Abstract base class for a feature associated with a spectrum.

    Properties:
      spectrum (read-only): the Spectrum object associated with this feature.
    """

    __metaclass__ = ABCMeta

    def __init__(self, spec, **kwargs):
        """Check and assign the spectrum property.

        Args:
          spec: a Spectrum object with which this feature is associated.
        """

        # if not isinstance(spec, spectrum.Spectrum):
        #     raise TypeError('spec should be a Spectrum object')
        # # circular import if we import spectrum!
        self._spec = spec
        super().__init__(**kwargs)

    @property
    def spectrum(self):
        """The Spectrum object with which this feature is associated."""

        return self._spec


class EnergyFeature(FeatureBase):
    """Abstract base class for an energy feature.

    Properties:
      energy_ch (read-only): The (measured) energy of the feature, in channels.
      cal_energy_kev: Energy assigned by user, e.g. for calibration.
    """

    __metaclass__ = ABCMeta

    def __init__(self, cal_energy_kev=None, **kwargs):
        """Assign cal_energy_kev if provided.

        Args:
          cal_energy_kev: the calibration energy of the feature in keV
            (optional).
        """

        self.cal_energy_kev = cal_energy_kev
        super().__init__(**kwargs)

    @property
    def energy_ch(self):
        """The (measured) characteristic energy of the feature, in channels.
        """

        return self._energy_ch

    @property
    def cal_energy_kev(self):
        """Energy assigned by user, e.g. for calibration."""

        return self._cal_energy_kev

    @cal_energy_kev.setter
    def cal_energy_kev(self, energy_kev):
        if energy_kev is not None:
            self._cal_energy_kev = float(energy_kev)
        else:
            self._cal_energy_kev = None


class AreaFeature(FeatureBase):
    """Abstract base class for an area (counts) feature.

    Properties:
      area_c (read-only): the area of the feature, in counts
      cal_area: the area assigned by the user for calibration (e.g. activity)
    """

    __metaclass__ = ABCMeta

    def __init__(self, cal_area=None, **kwargs):
        """Assign cal_area if provided.

        Args:
          cal_area: a float representing the calibration value corresponding to
            area (e.g. number of source emissions)
        """

        self.cal_area = cal_area
        super().__init__(**kwargs)

    @property
    def area_c(self):
        """The (measured) area of the feature, in counts."""
        return self._area_c

    @property
    def cal_area(self):
        """Area assigned by user for calibration.

        E.g. source emissions for efficiency calibration.
        """

        return self._cal_area

    @cal_area.setter
    def cal_area(self, area):
        if area is not None:
            self._cal_area = float(area)
        else:
            self._cal_area = None


class ArbitraryEnergyPoint(EnergyFeature):
    """An arbitrary energy calibration point."""

    def __init__(self, ch, kev, **kwargs):
        """
        Assign channel value and calibration value.

        Args:
          ch: the channel value
          kev: the keV value to assign
        """

        self._energy_ch = ch
        super().__init__(cal_energy_kev=kev, **kwargs)


class ArbitraryEfficiencyPoint(AreaFeature):
    """An arbitrary efficiency calibration point."""

    def __init__(self, counts, cal_area, energy_kev, **kwargs):
        """
        Assign peak area and calibration area (e.g. emissions).

        Args:
          counts: the area in counts
          cal_area: the calibration value (e.g. total source emissions)
          energy_kev: a float representing the photon energy this efficiency
            cal point is for.
        """

        self._area_c = counts
        self._energy_kev = energy_kev
        # TODO review how an efficiency point handles the energy value

        super().__init__(cal_area=cal_area, **kwargs)


class GrossROIPeak(SpectralFeature, EnergyFeature, AreaFeature):
    """A simplistic gross-area peak feature. For demonstration only.

    Properties:
      spectrum (read-only): the Spectrum object associated with this feature.
      energy_ch (read-only): The (measured) energy of the feature, in channels.
      energy_kev (read-only): energy_ch converted to kev by calibration.
      area_c (read-only): the area of the feature, in counts
      cal_energy_kev: Energy assigned by user, e.g. for calibration.
      cal_area: the area assigned by the user for calibration
      ROI_bounds_ch (read-only): the bounds of the ROI, in channels
    """

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

        super().__init__(spec=spec)
        self._left_ch, self._right_ch = ROI_bounds_ch
        self._area_c = self._spec.integrate(self._left_ch, self._right_ch)
        self._energy_ch = self._measure_centroid()

    def _measure_centroid(self):
        """Calculate energy centroid of region."""

        weighted_terms = np.sum(
            [ch * self._spec.data[ch]
             for ch in range(self._left_ch, self._right_ch + 1)])
        centroid = float(weighted_terms) / self._area_c
        return centroid

    @property
    def ROI_bounds_ch(self):
        """The bounds of the ROI, in channels, in a tuple of (left, right)."""

        return (self._left_ch, self._right_ch)

    @property
    def energy_kev(self):
        """The energy centroid, converted to keV by calibration."""

        # TODO there needs to be some shortcut function for channel to energy,
        #   accessible directly in the spectrum class, I think.
        # so this form can later be simplified.
        centroid_kev = self._spec.cal.channel_to_energy(self.energy_ch)
        return centroid_kev
