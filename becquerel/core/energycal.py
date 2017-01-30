"""
EnergyCal class.
"""

from __future__ import print_function
import numpy as np
import abc


class EnergyCalBase(object):
    """
    Abstract base class for an energy calibration.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def channel_to_energy(self, channel):
        """Convert channel(s) to energy(ies)."""
        return


class PolynomialCal(EnergyCalBase):
    """Polynomial calibration, from coefficients."""

    def __init__(self, coeffs):
        """
        Initialize energy calibration object by coefficients.

        coeffs is an iterable with 2 or 3 elements. The calibration is:
            energy = sum(coeffs[i] * ch**i)
        """

        self.coeffs = coeffs
        self.degree = len(self.coeffs) - 1
        if self.degree < 1:
            raise EnergyCalError(
                'Require at least 2 coefficients, got {} instead'.format(
                    self.degree + 1))

    @classmethod
    def from_file_obj(cls, file_obj):
        """
        Create a polynomial energy calibration from a spectrum file object.
        """

        cal = cls(file_obj.cal_coeff)
        return cal

    def channel_to_energy(self, channel):
        """Convert channels to energies."""

        channel = np.array(channel, dtype=float)
        energy = np.zeros_like(channel)
        for i, coeff in enumerate(self.coeffs):
            energy += coeff * channel**i
        return energy


class EnergyCalError(Exception):
    """Exception raised by EnergyCal."""

    pass
