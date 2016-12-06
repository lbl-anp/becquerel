"""
EnergyCal class.
"""

from __future__ import print_function
import numpy as np


class EnergyCal(object):
    """
    """

    def __init__(self, coeffs):
        """
        Initialize energy calibration object by coefficients.

        coeffs is an iterable with 2 or 3 elements. The calibration is:
            energy = sum(coeffs[i] * ch**i)
        """

        self.coeffs = np.squeeze(coeffs)
        if len(self.coeffs.shape) != 1:
            raise EnergyCalError('Coefficients input has wrong dimensions')

        self.degree = len(self.coeffs)
        if self.degree < 2 or self.degree > 3:
            raise EnergyCalError('Require 2 or 3 coefficients')

    @classmethod
    def from_file_obj(cls, fileobject):
        """
        Load the energy calibration from an existing file object e.g. SPEFile.
        """
        energycal = cls(fileobject.cal_coeffs)
        return energycal

    def channels_to_energies(self, channels):
        """Convert channels to energies."""
        channels = np.array(channels, dtype=float)
        energies = np.zeros_like(channels)
        for i, coeff in enumerate(self.coeffs):
            energies += coeff * channels**i
        return energies


class EnergyCalError(Exception):
    """Exception raised by EnergyCal."""
    pass
