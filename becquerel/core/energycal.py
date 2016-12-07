"""
EnergyCal class.
"""

from __future__ import print_function
import numpy as np


class EnergyCal(object):
    """
    Represents an energy calibration.
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

        self.degree = len(self.coeffs) - 1
        if self.degree < 1 or self.degree > 2:
            raise EnergyCalError(
                'Require 2 or 3 coefficients, got {} instead'.format(
                    self.degree + 1))

    @classmethod
    def from_file_obj(cls, fileobject):
        """
        Load the energy calibration from an existing file object e.g. SPEFile.
        """
        energycal = cls(fileobject.cal_coeffs)
        return energycal

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
