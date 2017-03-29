from abc import ABCMeta, abstractmethod
import numpy as np


class EnergyCalError(Exception):
    """Base class for errors in energycal.py"""

    pass


class BadInput(EnergyCalError):
    """Error related to energy cal input"""

    pass


class EnergyCalBase(object):
    """Abstract base class for energy calibration."""

    __metaclass__ = ABCMeta

    def __init__(self, chlist=None, kevlist=None):
        # initialize calibration points: channels, energies
        if chlist is not None:
            self._channels = np.array(chlist)
        if kevlist is not None:
            self._energies = np.array(kevlist)

        # initialize curve coefficients
        self._coeffs = dict()

        # initialize fit constraints?

    @classmethod
    def from_points(cls, chlist=None, kevlist=None, pairlist=None):
        """Construct EnergyCal from calibration points."""

        if pairlist and (chlist or kevlist):
            raise BadInput('Redundant calibration inputs')
        if (chlist and not kevlist) or (kevlist and not chlist):
            raise BadInput('Require both chlist and kevlist')
        if not chlist and not kevlist and not pairlist:
            raise BadInput('Calibration points are required')

        if pairlist:
            chlist, kevlist = zip(*pairlist)

        cal = cls(chlist=chlist, kevlist=kevlist)

        return cal

    @classmethod
    def from_coefficients(cls, coeffs):
        """Construct EnergyCal from equation coefficients."""

        pass

    @property
    def channels(self):
        return self._channels

    @property
    def energies(self):
        return self._energies

    @property
    def coeffs(self):
        return self._coeffs

    @abstractmethod
    def ch2kev(self, ch):
        """Convert channel(s) to energy value(s)."""

        pass
