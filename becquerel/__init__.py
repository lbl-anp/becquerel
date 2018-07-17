"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools

from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.utils import UncertaintiesError
from .core.plotting import SpectrumPlotter, PlottingError
from .core.peakfinder import (PeakFilter, BoxcarPeakFilter,
                              GaussianPeakFilter, PeakFinder, PeakFinderError)
from .core.autocal import AutoCalibrator, AutoCalibratorError

__all__ = ['core', 'parsers', 'tools',
           'Spectrum', 'SpectrumError', 'SpectrumPlotter', 'PlottingError',
           'UncalibratedError', 'LinearEnergyCal', 'EnergyCalError',
           'BadInput', 'UncertaintiesError',
           'PeakFilter', 'BoxcarPeakFilter', 'GaussianPeakFilter',
           'PeakFinder', 'PeakFinderError',
           'AutoCalibrator', 'AutoCalibratorError']
