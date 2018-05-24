"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools

from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.utils import UncertaintiesError
from .core.plotting import SpectrumPlotter, PlottingError

__all__ = ['core', 'parsers', 'tools',
           'Spectrum', 'SpectrumError', 'SpectrumPlotter', 'UncalibratedError',
           'LinearEnergyCal', 'EnergyCalError', 'BadInput',
           'UncertaintiesError']
