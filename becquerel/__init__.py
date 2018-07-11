"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools
from .__metadata__ import __name__, __package__, __description__, __url__
from .__metadata__ import __version__, __license__, __copyright__

from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.utils import UncertaintiesError
from .core.plotting import SpectrumPlotter, PlottingError

__all__ = ['core', 'parsers', 'tools',
           'Spectrum', 'SpectrumError', 'SpectrumPlotter', 'PlottingError',
           'UncalibratedError', 'LinearEnergyCal', 'EnergyCalError',
           'BadInput', 'UncertaintiesError']
