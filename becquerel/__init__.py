"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools

from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.plotting import plot_spectrum

__all__ = ['core', 'parsers', 'tools',
           'Spectrum', 'SpectrumError', 'UncalibratedError',
           'LinearEnergyCal', 'EnergyCalError', 'BadInput']
