"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools

from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.utils import UncertaintiesError
from .core.plotting import SpectrumPlotter, PlottingError

from .tools import nndc
from .tools.element import Element
from .tools.isotope import Isotope
from .tools.isotope_qty import IsotopeQuantity
from .tools import xcom
from .tools import materials

__all__ = ['core', 'parsers', 'tools',
           'Spectrum', 'SpectrumError', 'SpectrumPlotter', 'PlottingError',
           'UncalibratedError', 'LinearEnergyCal', 'EnergyCalError',
           'BadInput', 'UncertaintiesError']
