"""Becquerel: Tools for radiation spectral analysis."""

from . import core
from . import parsers
from . import tools
from .__metadata__ import __description__, __url__
from .__metadata__ import __version__, __license__, __copyright__

from .core.rebin import rebin, RebinError, RebinWarning
from .core.spectrum import Spectrum, SpectrumError, UncalibratedError
from .core.spectrum import SpectrumWarning
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.utils import UncertaintiesError
from .core.plotting import SpectrumPlotter, PlottingError
from .core.peakfinder import (PeakFilter, PeakFilterError,
                              GaussianPeakFilter, PeakFinder, PeakFinderError)
from .core.autocal import AutoCalibrator, AutoCalibratorError

from .core import utils
from .core import fitting
from .core.fitting import Fitter

from .tools import nndc
from .tools.element import Element
from .tools.isotope import Isotope
from .tools.isotope_qty import IsotopeQuantity
from .tools import xcom
from .tools import materials

import warnings
warnings.simplefilter('default', DeprecationWarning)

__all__ = ['core', 'parsers', 'tools',
           'rebin', 'RebinError', 'RebinWarning',
           'Spectrum', 'SpectrumError', 'SpectrumWarning', 'SpectrumPlotter',
           'PlottingError', 'UncalibratedError', 'LinearEnergyCal',
           'EnergyCalError', 'BadInput', 'UncertaintiesError',
           'PeakFilter', 'PeakFilterError',
           'GaussianPeakFilter', 'PeakFinder', 'PeakFinderError',
           'AutoCalibrator', 'AutoCalibratorError',
           '__description__', '__url__', '__version__', '__license__',
           '__copyright__']
