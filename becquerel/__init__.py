"""Becquerel: Tools for radiation spectral analysis."""

from .__metadata__ import (
    __description__,
    __url__,
    __version__,
    __license__,
    __copyright__,
)

from . import core
from .core import utils, fitting
from .core.autocal import AutoCalibrator, AutoCalibratorError
from .core.energycal import LinearEnergyCal, EnergyCalError, BadInput
from .core.calibration import Calibration, CalibrationError
from .core.fitting import Fitter
from .core.peakfinder import (
    PeakFilter,
    PeakFilterError,
    GaussianPeakFilter,
    PeakFinder,
    PeakFinderError,
)
from .core.plotting import SpectrumPlotter, PlottingError
from .core.rebin import rebin, RebinError, RebinWarning
from .core.spectrum import Spectrum, SpectrumError, UncalibratedError, SpectrumWarning
from .core.utils import UncertaintiesError

from . import parsers

from . import tools
from .tools import nndc, xcom, materials
from .tools.element import Element
from .tools.isotope import Isotope
from .tools.isotope_qty import IsotopeQuantity

import warnings

warnings.simplefilter("default", DeprecationWarning)

__all__ = [
    "__description__",
    "__url__",
    "__version__",
    "__license__",
    "__copyright__",
    "core",
    "utils",
    "fitting",
    "AutoCalibrator",
    "AutoCalibratorError",
    "LinearEnergyCal",
    "EnergyCalError",
    "BadInput",
    "Calibration",
    "CalibrationError",
    "Fitter",
    "PeakFilter",
    "PeakFilterError",
    "GaussianPeakFilter",
    "PeakFinder",
    "PeakFinderError",
    "SpectrumPlotter",
    "PlottingError",
    "rebin",
    "RebinError",
    "RebinWarning",
    "Spectrum",
    "SpectrumError",
    "UncalibratedError",
    "SpectrumWarning",
    "UncertaintiesError",
    "parsers",
    "tools",
    "nndc",
    "xcom",
    "materials",
    "Element",
    "Isotope",
    "IsotopeQuantity",
]
