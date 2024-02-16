"""Becquerel: Tools for radiation spectral analysis."""

import warnings

from . import core, parsers, tools
from .__metadata__ import (
    __copyright__,
    __description__,
    __license__,
    __url__,
    __version__,
)
from .core import fitting, utils
from .core.autocal import AutoCalibrator, AutoCalibratorError
from .core.calibration import Calibration, CalibrationError, CalibrationWarning
from .core.energycal import BadInput, EnergyCalError, LinearEnergyCal
from .core.fitting import Fitter
from .core.peakfinder import (
    GaussianPeakFilter,
    PeakFilter,
    PeakFilterError,
    PeakFinder,
    PeakFinderError,
)
from .core.plotting import PlottingError, SpectrumPlotter
from .core.rebin import RebinError, RebinWarning, rebin
from .core.spectrum import Spectrum, SpectrumError, SpectrumWarning, UncalibratedError
from .core.utils import UncertaintiesError
from .tools import materials, nndc, xcom
from .tools.element import Element
from .tools.isotope import Isotope
from .tools.isotope_qty import IsotopeQuantity

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
    "CalibrationWarning",
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
