"""Spectrum file parser classes.

Just instantiate a class with a filename:
    spec = SpeFile(filename)

Then the data are in
    spec.data [counts]
    spec.channels
    spec.energies
    spec.energy_bin_widths

"""

from .spe_file import SpeFile
from .spc_file import SpcFile
from .cnf_file import CnfFile
from .spectrum_file import SpectrumFileParsingError, SpectrumFileParsingWarning
