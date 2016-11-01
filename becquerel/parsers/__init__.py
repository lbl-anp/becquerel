"""Spectrum file parser classes.

Basic operation is:
    spec = SpeFile(filename)
    spec.read()
    spec.apply_calibration()

Then the data are in
    spec.data [counts]
    spec.channels
    spec.energies
    spec.energy_bin_widths

"""

from .spe_file import SpeFile
from .spc_file import SpcFile
from .cnf_file import CnfFile
