"""
Compute electron ranges using analytical formulae in Tabata's papers.

Tabata 1996a:
Tatsuo Tabata, Pedro Andreo, Kunihiko Shinoda.
"An analytic formula for the extrapolated range of electrons in condensed
materials."
Nucl. Instr. Meth. B 119 (1996) 463-470.

Tabata 2002:
Tatsuo Tabata, Vadim Moskvin, Pedro Andreo, Valentin Lazurik, Yuri Rogov.
"Extrapolated ranges of electrons determined from transmission and projected-
range straggling curves."
Rad. Phys. and Chem. 64 (2002) 161-167.
"""

from __future__ import print_function

import numpy as np

ELECTRON_REST_ENERGY_MEV = 0.510999

MIN_ENERGY_1996A_KEV = 1    # Tabata 1996a p. 2
SEMI_MIN_ENERGY_1996A_KEV = 10  # Tabata 1996a p. 2 (shell effects)
MAX_ENERGY_1996A_KEV = 100e3    # Tabata 1996a p. 2
MIN_ENERGY_2002_KEV = 10    # Tabata 2002 sec 2.3
MAX_ENERGY_2002_KEV = 50e3  # Tabata 2002 sec 1; 2.1

# paper notation is 1-indexed. so here, elements at index 0 are placeholders
TABATA_1996A_TABLE_1 = (
    0.0,        # (0)
    0.3879,     # 1
    0.2178,     # 2
    0.4541,     # 3
    0.03068,    # 4
    3.326e-16,  # 5
    13.24,      # 6
    1.316,      # 7
    14.03,      # 8
    0.7406,     # 9
    4.294e-3,   # 10
    1.684,      # 11
    0.2264,     # 12
    0.6127,     # 13
    0.1207,     # 14
)
TABATA_1996A_TABLE_2 = (
    0.0,        # (0)
    3.600,      # 1
    0.9882,     # 2
    1.191e-3,   # 3
    0.8622,     # 4
    1.02501,    # 5
    1.0803e-4,  # 6
    0.99628,    # 7
    1.303e-4,   # 8
    1.02441,    # 9
    1.2986e-4,  # 10
    1.030,      # 11
    1.110e-2,   # 12
    1.10e-6,    # 13
    0.959,      # 14
)
TABATA_2002_TABLE_2 = (
    0.0,        # (0)
    0.2946,     # 1
    0.2740,     # 2
    18.4,       # 3
    3.457,      # 4
    1.377,      # 5
    6.59,       # 6
    2.414,      # 7
    1.094,      # 8
    0.05242,    # 9
    0.3709,     # 10
    0.958,      # 11
    2.02,       # 12
    1.099,      # 13
    0.2808,     # 14
    0.2042,     # 15
)


class TabataError(Exception):
    """General error for Tabata module."""
    pass


class EnergyOutOfRange(TabataError):
    """Energy value out of bounds for calculation."""
    pass


def extrapolated_range_gcm2(energy_keV, Z, A, I_eV, ref=None):
    """Compute electron extrapolated range with Tabata's analytical formulae.

    Args:
      energy_keV: initial electron energy [keV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]
      ref: may be '1996a', '2002', 'CSDA', or None.
        '1996a': calculate extrapolated range using 1996a detour factor.
        '2002': calculate extrapolated range using 2002 detour factor.
        'CSDA': calculate CSDA range instead of extrapolated range.
        None: calculated extrapolated range using better paper (usually 2002).
        [Default: None]

    Returns:
      float of the computed extrapolated range, in g/cm^2.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    energy_keV = float(energy_keV)
    energy_MeV = energy_keV / 1e3
    I_eV = float(I_eV)

    if ref is None:
        ref = '2002'
        if ((ref > MAX_ENERGY_2002_KEV and ref <= MAX_ENERGY_1996A_KEV) or
                (ref < MIN_ENERGY_2002_KEV and ref >= MIN_ENERGY_1996A_KEV)):
            ref = '1996a'

    if ref == '1996' or ref.lower() == '1996a':
        _check_energy_1996a(energy_keV)
        f_d = _detour_factor_1996a(energy_MeV, Z)
    elif ref == '2002':
        _check_energy_2002(energy_keV)
        f_d = _detour_factor_2002(energy_MeV, Z)
    elif ref.upper() == 'CSDA':
        _check_energy_1996a(energy_keV)
        f_d = 1.
    else:
        raise ValueError("ref should be '1996a', '2002', 'CSDA', or None")

    range_gcm2 = f_d * CSDA_range_gcm2(energy_MeV, Z, A, I_eV)

    return range_gcm2


def extrapolated_range_mm(energy_keV, Z, A, I_eV, density_gcm3, ref=None):
    """Compute electron extrapolated range with Tabata's analytical formulae.

    Args:
      energy_keV: initial electron energy [keV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]
      density_gcm3: density of material [g/cm^3]
      ref: may be '1996a', '2002', 'CSDA', or None.
        '1996a': calculate extrapolated range using 1996a detour factor.
        '2002': calculate extrapolated range using 2002 detour factor.
        'CSDA': calculate CSDA range instead of extrapolated range.
        None: calculated extrapolated range using better paper (usually 2002).
        [Default: None]

    Returns:
      float of the computed extrapolated range, in mm.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    range_gcm2 = extrapolated_range_gcm2(energy_keV, Z, A, I_eV, ref=ref)
    range_mm = gcm2_to_mm(range_gcm2, density_gcm3)
    return range_mm


def CSDA_range_gcm2(energy_MeV, Z, A, I_eV):
    """Compute electron CSDA range, using analytical formula in Tabata 1996a.

    Args:
      energy_MeV: initial electron energy [MeV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]

    Returns:
      float of the CSDA range in g/cm^2.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    energy_MeV = float(energy_MeV)

    I = I_eV / (1e6 * ELECTRON_REST_ENERGY_MEV)
    t0 = energy_MeV / ELECTRON_REST_ENERGY_MEV

    d = TABATA_1996A_TABLE_2
    # Tabata 1996a equations 13--19
    c = (
        0.0,                    # (0)
        d[1] * A / Z**d[2],     # 1
        d[3] * Z**d[4],         # 2
        d[5] - d[6] * Z,        # 3
        d[7] - d[8] * Z,        # 4
        d[9] - d[10] * Z,       # 5
        d[11] / Z**d[12],       # 6
        d[13] * Z**d[14],       # 7
    )
    # Tabata 1996a equation 12
    B = np.log((t0 / (I + c[7] * t0))**2) + np.log(1 + t0/2)
    # Tabata 1996a equation 11
    range_gcm2 = c[1] / B * (
        np.log(1 + c[2] * t0**c[3]) / c[2] -
        c[4] * t0**c[5] / (1 + c[6] * t0))

    return range_gcm2


def CSDA_range_mm(energy_MeV, Z, A, I_eV, density_gcm3):
    """Compute electron CSDA range, using analytical formula in Tabata 1996a.

    Args:
      energy_MeV: initial electron energy [MeV]
      Z: atomic number of material
      A: atomic weight of material [AMU]
      I_eV: mean excitation energy of material [eV]
      density_gcm3: density of material [g/cm^3]

    Returns:
      float of the CSDA range in mm.

    Raises:
      EnergyOutOfRange: if energy is out of bounds of accuracy.
    """

    range_gcm2 = CSDA_range_gcm2(energy_MeV, Z, A, I_eV)
    range_mm = gcm2_to_mm(range_gcm2)
    return range_mm


def gcm2_to_mm(mass_thickness_gcm2, density_gcm3):
    """Convert a mass thickness to a length.

    Args:
      mass_thickness_gcm2: mass thickness [g/cm^2]
      density_gcm3: density of material [g/cm^3]

    Returns:
      float of the length in mm
    """

    density_gcm3 = float(density_gcm3)
    length_cm = mass_thickness_gcm2 / density_gcm3
    length_mm = length_cm * 10

    return length_mm


def _check_energy_1996a(energy_keV):
    """Check for energy value out of bounds for Tabata 1996a.

    Args:
      energy_keV: initial electron energy [keV]

    Raises:
      EnergyOutOfRange: if energy is out of bounds
    """

    if energy_keV < MIN_ENERGY_1996A_KEV:
        raise EnergyOutOfRange(
            'Range is inaccurate below {} keV'.format(MIN_ENERGY_1996A_KEV))
    elif energy_keV < SEMI_MIN_ENERGY_1996A_KEV:
        print(
            'Range is less accurate below {} keV, '.format(
                SEMI_MIN_ENERGY_1996A_KEV) +
            'because shell effects are ignored')
    elif energy_keV > MAX_ENERGY_1996A_KEV:
        raise EnergyOutOfRange(
            'Range is inaccurate above {} keV'.format(MAX_ENERGY_1996A_KEV))


def _check_energy_2002(energy_keV):
    """Check for energy value out of bounds for Tabata 2002.

    Args:
      energy_keV: initial electron energy [keV]

    Raises:
      EnergyOutOfRange: if energy is out of bounds
    """

    if energy_keV < MIN_ENERGY_2002_KEV:
        raise EnergyOutOfRange(
            'Range is inaccurate below {} keV'.format(MIN_ENERGY_2002_KEV))
    elif energy_keV > MAX_ENERGY_2002_KEV:
        raise EnergyOutOfRange(
            'Range is inaccurate above {} keV'.format(MAX_ENERGY_2002_KEV))


def _detour_factor_1996a(energy_MeV, Z):
    """Ratio of extrapolated range to CSDA range, according to Tabata 1996a.

    Args:
      energy_MeV: initial electron energy [MeV]
      Z: atomic number of material

    Returns:
      a float representing (extrapolated range) / (CSDA range)
    """

    energy_MeV = float(energy_MeV)
    t0 = energy_MeV / ELECTRON_REST_ENERGY_MEV
    b = TABATA_1996A_TABLE_1
    # Tabata 1996a equations 3--8
    a = (
        0.0,                                        # (0)
        b[1] * Z**b[2],                             # 1
        b[3] + b[4] * Z,                            # 2
        b[5] * Z**(b[6] - b[7] * np.log(Z)),        # 3
        b[8] / Z**(b[9]),                           # 4
        b[10] * Z**(b[11] - b[12] * np.log(Z)),     # 5
        b[13] * Z**b[14],                           # 6
    )
    # Tabata 1996a equation 2
    detour_factor = 1 / (
        a[1] + a[2] / (
            1 + a[3] / t0**a[4] + a[5] * t0**a[6]))

    return detour_factor


def _detour_factor_2002(energy_MeV, Z):
    """Ratio of extrapolated range to CSDA range, according to Tabata 2002.

    Args:
      energy_MeV: initial electron energy [MeV]
      Z: atomic number of material

    Returns:
      a float representing (extrapolated range) / (CSDA range)
    """

    energy_MeV = float(energy_MeV)
    t0 = energy_MeV / ELECTRON_REST_ENERGY_MEV
    b = TABATA_2002_TABLE_2
    # Tabata 2002 equations 9--14
    a = (
        0.0,                                        # (0)
        b[1] * Z**b[2],                             # 1
        b[3] * Z**(-b[4] + b[5] * np.log(Z)),       # 2
        b[6] * Z**(-b[7] + b[8] * np.log(Z)),       # 3
        b[9] * Z**(b[10]),                          # 4
        b[11] * Z**(-b[12] + b[13] * np.log(Z)),    # 5
        b[14] * Z**b[15],                           # 6
    )
    # Tabata 2002 equation 8 (identical to 1996a eq 2)
    detour_factor = 1 / (
        a[1] + a[2] / (
            1 + a[3] / t0**a[4] + a[5] * t0**a[6]))

    return detour_factor
