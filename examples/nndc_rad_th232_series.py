"""Create a table of gamma energies for the Th-232 decay chain."""

from __future__ import print_function
import pandas as pd
from becquerel.tools import nndc
pd.set_option('display.width', 220)


EMIN = 10
IMIN = 0.1

series = [
    'Th-232',
    'Ra-228',
    'Ac-228',
    'Th-228',
    'Ra-224',
    'Rn-220',
    'Po-216',
    'Pb-212',
    'Bi-212',
    'Po-212',  # Bi-212 -> Po-212 64.06%
    'Tl-208',  # Bi-212 -> Tl-208 35.94%
]


df_series = pd.DataFrame()
for isotope in series:
    print(isotope)
    try:
        isotope_id = isotope
        if isotope.endswith('m'):
            isotope_id = isotope[:-1]
        df = nndc.fetch_decay_radiation(
            nuc=isotope_id, type='Gamma', e_range=(EMIN, 1e4),
            i_range=(IMIN, 1000))
    except nndc.NoDataFound:
        continue
    df = df.loc[df['Parent Energy Level (MeV)'] == 0.]
    df = df.loc[df['Radiation'] == 'G']
    df = df.loc[df['Radiation Energy (keV)'] > EMIN]
    df = df.loc[df['Radiation Intensity (%)'] > IMIN]
    # handle special branching cases
    if isotope == 'Po-212':
        df['Radiation Intensity (%)'] *= 0.6406
    elif isotope == 'Tl-208':
        df['Radiation Intensity (%)'] *= 0.3594
    df_series = df_series.append(df)
df_series.sort('Radiation Energy (keV)')
fields = [
    'Z', 'Element', 'A', 'N', 'Decay Mode', 'Radiation', 'Radiation subtype',
    'Radiation Energy (keV)', 'Radiation Intensity (%)',
    'Daughter', 'T1/2 (s)']
print(df_series[fields])
