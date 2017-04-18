"""Create a table of gamma energies for the U-238 decay chain."""

from __future__ import print_function
import pandas as pd
from becquerel.tools import nndc
pd.set_option('display.width', 220)


EMIN = 10
IMIN = 0.1

series = [
    'U-238',
    'Th-234',
    'Pa-234m',
    'U-234',
    'Th-230',
    'Ra-226',
    'Rn-222',
    'Po-218',
    'Pb-214',
    'Bi-214',
    'Po-214',
    'Pb-210',
    'Bi-210',
    'Po-210',
]


df_series = pd.DataFrame()
for isotope in series:
    print(isotope)
    try:
        if isotope.endswith('m'):
            d = nndc.DecayRadiationQuery(
                nuc=isotope[:-1], type='Gamma', e_range=(EMIN, 1e4),
                i_range=(IMIN, 1000))
        else:
            d = nndc.DecayRadiationQuery(
                nuc=isotope, type='Gamma', e_range=(EMIN, 1e4),
                i_range=(IMIN, 1000))
    except nndc.NoDataFound:
        continue
    df = d._df
    if isotope.endswith('m'):
        df = df.loc[df['Parent Energy Level (MeV)'] > 0.]
    else:
        df = df.loc[df['Parent Energy Level (MeV)'] == 0.]
    df_series = df_series.append(df)
df_series.sort('Radiation Energy (keV)')
fields = [
    'Z', 'Element', 'A', 'N', 'Decay Mode', 'Radiation', 'Radiation subtype',
    'Radiation Energy (keV)', 'Radiation Intensity (%)',
    'Daughter', 'T1/2 (s)']
print(df_series[fields])
