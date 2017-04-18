"""Perform a decay radiation query to find isotopes with lines ~1460 keV."""

from __future__ import print_function
from becquerel.tools import nndc
import pandas as pd
pd.set_option('display.width', 220)


rad = nndc.fetch_decay_radiation(
    a_range=[0, 300], t_range=[1e6, None], e_range=[1460.3, 1461.3])
print('')
print('Decay Radiation near 1460.8 keV')
print('-------------------------------')
print('')
print(rad)
