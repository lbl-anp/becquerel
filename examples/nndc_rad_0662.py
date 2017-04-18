"""Perform a decay radiation query to find isotopes with lines ~662 keV."""

from __future__ import print_function
from becquerel.tools import nndc
import pandas as pd
pd.set_option('display.width', 220)


rad = nndc.DecayRadiationQuery(a_range=[130, 140], e_range=[661.4, 661.8])
print('')
print('Decay Radiation near 661.7 keV')
print('------------------------------')
print('')
print(rad)
