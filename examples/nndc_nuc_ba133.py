"""Perform the NNDC wallet card and decay radiation queries for Barium-133."""

from __future__ import print_function
from becquerel.tools import nndc
import pandas as pd
pd.set_option('display.width', 220)


card = nndc.NuclearWalletCardQuery(nuc='Ba-133')
print('')
print('Nuclear Wallet Card for Ba-133')
print('------------------------------')
print('')
print(card)

rad = nndc.DecayRadiationQuery(nuc='Ba-133')
print('')
print('Decay Radiation for Ba-133')
print('--------------------------')
print('')
print(rad)

rad = nndc.DecayRadiationQuery(
    nuc='Ba-133', elevel_range=(0, 0), type='Gamma', i_range=(1, 1000))
print('')
print('Decay Radiation for Ba-133 ground state, gammas only > 1%')
print('---------------------------------------------------------')
print('')
cols = [
    'Decay Mode', 'Radiation subtype', 'Radiation Energy (keV)',
    'Radiation Intensity (%)']
print(rad[cols])
