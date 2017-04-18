"""Perform the NNDC wallet card and decay radiation queries for Uranium-235."""

from __future__ import print_function
from becquerel.tools import nndc
import pandas as pd
pd.set_option('display.width', 220)


card = nndc.fetch_wallet_card(nuc='U-235')
print('')
print('Nuclear Wallet Card for U-235')
print('-----------------------------')
print('')
print(card)

rad = nndc.fetch_decay_radiation(nuc='U-235')
print('')
print('Decay Radiation for U-235')
print('-------------------------')
print('')
print(rad)

print('')
print('Decay Radiation for U-235 ground state, gammas only')
print('---------------------------------------------------')
print('')
cols = [
    'Decay Mode', 'Radiation subtype', 'Radiation Energy (keV)',
    'Radiation Intensity (%)']
selection = (rad['Parent Energy Level (MeV)'] == 0) & \
    (rad['Radiation'] == 'G') & \
    (rad['Radiation Intensity (%)'] > 1.)
print(rad[selection][cols])

print('')
print('Decay Radiation for U-235 ground state, alphas only')
print('---------------------------------------------------')
print('')
cols = [
    'Decay Mode', 'Radiation subtype', 'Radiation Energy (keV)',
    'Radiation Intensity (%)']
selection = (rad['Parent Energy Level (MeV)'] == 0) & \
    (rad['Radiation'] == 'A') & \
    (rad['Radiation Intensity (%)'] > 1.)
print(rad[selection][cols])
