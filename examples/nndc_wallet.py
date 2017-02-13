"""Test NNDC nuclear wallet card query."""

from __future__ import print_function
from becquerel.tools import nndc


for iso in ['K-40', 'Tc-99', 'Ar-40']:
    print('')
    print(iso)
    d = nndc.NuclearWalletCardQuery(nuc=iso)
    print('')
    print(d)
