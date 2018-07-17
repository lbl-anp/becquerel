"""A cache of all NNDC wallet card data."""

from __future__ import print_function
from future.builtins import super
import pandas as pd
import uncertainties
from . import nndc
from . import df_cache
from ..core.utils import isstring

# pylint: disable=no-self-use


def convert_float_ufloat(x):
    """Convert string to a float or a ufloat, including None ('') and NaN.

    Args:
      x: a string giving the number
    """

    if isstring(x):
        if '+/-' in x:
            tokens = x.split('+/-')
            return uncertainties.ufloat(float(tokens[0]), float(tokens[1]))
        if x == '':
            return None
    return float(x)


def format_ufloat(x, fmt='{:.12f}'):
    """Convert ufloat to a string, including None ('') and NaN.

    Args:
      x: a ufloat
    """

    if x is None:
        return ''
    return fmt.format(x)


class WalletCardCache(df_cache.DataFrameCache):
    """A cache of all isotope wallet cards from NNDC."""

    name = 'all_wallet_cards'

    def write_file(self):
        """Format ufloat columns before writing so they keep precision."""

        for col in ['Abundance (%)', 'Mass Excess (MeV)']:
            self.df[col] = self.df[col].apply(format_ufloat)
        super().write_file()

    def read_file(self):
        """Ensure some columns are properly converted to float/ufloat."""

        super().read_file()
        for col in ['Abundance (%)', 'Mass Excess (MeV)']:
            self.df[col] = self.df[col].apply(convert_float_ufloat)

    def fetch(self):
        """Fetch wallet card data from NNDC for all isotopes."""

        self.df = pd.DataFrame()
        z_edges = (0, 40, 80, 120)
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            df_chunk = nndc.fetch_wallet_card(z_range=(z0, z1 - 1))
            self.df = self.df.append(df_chunk)
        self.loaded = True


wallet_cache = WalletCardCache()
