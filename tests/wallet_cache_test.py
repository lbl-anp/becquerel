"""Test WalletCardCache class."""

import numpy as np
import uncertainties
import becquerel.tools.wallet_cache as wallet_cache
import pytest


@pytest.mark.parametrize(
    "arg, result",
    [
        ("", None),
        ("NaN", np.nan),
        ("nan", np.nan),
        ("5.0+/-1.0", uncertainties.ufloat(5.0, 1.0)),
        ("5.0", 5.0),
    ],
)
def test_convert_float_ufloat(arg, result):
    """Convert string to a float or a ufloat, including None ('') and NaN."""
    answer = wallet_cache.convert_float_ufloat(arg)
    if result is None:
        assert answer is None
    elif isinstance(result, uncertainties.core.Variable):
        assert isinstance(answer, type(result))
        assert np.isclose(answer.nominal_value, result.nominal_value)
    elif isinstance(result, float):
        assert isinstance(answer, type(result))
        if np.isnan(result):
            assert np.isnan(answer)
        else:
            assert np.isclose(answer, result)


@pytest.mark.parametrize(
    "arg, result",
    [
        (None, ""),
        (np.nan, "nan"),
        (uncertainties.ufloat(5.0, 1.0), "5.000000000000+/-1.000000000000"),
        (5.0, "5.000000000000"),
    ],
)
def test_format_ufloat(arg, result):
    """Convert ufloat to a string, including None ('') and NaN."""
    answer = wallet_cache.format_ufloat(arg)
    assert answer == result


@pytest.mark.webtest
class TestWalletCardCache:
    """Test functionality of wallet_cache."""

    def test_fetch(self):
        """Test wallet_cache.fetch()."""
        wallet_cache.wallet_cache.fetch()

    def test_write_file(self):
        """Test wallet_cache.write_file()."""
        wallet_cache.wallet_cache.write_file()

    def test_read_file(self):
        """Test wallet_cache.read_file()."""
        wallet_cache.wallet_cache.read_file()

    def test_load(self):
        """Test wallet_cache.load()."""
        wallet_cache.wallet_cache.load()
