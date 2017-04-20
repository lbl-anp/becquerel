"""Test NNDC data queries."""

from __future__ import print_function
import pandas as pd
from becquerel.tools import nndc
import pytest

# pylint: disable=protected-access,no-self-use,too-many-public-methods
# pylint: disable=attribute-defined-outside-init,missing-docstring


def is_close(f1, f2, ppm=1.):
    """True if f1 and f2 are within the given parts per million."""
    return abs((f1 - f2) / f2) < ppm / 1.e6


class TestParseFloatUncertainty(object):
    """Test _parse_float_uncertainty()."""

    def test_01(self):
        """Test _parse_float_uncertainty('257.123', '0.005')..............."""
        answer = nndc._parse_float_uncertainty('257.123', '0.005')
        assert is_close(answer.nominal_value, 257.123)
        assert is_close(answer.std_dev, 0.005)

    def test_02(self):
        """Test _parse_float_uncertainty('257.123', '0.015')..............."""
        answer = nndc._parse_float_uncertainty('257.123', '0.015')
        assert is_close(answer.nominal_value, 257.123)
        assert is_close(answer.std_dev, 0.015)

    def test_03(self):
        """Test _parse_float_uncertainty('257.123E0', '0.005')............."""
        answer = nndc._parse_float_uncertainty('257.123E0', '0.005')
        assert is_close(answer.nominal_value, 257.123)
        assert is_close(answer.std_dev, 0.005)

    def test_04(self):
        """Test _parse_float_uncertainty('257.123E+0', '0.005')............"""
        answer = nndc._parse_float_uncertainty('257.123E+0', '0.005')
        assert is_close(answer.nominal_value, 257.123)
        assert is_close(answer.std_dev, 0.005)

    def test_05(self):
        """Test _parse_float_uncertainty('257.123E4', '5E3')..............."""
        answer = nndc._parse_float_uncertainty('257.123E4', '5E3')
        assert is_close(answer.nominal_value, 2571230.)
        assert is_close(answer.std_dev, 5000.)

    def test_06(self):
        """Test _parse_float_uncertainty('257.123E+4', '5E-3')............."""
        answer = nndc._parse_float_uncertainty('257.123E+4', '5E-3')
        assert is_close(answer.nominal_value, 2571230.)
        assert is_close(answer.std_dev, 0.005)

    def test_07(self):
        """Test _parse_float_uncertainty('257.123E+4', '5E-1E-2').........."""
        answer = nndc._parse_float_uncertainty('257.123E+4', '5E-1E-2')
        assert is_close(answer.nominal_value, 2571230.)
        assert is_close(answer.std_dev, 0.005)

    def test_08(self):
        """Test _parse_float_uncertainty('257.123', '')...................."""
        answer = nndc._parse_float_uncertainty('257.123', '')
        assert isinstance(answer, float)
        assert is_close(answer, 257.123)

    def test_09(self):
        """Test _parse_float_uncertainty('8', '2')........................."""
        answer = nndc._parse_float_uncertainty('8', '2')
        assert is_close(answer.nominal_value, 8.)
        assert is_close(answer.std_dev, 2.)

    def test_10(self):
        """Test _parse_float_uncertainty('8', '').........................."""
        answer = nndc._parse_float_uncertainty('8', '')
        assert isinstance(answer, float)
        assert is_close(answer, 8.)

    def test_11(self):
        """Test _parse_float_uncertainty('100.0%', '')...................."""
        answer = nndc._parse_float_uncertainty('100.0%', '')
        assert is_close(answer, 100.)

    def test_12(self):
        """Test _parse_float_uncertainty('73.92+X', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+X', '')
        assert is_close(answer, 73.92)

    def test_13(self):
        """Test _parse_float_uncertainty('73.92+Y', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+Y', '')
        assert is_close(answer, 73.92)

    def test_14(self):
        """Test _parse_float_uncertainty('', '')..........................."""
        answer = nndc._parse_float_uncertainty('', '')
        assert answer is None

    def test_15(self):
        """Test _parse_float_uncertainty('****', '')......................."""
        answer = nndc._parse_float_uncertainty('****', '')
        assert answer is None

    def test_16(self):
        """Test _parse_float_uncertainty('~7', '1')........................"""
        answer = nndc._parse_float_uncertainty('~7', '1')
        assert answer is None

    def test_17(self):
        """Test _parse_float_uncertainty('1', '****')......................"""
        answer = nndc._parse_float_uncertainty('1', '****')
        assert answer is None

    def test_18(self):
        """Test _parse_float_uncertainty('73.92', 'AP')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'AP')
        assert answer is None

    def test_19(self):
        """Test _parse_float_uncertainty('73.92', 'CA')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'CA')
        assert answer is None

    def test_20(self):
        """Test _parse_float_uncertainty('X', '7') raises NNDCError........"""
        with pytest.raises(nndc.NNDCError):
            nndc._parse_float_uncertainty('X', '7')

    def test_21(self):
        """Test _parse_float_uncertainty('7', 'X') raises NNDCError........"""
        with pytest.raises(nndc.NNDCError):
            nndc._parse_float_uncertainty('7', 'X')


class NNDCQueryTests(object):
    """Tests common to NNDCQuery-derived classes."""

    def setup_method(self):
        self.cls = nndc._NNDCQuery

        def fetch_dummy(**kwargs):  # pylint: disable=unused-argument
            """Dummy fetch_ function for self.fetch."""
            return pd.DataFrame()

        self.fetch = fetch_dummy

    def stable_isotope_condition(self, df):
        """What should be true about the dataframe if the isotope is stable."""
        return len(df) > 0

    def test_query_nuc_Co60(self):
        """Test NNDCQuery: nuc='Co-60'....................................."""
        d = self.fetch(nuc='Co-60')
        assert len(d) > 0

    def test_query_nuc_He4(self):
        """Test NNDCQuery: nuc='He-4'......................................"""
        d = self.fetch(nuc='He-4')
        assert self.stable_isotope_condition(d)

    def test_query_nuc_V50(self):
        """Test NNDCQuery: nuc='V-50'......................................"""
        d = self.fetch(nuc='V-50')
        assert len(d) > 0

    def test_query_nuc_Ge70(self):
        """Test NNDCQuery: nuc='Ge-70'....................................."""
        d = self.fetch(nuc='Ge-70')
        assert self.stable_isotope_condition(d)

    def test_query_nuc_U238(self):
        """Test NNDCQuery: nuc='U-238'....................................."""
        d = self.fetch(nuc='U-238')
        assert len(d) > 0

    def test_query_nuc_Pa234m(self):
        """Test NNDCQuery: nuc='Pa-234m' raises exception.................."""
        with pytest.raises(nndc.NNDCError):
            self.fetch(nuc='Pa-234m')

    def test_query_nuc_Pa234(self):
        """Test NNDCQuery: nuc='Pa-234'...................................."""
        d = self.fetch(nuc='Pa-234')
        assert len(d) > 0

    def test_query_z_6(self):
        """Test NNDCQuery: z=6............................................."""
        d = self.fetch(z=6)
        assert len(d) > 0

    def test_query_a_12(self):
        """Test NNDCQuery: a=12............................................"""
        d = self.fetch(a=12)
        assert len(d) > 0

    def test_query_n_6(self):
        """Test NNDCQuery: n=6............................................."""
        d = self.fetch(n=6)
        assert len(d) > 0

    def test_query_z_6_a_12(self):
        """Test NNDCQuery: z=6, a=12......................................."""
        d = self.fetch(z=6, a=12)
        assert self.stable_isotope_condition(d)

    def test_query_n_6_a_12(self):
        """Test NNDCQuery: n=6, a=12......................................."""
        d = self.fetch(n=6, a=12)
        assert self.stable_isotope_condition(d)

    def test_query_z_6_a_12_n_6(self):
        """Test NNDCQuery: z=6, a=12, n=6.................................."""
        d = self.fetch(z=6, a=12, n=6)
        assert self.stable_isotope_condition(d)

    def test_query_zrange_1_20(self):
        """Test NNDCQuery: z_range=(1, 20)................................."""
        d = self.fetch(z_range=(1, 20))
        assert len(d) > 0

    def test_query_zrange_1_20_z_any(self):
        """Test NNDCQuery: z_range=(1, 20), z_any=True....................."""
        d = self.fetch(z_range=(1, 20), z_any=True)
        assert len(d) > 0

    def test_query_zrange_1_20_z_even(self):
        """Test NNDCQuery: z_range=(1, 20), z_even=True...................."""
        d = self.fetch(z_range=(1, 20), z_even=True)
        assert len(d) > 0

    def test_query_zrange_1_20_z_odd(self):
        """Test NNDCQuery: z_range=(1, 20), z_odd=True....................."""
        d = self.fetch(z_range=(1, 20), z_odd=True)
        assert len(d) > 0

    def test_query_zrange_30_50(self):
        """Test NNDCQuery: z_range=(30, 50)................................"""
        d = self.fetch(z_range=(30, 50))
        assert len(d) > 0

    def test_query_zrange_100_118(self):
        """Test NNDCQuery: z_range=(100, 118).............................."""
        d = self.fetch(z_range=(100, 118))
        assert len(d) > 0

    def test_query_zrange_230_250(self):
        """Test NNDCQuery: z_range=(230, 250) returns empty dataframe......"""
        d = self.fetch(z_range=(230, 250))
        assert len(d) == 0

    def test_query_arange_1_20(self):
        """Test NNDCQuery: a_range=(1, 20)................................."""
        d = self.fetch(a_range=(1, 20))
        assert len(d) > 0

    def test_query_arange_1_20_a_any(self):
        """Test NNDCQuery: a_range=(1, 20), a_any=True....................."""
        d = self.fetch(a_range=(1, 20), a_any=True)
        assert len(d) > 0

    def test_query_arange_1_20_a_even(self):
        """Test NNDCQuery: a_range=(1, 20), a_even=True...................."""
        d = self.fetch(a_range=(1, 20), a_even=True)
        assert len(d) > 0

    def test_query_arange_1_20_a_odd(self):
        """Test NNDCQuery: a_range=(1, 20), a_odd=True....................."""
        d = self.fetch(a_range=(1, 20), a_odd=True)
        assert len(d) > 0

    def test_query_nrange_1_20(self):
        """Test NNDCQuery: n_range=(1, 20)................................."""
        d = self.fetch(n_range=(1, 20))
        assert len(d) > 0

    def test_query_nrange_1_20_n_any(self):
        """Test NNDCQuery: n_range=(1, 20), n_any=True....................."""
        d = self.fetch(n_range=(1, 20), n_any=True)
        assert len(d) > 0

    def test_query_nrange_1_20_n_even(self):
        """Test NNDCQuery: n_range=(1, 20), n_even=True...................."""
        d = self.fetch(n_range=(1, 20), n_even=True)
        assert len(d) > 0

    def test_query_nrange_1_20_n_odd(self):
        """Test NNDCQuery: n_range=(1, 20), n_odd=True....................."""
        d = self.fetch(n_range=(1, 20), n_odd=True)
        assert len(d) > 0

    def test_query_nrange_1_20_z_even_a_even_n_any(self):
        """Test NNDCQuery: z_range=(1, 20), z_even, a_even, n_any.........."""
        d = self.fetch(
            z_range=(1, 20), z_even=True, a_even=True, n_any=True)
        assert len(d) > 0

    def test_query_exception_not_found(self):
        """Test NNDCQuery exception if website not found..................."""
        _URL_ORIG = self.cls._URL
        self.cls._URL = 'http://httpbin.org/status/404'
        with pytest.raises(nndc.NNDCError):
            self.fetch(nuc='Co-60')
        self.cls._URL = _URL_ORIG

    def test_query_exception_empty(self):
        """Test NNDCQuery exception if website is empty...................."""
        _URL_ORIG = self.cls._URL
        self.cls._URL = 'http://httpbin.org/post'
        with pytest.raises(nndc.NNDCError):
            self.fetch(nuc='Co-60')
        self.cls._URL = _URL_ORIG


@pytest.mark.webtest
class TestNuclearWalletCard(NNDCQueryTests):
    """Test NNDC nuclear_wallet_card query."""

    def setup_method(self):
        self.cls = nndc._NuclearWalletCardQuery
        self.fetch = nndc.fetch_wallet_card

    def test_query_nuc_Co60_BM(self):
        """Test fetch_wallet_card: nuc='Co-60', decay='B-'................."""
        d = self.fetch(nuc='Co-60', decay='B-')
        assert len(d) > 0

    def test_query_nuc_Pu239_SF(self):
        """Test fetch_wallet_card: nuc='Pu-239', decay='SF'................"""
        d = self.fetch(nuc='Pu-239', decay='SF')
        assert len(d) > 0


@pytest.mark.webtest
class TestDecayRadiationQuery(NNDCQueryTests):
    """Test NNDC decay_radiation."""

    def setup_method(self):
        self.cls = nndc._DecayRadiationQuery
        self.fetch = nndc.fetch_decay_radiation

    def stable_isotope_condition(self, df):
        """What should be true about the dataframe if the isotope is stable."""
        return len(df) == 0

    def test_decay_nuc_Co60_BM(self):
        """Test fetch_decay_radiation: nuc='Co-60', decay='B-'............."""
        d = self.fetch(nuc='Co-60', decay='B-')
        assert len(d) > 0

    def test_decay_nuc_Pu239_ANY(self):
        """Test fetch_decay_radiation: nuc='Pu-239', decay='ANY'..........."""
        d = self.fetch(nuc='Pu-239', decay='ANY')
        assert len(d) > 0

    def test_decay_nuc_Pu239_ANY_G(self):
        """Test fetch_decay_radiation: nuc='Pu-239', type='Gamma'.........."""
        d = self.fetch(nuc='Pu-239', type='Gamma')
        assert len(d) > 0

    def test_decay_nuc_200_300_ANY_G(self):
        """Test fetch_decay_radiation: z_range=(200, 300), type='Gamma'...."""
        d = self.fetch(z_range=(100, 120), type='Gamma')
        assert len(d) > 0
