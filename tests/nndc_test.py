"""Test NNDC data queries."""

from __future__ import print_function
from becquerel.tools import nndc
import pytest

# pylint: disable=protected-access,no-self-use,too-many-public-methods


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
        """Test _parse_float_uncertainty('257.123', '')...................."""
        answer = nndc._parse_float_uncertainty('257.123', '')
        assert is_close(answer, 257.123)

    def test_08(self):
        """Test _parse_float_uncertainty('8', '2')........................."""
        answer = nndc._parse_float_uncertainty('8', '2')
        assert is_close(answer.nominal_value, 8.)
        assert is_close(answer.std_dev, 2.)

    def test_09(self):
        """Test _parse_float_uncertainty('8', '').........................."""
        answer = nndc._parse_float_uncertainty('8', '')
        assert is_close(answer, 8.)

    def test_10(self):
        """Test _parse_float_uncertainty('73.92+X', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+X', '')
        assert is_close(answer, 73.92)

    def test_11(self):
        """Test _parse_float_uncertainty('73.92+Y', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+Y', '')
        assert is_close(answer, 73.92)

    def test_12(self):
        """Test _parse_float_uncertainty('73.92', 'AP')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'AP')
        assert is_close(answer, 73.92)

    def test_13(self):
        """Test _parse_float_uncertainty('73.92', 'CA')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'CA')
        assert is_close(answer, 73.92)


NNDC_WALLET_URL_ORIG = nndc._NuclearWalletCardQuery._URL


@pytest.mark.webtest
class TestNuclearWalletCard(object):
    """Test NNDC nuclear_wallet_card query."""

    def test_wallet_nuc_Co60(self):
        """Test fetch_wallet_card: nuc='Co-60'............................."""
        d = nndc.fetch_wallet_card(nuc='Co-60')
        assert len(d) > 0

    def test_wallet_nuc_He4(self):
        """Test fetch_wallet_card: nuc='He-4'.............................."""
        d = nndc.fetch_wallet_card(nuc='He-4')
        assert len(d) > 0

    def test_wallet_nuc_V50(self):
        """Test fetch_wallet_card: nuc='V-50'.............................."""
        d = nndc.fetch_wallet_card(nuc='V-50')
        assert len(d) > 0

    def test_wallet_nuc_Ge70(self):
        """Test fetch_wallet_card: nuc='Ge-70'............................."""
        d = nndc.fetch_wallet_card(nuc='Ge-70')
        assert len(d) > 0

    def test_wallet_nuc_U238(self):
        """Test fetch_wallet_card: nuc='U-238'............................."""
        d = nndc.fetch_wallet_card(nuc='U-238')
        assert len(d) > 0

    def test_wallet_nuc_Pa234m(self):
        """Test fetch_wallet_card: nuc='Pa-234m' raises exception.........."""
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_wallet_card(nuc='Pa-234m')

    def test_wallet_nuc_Pa234(self):
        """Test fetch_wallet_card: nuc='Pa-234'............................"""
        d = nndc.fetch_wallet_card(nuc='Pa-234')
        assert len(d) > 0

    def test_wallet_z_6(self):
        """Test fetch_wallet_card: z=6....................................."""
        d = nndc.fetch_wallet_card(z=6)
        assert len(d) > 0

    def test_wallet_a_12(self):
        """Test fetch_wallet_card: a=12...................................."""
        d = nndc.fetch_wallet_card(a=12)
        assert len(d) > 0

    def test_wallet_n_6(self):
        """Test fetch_wallet_card: n=6....................................."""
        d = nndc.fetch_wallet_card(n=6)
        assert len(d) > 0

    def test_wallet_z_6_a_12(self):
        """Test fetch_wallet_card: z=6, a=12..............................."""
        d = nndc.fetch_wallet_card(z=6, a=12)
        assert len(d) > 0

    def test_wallet_n_6_a_12(self):
        """Test fetch_wallet_card: n=6, a=12..............................."""
        d = nndc.fetch_wallet_card(n=6, a=12)
        assert len(d) > 0

    def test_wallet_z_6_a_12_n_6(self):
        """Test fetch_wallet_card: z=6, a=12, n=6.........................."""
        d = nndc.fetch_wallet_card(z=6, a=12, n=6)
        assert len(d) > 0

    def test_wallet_zrange_1_20(self):
        """Test fetch_wallet_card: z_range=(1, 20)........................."""
        d = nndc.fetch_wallet_card(z_range=(1, 20))
        assert len(d) > 0

    def test_wallet_zrange_30_50(self):
        """Test fetch_wallet_card: z_range=(30, 50)........................"""
        d = nndc.fetch_wallet_card(z_range=(30, 50))
        assert len(d) > 0

    def test_wallet_zrange_100_118(self):
        """Test fetch_wallet_card: z_range=(100, 118)......................"""
        d = nndc.fetch_wallet_card(z_range=(100, 118))
        assert len(d) > 0

    def test_wallet_zrange_230_250(self):
        """Test fetch_wallet_card: z_range=(230, 250) raises except........"""
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_wallet_card(z_range=(230, 250))

    def test_wallet_nuc_Co60_BM(self):
        """Test fetch_wallet_card: nuc='Co-60', decay='B-'................."""
        d = nndc.fetch_wallet_card(nuc='Co-60', decay='B-')
        assert len(d) > 0

    def test_wallet_nuc_Pu239_SF(self):
        """Test fetch_wallet_card: nuc='Pu-239', decay='SF'................"""
        d = nndc.fetch_wallet_card(nuc='Pu-239', decay='SF')
        assert len(d) > 0

    def test_wallet_exception_not_found(self):
        """Test fetch_wallet_card exception if website not found..........."""
        nndc._NuclearWalletCardQuery._URL = 'http://httpbin.org/status/404'
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_wallet_card(nuc='Co-60')
        nndc._NuclearWalletCardQuery._URL = NNDC_WALLET_URL_ORIG

    def test_wallet_exception_empty(self):
        """Test fetch_wallet_card exception if website is empty............"""
        nndc._NuclearWalletCardQuery._URL = 'http://httpbin.org/post'
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_wallet_card(nuc='Co-60')
        nndc._NuclearWalletCardQuery._URL = NNDC_WALLET_URL_ORIG


NNDC_DECAYRAD_URL_ORIG = nndc._DecayRadiationQuery._URL


@pytest.mark.webtest
class TestDecayRadiationQuery(object):
    """Test NNDC decay_radiation."""

    def test_decay_nuc_Co60(self):
        """Test fetch_decay_radiation: nuc='Co-60'........................."""
        d = nndc.fetch_decay_radiation(nuc='Co-60')
        assert len(d) > 0

    def test_decay_nuc_Co60_BM(self):
        """Test fetch_decay_radiation: nuc='Co-60', decay='B-'............."""
        d = nndc.fetch_decay_radiation(nuc='Co-60', decay='B-')
        assert len(d) > 0

    def test_decay_nuc_Pu239_ANY(self):
        """Test fetch_decay_radiation: nuc='Pu-239', decay='ANY'..........."""
        d = nndc.fetch_decay_radiation(nuc='Pu-239', decay='ANY')
        assert len(d) > 0

    def test_decay_nuc_Pu239_ANY_G(self):
        """Test fetch_decay_radiation: nuc='Pu-239', type='Gamma'.........."""
        d = nndc.fetch_decay_radiation(nuc='Pu-239', type='Gamma')
        assert len(d) > 0

    def test_decay_exception_not_found(self):
        """Test fetch_decay_radiation raises exception if website not found"""
        nndc._DecayRadiationQuery._URL = 'http://httpbin.org/status/404'
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_decay_radiation(nuc='Co-60')
        nndc._DecayRadiationQuery._URL = NNDC_DECAYRAD_URL_ORIG

    def test_decay_exception_empty(self):
        """Test fetch_decay_radiation raises exception if website is empty."""
        nndc._DecayRadiationQuery._URL = 'http://httpbin.org/post'
        with pytest.raises(nndc.NNDCError):
            nndc.fetch_decay_radiation(nuc='Co-60')
        nndc._DecayRadiationQuery._URL = NNDC_DECAYRAD_URL_ORIG

    def test_decay_nuc_200_300_ANY_G(self):
        """Test fetch_decay_radiation: z_range=(200, 300), type='Gamma'...."""
        d = nndc.fetch_decay_radiation(z_range=(100, 120), type='Gamma')
        assert len(d) > 0
