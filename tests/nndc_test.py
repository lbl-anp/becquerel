"""Test NNDC data queries."""

# pylint: disable=protected-access,no-self-use,no-member

from __future__ import print_function
import unittest
from becquerel.tools import nndc


def is_close(f1, f2, ppm=1.):
    """True if f1 and f2 are within the given parts per million."""
    return abs((f1 - f2) / f2) < ppm / 1.e6


class ParseFloatUncertaintyTests(unittest.TestCase):
    """Test _parse_float_uncertainty()."""

    def test_01(self):
        """Test _parse_float_uncertainty('257.123', '0.005')..............."""
        answer = nndc._parse_float_uncertainty('257.123', '0.005')
        self.assertTrue(is_close(answer.nominal_value, 257.123))
        self.assertTrue(is_close(answer.std_dev, 0.005))

    def test_02(self):
        """Test _parse_float_uncertainty('257.123', '0.015')..............."""
        answer = nndc._parse_float_uncertainty('257.123', '0.015')
        self.assertTrue(is_close(answer.nominal_value, 257.123))
        self.assertTrue(is_close(answer.std_dev, 0.015))

    def test_03(self):
        """Test _parse_float_uncertainty('257.123E0', '0.005')............."""
        answer = nndc._parse_float_uncertainty('257.123E0', '0.005')
        self.assertTrue(is_close(answer.nominal_value, 257.123))
        self.assertTrue(is_close(answer.std_dev, 0.005))

    def test_04(self):
        """Test _parse_float_uncertainty('257.123E+0', '0.005')............"""
        answer = nndc._parse_float_uncertainty('257.123E+0', '0.005')
        self.assertTrue(is_close(answer.nominal_value, 257.123))
        self.assertTrue(is_close(answer.std_dev, 0.005))

    def test_05(self):
        """Test _parse_float_uncertainty('257.123E4', '5E3')..............."""
        answer = nndc._parse_float_uncertainty('257.123E4', '5E3')
        self.assertTrue(is_close(answer.nominal_value, 2571230.))
        self.assertTrue(is_close(answer.std_dev, 5000.))

    def test_06(self):
        """Test _parse_float_uncertainty('257.123E+4', '5E-3')............."""
        answer = nndc._parse_float_uncertainty('257.123E+4', '5E-3')
        self.assertTrue(is_close(answer.nominal_value, 2571230.))
        self.assertTrue(is_close(answer.std_dev, 0.005))

    def test_07(self):
        """Test _parse_float_uncertainty('257.123', '')...................."""
        answer = nndc._parse_float_uncertainty('257.123', '')
        self.assertTrue(is_close(answer, 257.123))

    def test_08(self):
        """Test _parse_float_uncertainty('8', '2')........................."""
        answer = nndc._parse_float_uncertainty('8', '2')
        self.assertTrue(is_close(answer.nominal_value, 8.))
        self.assertTrue(is_close(answer.std_dev, 2.))

    def test_09(self):
        """Test _parse_float_uncertainty('8', '').........................."""
        answer = nndc._parse_float_uncertainty('8', '')
        self.assertTrue(is_close(answer, 8.))

    def test_10(self):
        """Test _parse_float_uncertainty('73.92+X', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+X', '')
        self.assertTrue(is_close(answer, 73.92))

    def test_11(self):
        """Test _parse_float_uncertainty('73.92+Y', '')...................."""
        answer = nndc._parse_float_uncertainty('73.92+Y', '')
        self.assertTrue(is_close(answer, 73.92))

    def test_12(self):
        """Test _parse_float_uncertainty('73.92', 'AP')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'AP')
        self.assertTrue(is_close(answer, 73.92))

    def test_13(self):
        """Test _parse_float_uncertainty('73.92', 'CA')...................."""
        answer = nndc._parse_float_uncertainty('73.92', 'CA')
        self.assertTrue(is_close(answer, 73.92))


# pylint: disable=W0212
NNDC_WALLET_URL_ORIG = nndc.NuclearWalletCardQuery._URL


class NuclearWalletCardTests(unittest.TestCase):
    """Test NNDC nuclear_wallet_card query."""

    def test_wallet_perform(self):
        """Test NuclearWalletCardQuery: perform=False......................"""
        d = nndc.NuclearWalletCardQuery(nuc='Co-60', perform=False)
        d.update()
        d.perform()
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_Co60(self):
        """Test NuclearWalletCardQuery: nuc='Co-60'........................"""
        d = nndc.NuclearWalletCardQuery(nuc='Co-60')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_He4(self):
        """Test NuclearWalletCardQuery: nuc='He-4'........................."""
        d = nndc.NuclearWalletCardQuery(nuc='He-4')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_V50(self):
        """Test NuclearWalletCardQuery: nuc='V-50'........................."""
        d = nndc.NuclearWalletCardQuery(nuc='V-50')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_Ge70(self):
        """Test NuclearWalletCardQuery: nuc='Ge-70'........................"""
        d = nndc.NuclearWalletCardQuery(nuc='Ge-70')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_U238(self):
        """Test NuclearWalletCardQuery: nuc='U-238'........................"""
        d = nndc.NuclearWalletCardQuery(nuc='U-238')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_Pa234m(self):
        """Test NuclearWalletCardQuery: nuc='Pa-234m' raises exception....."""
        with self.assertRaises(nndc.NNDCError):
            nndc.NuclearWalletCardQuery(nuc='Pa-234m')

    def test_wallet_nuc_Pa234(self):
        """Test NuclearWalletCardQuery: nuc='Pa-234'......................."""
        d = nndc.NuclearWalletCardQuery(nuc='Pa-234')
        self.assertTrue(len(d) > 0)

    def test_wallet_z_6(self):
        """Test NuclearWalletCardQuery: z=6................................"""
        d = nndc.NuclearWalletCardQuery(z=6)
        self.assertTrue(len(d) > 0)

    def test_wallet_a_12(self):
        """Test NuclearWalletCardQuery: a=12..............................."""
        d = nndc.NuclearWalletCardQuery(a=12)
        self.assertTrue(len(d) > 0)

    def test_wallet_n_6(self):
        """Test NuclearWalletCardQuery: n=6................................"""
        d = nndc.NuclearWalletCardQuery(n=6)
        self.assertTrue(len(d) > 0)

    def test_wallet_z_6_a_12(self):
        """Test NuclearWalletCardQuery: z=6, a=12.........................."""
        d = nndc.NuclearWalletCardQuery(z=6, a=12)
        self.assertTrue(len(d) > 0)

    def test_wallet_n_6_a_12(self):
        """Test NuclearWalletCardQuery: n=6, a=12.........................."""
        d = nndc.NuclearWalletCardQuery(n=6, a=12)
        self.assertTrue(len(d) > 0)

    def test_wallet_z_6_a_12_n_6(self):
        """Test NuclearWalletCardQuery: z=6, a=12, n=6....................."""
        d = nndc.NuclearWalletCardQuery(z=6, a=12, n=6)
        self.assertTrue(len(d) > 0)

    def test_wallet_zrange_1_20(self):
        """Test NuclearWalletCardQuery: z_range=(1, 20)...................."""
        d = nndc.NuclearWalletCardQuery(z_range=(1, 20))
        self.assertTrue(len(d) > 0)

    def test_wallet_zrange_30_50(self):
        """Test NuclearWalletCardQuery: z_range=(30, 50)..................."""
        d = nndc.NuclearWalletCardQuery(z_range=(30, 50))
        self.assertTrue(len(d) > 0)

    def test_wallet_zrange_100_118(self):
        """Test NuclearWalletCardQuery: z_range=(100, 118)................."""
        d = nndc.NuclearWalletCardQuery(z_range=(100, 118))
        self.assertTrue(len(d) > 0)

    def test_wallet_zrange_230_250(self):
        """Test NuclearWalletCardQuery: z_range=(230, 250) raises except..."""
        with self.assertRaises(nndc.NNDCError):
            nndc.NuclearWalletCardQuery(z_range=(230, 250))

    def test_wallet_nuc_Co60_BM(self):
        """Test NuclearWalletCardQuery: nuc='Co-60', decay='B-'............"""
        d = nndc.NuclearWalletCardQuery(nuc='Co-60', decay='B-')
        self.assertTrue(len(d) > 0)

    def test_wallet_nuc_Pu239_SF(self):
        """Test NuclearWalletCardQuery: nuc='Pu-239', decay='SF'..........."""
        d = nndc.NuclearWalletCardQuery(nuc='Pu-239', decay='SF')
        self.assertTrue(len(d) > 0)

    def test_wallet_exception_not_found(self):
        """Test NuclearWalletCardQuery exception if website not found......"""
        nndc.NuclearWalletCardQuery._URL = 'http://httpbin.org/status/404'
        with self.assertRaises(nndc.NNDCError):
            nndc.NuclearWalletCardQuery(nuc='Co-60')
        nndc.NuclearWalletCardQuery._URL = NNDC_WALLET_URL_ORIG

    def test_wallet_exception_empty(self):
        """Test NuclearWalletCardQuery exception if website is empty......."""
        nndc.NuclearWalletCardQuery._URL = 'http://httpbin.org/post'
        with self.assertRaises(nndc.NNDCError):
            nndc.NuclearWalletCardQuery(nuc='Co-60')
        nndc.NuclearWalletCardQuery._URL = NNDC_WALLET_URL_ORIG


# pylint: disable=W0212
NNDC_DECAYRAD_URL_ORIG = nndc.DecayRadiationQuery._URL


class DecayRadiationQueryTests(unittest.TestCase):
    """Test NNDC decay_radiation."""

    def test_decay_nuc_Co60(self):
        """Test DecayRadiationQuery: nuc='Co-60'..........................."""
        d = nndc.DecayRadiationQuery(nuc='Co-60')
        self.assertTrue(len(d) > 0)

    def test_decay_nuc_Co60_BM(self):
        """Test DecayRadiationQuery: nuc='Co-60', decay='B-'..............."""
        d = nndc.DecayRadiationQuery(nuc='Co-60', decay='B-')
        self.assertTrue(len(d) > 0)

    def test_decay_nuc_Pu239_ANY(self):
        """Test DecayRadiationQuery: nuc='Pu-239', decay='ANY'............."""
        d = nndc.DecayRadiationQuery(nuc='Pu-239', decay='ANY')
        self.assertTrue(len(d) > 0)

    def test_decay_nuc_Pu239_ANY_G(self):
        """Test DecayRadiationQuery: nuc='Pu-239', type='Gamma'............"""
        d = nndc.DecayRadiationQuery(nuc='Pu-239', type='Gamma')
        self.assertTrue(len(d) > 0)

    def test_decay_exception_not_found(self):
        """Test DecayRadiationQuery raises exception if website not found.."""
        nndc.DecayRadiationQuery._URL = 'http://httpbin.org/status/404'
        with self.assertRaises(nndc.NNDCError):
            nndc.DecayRadiationQuery(nuc='Co-60')
        nndc.DecayRadiationQuery._URL = NNDC_DECAYRAD_URL_ORIG

    def test_decay_exception_empty(self):
        """Test DecayRadiationQuery raises exception if website is empty..."""
        nndc.DecayRadiationQuery._URL = 'http://httpbin.org/post'
        with self.assertRaises(nndc.NNDCError):
            nndc.DecayRadiationQuery(nuc='Co-60')
        nndc.DecayRadiationQuery._URL = NNDC_DECAYRAD_URL_ORIG

    def test_decay_nuc_200_300_ANY_G(self):
        """Test DecayRadiationQuery: z_range=(200, 300), type='Gamma'......"""
        d = nndc.DecayRadiationQuery(z_range=(100, 120), type='Gamma')
        self.assertTrue(len(d) > 0)


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
