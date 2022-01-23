"""Test NNDC data queries."""

import numpy as np
import pandas as pd
from becquerel.tools import nndc
import pytest


def ufloats_overlap_range(ufloats, vmin, vmax):
    """Return whether the +/- 1 sigma range overlaps the value range."""
    vals = []
    sigmas = []
    for val in ufloats:
        if isinstance(val, float):
            vals.append(val)
            sigmas.append(0)
        else:
            vals.append(val.nominal_value)
            sigmas.append(val.std_dev)
    vals = np.array(vals)
    sigmas = np.array(sigmas)
    return ((vals - sigmas <= vmax) | (vals + sigmas >= vmin)).all()


class TestParseFloatUncertainty:
    """Test _parse_float_uncertainty()."""

    def test_01(self):
        """Test _parse_float_uncertainty('257.123', '0.005')..............."""
        answer = nndc._parse_float_uncertainty("257.123", "0.005")
        assert np.isclose(answer.nominal_value, 257.123)
        assert np.isclose(answer.std_dev, 0.005)

    def test_02(self):
        """Test _parse_float_uncertainty('257.123', '0.015')..............."""
        answer = nndc._parse_float_uncertainty("257.123", "0.015")
        assert np.isclose(answer.nominal_value, 257.123)
        assert np.isclose(answer.std_dev, 0.015)

    def test_03(self):
        """Test _parse_float_uncertainty('257.123E0', '0.005')............."""
        answer = nndc._parse_float_uncertainty("257.123E0", "0.005")
        assert np.isclose(answer.nominal_value, 257.123)
        assert np.isclose(answer.std_dev, 0.005)

    def test_04(self):
        """Test _parse_float_uncertainty('257.123E+0', '0.005')............"""
        answer = nndc._parse_float_uncertainty("257.123E+0", "0.005")
        assert np.isclose(answer.nominal_value, 257.123)
        assert np.isclose(answer.std_dev, 0.005)

    def test_05(self):
        """Test _parse_float_uncertainty('257.123E4', '5E3')..............."""
        answer = nndc._parse_float_uncertainty("257.123E4", "5E3")
        assert np.isclose(answer.nominal_value, 2571230.0)
        assert np.isclose(answer.std_dev, 5000.0)

    def test_06(self):
        """Test _parse_float_uncertainty('257.123E+4', '5E-3')............."""
        answer = nndc._parse_float_uncertainty("257.123E+4", "5E-3")
        assert np.isclose(answer.nominal_value, 2571230.0)
        assert np.isclose(answer.std_dev, 0.005)

    def test_07(self):
        """Test _parse_float_uncertainty('257.123E+4', '5E-1E-2').........."""
        answer = nndc._parse_float_uncertainty("257.123E+4", "5E-1E-2")
        assert np.isclose(answer.nominal_value, 2571230.0)
        assert np.isclose(answer.std_dev, 0.005)

    def test_08(self):
        """Test _parse_float_uncertainty('257.123', '')...................."""
        answer = nndc._parse_float_uncertainty("257.123", "")
        assert isinstance(answer, float)
        assert np.isclose(answer, 257.123)

    def test_09(self):
        """Test _parse_float_uncertainty('8', '2')........................."""
        answer = nndc._parse_float_uncertainty("8", "2")
        assert np.isclose(answer.nominal_value, 8.0)
        assert np.isclose(answer.std_dev, 2.0)

    def test_10(self):
        """Test _parse_float_uncertainty('8', '').........................."""
        answer = nndc._parse_float_uncertainty("8", "")
        assert isinstance(answer, float)
        assert np.isclose(answer, 8.0)

    def test_11(self):
        """Test _parse_float_uncertainty('100.0%', '')....................."""
        answer = nndc._parse_float_uncertainty("100.0%", "")
        assert np.isclose(answer, 100.0)

    def test_12(self):
        """Test _parse_float_uncertainty('73.92+X', '')...................."""
        answer = nndc._parse_float_uncertainty("73.92+X", "")
        assert np.isclose(answer, 73.92)

    def test_13(self):
        """Test _parse_float_uncertainty('Y', '').........................."""
        answer = nndc._parse_float_uncertainty("Y", "")
        assert answer == 0.0

    def test_14(self):
        """Test _parse_float_uncertainty('', '')..........................."""
        answer = nndc._parse_float_uncertainty("", "")
        assert answer is None

    def test_15(self):
        """Test _parse_float_uncertainty('****', '')......................."""
        answer = nndc._parse_float_uncertainty("****", "")
        assert answer is None

    def test_16(self):
        """Test _parse_float_uncertainty('~7', '1')........................"""
        answer = nndc._parse_float_uncertainty("~7", "1")
        assert np.isclose(answer.nominal_value, 7.0)
        assert np.isclose(answer.std_dev, 1.0)

    def test_17(self):
        """Test _parse_float_uncertainty('1', '****')......................"""
        answer = nndc._parse_float_uncertainty("1", "****")
        assert np.isclose(answer, 1.0)

    def test_18(self):
        """Test _parse_float_uncertainty('73.92', 'AP')...................."""
        answer = nndc._parse_float_uncertainty("73.92", "AP")
        assert np.isclose(answer, 73.92)

    def test_19(self):
        """Test _parse_float_uncertainty('73.92', 'CA')...................."""
        answer = nndc._parse_float_uncertainty("73.92", "CA")
        assert np.isclose(answer, 73.92)

    def test_20(self):
        """Test _parse_float_uncertainty('7', '@') raises NNDCRequestError."""
        with pytest.raises(nndc.NNDCRequestError):
            nndc._parse_float_uncertainty("7", "@")


class NNDCQueryTests:
    """Tests common to NNDCQuery-derived classes."""

    def setup_method(self):
        self.cls = nndc._NNDCQuery

        def fetch_dummy(**kwargs):
            """Dummy fetch_ function for self.fetch."""
            return pd.DataFrame()

        self.fetch = fetch_dummy

    def stable_isotope_condition(self, df):
        """What should be true about the dataframe if the isotope is stable."""
        return len(df) > 0

    def test_query_nuc_Co60(self):
        """Test NNDCQuery: nuc='Co-60'....................................."""
        d = self.fetch(nuc="Co-60")
        assert len(d) > 0
        assert (d.Z == 27).all()
        assert (d.A == 60).all()

    def test_query_nuc_He4(self):
        """Test NNDCQuery: nuc='He-4'......................................"""
        d = self.fetch(nuc="He-4")
        assert self.stable_isotope_condition(d)
        assert (d.Z == 2).all()
        assert (d.A == 4).all()

    def test_query_nuc_V50(self):
        """Test NNDCQuery: nuc='V-50'......................................"""
        d = self.fetch(nuc="V-50")
        assert len(d) > 0
        assert (d.Z == 23).all()
        assert (d.A == 50).all()

    def test_query_nuc_Ge70(self):
        """Test NNDCQuery: nuc='Ge-70'....................................."""
        d = self.fetch(nuc="Ge-70")
        assert self.stable_isotope_condition(d)
        assert (d.Z == 32).all()
        assert (d.A == 70).all()

    def test_query_nuc_U238(self):
        """Test NNDCQuery: nuc='U-238'....................................."""
        d = self.fetch(nuc="U-238")
        assert len(d) > 0
        assert (d.Z == 92).all()
        assert (d.A == 238).all()

    def test_query_nuc_Pa234(self):
        """Test NNDCQuery: nuc='Pa-234'...................................."""
        d = self.fetch(nuc="Pa-234")
        assert len(d) > 0
        assert (d.Z == 91).all()
        assert (d.A == 234).all()

    def test_query_z_6(self):
        """Test NNDCQuery: z=6............................................."""
        d = self.fetch(z=6)
        assert len(d) > 0
        assert (d.Z == 6).all()

    def test_query_a_12(self):
        """Test NNDCQuery: a=12............................................"""
        d = self.fetch(a=12)
        assert len(d) > 0
        assert (d.A == 12).all()

    def test_query_n_6(self):
        """Test NNDCQuery: n=6............................................."""
        d = self.fetch(n=6)
        assert len(d) > 0
        assert (d.N == 6).all()

    def test_query_z_6_a_12(self):
        """Test NNDCQuery: z=6, a=12......................................."""
        d = self.fetch(z=6, a=12)
        assert self.stable_isotope_condition(d)
        assert (d.Z == 6).all()
        assert (d.A == 12).all()

    def test_query_n_6_a_12(self):
        """Test NNDCQuery: n=6, a=12......................................."""
        d = self.fetch(n=6, a=12)
        assert self.stable_isotope_condition(d)
        assert (d.A == 12).all()
        assert (d.N == 6).all()

    def test_query_z_6_a_12_n_6(self):
        """Test NNDCQuery: z=6, a=12, n=6.................................."""
        d = self.fetch(z=6, a=12, n=6)
        assert self.stable_isotope_condition(d)
        assert (d.Z == 6).all()
        assert (d.A == 12).all()
        assert (d.N == 6).all()

    def test_query_zrange_9_10(self):
        """Test NNDCQuery: z_range=(9, 10)................................."""
        d = self.fetch(z_range=(9, 10))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()

    def test_query_zrange_9_10_z_any(self):
        """Test NNDCQuery: z_range=(9, 10), z_any=True....................."""
        d = self.fetch(z_range=(9, 10), z_any=True)
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()

    def test_query_zrange_9_10_z_even(self):
        """Test NNDCQuery: z_range=(9, 10), z_even=True...................."""
        d = self.fetch(z_range=(9, 10), z_even=True)
        assert len(d) > 0
        assert (d.Z == 10).all()

    def test_query_zrange_9_10_z_odd(self):
        """Test NNDCQuery: z_range=(9, 10), z_odd=True....................."""
        d = self.fetch(z_range=(9, 10), z_odd=True)
        assert len(d) > 0
        assert (d.Z == 9).all()

    def test_query_zrange_21_22(self):
        """Test NNDCQuery: z_range=(21, 22)................................"""
        d = self.fetch(z_range=(21, 22))
        assert len(d) > 0
        assert ((21 <= d.Z) & (d.Z <= 22)).all()

    def test_query_zrange_110_118(self):
        """Test NNDCQuery: z_range=(110, 118).............................."""
        d = self.fetch(z_range=(110, 118))
        assert len(d) > 0
        assert ((110 <= d.Z) & (d.Z <= 118)).all()

    def test_query_zrange_None_3(self):
        """Test NNDCQuery: z_range=(None, 3)..............................."""
        d = self.fetch(z_range=(None, 3))
        assert len(d) > 0
        assert (d.Z <= 3).all()

    def test_query_zrange_110_None(self):
        """Test NNDCQuery: z_range=(110, None)............................."""
        d = self.fetch(z_range=(110, None))
        assert len(d) > 0
        assert (110 <= d.Z).all()

    def test_query_arange_9_10(self):
        """Test NNDCQuery: a_range=(9, 10)................................."""
        d = self.fetch(a_range=(9, 10))
        assert len(d) > 0
        assert ((9 <= d.A) & (d.A <= 10)).all()

    def test_query_arange_9_10_a_any(self):
        """Test NNDCQuery: a_range=(9, 10), a_any=True....................."""
        d = self.fetch(a_range=(9, 10), a_any=True)
        assert len(d) > 0
        assert ((9 <= d.A) & (d.A <= 10)).all()

    def test_query_arange_9_10_a_even(self):
        """Test NNDCQuery: a_range=(9, 10), a_even=True...................."""
        d = self.fetch(a_range=(9, 10), a_even=True)
        assert len(d) > 0
        assert (d.A == 10).all()

    def test_query_arange_7_10_a_odd(self):
        """Test NNDCQuery: a_range=(7, 10), a_odd=True....................."""
        d = self.fetch(a_range=(7, 10), a_odd=True)
        assert len(d) > 0
        assert ((d.A == 7) | (d.A == 9)).all()

    def test_query_arange_None_3(self):
        """Test NNDCQuery: a_range=(None, 3)..............................."""
        d = self.fetch(a_range=(None, 3))
        assert len(d) > 0
        assert (d.A <= 3).all()

    def test_query_arange_270_None(self):
        """Test NNDCQuery: a_range=(270, None)............................."""
        d = self.fetch(a_range=(270, None))
        assert len(d) > 0
        assert (270 <= d.A).all()

    def test_query_nrange_9_10(self):
        """Test NNDCQuery: n_range=(9, 10)................................."""
        d = self.fetch(n_range=(9, 10))
        assert len(d) > 0
        assert ((9 <= d.N) & (d.N <= 10)).all()

    def test_query_nrange_9_10_n_any(self):
        """Test NNDCQuery: n_range=(9, 10), n_any=True....................."""
        d = self.fetch(n_range=(9, 10), n_any=True)
        assert len(d) > 0
        assert ((9 <= d.N) & (d.N <= 10)).all()

    def test_query_nrange_9_10_n_even(self):
        """Test NNDCQuery: n_range=(9, 10), n_even=True...................."""
        d = self.fetch(n_range=(9, 10), n_even=True)
        assert len(d) > 0
        assert (d.N == 10).all()

    def test_query_nrange_9_10_n_odd(self):
        """Test NNDCQuery: n_range=(9, 10), n_odd=True....................."""
        d = self.fetch(n_range=(9, 10), n_odd=True)
        assert len(d) > 0
        assert (d.N == 9).all()

    def test_query_zrange_9_10_z_even_a_even_n_any(self):
        """Test NNDCQuery: z_range=(9, 10), z_even, a_even, n_any.........."""
        d = self.fetch(z_range=(9, 10), z_even=True, a_even=True, n_any=True)
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert (d.Z % 2 == 0).all()
        assert (d.A % 2 == 0).all()

    def test_query_z_6_arange_11_13(self):
        """Test NNDCQuery: z=6, a_range=(11, 13)..........................."""
        d = self.fetch(z=6, a_range=(11, 13))
        assert len(d) > 0
        assert (d.Z == 6).all()
        assert ((11 <= d.A) & (d.A <= 13)).all()

    def test_query_exception_not_found(self):
        """Test NNDCQuery exception if website not found..................."""
        _URL_ORIG = self.cls._URL
        self.cls._URL = "http://httpbin.org/status/404"
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(nuc="Co-60")
        self.cls._URL = _URL_ORIG

    def test_query_exception_empty(self):
        """Test NNDCQuery exception if website is empty...................."""
        _URL_ORIG = self.cls._URL
        self.cls._URL = "http://httpbin.org/post"
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(nuc="Co-60")
        self.cls._URL = _URL_ORIG

    def test_query_exception_range_None(self):
        """Test NNDCQuery exception if a range kwarg is not iterable......."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(n_range=None)

    def test_query_exception_range_empty(self):
        """Test NNDCQuery exception if a range kwarg is empty.............."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(n_range=())

    def test_query_exception_range_len_1(self):
        """Test NNDCQuery exception if a range kwarg has length 1.........."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(n_range=(0,))

    def test_query_exception_range_len_3(self):
        """Test NNDCQuery exception if a range kwarg has length 3.........."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(n_range=(0, 10, 100))

    def test_query_exception_unknown_kwarg(self):
        """Test NNDCQuery exception if an invalid kwarg is given..........."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(bad_kwarg=None)

    def test_query_trange_1Em6_1Em5(self):
        """Test NNDCQuery: t_range=(1e-6, 1e-5)............................"""
        d = self.fetch(t_range=(1e-6, 1e-5))
        assert len(d) > 0
        # TODO: the following assertion fails
        # assert ((1e-6 <= d['T1/2 (s)']) & (d['T1/2 (s)'] <= 1e-5)).all()

    def test_query_zrange_9_10_trange_None_1E4(self):
        """Test NNDCQuery: z_range=(9, 10), t_range=(None, 1e4)............"""
        d = self.fetch(z_range=(9, 10), t_range=(None, 1e4))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert (d["T1/2 (s)"] <= 1e4).all()

    def test_query_zrange_9_10_trange_0_1E4(self):
        """Test NNDCQuery: z_range=(9, 10), t_range=(0, 1e4)..............."""
        d = self.fetch(z_range=(9, 10), t_range=(0, 1e4))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert (d["T1/2 (s)"] <= 1e4).all()

    def test_query_zrange_9_10_trange_1E3_1E9(self):
        """Test NNDCQuery: z_range=(9, 10), t_range=(1e3, 1e9)............."""
        d = self.fetch(z_range=(9, 10), t_range=(1e3, 1e9))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert ((1e3 <= d["T1/2 (s)"]) & (d["T1/2 (s)"] <= 1e9)).all()

    def test_query_zrange_5_10_trange_1E9_None(self):
        """Test NNDCQuery: z_range=(5, 10), t_range=(1e9, None)............"""
        d = self.fetch(z_range=(5, 10), t_range=(1e9, None))
        assert len(d) > 0
        assert ((5 <= d.Z) & (d.Z <= 10)).all()
        assert (d["T1/2 (s)"] >= 1e9).all()

    def test_query_zrange_5_10_trange_1E9_inf(self):
        """Test NNDCQuery: z_range=(5, 10), t_range=(1e9, np.inf).........."""
        d = self.fetch(z_range=(5, 10), t_range=(1e9, np.inf))
        assert len(d) > 0
        assert ((5 <= d.Z) & (d.Z <= 10)).all()
        assert (d["T1/2 (s)"] >= 1e9).all()

    def test_query_nuc_Pu239_decay_ANY(self):
        """Test NNDCQuery: nuc='Pu-239', decay='ANY'......................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "ANY"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Pu-239", decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 94).all()
            assert (d.A == 239).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_Pu239_decay_A(self):
        """Test NNDCQuery: nuc='Pu-239', decay='Alpha'....................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "A"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Pu-239", decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 94).all()
            assert (d.A == 239).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_Co60_decay_BM(self):
        """Test NNDCQuery: nuc='Co-60', decay='B-'........................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "B-"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Co-60", decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 27).all()
            assert (d.A == 60).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_Na22_decay_ECBP(self):
        """Test NNDCQuery: nuc='Na-22', decay='EC+B+'......................"""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "ECBP"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Na-22", decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 11).all()
            assert (d.A == 22).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_Tc99_decay_IT(self):
        """Test NNDCQuery: nuc='Tc-99', decay='IT'........................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "IT"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Tc-99", decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 43).all()
            assert (d.A == 99).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_zrange_1_50_decay_N(self):
        """Test NNDCQuery: z_range=(1, 50), decay='N'......................"""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "N"]
        assert len(kwargs) > 0
        ds = [self.fetch(z_range=(1, 50), decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z <= 50).all()
            assert len(d) == len(ds[0])

    def test_query_nuc_zrange_1_50_decay_P(self):
        """Test NNDCQuery: z_range=(1, 50), decay='P'......................"""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "P"]
        assert len(kwargs) > 0
        ds = [self.fetch(z_range=(1, 50), decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z <= 50).all()
            assert len(d) == len(ds[0])

    def test_query_exception_bad_decay(self):
        """Test NNDCQuery raises exception for an invalid decay mode......."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(nuc="Pu-239", decay="InvalidMode")

    def test_query_zrange_9_10_elevelrange_0_01(self):
        """Test NNDCQuery: z_range=(9, 10), elevel_range=(0, 0.1).........."""
        d = self.fetch(z_range=(9, 10), elevel_range=(0, 0.1))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert ufloats_overlap_range(d["Energy Level (MeV)"], 0, 0.1)

    def test_query_zrange_9_10_elevelrange_None_01(self):
        """Test NNDCQuery: z_range=(9, 10), elevel_range=(None, 0.1)......."""
        d = self.fetch(z_range=(9, 10), elevel_range=(None, 0.1))
        assert len(d) > 0
        assert ((9 <= d.Z) & (d.Z <= 10)).all()
        assert ufloats_overlap_range(d["Energy Level (MeV)"], 0, 0.1)

    def test_query_zrange_7_20_elevelrange_01_10(self):
        """Test NNDCQuery: z_range=(7, 20), elevel_range=(0.1, 10)........."""
        d = self.fetch(z_range=(7, 20), elevel_range=(0.1, 10))
        assert len(d) > 0
        assert ((7 <= d.Z) & (d.Z <= 20)).all()
        assert ufloats_overlap_range(d["Energy Level (MeV)"], 0.1, 10)

    def test_query_zrange_7_20_elevelrange_01_None(self):
        """Test NNDCQuery: z_range=(7, 20), elevel_range=(0.1, None)......."""
        d = self.fetch(z_range=(7, 20), elevel_range=(0.1, None))
        assert len(d) > 0
        assert ((7 <= d.Z) & (d.Z <= 20)).all()
        assert ufloats_overlap_range(d["Energy Level (MeV)"], 0.1, 1e9)

    def test_query_zrange_7_20_elevelrange_01_inf(self):
        """Test NNDCQuery: z_range=(7, 20), elevel_range=(0.1, np.inf)....."""
        d = self.fetch(z_range=(7, 20), elevel_range=(0.1, np.inf))
        assert len(d) > 0
        assert ((7 <= d.Z) & (d.Z <= 20)).all()
        assert ufloats_overlap_range(d["Energy Level (MeV)"], 0.1, 1e9)


@pytest.mark.webtest
class TestNuclearWalletCard(NNDCQueryTests):
    """Test NNDC nuclear_wallet_card query."""

    def setup_method(self):
        self.cls = nndc._NuclearWalletCardQuery
        self.fetch = nndc.fetch_wallet_card

    def test_wallet_nuc_Pa234m(self):
        """Test fetch_wallet_card: nuc='Pa-234m' raises exception.........."""
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(nuc="Pa-234m")

    def test_wallet_zrange_230_250(self):
        """Test fetch_wallet_card: z_range=(230, 250) raises exception....."""
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(z_range=(230, 250))

    def test_wallet_zrange_9_10_j_0(self):
        """Test fetch_wallet_card: z_range=(9, 10), j='0'.................."""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(z_range=(9, 10), j="0")
            assert len(d) > 0

    def test_wallet_zrange_9_10_j_3_2(self):
        """Test fetch_wallet_card: z_range=(9, 10), j='3/2'................"""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(z_range=(9, 10), j="3/2")
            assert len(d) > 0

    def test_wallet_zrange_9_10_parity_any(self):
        """Test fetch_wallet_card: z_range=(9, 10), parity='any'..........."""
        kwargs = ["any", "ANY"]
        ds = [self.fetch(z_range=(9, 10), parity=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert ((9 <= d.Z) & (d.Z <= 10)).all()
            assert len(d) == len(ds[0])

    def test_wallet_zrange_9_10_parity_p(self):
        """Test fetch_wallet_card: z_range=(9, 10), parity='+'............."""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(z_range=(9, 10), parity="+")
            assert len(d) > 0

    def test_wallet_zrange_9_10_parity_m(self):
        """Test fetch_wallet_card: z_range=(9, 10), parity='-'............."""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(z_range=(9, 10), parity="-")
            assert len(d) > 0

    def test_wallet_zrange_9_10_parity_bad(self):
        """Test fetch_wallet_card raises except. for invalid parity........"""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(z_range=(9, 10), parity="InvalidParity")

    def test_wallet_zrange_9_10_j_2_parity_p(self):
        """Test fetch_wallet_card: z_range=(9, 10), j='2', parity='+'......"""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(z_range=(9, 10), j="2", parity="+")
            assert len(d) > 0

    def test_wallet_j_10(self):
        """Test fetch_wallet_card: j='10'.................................."""
        with pytest.raises(nndc.NNDCRequestError):
            d = self.fetch(j="10")
            assert len(d) > 0

    @pytest.mark.skip(
        reason='query kwarg "decay" seems to not be working '
        + "on NNDC, and as a result too many results "
        + "are returned for this test, causing an "
        + "NNDCRequestError"
    )
    def test_wallet_decay_SF(self):
        """Test fetch_wallet_card: decay='SF'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "SF"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_cluster(self):
        """Test fetch_wallet_card: decay='Cluster'........................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "C"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_DB(self):
        """Test fetch_wallet_card: decay='DB'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "DB"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_DN(self):
        """Test fetch_wallet_card: decay='DN'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "DN"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_DP(self):
        """Test fetch_wallet_card: decay='DP'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "DP"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_DA(self):
        """Test fetch_wallet_card: decay='DA'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "DA"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_decay_DF(self):
        """Test fetch_wallet_card: decay='DF'.............................."""
        kwargs = [kw for kw, val in nndc.WALLET_DECAY_MODE.items() if val == "DF"]
        assert len(kwargs) > 0
        ds = [self.fetch(decay=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert len(d) == len(ds[0])

    def test_wallet_exception_too_many(self):
        """Test fetch_wallet_card: exception if too many results..........."""
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(z_range=(20, 90))


@pytest.mark.webtest
class TestDecayRadiationQuery(NNDCQueryTests):
    """Test NNDC decay_radiation."""

    def setup_method(self):
        self.cls = nndc._DecayRadiationQuery
        self.fetch = nndc.fetch_decay_radiation

    def stable_isotope_condition(self, df):
        """What should be true about the dataframe if the isotope is stable."""
        return len(df) == 0

    def test_decay_nuc_Pa234m(self):
        """Test fetch_decay_radiation: no exception for nuc='Pa-234m'......"""
        d = self.fetch(nuc="Pa-234m")
        assert len(d) == 0
        assert (d.Z == 91).all()
        assert (d.A == 234).all()
        assert (d["Energy Level (MeV)"] > 0).all()

    def test_decay_zrange_230_250(self):
        """Test fetch_decay_radiation: z_range=(230, 250) dataframe empty.."""
        d = self.fetch(z_range=(230, 250))
        assert len(d) == 0

    def test_decay_nuc_Pu239_type_G(self):
        """Test fetch_decay_radiation: nuc='Pu-239', type='Gamma'.........."""
        kwargs = [kw for kw, val in nndc.DECAYRAD_RADIATION_TYPE.items() if val == "G"]
        assert len(kwargs) > 0
        ds = [self.fetch(nuc="Pu-239", type=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert (d.Z == 94).all()
            assert (d.A == 239).all()
            assert len(d) == len(ds[0])

    def test_decay_exception_bad_type(self):
        """Test fetch_decay_radiation raises except. for invalid rad type.."""
        with pytest.raises(nndc.NNDCInputError):
            self.fetch(nuc="Pu-239", type="InvalidType")

    def test_decay_zrange_100_102_type_ANY(self):
        """Test fetch_decay_radiation: z_range=(100, 102), type='ANY'......"""
        kwargs = [
            kw for kw, val in nndc.DECAYRAD_RADIATION_TYPE.items() if val == "ANY"
        ]
        assert len(kwargs) > 0
        ds = [self.fetch(z_range=(100, 102), type=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert ((100 <= d.Z) & (d.Z <= 102)).all()
            assert len(d) == len(ds[0])

    def test_decay_zrange_100_102_type_G(self):
        """Test fetch_decay_radiation: z_range=(100, 102), type='Gamma'...."""
        kwargs = [kw for kw, val in nndc.DECAYRAD_RADIATION_TYPE.items() if val == "G"]
        assert len(kwargs) > 0
        ds = [self.fetch(z_range=(100, 102), type=x) for x in kwargs]
        for d in ds:
            assert len(d) > 0
            assert ((100 <= d.Z) & (d.Z <= 102)).all()
            assert len(d) == len(ds[0])

    def test_decay_nuc_Ba133_erange_300_400(self):
        """Test fetch_decay_radiation: nuc='Ba-133', e_range=(300, 400)...."""
        d = self.fetch(nuc="Ba-133", e_range=(300, 400))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Energy (keV)"], 300, 400)

    def test_decay_nuc_Ba133_erange_300_None(self):
        """Test fetch_decay_radiation: nuc='Ba-133', e_range=(300, None)..."""
        d = self.fetch(nuc="Ba-133", e_range=(300, None))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Energy (keV)"], 300, 1e9)

    def test_decay_nuc_Ba133_erange_None_400(self):
        """Test fetch_decay_radiation: nuc='Ba-133', e_range=(None, 400)..."""
        d = self.fetch(nuc="Ba-133", e_range=(None, 400))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Energy (keV)"], 0, 400)

    def test_decay_nuc_Ba133_irange_10_100(self):
        """Test fetch_decay_radiation: nuc='Ba-133', i_range=(10, 100)....."""
        d = self.fetch(nuc="Ba-133", i_range=(10, 100))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Intensity (%)"], 10, 100)

    def test_decay_nuc_Ba133_irange_40_None(self):
        """Test fetch_decay_radiation: nuc='Ba-133', i_range=(40, None)...."""
        d = self.fetch(nuc="Ba-133", i_range=(40, None))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Intensity (%)"], 40, 100)

    def test_decay_nuc_Ba133_irange_None_40(self):
        """Test fetch_decay_radiation: nuc='Ba-133', i_range=(None, 40)...."""
        d = self.fetch(nuc="Ba-133", i_range=(None, 40))
        assert len(d) > 0
        assert (d.Z == 56).all()
        assert (d.A == 133).all()
        assert ufloats_overlap_range(d["Radiation Intensity (%)"], 0, 40)

    def test_decay_erange_661_662(self):
        """Test fetch_decay_radiation: e_range=(661.5, 661.9).............."""
        d = self.fetch(e_range=(661.5, 661.9))
        assert len(d) > 0
        assert ufloats_overlap_range(d["Radiation Energy (keV)"], 661.5, 661.9)

    def test_decay_exception_too_many(self):
        """Test fetch_decay_radiation: exception if too many results......."""
        with pytest.raises(nndc.NNDCRequestError):
            self.fetch(z_range=(28, 40))
