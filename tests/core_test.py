"""Test becquerel's Spectrum."""

from __future__ import print_function
import unittest
import numpy as np

import becquerel as bq

from parsers_test import SAMPLES

TEST_DATA_LENGTH = 256
TEST_COUNTS = 4
TEST_GAIN = 8.23
TEST_EDGES_KEV = np.arange(TEST_DATA_LENGTH + 1) * TEST_GAIN


class SpectrumFromFileTests(unittest.TestCase):
    """Test Spectrum.from_file() class method."""

    def run_from_file(self, extension):
        """Run the test of from_file() for files with the given extension."""
        filenames = SAMPLES.get(extension, [])
        self.assertTrue(len(filenames) >= 1)
        for filename in filenames:
            bq.core.Spectrum.from_file(filename)

    def test_spe(self):
        """Test Spectrum.from_file for SPE file........................."""
        self.run_from_file('.spe')

    def test_spc(self):
        """Test Spectrum.from_file for SPC file........................."""
        self.run_from_file('.spc')

    def test_cnf(self):
        """Test Spectrum.from_file for CNF file........................."""
        self.run_from_file('.cnf')


class SpectrumConstructorTests(unittest.TestCase):
    """Test Spectrum.__init__()."""

    def test_uncal(self):
        """Test simple uncalibrated construction."""

        spec = get_test_uncal_spectrum()
        self.assertEqual(len(spec.data), TEST_DATA_LENGTH)
        self.assertFalse(spec.is_calibrated)

    def test_cal(self):
        """Test simple calibrated construction."""

        spec = get_test_cal_spectrum()
        self.assertEqual(len(spec.data), TEST_DATA_LENGTH)
        self.assertEqual(len(spec.bin_edges_kev), TEST_DATA_LENGTH + 1)
        self.assertEqual(len(spec.energies_kev), TEST_DATA_LENGTH)
        self.assertTrue(spec.is_calibrated)

    def test_init_exceptions(self):
        """Test errors on initialization."""

        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum([])
        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum(
                get_test_data(), bin_edges_kev=TEST_EDGES_KEV[:-1])

        bad_edges = TEST_EDGES_KEV.copy()
        bad_edges[12] = bad_edges[9]
        with self.assertRaises(bq.core.SpectrumError):
            bq.core.Spectrum(get_test_data(), bin_edges_kev=bad_edges)

    def test_uncalibrated_exception(self):
        """Test UncalibratedError."""

        spec = get_test_uncal_spectrum()
        with self.assertRaises(bq.core.UncalibratedError):
            spec.energies_kev


class SpectrumAddSubtractTests(unittest.TestCase):
    """Test addition and subtraction of spectra"""

    def test_uncal_add_sub(self):
        """Test basic addition/subtraction of uncalibrated spectra"""

        spec1 = get_test_uncal_spectrum()
        spec2 = get_test_uncal_spectrum()
        tot = spec1 + spec2
        self.assertTrue(np.all(tot.data == spec1.data + spec2.data))
        diff = spec1 - spec2
        self.assertTrue(np.all(diff.data == spec1.data - spec2.data))

    def test_cal_uncal_add_sub(self):
        """Test basic addition of a calibrated with an uncalibrated spectrum.

        NOTE: not implemented yet - so check that it errors.
        """

        spec1 = get_test_uncal_spectrum()
        spec2 = get_test_cal_spectrum()
        with self.assertRaises(NotImplementedError):
            spec1 + spec2
        with self.assertRaises(NotImplementedError):
            spec1 - spec2

    def test_cal_add_sub(self):
        """Test basic addition of calibrated spectra.

        NOTE: not implemented yet - so check that it errors.
        """

        spec1 = get_test_cal_spectrum()
        spec2 = get_test_cal_spectrum()
        with self.assertRaises(NotImplementedError):
            spec1 + spec2
        with self.assertRaises(NotImplementedError):
            spec1 - spec2

    def test_add_sub_type_error(self):
        """Check that adding/subtracting a non-Spectrum gives a TypeError."""

        spec1 = get_test_uncal_spectrum()
        with self.assertRaises(TypeError):
            spec1 + 5
        with self.assertRaises(TypeError):
            spec1 - 5
        with self.assertRaises(TypeError):
            spec1 + 'asdf'
        with self.assertRaises(TypeError):
            spec1 - 'asdf'
        with self.assertRaises(TypeError):
            spec1 + get_test_data()
        with self.assertRaises(TypeError):
            spec1 - get_test_data()

    def test_add_sub_wrong_length(self):
        """
        Adding/subtracting spectra of different lengths gives a SpectrumError.
        """

        spec1 = get_test_uncal_spectrum()
        spec2 = bq.core.Spectrum(get_test_data(length=TEST_DATA_LENGTH * 2))
        with self.assertRaises(bq.core.SpectrumError):
            spec1 + spec2
        with self.assertRaises(bq.core.SpectrumError):
            spec1 - spec2


class SpectrumMultiplyDivideTests(unittest.TestCase):
    """Test multiplication and division of spectra"""

    def test_uncal_mul_div(self):
        """
        Basic multiplication/division of uncalibrated spectrum by a scalar.
        """

        spec = get_test_uncal_spectrum()
        doubled = spec * 2
        self.assertTrue(np.all(doubled.data == 2 * spec.data))
        halved = spec / 2
        self.assertTrue(np.all(halved.data == spec.data / 2.0))

    def test_cal_mul_div(self):
        """Basic multiplication/division of calibrated spectrum by a scalar."""

        spec = get_test_cal_spectrum()
        doubled = spec * 2
        self.assertTrue(np.all(doubled.data == 2 * spec.data))
        halved = spec / 2
        self.assertTrue(np.all(halved.data == spec.data / 2.0))
        halved_again = spec * 0.5
        self.assertTrue(np.all(halved_again.data == spec.data * 0.5))

    def test_mul_div_type_error(self):
        """Multiplication/division with a non-scalar gives a TypeError."""

        spec = get_test_uncal_spectrum()

        with self.assertRaises(TypeError):
            spec * spec
        with self.assertRaises(TypeError):
            spec / spec
        with self.assertRaises(TypeError):
            spec * 'asdf'
        with self.assertRaises(TypeError):
            spec / 'asdf'
        with self.assertRaises(TypeError):
            spec * get_test_data()
        with self.assertRaises(TypeError):
            spec / get_test_data()

    def test_mul_div_bad_factor(self):
        """Multiplication/division with zero/inf/nan gives a SpectrumError."""

        spec = get_test_uncal_spectrum()

        with self.assertRaises(bq.core.SpectrumError):
            spec * 0
        with self.assertRaises(bq.core.SpectrumError):
            spec / 0
        with self.assertRaises(bq.core.SpectrumError):
            spec * np.inf
        with self.assertRaises(bq.core.SpectrumError):
            spec / np.inf
        with self.assertRaises(bq.core.SpectrumError):
            spec * np.nan
        with self.assertRaises(bq.core.SpectrumError):
            spec / np.nan


class PeaksTests(unittest.TestCase):
    """Test core.peaks"""

    def test_01(self):
        """Test peaks.ArbitraryEnergyPoint"""

        ch = 2345
        kev = 661.66
        energy_pt = bq.core.peaks.ArbitraryEnergyPoint(ch, kev)
        self.assertEqual(energy_pt.energy_ch, ch)
        self.assertEqual(energy_pt.cal_energy_kev, kev)

    def test_02(self):
        """Test peaks.ArbitraryEfficiencyPoint"""

        counts = 1000
        emissions = 65432
        kev = 661.66
        eff_pt = bq.core.peaks.ArbitraryEfficiencyPoint(
            counts, emissions, energy_kev=kev)
        self.assertEqual(eff_pt.area_c, counts)
        self.assertEqual(eff_pt.cal_area, emissions)

    def test_03(self):
        """Test peaks.GrossROIPeak construction and basic properties"""

        spec = get_test_uncal_spectrum()
        roi = (32, 48)
        pk = bq.core.peaks.GrossROIPeak(spec, roi)
        self.assertIs(pk.spectrum, spec)
        self.assertTrue(np.all(np.array(pk.ROI_bounds_ch) == np.array(roi)))

    def test_04(self):
        """Test error on bad ROI_bounds"""

        spec = get_test_uncal_spectrum()
        roi = (125,)
        with self.assertRaises(ValueError):
            bq.core.peaks.GrossROIPeak(spec, roi)
        roi = (125, 133, 139)
        with self.assertRaises(ValueError):
            bq.core.peaks.GrossROIPeak(spec, roi)
        roi = 'string'
        with self.assertRaises(ValueError):
            bq.core.peaks.GrossROIPeak(spec, roi)


class EnergyCalTests(unittest.TestCase):
    """Test core.energycal"""

    def test_simple_cal(self):
        """Test energycal.SimplePolyCal"""

        cal = bq.core.energycal.SimplePolyCal(coeffs=(1, 0.37))
        self.assertEqual(cal.ch2kev(100), 38)

    def test_fit_poly_cal(self):
        """Test energycal.FitPolyCal"""

        pts = []
        pts.append(bq.core.peaks.ArbitraryEnergyPoint(32, 661.66))
        pts.append(bq.core.peaks.ArbitraryEnergyPoint(88, 1460.83))
        cal = bq.core.energycal.FitPolyCal(peaks_list=pts, order=1)
        self.assertTrue(np.all(
            np.isclose(cal.ch2kev([32, 88]), [661.66, 1460.83])))

    def test_add_rm_peaks(self):
        """Test features of energycal.FitEnergyCalBase"""

        pts = []
        pts.append(bq.core.peaks.ArbitraryEnergyPoint(32, 661.66))
        pts.append(bq.core.peaks.ArbitraryEnergyPoint(88, 1460.83))
        cal = bq.core.energycal.FitPolyCal(peaks_list=pts, order=1)
        new_pt = bq.core.peaks.ArbitraryEnergyPoint(127, 2614)
        cal.add_peak(new_pt)
        cal.rm_peak(pts[0])


def get_test_data(length=TEST_DATA_LENGTH, expectation_val=TEST_COUNTS):
    """Build a vector of random counts."""
    return np.random.poisson(lam=expectation_val, size=length).astype(np.int)


def get_test_uncal_spectrum():
    uncal = bq.core.Spectrum(get_test_data())
    return uncal


def get_test_cal_spectrum():
    cal = bq.core.Spectrum(get_test_data(), bin_edges_kev=TEST_EDGES_KEV)
    return cal


def main():
    """Run unit tests."""
    unittest.main()


if __name__ == '__main__':
    main()
