"""Test Spectrum I/O for different file types."""

import os
from pathlib import Path

import numpy as np
import pytest
from h5_tools_test import TEST_OUTPUTS
from parsers_test import SAMPLES
from spectrum_test import make_spec

import becquerel as bq


@pytest.mark.parametrize("extension", SAMPLES.keys())
@pytest.mark.parametrize("path_type", [str, Path])
def test_spectrum_from_file(extension, path_type):
    """Test Spectrum.from_file for the given extension and path_type."""
    filenames = SAMPLES[extension]
    assert len(filenames) >= 1
    for filename in filenames:
        spec = bq.Spectrum.from_file(path_type(filename))
        assert spec.livetime is not None


def test_spectrum_from_file_raises():
    """Test Spectrum.from_file raises error for an unsupported file type."""
    with pytest.raises(NotImplementedError):
        bq.Spectrum.from_file("foo.bar")


@pytest.mark.parametrize(
    "kind",
    [
        "uncal",
        "cal",
        "cal_new",
        "applied_energy_cal",
        "cal_cps",
        "uncal_long",
        "uncal_cps",
    ],
)
def test_write_h5(kind):
    """Test writing different Spectrums to HDF5 files."""
    spec = make_spec(kind, lt=600.0)
    fname = os.path.join(TEST_OUTPUTS, "spectrum_io__test_write_h5__" + kind + ".h5")
    spec.write(fname)


@pytest.mark.parametrize(
    "kind",
    [
        "uncal",
        "cal",
        "cal_new",
        "applied_energy_cal",
        "cal_cps",
        "uncal_long",
        "uncal_cps",
    ],
)
def test_from_file_h5(kind):
    """Test Spectrum.from_file works for HDF5 files."""
    fname = os.path.join(TEST_OUTPUTS, "spectrum_io__test_write_h5__" + kind + ".h5")
    spec = bq.Spectrum.from_file(fname)
    assert spec.livetime is not None
    if kind == "applied_energy_cal":
        assert spec.is_calibrated and spec.energy_cal is not None


@pytest.mark.parametrize("extension", SAMPLES.keys())
def test_spectrum_samples_write_read_h5(extension):
    """Test Spectrum HDF5 I/O using sample files."""
    filenames = SAMPLES[extension]
    assert len(filenames) >= 1
    for filename in filenames:
        spec = bq.Spectrum.from_file(filename)
        fname2 = os.path.splitext(filename)[0] + ".h5"
        fname2 = os.path.join(
            TEST_OUTPUTS, "spectrum_io__sample_write_h5__" + os.path.split(fname2)[1]
        )
        spec.write(fname2)
        spec = bq.Spectrum.from_file(fname2)
        assert spec.livetime is not None


def test_from_file_cal_kwargs():
    """Test Spectrum.from_file overrides calibration with cal_kwargs."""
    fname = os.path.join(
        TEST_OUTPUTS, "spectrum_io__test_write_h5__applied_energy_cal.h5"
    )
    domain = [-100, 10000]
    rng = [-10, 1000]
    params = [0.6]
    # load without the calibration override
    spec = bq.Spectrum.from_file(fname)
    assert not np.allclose(spec.energy_cal.domain, domain)
    assert not np.allclose(spec.energy_cal.rng, rng)
    assert not np.allclose(spec.energy_cal.params, params)
    # load with the calibration override
    spec = bq.Spectrum.from_file(
        fname, cal_kwargs={"domain": domain, "rng": rng, "params": params}
    )
    assert np.allclose(spec.energy_cal.domain, domain)
    assert np.allclose(spec.energy_cal.rng, rng)
    assert np.allclose(spec.energy_cal.params, params)


@pytest.mark.parametrize("extension", SAMPLES.keys())
def test_spectrum_from_file_cal_kwargs(extension):
    """Test Spectrum.from_file with calibration overrides."""
    filenames = SAMPLES[extension]
    assert len(filenames) >= 1
    for filename in filenames:
        domain = [-1e7, 1e7]
        rng = [-1e7, 1e7]
        # load without the calibration override
        spec = bq.Spectrum.from_file(filename)
        if spec.energy_cal is None:
            continue
        assert not np.allclose(spec.energy_cal.domain, domain)
        assert not np.allclose(spec.energy_cal.rng, rng)
        # load with the calibration override
        spec = bq.Spectrum.from_file(
            filename, cal_kwargs={"domain": domain, "rng": rng}
        )
        assert np.allclose(spec.energy_cal.domain, domain)
        assert np.allclose(spec.energy_cal.rng, rng)
