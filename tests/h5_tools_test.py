"""Test HDF5 I/O tools."""

import os
import h5py
import numpy as np
import pytest
from becquerel.io.h5 import ensure_string, is_h5_filename, open_h5, write_h5, read_h5


TEST_OUTPUTS = os.path.join(os.path.split(__file__)[0], "test_outputs")
if not os.path.exists(TEST_OUTPUTS):
    os.mkdir(TEST_OUTPUTS)

DSETS = {
    "dset_1d": np.ones(100, dtype=int),
    "dset_2d": np.ones((30, 30), dtype=float),
    "dset_0d": 7,
    "dset_str": "test",
}

ATTRS = {
    "str": "testing",
    "float": 1.3,
    "list": [1, 2, 3],
    "ndarray": np.ones(10),
}


def test_ensure_string():
    """Test the ensure_string function."""
    assert "abc" == ensure_string("abc")
    assert "abc" == ensure_string(b"abc")
    with pytest.raises(TypeError):
        ensure_string(None)


def test_is_h5_filename():
    """Test functionality of is_h5_filename."""
    assert is_h5_filename("test.h5")
    assert is_h5_filename("test.hdf5")
    assert is_h5_filename("test.H5")
    assert is_h5_filename("test.HDF5")
    assert not is_h5_filename("test.csv")
    assert not is_h5_filename("test.spe")
    assert not is_h5_filename(None)


def write_test_open_h5_file(fname):
    """Write the test file for test_open_h5."""
    with h5py.File(fname, "w") as file:
        file.create_dataset("test_dset", data=[1, 2, 3])
        group = file.create_group("test_group")
        group.create_dataset("group_dset", data=[1, 2, 3])


def test_open_h5():
    """Test open_h5 for different inputs."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_open_h5.h5")

    # filename cases
    write_test_open_h5_file(fname)
    with open_h5(fname, "r") as f:
        print(list(f.keys()))

    write_test_open_h5_file(fname)
    with open_h5(fname, "w") as f:
        print(list(f.keys()))
        f.create_dataset("test_dset_1", data=[1, 2, 3])

    write_test_open_h5_file(fname)
    with open_h5(fname, "r+") as f:
        print(list(f.keys()))
        f.create_dataset("test_dset_2", data=[1, 2, 3])

    # h5py.File cases
    write_test_open_h5_file(fname)
    with h5py.File(fname, "r") as file:
        print(list(file.keys()))
        with open_h5(file) as f:
            print(list(f.keys()))
            with pytest.raises(ValueError):
                f.create_dataset("test_dset_3", data=[1, 2, 3])

    write_test_open_h5_file(fname)
    with h5py.File(fname, "r+") as file:
        print(list(file.keys()))
        with open_h5(file) as f:
            print(list(f.keys()))
            f.create_dataset("test_dset_4", data=[1, 2, 3])

    write_test_open_h5_file(fname)
    with h5py.File(fname, "w") as file:
        print(list(file.keys()))
        with open_h5(file) as f:
            print(list(f.keys()))
            f.create_dataset("test_dset_5", data=[1, 2, 3])

    # h5py.Group cases
    write_test_open_h5_file(fname)
    with h5py.File(fname, "r") as file:
        print(list(file.keys()))
        group = file["test_group"]
        with open_h5(group) as f:
            print(list(f.keys()))
            with pytest.raises(ValueError):
                f.create_dataset("group_dset_1", data=[1, 2, 3])

    write_test_open_h5_file(fname)
    with h5py.File(fname, "r+") as file:
        print(list(file.keys()))
        group = file["test_group"]
        with open_h5(group) as f:
            print(list(f.keys()))
            f.create_dataset("group_dset_2", data=[1, 2, 3])

    write_test_open_h5_file(fname)
    with h5py.File(fname, "w") as file:
        print(list(file.keys()))
        group = file.create_group("test_group")
        with open_h5(group) as f:
            print(list(f.keys()))
            f.create_dataset("group_dset_3", data=[1, 2, 3])


def check_dsets_attrs(dsets1, attrs1, dsets2, attrs2):
    """Check that the dataset and attribute dicts are identical."""
    assert set(dsets1.keys()) == set(dsets2.keys())
    assert set(attrs1.keys()) == set(attrs2.keys())
    for key in dsets1.keys():
        if "str" in key:
            assert ensure_string(dsets1[key]) == ensure_string(dsets2[key])
        else:
            assert np.allclose(dsets1[key], dsets2[key])
    for key in attrs1.keys():
        if "str" in key:
            assert ensure_string(attrs1[key]) == ensure_string(attrs2[key])
        else:
            assert np.allclose(attrs1[key], attrs2[key])


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_write_h5_filename(dsets, attrs):
    """Write data to h5 given its filename."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_filename.h5")
    write_h5(fname, dsets, attrs)


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_write_h5_file(dsets, attrs):
    """Write data to h5 given an open h5py.File."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_file.h5")
    with h5py.File(fname, "w") as file:
        write_h5(file, dsets, attrs)


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_write_h5_group(dsets, attrs):
    """Write data to h5 given an h5py.Group."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_group.h5")
    with h5py.File(fname, "w") as file:
        group = file.create_group("test_group")
        write_h5(group, dsets, attrs)


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_read_h5_filename(dsets, attrs):
    """Read data from h5 given its filename."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_filename.h5")
    dsets2, attrs2, skipped = read_h5(fname)
    check_dsets_attrs(dsets, attrs, dsets2, attrs2)
    assert len(skipped) == 0


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_read_h5_file(dsets, attrs):
    """Read data from h5 given an open h5py.File."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_file.h5")
    with h5py.File(fname, "r") as file:
        dsets2, attrs2, skipped = read_h5(file)
    check_dsets_attrs(dsets, attrs, dsets2, attrs2)
    assert len(skipped) == 0


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_read_h5_group(dsets, attrs):
    """Read data from h5 given an h5py.Group."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__test_write_h5_group.h5")
    with h5py.File(fname, "r") as file:
        group = file["test_group"]
        dsets2, attrs2, skipped = read_h5(group)
    check_dsets_attrs(dsets, attrs, dsets2, attrs2)
    assert len(skipped) == 0


@pytest.mark.parametrize("dsets", [DSETS])
@pytest.mark.parametrize("attrs", [ATTRS])
def test_read_h5_ignore_group(dsets, attrs):
    """Read data from h5 and ignore a data group within the group."""
    fname = os.path.join(TEST_OUTPUTS, "io_h5__write_h5_ignore_group.h5")

    # write the file with an extra group
    with h5py.File(fname, "w") as file:
        file.create_group("test_group")
        write_h5(file, dsets, attrs)

    # read the file, ignoring the group
    with h5py.File(fname, "r") as file:
        print("file:", file)
        dsets2, attrs2, skipped = read_h5(file)
    check_dsets_attrs(dsets, attrs, dsets2, attrs2)
    assert skipped == ["test_group"]
