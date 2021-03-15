"""Test Calibration class."""

import os
from becquerel.core.calibration import (
    CalibrationError,
    Calibration,
    LinearCalibration,
    PolynomialCalibration,
    SqrtPolynomialCalibration,
)
import pytest
from h5_tools_test import TEST_IO


name_cls_args = [
    ["cal1", Calibration, ("p[0]", [1.0])],
    ["cal2", Calibration, ("p[0] + p[1] * x", [1.0, 5.0])],
    ["cal3", Calibration, ("sqrt(p[0] + p[1] * x)", [1.0, 5.0])],
    ["lin", LinearCalibration, ([2.0, 3.0],)],
    ["poly2", PolynomialCalibration, ([2.0, 3.0],)],
    ["poly3", PolynomialCalibration, ([2.0, 3.0, 7.0],)],
    ["poly4", PolynomialCalibration, ([2.0, 3.0, 7.0, 5.0],)],
    ["sqrt4", SqrtPolynomialCalibration, ([2.0, 3.0, 7.0, 5.0],)],
]


@pytest.mark.parametrize("name, cls, args", name_cls_args)
def test_calibration(name, cls, args):
    """Test the Calibration class."""
    fname = os.path.join(TEST_IO, f"__test_calibration_{name}.h5")
    # test __init__()
    cal = cls(*args, comment="Test of class " + cls.__name__)
    print("attrs (test):", cal.attrs)
    # test write()
    cal.write(fname)
    # test read()
    cal2 = Calibration.read(fname)
    # test __eq__()
    assert cal2 == cal
    # test copy()
    cal3 = cal.copy()
    assert cal3 == cal
    # test __call__()
    cal(1.0)


@pytest.mark.parametrize("name, cls, args", name_cls_args)
def test_calibration_set_add_points(name, cls, args):
    """Test Calibration.set_points and add_points methods."""
    fname = os.path.join(TEST_IO, f"__test_calibration_{name}_points.h5")
    cal = cls(*args, comment="Test of class " + cls.__name__)
    # test set_points
    cal.set_points()
    cal.set_points((0, 100), (0, 100))
    cal.set_points([], [])
    # test add_points
    for px, py in [[(), ()], [(0, 100), (0, 100)]]:
        cal.add_points(px, py)
    # test write()
    cal.write(fname)
    # test read()
    cal2 = Calibration.read(fname)
    # test __eq__()
    assert cal2 == cal
