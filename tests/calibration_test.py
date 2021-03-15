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


@pytest.mark.parametrize(
    "name, cls, args",
    [
        ["cal1", Calibration, ("p[0]", [1.0])],
        ["cal2", Calibration, ("p[0] + p[1] * x", [1.0, 5.0])],
        ["cal3", Calibration, ("sqrt(p[0] + p[1] * x)", [1.0, 5.0])],
        ["lin", LinearCalibration, ([2.0, 3.0],)],
        ["poly2", PolynomialCalibration, ([2.0, 3.0],)],
        ["poly3", PolynomialCalibration, ([2.0, 3.0, 7.0],)],
        ["poly4", PolynomialCalibration, ([2.0, 3.0, 7.0, 5.0],)],
        ["sqrt4", SqrtPolynomialCalibration, ([2.0, 3.0, 7.0, 5.0],)],
    ],
)
def test_calibration(name, cls, args):
    """Test the Calibration class."""
    fname = os.path.join(TEST_IO, f"__test_calibration_{name}.h5")
    cal = cls(*args, comment="Test of class " + cls.__name__)
    print("attrs (test):", cal.attrs)
    cal.write(fname)
    cal2 = Calibration.read(fname)
    assert cal2 == cal
    cal3 = cal.copy()
    assert cal3 == cal

