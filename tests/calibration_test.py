"""Test Calibration class."""

import os
import matplotlib.pyplot as plt
import numpy as np
from becquerel.io import h5
from becquerel.core.calibration import (
    _eval_expression,
    _param_indices,
    _validate_expression,
    _fit_expression,
    CalibrationError,
    Calibration,
    LinearCalibration,
    PolynomialCalibration,
    SqrtPolynomialCalibration,
    InterpolatedCalibration,
)
import pytest
from h5_tools_test import TEST_OUTPUTS


def test_eval_expression():
    """Test calibration._eval_expression."""
    _eval_expression("p[0] + p[1] * x", [1.0, 5.0], 2.0)
    # no parameters
    _eval_expression("1.0 + 5.0 * x", [], 2.0)
    # bad syntax
    with pytest.raises(CalibrationError):
        _eval_expression("p[0] + p[1] x", [1.0, 5.0], 2.0)
    # unknown symbol
    with pytest.raises(CalibrationError):
        _eval_expression("p[0] + p[1] * x + z", [1.0, 5.0], 2.0)
    # unknown function
    with pytest.raises(CalibrationError):
        _eval_expression("p[0] + p[1] * f(x, x)", [1.0, 5.0], 2.0)
    # negative argument
    with pytest.raises(CalibrationError):
        _eval_expression("p[0] + p[1] * x", [1.0, 5.0], -2.0)
    # result is a complex number
    with pytest.raises(CalibrationError):
        _eval_expression("p[0] + p[1] * x", [1j, 5.0], 2.0)


def test_expression_param_indices():
    """Test calibration._param_indices."""
    assert np.allclose(_param_indices("p[0] + p[1] * x"), [0, 1])
    # does not result in an error when calling this function
    assert np.allclose(_param_indices("p[0] + p[2] * x"), [0, 2])
    # neither does this, but notice the order!
    assert np.allclose(_param_indices("p[0] + p[-1] * x"), [-1, 0])
    # having no parameters is supported
    assert len(_param_indices("1.0 + 5.0 * x")) == 0
    # error if indices are not integers
    with pytest.raises(ValueError):
        _param_indices("p[0.2] + p[1] * x")
    with pytest.raises(ValueError):
        _param_indices("p[:] + p[1] * x")
    with pytest.raises(ValueError):
        _param_indices("p[] + p[1] * x")
    with pytest.raises(ValueError):
        _param_indices("p[a] + p[1] * x")


def test_validate_expression():
    """Test calibration._validate_expression."""
    # python syntax error
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1] None")
    # parentheses not matching
    with pytest.raises(CalibrationError):
        _validate_expression("(p[0] + p[1] * x")
    # parentheses not matching
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1] * x]")
    # "x" must appear in the formula
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1]")
    # having no parameters is supported
    assert _validate_expression("1.0 + 5.0 * x")
    assert _validate_expression("1.0 + 5.0 * x", params=[])
    # square brackets after "p" must only enclose integers
    with pytest.raises(CalibrationError):
        _validate_expression("p[0.2] + p[1] * x")
    with pytest.raises(CalibrationError):
        _validate_expression("p[:] + p[1] * x")
    with pytest.raises(CalibrationError):
        _validate_expression("p[] + p[1] * x")
    with pytest.raises(CalibrationError):
        _validate_expression("p[a] + p[1] * x")
    # minimum parameter index is > 0
    with pytest.raises(CalibrationError):
        _validate_expression("p[1] + p[2] * x")
    # parameter indices not consecutive
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[2] * x")
    # mismtach between number of parameters in expression and length of params
    _validate_expression("p[0] + p[1] * x")
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1] * x", params=[1.0])
    _validate_expression("p[0] + p[1] * x", params=[1.0, 5.0])
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1] * x", params=[1.0, 5.0, 2.0])
    # expression is okay except for an unknown function
    with pytest.raises(CalibrationError):
        _validate_expression("p[0] + p[1] * f(x, x)", [1.0, 5.0])
    # expression looks okay until it is evaluated on a float
    with pytest.raises(CalibrationError):
        _validate_expression("np.sqrt(p[0] + p[1] * x)", [1j, 5.0])
    # expression looks okay until it is evaluated on an array
    with pytest.raises(CalibrationError):
        _validate_expression("sqrt(p[0] + p[1] * x)", [1.0, 5.0])


def test_fit_expression():
    """Test calibration._fit_expression."""
    # this case should work
    _fit_expression("p[0] + p[1] * x", [0, 1000], [0, 1000])
    # provide initial guesses
    _fit_expression("p[0] + p[1] * x", [0, 1000], [0, 1000], params0=[0.0, 1.0])
    # bad number of guesses
    with pytest.raises(CalibrationError):
        _fit_expression(
            "p[0] + p[1] * x", [0, 1000], [0, 1000], params0=[0.0, 1.0, 2.0]
        )
    # not enough points
    with pytest.raises(CalibrationError):
        _fit_expression("p[0] + p[1] * x", [1000], [1000], params0=[0.0, 1.0])
    # fit returns success=False (not allowed sufficient # of evaluations)
    with pytest.raises(CalibrationError):
        _fit_expression(
            "p[0] + p[1] * x",
            [0, 500, 1000],
            [0, 500, 1000],
            params0=[1.0, 1.0],
            max_nfev=1,
            ftol=1e-15,
            gtol=1e-15,
            xtol=1e-15,
        )


name_cls_args = [
    ["cal1", Calibration, ("p[0] * x", [5.0])],
    ["cal2", Calibration, ("p[0] + p[1] * x", [1.0, 5.0])],
    [
        "cal3",
        Calibration,
        ("np.sqrt(p[0] + p[1] * x + p[2] * x ** 2)", [2.0, 1.0, 1.0e-2]),
    ],
    ["cal4", Calibration, ("p[0] + p[1] * np.exp(x / p[2])", [1.0, 5.0, 1000.0])],
    ["lin", LinearCalibration, ([2.0, 3.0],)],
    ["poly1", PolynomialCalibration, ([2.0, 1.0],)],
    ["poly2", PolynomialCalibration, ([2.0, 1.0, 1.0e-2],)],
    ["sqrt2", SqrtPolynomialCalibration, ([2.0, 1.0, 1.0e-2],)],
    ["interp", InterpolatedCalibration, ()],
]


@pytest.mark.parametrize("name, cls, args", name_cls_args)
def test_calibration(name, cls, args):
    """Test the Calibration class."""
    fname = os.path.join(TEST_OUTPUTS, f"calibration__init__{name}.h5")
    # test __init__()
    cal = cls(*args, comment="Test of class " + cls.__name__)
    # test protections on setting parameters
    with pytest.raises(CalibrationError):
        cal.params = None
    with pytest.raises(CalibrationError):
        cal.params = np.ones((1, 2))
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
    str(cal)
    repr(cal)


@pytest.mark.parametrize("name, cls, args", name_cls_args)
def test_calibration_set_add_points(name, cls, args):
    """Test Calibration.set_points and add_points methods."""
    fname = os.path.join(TEST_OUTPUTS, f"calibration__add_points__{name}.h5")
    cal = cls(*args, comment="Test of class " + cls.__name__)
    # test set_points
    cal.set_points()
    cal.set_points(1000, 1000)
    cal.set_points((0, 1000), (0, 1000))
    cal.set_points([], [])
    # test add_points
    cal.add_points()  # does nothing
    for px, py in [[(), ()], [1000, 1000], [(0, 1000), (0, 1000)]]:
        cal.add_points(px, py)
    # test __str__() and __repr__()
    str(cal)
    repr(cal)
    # test write()
    cal.write(fname)
    # test read()
    cal2 = Calibration.read(fname)
    # test __eq__()
    assert cal2 == cal
    # points_x is not 1D
    with pytest.raises(CalibrationError):
        points_1d = np.array([0, 1000, 2000])
        points_2d = np.reshape(points_1d, (1, 3))
        cal.add_points(points_2d, points_1d)
    # points_y is not 1D
    with pytest.raises(CalibrationError):
        points_1d = np.array([0, 1000, 2000])
        points_2d = np.reshape(points_1d, (1, 3))
        cal.add_points(points_1d, points_2d)
    # points have different lengths
    with pytest.raises(CalibrationError):
        points_1d = np.array([0, 1000, 2000])
        points_2d = np.reshape(points_1d, (1, 3))
        cal.add_points([0, 1000, 2000], [0, 2000])
    # points_x contains negative values
    with pytest.raises(CalibrationError):
        cal.add_points([0, -2000], [0, 2000])
    # points_y contains negative values
    with pytest.raises(CalibrationError):
        cal.add_points([0, 2000], [0, -2000])


@pytest.mark.parametrize("name, cls, args", name_cls_args)
def test_calibration_fit_from_points(name, cls, args):
    """Test Calibration.fit and from_points methods."""
    points_x = [100, 500, 1000, 1500, 2500]
    points_y = [18, 42, 63, 82, 117]
    # test fit()
    cal1 = cls(*args, comment="Test of class " + cls.__name__)
    cal1.add_points(points_x, points_y)
    cal1.fit()
    # test from_points()
    if cls == Calibration:
        cal2 = cls.from_points(args[0], points_x, points_y, args[1])
        cal3 = cls.from_points(
            args[0], points_x, points_y, args[1], include_origin=True
        )
    elif cls == InterpolatedCalibration:
        cal2 = cls.from_points(points_x, points_y)
        cal3 = cls.from_points(points_x, points_y, include_origin=True)
    else:
        cal2 = cls.from_points(points_x, points_y, args[0])
        cal3 = cls.from_points(points_x, points_y, args[0], include_origin=True)
    assert cal2 == cal1

    plt.figure()
    if cls == InterpolatedCalibration:
        plt.title(cls.__name__)
    else:
        plt.title(cal1.expression)
    x_fine1 = np.linspace(min(points_x), max(points_x), num=500)
    x_fine3 = np.linspace(0, max(points_x), num=500)
    plt.plot(
        x_fine1,
        cal1(x_fine1),
        "b-",
        lw=2,
        alpha=0.5,
        label="fitted function (include_origin=False)",
    )
    plt.plot(
        x_fine3,
        cal3(x_fine3),
        "g-",
        alpha=0.5,
        label="fitted function (include_origin=True)",
    )
    plt.plot(points_x, points_y, "ro", label="calibration points")
    plt.xlabel("x")
    plt.xlabel("y")
    plt.xlim(0)
    plt.ylim(0)
    plt.legend()
    plt.savefig(os.path.join(TEST_OUTPUTS, f"calibration__fit__{name}.png"))


def test_calibration_misc():
    """Miscellaneous tests to increase test coverage."""
    cal1 = LinearCalibration([2.0, 3.0])
    cal2 = PolynomialCalibration([2.0, 3.0, 7.0])
    with pytest.raises(CalibrationError):
        cal1 != 0
    assert cal1 != cal2

    # bad number of arguments
    with pytest.raises(CalibrationError):
        LinearCalibration([2.0])
    LinearCalibration([2.0, 3.0])
    with pytest.raises(CalibrationError):
        LinearCalibration([2.0, 3.0, 4.0])

    # bad number of arguments
    with pytest.raises(CalibrationError):
        PolynomialCalibration([2.0])
    PolynomialCalibration([2.0, 3.0])
    PolynomialCalibration([2.0, 3.0, 4.0])

    # bad number of arguments
    with pytest.raises(CalibrationError):
        SqrtPolynomialCalibration([2.0])
    SqrtPolynomialCalibration([2.0, 3.0])
    SqrtPolynomialCalibration([2.0, 3.0, 4.0])


def test_calibration_read_failures():
    """Test miscellaneous HDF5 reading failures."""
    fname = os.path.join(TEST_OUTPUTS, "calibration__read_failures.h5")
    cal = LinearCalibration([2.0, 3.0])
    cal.add_points([0, 1000, 2000], [0, 1000, 2000])

    # remove the params from the file
    cal.write(fname)
    with h5.open_h5(fname, "r+") as f:
        del f["params"]
    with pytest.raises(CalibrationError):
        Calibration.read(fname)

    # remove the expression from the file
    cal.write(fname)
    with h5.open_h5(fname, "r+") as f:
        del f["expression"]
    with pytest.raises(CalibrationError):
        Calibration.read(fname)

    # remove points_x from the file
    cal.write(fname)
    with h5.open_h5(fname, "r+") as f:
        del f["points_x"]
    Calibration.read(fname)

    # add unexpected dataset to the file
    cal.write(fname)
    with h5.open_h5(fname, "r+") as f:
        f.create_dataset("unexpected", data=[0, 1, 2])
    with pytest.raises(CalibrationError):
        Calibration.read(fname)
