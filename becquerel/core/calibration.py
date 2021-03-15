"""Generic calibration class."""

import copy
import asteval
import black
import blib2to3
import numpy as np
import scipy.optimize
from .. import io

safe_eval = asteval.Interpreter(use_numpy=True)


class CalibrationError(Exception):
    """Base class for calibration errors."""

    pass


class Calibration(object):
    """Base class for calibrations.

    A calibration is a nonnegative scalar function of a nonnegative scalar
    argument, parametrized by an array of scalars. Examples of calibrations are
    energy calibrations (mapping raw channels to energy in keV), energy
    resolution calibrations (mapping energy to energy FWHM or sigma), and
    efficiency calibrations (mapping energy to fraction of photopeak
    detected).
    """

    def __init__(self, expression: str, params, **attrs):
        """Create a calibration described by the expression and parameters.

        Parameters
        ----------
        expression : string
            The expression that defines the calibration function as a
            function of argument "x". Parameters are referenced as "p",
            i.e., "p[j]" is the jth parameter, and all parameters must be
            explicitly indexed in the expression. Can be a single-line formula
            like "p[0] + p[1] * x" or a code block.
        params : array_like
            List of floating point parameters for the calibration function
        attrs : dict
            Other information to be stored with the calibration.
        """
        self.expression = expression
        self.params = params
        self.attrs = attrs
        self.set_points()

    @staticmethod
    def eval_expression(expression, params, x):
        """Evaluate the expression at x.

        TODO: docstring
        """
        x = np.asarray(x)
        assert np.all(x >= 0)
        safe_eval.symtable["p"] = params
        safe_eval.symtable["x"] = x
        y = safe_eval(expression)
        assert np.all(y >= 0)
        return y

    @staticmethod
    def param_indices(expr):
        """Find all integer parameter indices of the expression.

        The expression must explicitly call each parameter as "p[j]", where
        j is the index of the parameter.

        Parameters
        ----------
        expression : string
            The expression that defines the calibration function.

        Returns
        -------
        param_indices : array_like
            List of integer parameter indices appearing in the expression.
        """
        # find parameter indices
        tokens = expr.split("p[")
        param_indices = [int(token.split("]")[0]) for token in tokens[1:]]
        param_indices = np.array(sorted(np.unique(param_indices)))
        return param_indices

    @staticmethod
    def validate_expression(expr, params=None):
        """Perform checks on the expression.

        The expression must explicitly call each parameter as "p[j]", where
        j is the index of the parameter, and the indices for n parameters
        range from 0 to n - 1. The expression is checked for how many
        parameters there are and their length is checked if `params` is given
        to ensure each is used at least once.

        Parameters
        ----------
        expression : string
            The expression that defines the calibration function. It will
            be checked for syntax, whether it uses all the parameters,
            and whether it can be evaluated.
        params : array_like
            List of floating point parameters for the calibration function.
            The expression will be checked whether it includes all of
            the parameters.

        Returns
        -------
        expression : string
            Expression having been validated and reformatted using black.
        """
        print("Initial expression:", expr)
        # apply black formatting for consistency and error checking
        try:
            expr = black.format_str(expr, mode=black.FileMode())
        except (black.InvalidInput, blib2to3.pgen2.tokenize.TokenError):
            raise CalibrationError(f"Error while running black on expression {expr}")
        print("After black:       ", expr)

        # make sure square brackets only occur with "p"
        for j in range(1, len(expr)):
            if expr[j] == "[":
                if expr[j - 1] != "p":
                    raise CalibrationError(
                        f"Character preceding '[' must be 'p':\n{expr[:j]}  {expr[j:]}"
                    )

        # make sure each parameter appears at least once
        try:
            param_indices = Calibration.param_indices(expr)
        except ValueError:
            raise CalibrationError(f"Unable to extract indices to parameters:\n{expr}")
        print("indices:", param_indices)
        if param_indices.min() != 0:
            raise CalibrationError(
                f"Minimum parameter index in expression is not 0:\n{expr}\n{param_indices}"
            )
        if not np.allclose(np.diff(param_indices), 1):
            raise CalibrationError(
                f"Parameter indices in expression are not contiguous:\n{expr}\n{param_indices}"
            )
        if params is not None:
            if len(param_indices) != len(params):
                raise CalibrationError(
                    f"Not enough parameter indices in expression:\n{expr}\n{param_indices}"
                )

        # make sure the expression can be evaluated
        if params is not None:
            try:
                y = Calibration.eval_expression(expr, params, 200.0)
                print("y:", y)
            except TypeError:
                raise CalibrationError(
                    f"Cannot evaluate expression for a float:\n{expr}\n{safe_eval.symtable['x']}"
                )
            try:
                y = Calibration.eval_expression(expr, params, [200.0, 500.0])
                print("y:", y)
            except TypeError:
                raise CalibrationError(
                    f"Cannot evaluate expression for an array:\n{expr}\n{safe_eval.symtable['x']}"
                )

        print("Final expression:  ", expr)
        return expr

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expr):
        expr = self.validate_expression(expr)
        self._expression = expr

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        params = np.array(p)
        if params.ndim != 1:
            raise CalibrationError(f"Parameters must be a 1-D array: {params}")
        self.validate_expression(self.expression, params=params)
        self._params = params

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = copy.deepcopy(attrs)

    @property
    def points_x(self):
        return self._points_x

    @property
    def points_y(self):
        return self._points_y

    def add_points(self, points_x=None, points_y=None):
        """Add the calibration point values to the internal list.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        """
        if points_x is None:
            points_x = []
        if points_y is None:
            points_y = []
        points_x = np.asarray(points_x)
        points_y = np.asarray(points_y)
        assert points_x.ndim == 1
        assert points_y.ndim == 1
        assert len(points_x) == len(points_y)
        self._points_x = np.append(self._points_x, points_x)
        self._points_y = np.append(self._points_y, points_y)
        # sort points in increasing order of x
        i = np.argsort(self._points_x)
        self._points_x = self._points_x[i]
        self._points_y = self._points_y[i]
        # check all values are positive
        assert np.all(self._points_x >= 0)
        assert np.all(self._points_y >= 0)

    def set_points(self, points_x=None, points_y=None):
        """Remove existing points and set the calibration point values.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        """
        self._points_x = []
        self._points_y = []
        if points_x is None:
            points_x = []
        if points_y is None:
            points_y = []
        self.add_points(points_x=points_x, points_y=points_y)

    def __eq__(self, other):
        """Determine if the two calibrations are identical."""
        if not isinstance(other, Calibration):
            raise CalibrationError(
                f"Attempting to compare {self.__class__} and {other.__class__}"
            )
        if len(self.params) != len(other.params):
            return False
        return (self.expression == other.expression) and np.allclose(
            self.params, other.params
        )

    def copy(self):
        """Make a complete copy of the calibration."""
        cal = Calibration(self.expression, self.params, **self.attrs)
        cal.set_points(cal.points_x, cal.points_y)
        return cal

    def __call__(self, x):
        """Call the calibration function.

        Parameters
        ----------
        x : float or array_like
            The nonnegative scalar argument to the function (e.g., raw channel).

        Returns
        -------
        calibration : float or np.ndarray
            The value of the calibration function at x.
        """
        return self.eval_expression(self.expression, self.params, x)

    @classmethod
    def read(cls, name):
        """Read the class from HDF5.

        Parameters
        ----------
        name : str, h5py.File, h5py.Group
            The filename or an open h5py File or Group.

        Returns
        -------
        calibration : becquerel.Calibration
        """
        dsets, attrs, skipped = io.h5.read_h5(name)
        assert len(dsets.keys()) in [2, 4]
        assert "params" in dsets
        assert "expression" in dsets
        expr = io.h5.ensure_string(dsets["expression"])
        cal = cls(expr, dsets["params"], **attrs)
        if "points_x" in dsets and "points_y" in dsets:
            cal.set_points(dsets["points_x"], dsets["points_y"])
        for key in attrs:
            if isinstance(attrs[key], bytes):
                attrs[key] = io.h5.ensure_string(attrs[key])
        return cal

    def write(self, name):
        """Write the class to HDF5.

        TODO: docstring
        """
        dsets = {
            "expression": self.expression,
            "params": self.params,
            "points_x": self.points_x,
            "points_y": self.points_y,
        }
        attrs = copy.deepcopy(self.attrs)
        io.h5.write_h5(name, dsets, attrs)


class LinearCalibration(Calibration):
    """Linear calibration."""

    def __init__(self, params, **attrs):
        """Create a linear calibration with the given parameters.

        Calibration expression is "p[0] + p[1] * x".

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        assert len(params) == 2
        expr = "p[0] + p[1] * x"
        super().__init__(expr, params, **attrs)


class PolynomialCalibration(Calibration):
    """Polynomial calibration of any order."""

    def __init__(self, params, **attrs):
        """Create a polynomial calibration with the given parameters.

        Calibration expression is
            "p[0] + p[1] * x + p[2] * x**2 + ..."

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        order = len(params) - 1
        assert order >= 0
        expr = "p[0]"
        for n in range(1, order + 1):
            expr += f" + p[{n}] * x ** {n}"
        super().__init__(expr, params, **attrs)


class SqrtPolynomialCalibration(Calibration):
    """Square root of a polynomial of any order."""

    def __init__(self, params, **attrs):
        """Create a square root of a polynomial with the given parameters.

        Calibration expression is
            "sqrt(p[0] + p[1] * x + p[2] * x**2 + ...)"

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        order = len(params) - 1
        assert order >= 0
        expr = "sqrt(p[0]"
        for n in range(1, order + 1):
            expr += f" + p[{n}] * x ** {n}"
        expr += ")"
        super().__init__(expr, params, **attrs)

