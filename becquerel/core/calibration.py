"""Generic calibration class."""

from abc import abstractmethod
import ast
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


def _check_points(points_x, points_y):
    """Perform various checks on the sets of calibration points.

    Ensure the arrays of points are both 1-D and have the same length,
    that all values are >= 0, and then put them in the order of ascending
    x values.

    Parameters
    ----------
    points_x : float or array_like
        The x-value or values of calibration points
    points_y : float or array_like
        The y-value or values of calibration points

    Returns
    -------
    points_x : array_like
        The x-value or values of calibration points
    points_y : array_like
        The y-value or values of calibration points
    """
    if points_x is None:
        points_x = []
    if points_y is None:
        points_y = []
    points_x = np.asarray(points_x)
    points_y = np.asarray(points_y)
    if points_x.ndim != 1:
        raise CalibrationError(f"Calibration x points must be 1-D: {points_x}")
    if points_y.ndim != 1:
        raise CalibrationError(f"Calibration y points must be 1-D: {points_y}")
    if len(points_x) != len(points_y):
        raise CalibrationError(
            f"Number of x and y calibration points must match: {len(points_x)}, {len(points_y)}"
        )
    # sort points in increasing order of x
    i = np.argsort(points_x)
    points_x = points_x[i]
    points_y = points_y[i]
    # check all values are positive
    if not np.all(points_x >= 0):
        raise CalibrationError(f"All calibration x points must be >= 0: {points_x}")
    if not np.all(points_y >= 0):
        raise CalibrationError(f"All calibration y points must be >= 0: {points_y}")
    return points_x, points_y


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

        Parameters
        ----------
        expression : string
            The expression that defines the calibration function.
        params : array_like
            List of floating point parameters for the calibration function
        x : float or array_like
            The argument at which to evaluate the expression.

        Returns
        -------
        y : float or array_like
            Result of evaluating the expression for x.
        """
        x = np.asarray(x)
        if not np.all(x >= 0):
            raise CalibrationError(f"x must be >= 0: {x}")
        safe_eval.symtable["p"] = params
        safe_eval.symtable["x"] = x
        y = safe_eval(expression)
        if len(safe_eval.error) > 0:
            raise CalibrationError(
                f"asteval failed with errors:\n"
                + "\n".join(str(err.get_error()) for err in safe_eval.error)
            )
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
        # apply black formatting for consistency and error checking
        try:
            expr = black.format_str(expr, mode=black.FileMode())
        except (black.InvalidInput, blib2to3.pgen2.tokenize.TokenError):
            raise CalibrationError(f"Error while running black on expression:\n{expr}")

        # make sure "x" appears in the formula
        x_appears = False
        for node in ast.walk(ast.parse(expr)):
            if type(node) is ast.Name:
                if node.id == "x":
                    x_appears = True
        if not x_appears:
            raise CalibrationError(
                f'Independent variable "x" must appear in the expression:\n{expr}'
            )

        # make sure each parameter appears at least once
        try:
            param_indices = Calibration.param_indices(expr)
        except ValueError:
            raise CalibrationError(f"Unable to extract indices to parameters:\n{expr}")
        if len(param_indices) > 0:
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
            except (TypeError, NameError):
                raise CalibrationError(
                    f"Cannot evaluate expression for a float:\n{expr}\n{safe_eval.symtable['x']}"
                )
            try:
                y = Calibration.eval_expression(expr, params, [200.0, 500.0])
            except (TypeError, NameError):
                raise CalibrationError(
                    f"Cannot evaluate expression for an array:\n{expr}\n{safe_eval.symtable['x']}"
                )

        return expr.strip()

    @staticmethod
    def fit_expression(expr, points_x, points_y, params0=None):
        """Fit the expression using the calibration points.

        Performs least squares via scipy.optimize.least_squares.

        Parameters
        ----------
        expression : string
            The expression that defines the calibration function.
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        params0 : float or array_like
            Initial guesses for the parameters. By default an array of ones
            with its length inferred from the number of parameters
            referenced in the expression.

        Returns
        -------
        params : array_like
            Parameters that result from the fit.
        """
        expr = Calibration.validate_expression(expr)
        points_x, points_y = _check_points(points_x, points_y)

        # check that we have the expected number of parameters
        n_params = len(Calibration.param_indices(expr))
        if params0 is None:
            params0 = np.ones(n_params)
        else:
            params0 = np.asarray(params0).flatten()
        if len(params0) != n_params:
            raise CalibrationError(
                f"Starting parameters have length {len(params0)}, but expression requires {n_params} parameters"
            )
        expr = Calibration.validate_expression(expr, params=params0)

        # check that we have enough points
        if len(points_x) < n_params:
            raise CalibrationError(
                f"Expression has {n_params} free parameters but there are only {len(points_x)} points to fit"
            )

        # define the residuals for least squares
        def residuals(p, xs, ys):
            fs = Calibration.eval_expression(expr, p, xs)
            return ys - fs

        # perform the fit
        results = scipy.optimize.least_squares(
            residuals,
            params0,
            args=(points_x, points_y),
        )
        if not results.success:
            raise CalibrationError(results.message)
        params = results.x
        return params

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
        self._attrs = dict(copy.deepcopy(attrs))

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
        points_x, points_y = _check_points(points_x, points_y)
        self._points_x = np.append(self._points_x, points_x)
        self._points_y = np.append(self._points_y, points_y)
        self._points_x, self._points_y = _check_points(self._points_x, self._points_y)

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
        if "params" not in dsets:
            raise CalibrationError(f'Expected dataset "params"')
        if "expression" not in dsets:
            raise CalibrationError(f'Expected dataset "expression"')
        if not (len(dsets.keys()) in [2, 4]):
            unexpected = set(dsets.keys()) - set(
                ["expression", "params", "points_x", "points_y"]
            )
            raise CalibrationError(f"Unexpected dataset names in file: {unexpected}")
        expr = io.h5.ensure_string(dsets["expression"])
        cal = cls(expr, dsets["params"], **attrs)
        if "points_x" in dsets and "points_y" in dsets:
            cal.set_points(dsets["points_x"], dsets["points_y"])
        for key in attrs:
            if isinstance(attrs[key], (str, bytes)):
                attrs[key] = io.h5.ensure_string(attrs[key])
        return cal

    def write(self, name):
        """Write the class to HDF5.

        Parameters
        ----------
        name : str, h5py.File, h5py.Group
            The filename or an open h5py File or Group.
        """
        dsets = {
            "expression": self.expression,
            "params": self.params,
            "points_x": self.points_x,
            "points_y": self.points_y,
        }
        attrs = copy.deepcopy(self.attrs)
        io.h5.write_h5(name, dsets, attrs)

    def fit(self):
        """Fit the calibration to the stored calibration points."""
        params = self.fit_expression(
            self.expression, self.points_x, self.points_y, params0=self.params
        )
        self.params = params

    @classmethod
    def from_points(cls, expr, points_x, points_y, params0=None, **attrs):
        """Create a calibration with the expression and fit the points.

        Parameters
        ----------
        expr : string
            The expression that defines the calibration function.
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        params0 : float or array_like
            Initial guesses for the parameters. By default an array of ones
            with its length inferred from the number of parameters
            referenced in the expression.
        attrs : dict
            Other information to be stored with the calibration.

        Returns
        -------
        cal : Calibration
            The Calibration instance with the given expression fitted to
            the points.
        """
        params = cls.fit_expression(expr, points_x, points_y, params0=params0)
        cal = cls(expr, params, **attrs)
        cal.add_points(points_x, points_y)
        return cal


class AutoExpressionCalibration(Calibration):
    """A Calibration class that automatically generates its expression."""

    @staticmethod
    @abstractmethod
    def make_expression(params):
        """Build the expression for this class given the parameters.

        Parameters
        ---------
        params : array_like
            List of floating point parameters for the calibration function

        Returns
        -------
        expr : string
            The expression that defines the calibration function.
        """
        pass

    def __init__(self, params, **attrs):
        """Create a calibration with an auto-generated formula.

        Parameters
        ----------
        params : array_like
            Coefficients of the calibration function, used to infer the
            calibration function formula.
        attrs : dict
            Other information to be stored with the calibration.
        """
        expr = self.make_expression(params)
        super().__init__(expr, params, **attrs)

    @classmethod
    def from_points(cls, points_x, points_y, params0, **attrs):
        """Create a calibration instance and fit the points.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        params0 : float or array_like
            Initial guesses for the parameters. By default an array of ones
            with its length inferred from the number of parameters
            referenced in the expression.
        attrs : dict
            Other information to be stored with the calibration.

        Returns
        -------
        cal : Calibration
            The Calibration instance with the given expression fitted to
            the points.
        """
        expr = cls.make_expression(params0)
        params = cls.fit_expression(expr, points_x, points_y, params0=params0)
        cal = Calibration(expr, params, **attrs)
        cal.add_points(points_x, points_y)
        return cal


class LinearCalibration(AutoExpressionCalibration):
    """Linear calibration."""

    @staticmethod
    def make_expression(params):
        """Create a linear expression.

        The calibration expression is "p[0] + p[1] * x".

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        """
        if len(params) != 2:
            raise CalibrationError("LinearCalibration expects 2 parameters")
        return "p[0] + p[1] * x"


class PolynomialCalibration(AutoExpressionCalibration):
    """Polynomial calibration of any order."""

    @staticmethod
    def make_expression(params):
        """Create a polynomial expression for the given parameters.

        The calibration expression is
            "p[0] + p[1] * x + p[2] * x**2 + ..."

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        """
        order = len(params) - 1
        if order <= 0:
            raise CalibrationError(
                "PolynomialCalibration expects an order of at least 1"
            )
        expr = "p[0]"
        for n in range(1, order + 1):
            expr += f" + p[{n}] * x ** {n}"
        return expr


class SqrtPolynomialCalibration(AutoExpressionCalibration):
    """Square root of a polynomial of any order."""

    @staticmethod
    def make_expression(params):
        """Create a square root polynomial expression for the given parameters.

        The calibration expression is
            "sqrt(p[0] + p[1] * x + p[2] * x**2 + ...)"

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        """
        order = len(params) - 1
        if order <= 0:
            raise CalibrationError(
                "SqrtPolynomialCalibration expects an order of at least 1"
            )
        expr = "sqrt(p[0]"
        for n in range(1, order + 1):
            expr += f" + p[{n}] * x ** {n}"
        expr += ")"
        return expr


class InterpolatedCalibration(Calibration):
    """A calibration that works by interpolating a series of points."""

    @staticmethod
    def make_expression(points_x, points_y):
        """Build the interpolation expression given the points.

        Parameters
        ---------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points

        Returns
        -------
        expr : string
            The expression that defines the calibration function.
        """
        points_x, points_y = _check_points(points_x, points_y)
        xp = ", ".join([f"{x:.9e}" for x in points_x])
        yp = ", ".join([f"{y:.9e}" for y in points_y])
        expr = ""
        expr += f"assert all(x >= {points_x.min():.9e})\n"
        expr += f"assert all(x <= {points_x.max():.9e})\n"
        expr += f"interp(x, [{xp}], [{yp}])"
        return expr

    def __init__(self, **attrs):
        """Create a calibration that interpolates the points.

        The calibration will be valid until at least two points are added.

        Parameters
        ----------
        attrs : dict
            Other information to be stored with the calibration.
        """
        super().__init__("x", [], **attrs)

    def add_points(self, points_x=None, points_y=None):
        """Add the calibration point values to the internal list.

        Update the interpolation expression once there are at least two points.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        """
        super().add_points(points_x, points_y)
        if len(self.points_x) >= 2:
            self.expression = self.make_expression(self.points_x, self.points_y)

    @classmethod
    def from_points(cls, points_x, points_y, **attrs):
        """Create a calibration class that interpolates the points.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        attrs : dict
            Other information to be stored with the calibration.

        Returns
        -------
        cal : Calibration
            The Calibration instance with the given expression fitted to
            the points.
        """
        cal = cls(**attrs)
        cal.add_points(points_x, points_y)
        return cal

    def fit(self):
        """Fit the calibration to the stored calibration points.

        Since this function is interpolated, do nothing if this method is
        called.
        """
        pass
