"""Generic calibration class."""

from abc import abstractmethod
import ast
import copy
import asteval
import black
import blib2to3
import numpy as np
import scipy
import scipy.optimize
from .. import io

CLIP_MAX = 1e6  # maximum value for a calibration function

safe_eval = asteval.Interpreter(use_numpy=False)
safe_eval.symtable["np"] = np
safe_eval.symtable["numpy"] = np
safe_eval.symtable["scipy"] = scipy


class CalibrationError(Exception):
    """Base class for calibration errors."""

    pass


def _eval_expression(expression, params, x):
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
            "asteval failed with errors:\n"
            + "\n".join(str(err.get_error()) for err in safe_eval.error)
        )
    if not np.all(np.isreal(y)):
        raise CalibrationError(f"Function evaluation resulted in complex values: {y}")
    # clip values of y
    y = np.clip(y, 0, CLIP_MAX)
    return y


def _param_indices(expression):
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
    tokens = expression.split("p[")
    param_indices = [int(token.split("]")[0]) for token in tokens[1:]]
    param_indices = np.array(sorted(np.unique(param_indices)))
    return param_indices


def _validate_expression(expression, params=None):
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
        expression = black.format_str(expression, mode=black.FileMode())
    except (black.InvalidInput, blib2to3.pgen2.tokenize.TokenError):
        raise CalibrationError(
            f"Error while running black on expression:\n{expression}"
        )

    # make sure "x" appears in the formula
    x_appears = False
    for node in ast.walk(ast.parse(expression)):
        if type(node) is ast.Name:
            if node.id == "x":
                x_appears = True
    if not x_appears:
        raise CalibrationError(
            f'Independent variable "x" must appear in the expression:\n{expression}'
        )

    # make sure each parameter appears at least once
    try:
        param_indices = _param_indices(expression)
    except ValueError:
        raise CalibrationError(
            f"Unable to extract indices to parameters:\n{expression}"
        )
    if len(param_indices) > 0:
        if param_indices.min() != 0:
            raise CalibrationError(
                f"Minimum parameter index in expression is not 0:\n{expression}\n{param_indices}"
            )
        if not np.allclose(np.diff(param_indices), 1):
            raise CalibrationError(
                f"Parameter indices in expression are not contiguous:\n{expression}\n{param_indices}"
            )
    if params is not None:
        if len(param_indices) != len(params):
            raise CalibrationError(
                f"Not enough parameter indices in expression:\n{expression}\n{param_indices}"
            )

    # make sure the expression can be evaluated
    if params is not None:
        try:
            y = _eval_expression(expression, params, 200.0)
        except CalibrationError:
            raise CalibrationError(
                f"Cannot evaluate expression for a float:\n{expression}\n{safe_eval.symtable['x']}"
            )
        try:
            y = _eval_expression(expression, params, [200.0, 500.0])
        except CalibrationError:
            raise CalibrationError(
                f"Cannot evaluate expression for an array:\n{expression}\n{safe_eval.symtable['x']}"
            )

    return expression.strip()


def _fit_expression(expression, points_x, points_y, params0=None, **kwargs):
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
    kwargs : dict
        Kwargs to pass to the minimization routine.

    Returns
    -------
    params : array_like
        Parameters that result from the fit.
    """
    expression = _validate_expression(expression)
    points_x, points_y = _check_points(points_x, points_y)

    # check that we have the expected number of parameters
    n_params = len(_param_indices(expression))
    if params0 is None:
        params0 = np.ones(n_params)
    else:
        params0 = np.asarray(params0).flatten()
    if len(params0) != n_params:
        raise CalibrationError(
            f"Starting parameters have length {len(params0)}, but expression requires {n_params} parameters"
        )
    expression = _validate_expression(expression, params=params0)

    # check that we have enough points
    if len(points_x) < n_params:
        raise CalibrationError(
            f"Expression has {n_params} free parameters but there are only {len(points_x)} points to fit"
        )

    # skip fitting if there are zero parameters to fit
    if n_params == 0:
        return np.array([])

    # define the residuals for least squares
    def residuals(p, xs, ys):
        fs = _eval_expression(expression, p, xs)
        return ys - fs

    # perform the fit
    results = scipy.optimize.least_squares(
        residuals,
        params0,
        args=(points_x, points_y),
        **kwargs,
    )
    if not results.success:
        raise CalibrationError(results.message)
    params = results.x
    return params


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
    points_x = np.atleast_1d(points_x)
    points_y = np.atleast_1d(points_y)
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


def _polynomial_expression(params):
    """Create a polynomial expression of any order.

    The calibration function expression is
        "p[0] + p[1] * x + p[2] * x**2 + ..."

    Parameters
    ----------
    params : array_like
        Coefficients beginning with 0th order.

    Returns
    -------
    expression : str
        The polynomial expression.
    """
    order = len(params) - 1
    if order <= 0:
        raise CalibrationError("Polynomial expression expects an order of at least 1")
    expr = "p[0]"
    for n in range(1, order + 1):
        expr += f" + p[{n}] * x ** {n}"
    return expr


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

    def __str__(self):
        """A string version of the calibration."""
        result = ""
        result += "expression:\n"
        lines = str(self.expression).split("\n")
        for line in lines:
            result += " " * 4 + line + "\n"
        result += "params:\n"
        result += " " * 4 + str(self.params) + "\n"
        if len(self.points_x) > 0:
            result += "calibration points (x):\n"
            result += " " * 4 + str(self.points_x) + "\n"
            result += "calibration points (y):\n"
            result += " " * 4 + str(self.points_y) + "\n"
        if len(self.attrs.keys()) > 0:
            result += "other attributes:\n"
            result += " " * 4 + str(self.attrs)
        return result

    def __repr__(self):
        """A string representation of the calibration."""
        result = "Calibration("
        result += repr(self.expression) + ", "
        result += repr(self.params)
        if len(self.attrs) > 0:
            for key in self.attrs:
                result += f", {key}={repr(self.attrs[key])}"
        result += ")"
        return result

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        expression = _validate_expression(expression)
        self._expression = expression

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        params = np.array(p)
        if params.ndim != 1:
            raise CalibrationError(f"Parameters must be a 1-D array: {params}")
        _validate_expression(self.expression, params=params)
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

    @property
    def fit_y(self):
        """Calibration evaluated at the input x values."""
        return self(self.points_x)

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
        return _eval_expression(self.expression, self.params, x)

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
        unexpected = set(dsets.keys()) - set(
            ["expression", "params", "points_x", "points_y"]
        )
        if len(unexpected) > 0:
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

    def fit(self, **kwargs):
        """Fit the calibration to the stored calibration points.

        Parameters
        ----------
        kwargs : dict
            Kwargs to pass to the minimization routine.
        """
        params = _fit_expression(
            self.expression,
            self.points_x,
            self.points_y,
            params0=self.params,
            **kwargs,
        )
        self.params = params

    @classmethod
    def from_points(
        cls,
        expression,
        points_x,
        points_y,
        params0=None,
        include_origin=False,
        fit_kwargs={},
        **attrs,
    ):
        """Create a Calibration with the expression and fit the points.

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
        include_origin : bool
            Whether to add and fit with the point (0, 0) in addition to the
            others.
        fit_kwargs : dict
            Kwargs to pass to the minimization routine.
        attrs : dict
            Other information to be stored with the calibration.

        Returns
        -------
        cal : Calibration
            The Calibration instance with the given expression fitted to
            the points.
        """
        points_x, points_y = _check_points(points_x, points_y)
        if include_origin:
            points_x = np.append(0, points_x)
            points_y = np.append(0, points_y)
        points_x, points_y = _check_points(points_x, points_y)
        params = _fit_expression(
            expression, points_x, points_y, params0=params0, **fit_kwargs
        )
        cal = cls(expression, params, **attrs)
        cal.set_points(points_x, points_y)
        return cal

    @classmethod
    def from_linear(cls, params, **attrs):
        """Create a Calibration with a linear function.

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        expr = "p[0] + p[1] * x"
        if len(params) != 2:
            raise CalibrationError("Linear calibration expects 2 parameters")
        return cls(expr, params, **attrs)

    @classmethod
    def from_polynomial(cls, params, **attrs):
        """Create a Calibration with a polynomial function of any order.

        The calibration function expression is
            "p[0] + p[1] * x + p[2] * x**2 + ..."

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        expr = _polynomial_expression(params)
        return cls(expr, params, **attrs)

    @classmethod
    def from_sqrt_polynomial(cls, params, **attrs):
        """Create a square root of a polynomial function of any order.

        The calibration function expression is
            "np.sqrt(p[0] + p[1] * x + p[2] * x**2 + ...)"

        Parameters
        ----------
        params : array_like
            Coefficients beginning with 0th order.
        attrs : dict
            Other information to be stored with the calibration.
        """
        expr = _polynomial_expression(params)
        expr = "np.sqrt(" + expr + ")"
        return cls(expr, params, **attrs)

    @classmethod
    def from_interpolation(cls, points_x, points_y, **attrs):
        """Create a Calibration that interpolates the calibration points.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        attrs : dict
            Other information to be stored with the calibration.
        """
        points_x, points_y = _check_points(points_x, points_y)
        if len(points_x) < 2:
            raise CalibrationError("Interpolated calibration expects at least 2 points")
        xp = np.array2string(points_x, precision=9, separator=", ")
        yp = np.array2string(points_y, precision=9, separator=", ")
        expr = ""
        expr += f"assert np.all(x >= {points_x.min():.9e})\n"
        expr += f"assert np.all(x <= {points_x.max():.9e})\n"
        expr += f"np.interp(x, {xp}, {yp})"
        return cls(expr, [], **attrs)

    def fit_R_squared(self):
        """Calibration fit R^2 value.

        Reference
        ---------
        stackoverflow.com/questions/19189362
        """

        # residual sum of squares
        ss_res = np.sum((self.points_y - self.fit_y) ** 2)

        # total sum of squares
        ss_tot = np.sum((self.points_y - np.mean(self.points_y)) ** 2)

        # r-squared
        return 1 - (ss_res / ss_tot)

    def fit_chi_squared(self):
        """Calibration fit chi^2 value."""

        if self.points_y.shape != self.fit_y.shape:
            raise ValueError(
                "y and fit_y must have same shapes:", self.y.shape, self.fit_y.shape
            )
        # Mask out zeros
        fit_y = self.fit_y[self.points_y > 0]
        points_y = self.points_y[self.points_y > 0]
        return np.sum((points_y - fit_y) ** 2 / points_y)
