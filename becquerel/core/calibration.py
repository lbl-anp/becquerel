"""Class to describe a generic calibration function."""

import ast
import copy
import asteval
import black
import blib2to3
import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from .. import io

DEFAULT_DOMAIN = (0, 1e5)
DEFAULT_RANGE = (0, 1e5)

safe_eval = asteval.Interpreter(use_numpy=False)
safe_eval.symtable["np"] = np
safe_eval.symtable["numpy"] = np
safe_eval.symtable["scipy"] = scipy


class CalibrationError(Exception):
    """Base class for calibration errors."""

    pass


def _validate_domain_range(domain, rng):
    """Validate that domain and range contain finite values.

    Parameters
    ----------
    domain : array_like
        The domain of the function. Will raise an error if the independent
        variable is outside this interval. Must be finite.
        By default DEFAULT_DOMAIN.
    rng : array_like
        The range of the function. Expression outputs will be clipped to this
        interval. Must be finite. By default DEFAULT_RANGE.
    """
    # must be length-2 iterables
    try:
        len(domain)
    except TypeError:
        raise CalibrationError(f"Domain must be length-2 iterable: {domain}")
    domain = np.asarray(domain)
    if not (len(domain) == 2 and domain.ndim == 1):
        raise CalibrationError(f"Domain must be length-2 iterable: {domain}")
    try:
        len(rng)
    except TypeError:
        raise CalibrationError(f"Range must be length-2 iterable: {rng}")
    rng = np.asarray(rng)
    if not (len(rng) == 2 and rng.ndim == 1):
        raise CalibrationError(f"Range must contain two values: {rng}")
    # must contain finite values
    if not np.isfinite(domain[0]) or not np.isfinite(domain[1]):
        raise CalibrationError(f"Domain must contain finite values: {domain}")
    if not np.isfinite(rng[0]) or not np.isfinite(rng[1]):
        raise CalibrationError(f"Range must contain finite values: {rng}")
    # must be in ascending order
    if not (domain[1] > domain[0]):
        raise CalibrationError(f"Domain must contain ascending values: {domain}")
    if not (rng[1] > rng[0]):
        raise CalibrationError(f"Range must contain ascending values: {rng}")


def _eval_expression(
    expression,
    params,
    x,
    ind_var="x",
    aux_params=None,
    domain=DEFAULT_DOMAIN,
    rng=DEFAULT_RANGE,
):
    """Evaluate the expression at x.

    Parameters
    ----------
    expression : string
        The expression that defines the calibration function.
    params : array_like
        List of floating point parameters for the calibration function,
        referred to by the symbol "p".
    x : float or array_like
        The argument at which to evaluate the expression.
    ind_var : str
        The symbol of the independent variable. Default "x", "y" also allowed.
    aux_params : array_like
        Auxiliary floating point parameters for the calibration function,
        referred to by the symbol "a". By default an empty array.
    domain : array_like
        The domain of the function. Will raise an error if the independent
        variable is outside this interval. Must be finite. By default
        DEFAULT_DOMAIN.
    rng : array_like
        The range of the function. Expression outputs will be clipped to this
        interval. Must be finite. By default DEFAULT_RANGE.

    Returns
    -------
    y : float or array_like
        Result of evaluating the expression for x.
    """
    _validate_domain_range(domain, rng)
    x = np.asarray(x)
    if not np.all(x >= domain[0]):
        raise CalibrationError(f"{ind_var} must be >= {domain[0]}: {x}")
    if not np.all(x <= domain[1]):
        raise CalibrationError(f"{ind_var} must be <= {domain[1]}: {x}")
    if ind_var not in ["x", "y"]:
        raise CalibrationError(f"Independent variable {ind_var} must be 'x' or 'y'")
    if aux_params is None:
        aux_params = np.array([])
    else:
        aux_params = np.asarray(aux_params)
    safe_eval.symtable["p"] = params
    safe_eval.symtable["a"] = aux_params
    safe_eval.symtable[ind_var] = x
    y = safe_eval(expression)
    if len(safe_eval.error) > 0:
        raise CalibrationError(
            "asteval failed with errors:\n"
            + "\n".join(str(err.get_error()) for err in safe_eval.error)
        )
    if not np.all(np.isreal(y)):
        raise CalibrationError(f"Function evaluation resulted in complex values: {y}")
    # clip values of y to the range
    y = np.clip(y, rng[0], rng[1])
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


def _validate_expression(
    expression,
    ind_var="x",
    params=None,
    aux_params=None,
    domain=DEFAULT_DOMAIN,
    rng=DEFAULT_RANGE,
    n_eval=100,
):
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
    ind_var : str
        The symbol of the independent variable. Default "x", "y" also allowed.
    params : array_like
        List of floating point parameters for the calibration function,
        referred to by the symbol "p". The expression will be checked whether
        it includes all of the parameters.
    aux_params : array_like
        Auxiliary floating point parameters for the calibration function,
        referred to by the symbol "a". By default an empty array.
    domain : array_like
        The domain of the function. Will draw test values from inside this
        interval. Must be finite. By default DEFAULT_DOMAIN.
    rng : array_like
        The range of the function. Expression outputs will be clipped to this
        interval. Must be finite. By default DEFAULT_RANGE.
    n_eval : int
        Number of points in the domain the evaluate at for testing.

    Returns
    -------
    expression : string
        Expression having been validated and reformatted using black.
    """
    _validate_domain_range(domain, rng)

    # apply black formatting for consistency and error checking
    try:
        expression = black.format_str(expression, mode=black.FileMode())
    except (black.InvalidInput, blib2to3.pgen2.tokenize.TokenError):
        raise CalibrationError(
            f"Error while running black on expression:\n{expression}"
        )

    # make sure `ind_var` appears in the formula
    if ind_var not in ["x", "y"]:
        raise CalibrationError(f"Independent variable {ind_var} must be 'x' or 'y'")
    ind_var_appears = False
    for node in ast.walk(ast.parse(expression)):
        if type(node) is ast.Name:
            if node.id == ind_var:
                ind_var_appears = True
    if not ind_var_appears:
        raise CalibrationError(
            f'Independent variable "{ind_var}" must appear in the expression:\n'
            f"{expression}"
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
                "Minimum parameter index in expression is not 0:\n"
                f"{expression}\n{param_indices}"
            )
        if not np.allclose(np.diff(param_indices), 1):
            raise CalibrationError(
                "Parameter indices in expression are not contiguous:\n"
                f"{expression}\n{param_indices}"
            )
    if params is not None:
        if len(param_indices) != len(params):
            raise CalibrationError(
                "Not enough parameter indices in expression:\n"
                f"{expression}\n{param_indices}"
            )

    # make sure the expression can be evaluated
    if params is not None:
        x_arr = np.linspace(domain[0], domain[1], num=n_eval)
        for x_val in x_arr:
            try:
                _eval_expression(
                    expression,
                    params,
                    x_val,
                    ind_var=ind_var,
                    aux_params=aux_params,
                    domain=domain,
                    rng=rng,
                )
            except CalibrationError:
                raise CalibrationError(
                    f"Cannot evaluate expression for float {ind_var} = {x_val}:\n"
                    f"{expression}\n{safe_eval.symtable['x']}"
                )
        try:
            _eval_expression(
                expression,
                params,
                x_arr,
                ind_var=ind_var,
                aux_params=aux_params,
                domain=domain,
                rng=rng,
            )
        except CalibrationError:
            raise CalibrationError(
                f"Cannot evaluate expression for array {ind_var} = {x_arr}:\n"
                f"{expression}\n{safe_eval.symtable['x']}"
            )

    return expression.strip()


def _fit_expression(
    expression,
    points_x,
    points_y,
    weights=None,
    params0=None,
    aux_params=None,
    domain=DEFAULT_DOMAIN,
    rng=DEFAULT_RANGE,
    **kwargs,
):
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
    weights : float or array_like
        The weights to be used when fitting. Fitting will minimize the sum
        of squared errors weighted by these weights, i.e.,
        sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
        ones with the same length as the points.
    params0 : float or array_like
        Initial guesses for the parameters. By default an array of ones
        with its length inferred from the number of parameters
        referenced in the expression.
    aux_params : array_like
        Auxiliary floating point parameters for the calibration function,
        referred to by the symbol "a". By default an empty array.
    domain : array_like
        The domain of the function. Will raise an error if the independent
        variable is outside this interval. Must be finite. By default
        DEFAULT_DOMAIN.
    rng : array_like
        The range of the function. Expression outputs will be clipped to this
        interval. Must be finite. By default DEFAULT_RANGE.
    kwargs : dict
        Kwargs to pass to the minimization routine.

    Returns
    -------
    params : array_like
        Parameters that result from the fit.
    """
    expression = _validate_expression(
        expression, aux_params=aux_params, domain=domain, rng=rng
    )
    points_x, points_y, weights = _check_points(
        points_x, points_y, weights=weights, domain=domain, rng=rng
    )

    # check that we have the expected number of parameters
    n_params = len(_param_indices(expression))
    if params0 is None:
        params0 = np.ones(n_params)
    else:
        params0 = np.asarray(params0).flatten()
    if len(params0) != n_params:
        raise CalibrationError(
            f"Starting parameters have length {len(params0)}, but expression "
            f"requires {n_params} parameters"
        )
    expression = _validate_expression(
        expression, params=params0, aux_params=aux_params, domain=domain, rng=rng
    )

    # check that we have enough points
    if len(points_x) < n_params:
        raise CalibrationError(
            f"Expression has {n_params} free parameters but there are only "
            f"{len(points_x)} points to fit"
        )

    # skip fitting if there are zero parameters to fit
    if n_params == 0:
        return np.array([])

    # define the residuals for least squares
    def residuals(p, xs, ys, ws):
        fs = _eval_expression(
            expression, p, xs, aux_params=aux_params, domain=domain, rng=rng
        )
        return np.sqrt(ws) * (ys - fs)

    # perform the fit
    results = scipy.optimize.least_squares(
        residuals,
        params0,
        args=(points_x, points_y, weights),
        **kwargs,
    )
    if not results.success:
        raise CalibrationError(results.message)
    params = results.x
    return params


def _check_points(
    points_x, points_y, weights=None, domain=DEFAULT_DOMAIN, rng=DEFAULT_RANGE
):
    """Perform various checks on the sets of calibration points.

    Ensure the arrays of points are both 1-D and have the same length,
    that all values are >= 0, that they fall within the domain and range,
    and then put them in the order of ascending x values.

    Parameters
    ----------
    points_x : float or array_like
        The x-value or values of calibration points
    points_y : float or array_like
        The y-value or values of calibration points
    weights : float or array_like
        The weights to be used when fitting. Fitting will minimize the sum
        of squared errors weighted by these weights, i.e.,
        sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
        ones with the same length as the points.
    domain : array_like
        The domain of the function. Will raise an error if the independent
        variable is outside this interval. Must be finite. By default
        DEFAULT_DOMAIN.
    rng : array_like
        The range of the function. Expression outputs will be clipped to this
        interval. Must be finite. By default DEFAULT_RANGE.

    Returns
    -------
    points_x : array_like
        The x-value or values of calibration points
    points_y : array_like
        The y-value or values of calibration points
    weights : array_like
        The weights to be used when fitting to the points.
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
            f"Number of x and y calibration points must match: "
            f"{len(points_x)}, {len(points_y)}"
        )
    # handle the weights
    if weights is None:
        weights = np.ones_like(points_x)
    else:
        weights = np.atleast_1d(weights)
    if weights.ndim != 1:
        raise CalibrationError(f"Calibration weights must be 1-D: {weights}")
    if len(weights) != len(points_x):
        raise CalibrationError(
            f"Number of weights must match points: {len(weights)}, {len(points_x)}"
        )
    if (weights < 0.0).any():
        raise CalibrationError(f"Weights cannot be negative: {weights}")
    # sort points in increasing order of x
    i = np.argsort(points_x)
    points_x = points_x[i]
    points_y = points_y[i]
    weights = weights[i]
    # check domain and range
    if np.any((points_x < domain[0]) | (domain[1] < points_x)):
        raise CalibrationError(
            f"Some x points are outside of domain {domain}: {points_x}"
        )
    if np.any((points_y < rng[0]) | (rng[1] < points_y)):
        raise CalibrationError(f"Some y points are outside of range {rng}: {points_y}")
    return points_x, points_y, weights


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


class Calibration:
    """Base class for calibrations.

    A calibration is a scalar function of a scalar argument, parametrized by
    an array of scalars. Examples of calibrations are energy calibrations
    (mapping raw channels to energy in keV), energy resolution calibrations
    (mapping energy to energy FWHM or sigma), and efficiency calibrations
    (mapping energy to fraction of photopeak counts detected).
    """

    def __init__(
        self,
        expression: str,
        params,
        aux_params=None,
        inv_expression=None,
        domain=DEFAULT_DOMAIN,
        rng=DEFAULT_RANGE,
        **attrs,
    ):
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
        aux_params : array_like
            Auxiliary floating point parameters for the calibration function,
            referred to by the symbol "a". By default an empty array.
        inv_expression : str
            String giving the inverse of the function. If such an expression
            is available in closed form, it will speed up calls to inverse().
        domain : array_like
            The domain of the function. Will raise an error if the independent
            variable is outside this interval. Must be finite. By default
            DEFAULT_DOMAIN.
        rng : array_like
            The range of the function. Expression outputs will be clipped to
            this interval. Must be finite. By default DEFAULT_RANGE.
        attrs : dict
            Other information to be stored with the calibration.
        """
        self.params = params
        self.aux_params = aux_params
        self.domain = domain
        self.range = rng
        self.attrs = attrs
        self.expression = expression
        self.inv_expression = inv_expression
        self.set_points()

    def __str__(self):
        """A string version of the calibration."""
        result = ""
        result += "expression:\n"
        lines = str(self.expression).split("\n")
        for line in lines:
            result += " " * 4 + line + "\n"
        result += "params (p):\n"
        result += " " * 4 + str(self.params) + "\n"
        if len(self.aux_params) > 0:
            result += "auxiliary params (a):\n"
            result += " " * 4 + str(self.aux_params) + "\n"
        if self.inv_expression is not None:
            result += "inv_expression:\n"
            lines = str(self.inv_expression).split("\n")
            for line in lines:
                result += " " * 4 + line + "\n"
        result += "domain:\n"
        result += " " * 4 + str(self.domain) + "\n"
        result += "range:\n"
        result += " " * 4 + str(self.range) + "\n"
        if len(self.points_x) > 0:
            result += "calibration points (x):\n"
            result += " " * 4 + str(self.points_x) + "\n"
            result += "calibration points (y):\n"
            result += " " * 4 + str(self.points_y) + "\n"
            result += "calibration weights:\n"
            result += " " * 4 + str(self.weights) + "\n"
        if len(self.attrs.keys()) > 0:
            result += "other attributes:\n"
            result += " " * 4 + str(self.attrs)
        return result

    def __repr__(self):
        """A string representation of the calibration."""
        result = "Calibration("
        result += repr(self.expression)
        result += ", "
        result += repr(self.params)
        if len(self.aux_params) > 0:
            result += ", aux_params="
            result += repr(self.aux_params)
        if self.inv_expression is not None:
            result += ", inv_expression="
            result += repr(self.inv_expression)
        result += ", domain="
        result += repr(self.domain)
        result += ", rng="
        result += repr(self.range)
        if len(self.attrs) > 0:
            for key in self.attrs:
                result += ", "
                result += f"{key}={repr(self.attrs[key])}"
        result += ")"
        return result

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expression):
        expression = _validate_expression(
            expression, domain=self.domain, rng=self.range
        )
        self._expression = expression

    @property
    def inv_expression(self):
        return self._inv_expression

    @inv_expression.setter
    def inv_expression(self, inv_expression):
        if inv_expression is None:
            self._inv_expression = inv_expression
        else:
            inv_expression = _validate_expression(
                inv_expression, ind_var="y", domain=self.range, rng=self.domain
            )
            self._inv_expression = inv_expression

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, p):
        params = np.array(p)
        if params.ndim != 1:
            raise CalibrationError(f"Parameters must be a 1-D array: {params}")
        self._params = params

    @property
    def aux_params(self):
        return self._aux_params

    @aux_params.setter
    def aux_params(self, a):
        if a is None:
            self._aux_params = np.array([])
        else:
            self._aux_params = np.array(a)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        _validate_domain_range(domain, (0, 1))
        self._domain = tuple(domain)

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, rng):
        _validate_domain_range((0, 1), rng)
        self._range = tuple(rng)

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
    def weights(self):
        return self._weights

    @property
    def fit_y(self):
        """Calibration evaluated at the input x values."""
        return self(self.points_x)

    def add_points(self, points_x=None, points_y=None, weights=None):
        """Add the calibration point values to the internal list.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        weights : float or array_like
            The weights to be used when fitting. Fitting will minimize the sum
            of squared errors weighted by these weights, i.e.,
            sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
            ones with the same length as the points.
        """
        points_x, points_y, weights = _check_points(
            points_x, points_y, weights=weights, domain=self.domain, rng=self.range
        )
        self._points_x = np.append(self._points_x, points_x)
        self._points_y = np.append(self._points_y, points_y)
        self._weights = np.append(self._weights, weights)
        self._points_x, self._points_y, self._weights = _check_points(
            self._points_x,
            self._points_y,
            weights=self._weights,
            domain=self.domain,
            rng=self.range,
        )

    def set_points(self, points_x=None, points_y=None, weights=None):
        """Remove existing points and set the calibration point values.

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        weights : float or array_like
            The weights to be used when fitting. Fitting will minimize the sum
            of squared errors weighted by these weights, i.e.,
            sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
            ones with the same length as the points.
        """
        self._points_x = []
        self._points_y = []
        self._weights = []
        self.add_points(points_x=points_x, points_y=points_y, weights=weights)

    def __eq__(self, other):
        """Determine if the two calibrations are identical."""
        if not isinstance(other, Calibration):
            raise CalibrationError(
                f"Attempting to compare {self.__class__} and {other.__class__}"
            )
        if len(self.params) != len(other.params):
            return False
        return (
            (self.expression == other.expression)
            and np.allclose(self.params, other.params)
            and np.allclose(self.aux_params, other.aux_params)
        )

    def copy(self):
        """Make a complete copy of the calibration."""
        cal = Calibration(
            self.expression,
            self.params,
            aux_params=self.aux_params,
            inv_expression=self.inv_expression,
            domain=self.domain,
            rng=self.range,
            **self.attrs,
        )
        cal.set_points(cal.points_x, cal.points_y)
        return cal

    def __call__(self, x):
        """Call the calibration function.

        Parameters
        ----------
        x : float or array_like
            The scalar argument(s) to the function (e.g., raw channel).

        Returns
        -------
        calibration : float or np.ndarray
            The value(s) of the calibration function at x.
        """
        return _eval_expression(
            self.expression,
            self.params,
            x,
            aux_params=self.aux_params,
            domain=self.domain,
            rng=self.range,
        )

    def inverse(self, y, x0=None, **kwargs):
        """Call the inverse of the calibration function.

        Parameters
        ----------
        y : float or array_like
            The value of the calibration function that we want to find the
            argument for.
        x0 : float or array_like
            A guess of the inverse of the function. If inverse expression
            does not exist, this can speed up the process of calculating
            the inverse.
        kwargs : dict
            Kwargs to be sent to scipy.optimize.root_scalar.

        Returns
        -------
        x : float or np.ndarray
            The argument of the inverse of the calibration function at y.
        """
        if np.any((y < self.range[0]) | (self.range[1] < y)):
            raise CalibrationError(f"Value {y} is outside the range {self.range}")
        if self.inv_expression is None:
            bracket = self.domain
            if isinstance(y, float):
                result = scipy.optimize.root_scalar(
                    lambda _x: self(_x) - y, x0=x0, bracket=bracket, **kwargs
                )
                x = result.root
            else:
                y = np.asarray(y)
                x = np.zeros_like(y)
                if x0 is not None:
                    x0 = np.asarray(x0)
                for j in range(len(x)):
                    if x0 is not None:
                        x0j = x0[j]
                    else:
                        x0j = None
                    result = scipy.optimize.root_scalar(
                        lambda _x: self(_x) - y[j], x0=x0j, bracket=bracket, **kwargs
                    )
                    x[j] = result.root
        else:
            x = _eval_expression(
                self.inv_expression,
                self.params,
                y,
                ind_var="y",
                aux_params=self.aux_params,
                domain=self.range,  # domain and range are swapped for inverse
                rng=self.domain,
            )
        # perform a final check on the calculated inverse
        if not np.allclose(self(x), y):
            raise CalibrationError(
                f"Inverse function did not result in matching y values:\n{self(x)}\n{y}"
            )
        return x

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
            raise CalibrationError('Expected dataset "params"')
        if "expression" not in dsets:
            raise CalibrationError('Expected dataset "expression"')
        if "domain" not in dsets:
            raise CalibrationError('Expected dataset "domain"')
        if "range" not in dsets:
            raise CalibrationError('Expected dataset "range"')
        unexpected = set(dsets.keys()) - {
            "expression",
            "params",
            "aux_params",
            "inv_expression",
            "domain",
            "range",
            "points_x",
            "points_y",
            "weights",
        }
        if len(unexpected) > 0:
            raise CalibrationError(f"Unexpected dataset names in file: {unexpected}")
        if "aux_params" in dsets:
            aux_params = dsets["aux_params"][()]
        else:
            aux_params = None
        expr = io.h5.ensure_string(dsets["expression"])
        if "inv_expression" in dsets:
            inv_expr = io.h5.ensure_string(dsets["inv_expression"])
        else:
            inv_expr = None
        cal = cls(
            expr,
            dsets["params"],
            aux_params=aux_params,
            inv_expression=inv_expr,
            domain=dsets["domain"],
            rng=dsets["range"],
            **attrs,
        )
        if "points_x" in dsets and "points_y" in dsets and "weights" in dsets:
            cal.set_points(dsets["points_x"], dsets["points_y"], dsets["weights"])
        elif "points_x" in dsets and "points_y" in dsets:
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
            "domain": self.domain,
            "range": self.range,
            "points_x": self.points_x,
            "points_y": self.points_y,
            "weights": self.weights,
        }
        if len(self.aux_params) > 0:
            dsets["aux_params"] = self.aux_params
        if self.inv_expression is not None:
            dsets["inv_expression"] = self.inv_expression
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
            weights=self.weights,
            params0=self.params,
            aux_params=self.aux_params,
            domain=self.domain,
            rng=self.range,
            **kwargs,
        )
        self.params = params

    def fit_points(self, points_x, points_y, weights=None, params0=None, **kwargs):
        """Set the calibration point values and fit them.

        Convenience function for calling set_points() and fit().

        Parameters
        ----------
        points_x : float or array_like
            The x-value or values of calibration points
        points_y : float or array_like
            The y-value or values of calibration points
        weights : float or array_like
            The weights to be used when fitting. Fitting will minimize the sum
            of squared errors weighted by these weights, i.e.,
            sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
            ones with the same length as the points.
        params0 : float or array_like
            Initial guesses for the parameters. By default an array of ones
            with its length inferred from the number of parameters
            referenced in the expression.
        kwargs : dict
            Kwargs to pass to the minimization routine.
        """
        self.set_points(points_x=points_x, points_y=points_y, weights=weights)
        if params0 is not None:
            self.params = params0
        self.fit(**kwargs)

    @classmethod
    def from_points(
        cls,
        expression,
        points_x,
        points_y,
        weights=None,
        params0=None,
        aux_params=None,
        domain=DEFAULT_DOMAIN,
        rng=DEFAULT_RANGE,
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
        weights : float or array_like
            The weights to be used when fitting. Fitting will minimize the sum
            of squared errors weighted by these weights, i.e.,
            sum_i (w_i*(y_i - f(x_i))**2)). By default will use an array of
            ones with the same length as the points.
        params0 : float or array_like
            Initial guesses for the parameters. By default an array of ones
            with its length inferred from the number of parameters
            referenced in the expression.
        aux_params : array_like
            Auxiliary floating point parameters for the calibration function,
            referred to by the symbol "a". By default an empty array.
        domain : array_like
            The domain of the function. Will raise an error if the independent
            variable is outside this interval. Must be finite. By default
            DEFAULT_DOMAIN.
        rng : array_like
            The range of the function. Expression outputs will be clipped to
            this interval. Must be finite. By default DEFAULT_RANGE.
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
        points_x, points_y, weights = _check_points(
            points_x, points_y, weights=weights, domain=domain, rng=rng
        )
        params = _fit_expression(
            expression,
            points_x,
            points_y,
            weights=weights,
            params0=params0,
            aux_params=aux_params,
            domain=domain,
            rng=rng,
            **fit_kwargs,
        )
        cal = cls(
            expression, params, aux_params=aux_params, domain=domain, rng=rng, **attrs
        )
        cal.set_points(points_x, points_y, weights)
        return cal

    @classmethod
    def from_linear(cls, params, domain=DEFAULT_DOMAIN, rng=DEFAULT_RANGE, **attrs):
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
        return cls(expr, params, domain=domain, rng=rng, **attrs)

    @classmethod
    def from_polynomial(cls, params, domain=DEFAULT_DOMAIN, rng=DEFAULT_RANGE, **attrs):
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
        return cls(expr, params, domain=domain, rng=rng, **attrs)

    @classmethod
    def from_sqrt_polynomial(
        cls, params, domain=DEFAULT_DOMAIN, rng=DEFAULT_RANGE, **attrs
    ):
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
        return cls(expr, params, domain=domain, rng=rng, **attrs)

    @classmethod
    def from_interpolation(
        cls, points_x, points_y, domain=DEFAULT_DOMAIN, rng=DEFAULT_RANGE, **attrs
    ):
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
        points_x, points_y, _ = _check_points(
            points_x, points_y, domain=domain, rng=rng
        )
        if len(points_x) < 2:
            raise CalibrationError("Interpolated calibration expects at least 2 points")
        # place the points into array of auxiliary params "a"
        aux_params = np.array([points_x, points_y])
        expr = ""
        expr += (
            "scipy.interpolate.interp1d(a[0, :], a[1, :], fill_value='extrapolate')(x)"
        )
        return cls(expr, [], aux_params=aux_params, domain=domain, rng=rng, **attrs)

    @property
    def fit_R_squared(self):
        """Calibration fit R^2 value.

        Reference
        ---------
        stackoverflow.com/questions/19189362
        """

        # residual sum of squares
        ss_res = np.sum(self.weights * (self.points_y - self.fit_y) ** 2)

        # total sum of squares
        ss_tot = np.sum(self.weights * (self.points_y - np.mean(self.points_y)) ** 2)

        # r-squared
        return 1 - (ss_res / ss_tot)

    @property
    def fit_chi_squared(self):
        """Calibration fit chi^2 value."""

        if self.points_y.shape != self.fit_y.shape:
            raise ValueError(
                "y and fit_y must have same shapes:", self.y.shape, self.fit_y.shape
            )
        # Mask out zeros
        fit_y = self.fit_y[self.points_y > 0]
        points_y = self.points_y[self.points_y > 0]
        weights = self.weights[self.points_y > 0]
        return np.sum(weights * (points_y - fit_y) ** 2)

    @property
    def fit_degrees_of_freedom(self):
        """Calibration fit number of degrees of freedom."""
        return len(self.points_x) - len(self.params)

    @property
    def fit_reduced_chi_squared(self):
        """Calibration fit reduced chi^2 value."""
        return self.fit_chi_squared / self.fit_degrees_of_freedom

    def plot(self, ax=None):
        """Plot the calibration, and residuals if points exist.

        Parameters
        ----------
        ax : np.ndarray of shape (2,), or matplotlib axes object, optional
            Plot axes to use. If None, create new axes.
        """

        # Handle whether we have fit points or just a fit function
        has_points = self.points_x.size > 0

        if ax is None:
            fig, ax = plt.subplots(1 + has_points, 1, sharex=True)

        if has_points:
            assert ax.shape == (2,)
            ax_cal, ax_res = ax
            xmin, xmax = self.points_x.min(), self.points_x.max()
        else:
            ax_cal = ax
            xmin, xmax = self.domain

        # Plot calibration curve
        xx = np.linspace(xmin, xmax, 1000)
        yy = self(xx)
        ax_cal.plot(xx, yy, alpha=1.0 - 0.7 * has_points)
        ax_cal.set_ylabel("$y$")

        if has_points:
            # Plot calibration points
            ax_cal.scatter(self.points_x, self.points_y)

            # Plot residuals
            ax_res.scatter(self.points_x, self(self.points_x) - self.points_y)
            ax_res.set_xlabel("$x$")
            ax_res.set_ylabel("$y-x$")
            ax_res.axhline(0, linestyle="dashed", linewidth=1, c="k")
        else:
            ax_cal.set_xlabel("x")
