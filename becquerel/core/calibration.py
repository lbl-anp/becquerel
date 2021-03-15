"""Generic calibration class."""

import copy
import asteval
import black
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
            present in the expression. Can be a single-line formula like
            "p[0] + p[1] * x" or a code block.
        params : array_like
            List of floating point parameters for the calibration function
        attrs : dict
            Other information to be stored with the calibration.
        """
        if "expression" in attrs.keys():
            raise CalibrationError(
                f'Keyword "expression" cannot be used in attrs: {attrs}'
            )
        self.expression = expression
        self.params = params
        self.attrs = attrs

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
    def validate_expression(expr, params=None):
        """Perform checks on the expression.

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
        expr = black.format_str(expr, mode=black.FileMode())
        print("After black:       ", expr)

        # make sure all parentheses match
        left_parens = []
        for j, char in enumerate(expr):
            if char in ["(", "["]:
                left_parens.append(char)
            elif char == ")":
                assert left_parens[-1] == "("
                left_parens = left_parens[:-1]
            elif char == "]":
                assert left_parens[-1] == "["
                left_parens = left_parens[:-1]

        # make sure square brackets only occur with "p"
        for j in range(1, len(expr)):
            if expr[j] == "[":
                if expr[j - 1] != "p":
                    raise CalibrationError(
                        f"Character preceding '[' must be 'p':\n{expr[:j]}  {expr[j:]}"
                    )

        # make sure each parameter appears at least once
        tokens = expr.split("p[")
        print("tokens:", tokens)
        param_indices = [int(token.split("]")[0]) for token in tokens[1:]]
        param_indices = np.array(sorted(np.unique(param_indices)))
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
            if param_indices.max() != len(params) - 1:
                raise CalibrationError(
                    f"Maximum parameter index in expression is not {len(params) - 1}:\n{expr}\n{param_indices}"
                )
            if len(param_indices) != len(params):
                raise CalibrationError(
                    f"Not enough parameter indices in expression:\n{expr}\n{param_indices}"
                )

        # make sure the expression can be evaluated
        if params is not None:
            try:
                y = Calibration.eval_expression(expr, params, 200.0)
                print("y:", y)
            except (NotImplementedError,):
                raise CalibrationError(
                    f"Cannot evaluate expression for a float:\n{expr}\n{safe_eval.symtable['x']}"
                )
            try:
                y = Calibration.eval_expression(expr, params, [200.0, 500.0])
                print("y:", y)
            except (NotImplementedError,):
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
        return Calibration(self.expression, self.params, **self.attrs)

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
        assert len(dsets.keys()) == 1
        assert "params" in dsets.keys()
        assert "expression" in attrs.keys()
        expr = attrs.pop("expression")
        cal = cls(expr, dsets["params"], **attrs)
        return cal

    def write(self, name):
        """Write the class to HDF5.

        TODO: docstring
        """
        dsets = {"params": self.params}
        attrs = copy.deepcopy(self.attrs)
        attrs["expression"] = self.expression
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

