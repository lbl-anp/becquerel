"""General utility functions to be shared among core modules."""

from __future__ import print_function
from uncertainties import UFloat, unumpy
import numpy as np


class UncertaintiesError(Exception):
    """Raised when uncertainties are badly specified in an input."""

    pass


def all_ufloats(x):
    """Check if every element of x is a UFloat.

    Args:
      x: an iterable or scalar to check

    Raises:
      UncertaintiesError: if some but not all elements of x are UFloats

    Returns:
      True if all elements of x are UFloats.
      False if no elements of x are UFloats.
    """

    try:
        are_ufloats = [isinstance(xi, UFloat) for xi in x]
    except TypeError:
        return isinstance(x, UFloat)
    else:
        if all(are_ufloats):
            return True
        elif any(are_ufloats):
            raise UncertaintiesError(
                'Input should be all UFloats or no UFloats')
        else:
            return False


def handle_uncs(x_array, x_uncs, default_unc_func):
    """Handle two methods of specifying uncertainties (UFloats or manually).

    Args:
      x_array: a list/tuple/array that may contain UFloats
      x_uncs: a list/tuple/array that may contain manual uncertainty values
      default_unc_func: a function that will take as input x_array and
        return a set of default values for x_uncs (for if x_uncs not specified
        and x_array not UFloats)

    Raises:
      UncertaintiesError: if both UFloats and manual uncertainties are
        specified

    Returns:
      a np.array of UFloats
    """

    ufloats = all_ufloats(x_array)

    if ufloats and x_uncs is None:
        return np.array(x_array)
    elif ufloats:
        raise UncertaintiesError('Specify uncertainties with UFloats or ' +
                                 'by separate argument, but not both')
    elif x_uncs is not None:
        return unumpy.uarray(x_array, x_uncs)
    else:
        return unumpy.uarray(x_array, default_unc_func(x_array))
