"""General utility functions to be shared among core modules."""

from __future__ import print_function
import sys
import datetime
from dateutil.parser import parse as dateutil_parse
from uncertainties import UFloat, unumpy
import numpy as np
try:
    # Python 3.x
    from collections.abc import Iterable
except ImportError:
    # Python 2.x
    from collections import Iterable

EPS = np.finfo(float).eps

VECTOR_TYPES = (list, tuple, np.ndarray)

if sys.version_info < (3, 2):

    class ResourceWarning(Warning):
        pass

else:

    ResourceWarning = ResourceWarning

if sys.version_info < (3,):

    def isstring(s):
        """Test for strings in python2"""
        return isinstance(s, basestring)

else:

    def isstring(s):
        """Test for strings in python3"""
        return isinstance(s, str)


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
        return np.asarray(x_array)
    elif ufloats:
        raise UncertaintiesError('Specify uncertainties with UFloats or ' +
                                 'by separate argument, but not both')
    elif x_uncs is not None:
        return unumpy.uarray(x_array, x_uncs)
    else:
        return unumpy.uarray(x_array, default_unc_func(x_array))


def handle_datetime(input_time, error_name='datetime arg', allow_none=False):
    """Parse an argument as a datetime, date+time string, or None.

    Args:
      input_time: the input argument to be converted to a datetime
      error_name: the name to be displayed if an error is raised.
        (default: 'datetime arg')
      allow_none: whether a None is allowed as an input and return value.
        (default: False)

    Raises:
      TypeError: if input_time is not a string, datetime, or None

    Returns:
      a datetime.datetime, or None
    """

    if isinstance(input_time, datetime.datetime):
        return input_time
    elif isstring(input_time):
        return dateutil_parse(input_time)
    elif input_time is None and allow_none:
        return None
    else:
        raise TypeError(
            'Unknown type for {}: {}'.format(error_name, input_time))


def bin_centers_from_edges(edges_kev):
    """Calculate bin centers from bin edges.

    Args:
      edges_kev: an iterable representing bin edge values

    Returns:
      np.array of length (len(edges_kev) - 1), representing bin center
        values with the same units as the input
    """

    edges_kev = np.array(edges_kev)
    centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
    return centers_kev
