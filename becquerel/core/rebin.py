import numpy as np
import numba as nb


def _check_ndim_and_dtype(arr, ndim, dtype, arr_name='array'):
    assert isinstance(arr, np.ndarray), \
        '{} is not a numpy array: {}'.format(arr_name, arr)
    assert arr.ndim == ndim, \
        '{}({}) is not {}D'.format(arr_name, arr.shape, ndim)
    assert np.issubdtype(arr.dtype, dtype), \
        '{}({}) is not type: {}'.format(arr_name, arr.dtype, dtype)


def _check_strictly_increasing(arr, arr_name='array'):
    assert np.all(np.diff(arr) > 0), \
        "{} is not strictly incraesing: {}".format(arr_name, arr)


@nb.jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _linear_offset(slope, cts, low, high):
    """
    Calculate the offset of the linear aproximation of slope when splitting
    counts between bins.
    """
    if np.abs(slope) < 1e-6:
        offset = cts / (high - low)
    else:
        offset = (cts - slope / 2. * (high**2 - low**2)) / (high - low)
    return offset


@nb.jit(nb.f8(nb.f8, nb.f8, nb.f8), nopython=True)
def _slope_integral(x, m, b):
    '''
    Calculate the integral of our quadratic given some slope and offset.
    '''
    return m * x**2 / 2 + b * x


@nb.jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _counts(slope, offset, cts, low, high):
    '''
    Computes the area under a linear approximation of the changing count
    rate in the vincity of relevant bins.  Edges of this integration
    are low and high while offset is provided from _linear_offset and cts
    from the bin being partitioned.
    '''
    return (_slope_integral(high, slope, offset) -
            _slope_integral(low, slope, offset))


@nb.jit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        locals={'in_idx': nb.u4, 'out_idx': nb.u4, 'counts': nb.f8,
                'slope': nb.f8, 'offset': nb.f8, 'low': nb.f8, 'high': nb.f8},
        nopython=True)
def _rebin(in_spectrum, in_edges, out_edges, slopes):
    """


    Keeps a running counter of two loop indices: in_idx & out_idx
    """
    out_spectrum = np.zeros(out_edges.shape[0] - 1) # init output
    # in_idx: input bin or left edge
    #         init to the first in_bin which overlaps the 0th out_bin
    in_idx = max(0, np.searchsorted(in_edges, out_edges[0]) - 1)
    # Under-flow handling: Put all counts from in_bins that are completely to
    # the left of the 0th out_bin into the 0th out_bin
    out_spectrum[0] = np.sum(in_spectrum[:in_idx])
    # out_idx: output bin or left edge
    #          init to the first out_bin which overlaps the 0th in_bin
    out_idx = max(0, np.searchsorted(out_edges, in_edges[0]) - 1)
    # loop through input bins (starting from the above in_idx value):
    for in_idx in range(in_idx, len(in_spectrum)):
        # input bin info/data
        in_left_edge = in_edges[in_idx]
        in_right_edge = in_edges[in_idx + 1]
        counts = in_spectrum[in_idx]
        slope = slopes[in_idx]
        offset = _linear_offset(slope, counts, in_left_edge, in_right_edge)
        # loop through output bins that overlap with the current input bin
        for out_idx in range(out_idx, len(out_spectrum)):
            out_left_edge = out_edges[out_idx]
            out_right_edge = out_edges[out_idx + 1]
            if out_left_edge >= in_right_edge:
                out_idx -= 1  # rewind back to previous out_bin; not done yet
                break  # break out of out_bins loop, move on to next in_bin
            # Low edge for interpolation
            if out_idx == 0:
                # under-flow handling:
                # all counts in this in_bin goes into the 0th out_bin
                low = in_left_edge  # == min(in_left_edge, out_edges[0])
            else:
                low = max(in_left_edge, out_left_edge)
            # High edge for interpolation
            if out_idx == len(out_spectrum) - 1:
                # over-flow handling:
                # all counts in this in_bin goes into this out_bin
                high = in_right_edge
            else:
                high = min(in_right_edge, out_right_edge)
            # Calc counts for this bin
            out_spectrum[out_idx] += _counts(slope, offset, counts, low, high)
    return out_spectrum


def rebin(in_spectrum, in_edges, out_edges, slopes=None):
    """
    Spectrum rebining via interpolation.

    Args:
        in_spectrum (np.ndarray): an array of input spectrum counts
            [num_channels_in]
        in_edges (np.ndarray): an array of the input bin edges
            [num_channels_in + 1]
        out_edges (np.ndarray): an array of the output bin edges
            [num_channels_out]
        slopes (np.ndarray|None): (optional) an array of input bin slopes for
            quadratic interpolation
            [num_channels_in + 1]

    Raises:
        AssertionError: for bad input arguments
    """
    # Init slopes
    if slopes is None:
        slopes = np.zeros_like(in_spectrum, dtype=np.float)
    # Check for inputs
    _check_ndim_and_dtype(in_spectrum, 1, np.float, 'in_spectrum')
    _check_ndim_and_dtype(in_edges, 1, np.float, 'in_edges')
    _check_ndim_and_dtype(out_edges, 1, np.float, 'out_edges')
    _check_ndim_and_dtype(slopes, 1, np.float, 'slopes')
    _check_strictly_increasing(in_edges, 'in_edges')
    _check_strictly_increasing(out_edges, 'out_edges')
    # Check slopes
    assert slopes.shape == in_spectrum.shape, \
        "shape of slopes({}) differs from in_spectra({})".format(
            slopes.shape, in_spectrum.shape)
    # Check input spectrum
    assert in_spectrum.shape[0] == in_edges.shape[0] - 1, \
        "in_spectrum({}) is not 1 channel shorter than in_edges({})".format(
            in_spectrum.shape, in_edges.shape)
    return _rebin(in_spectrum, in_edges, out_edges, slopes)


@nb.jit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]),
        locals={'i': nb.u4}, nopython=True)
def _rebin2d(in_spectra, in_edges, out_edges, slopes):
    # Init output
    out_spectra = np.zeros((in_spectra.shape[0], out_edges.shape[0] - 1))
    for i in np.arange(in_spectra.shape[0]):
        out_spectra[i, :] = _rebin(in_spectra[i, :], in_edges[i, :],
                                   out_edges, slopes[i, :])
    return out_spectra


def rebin2d(in_spectra, in_edges, out_edges, slopes=None):
    """
    Spectra rebining via interpolation.

    Args:
        in_spectra (np.ndarray): an array of individual input spectra counts
            [num_spectra, num_channels_in]
        in_edges (np.ndarray): an array of individual input bin edges
            [num_spectra, num_channels_in + 1]
        out_edges (np.ndarray): an array of the output bin edges
            [num_channels_out]
        slopes (np.ndarray|None): (optional) an array of individual input bin
            slopes for quadratic interpolation
            [num_spectra, num_channels_in + 1]

    Raises:
        AssertionError: for bad input arguments
    """
    # Init slopes
    if slopes is None:
        slopes = np.zeros_like(in_spectra, dtype=np.float)
    # Check for inputs
    _check_ndim_and_dtype(in_spectra, 2, np.float, 'in_spectrum')
    _check_ndim_and_dtype(in_edges, 2, np.float, 'in_edges')
    _check_ndim_and_dtype(out_edges, 1, np.float, 'out_edges')
    _check_ndim_and_dtype(slopes, 2, np.float, 'slopes')
    _check_strictly_increasing(in_edges, 'in_edges')
    _check_strictly_increasing(out_edges, 'out_edges')
    # Check slopes
    assert slopes.shape == in_spectra.shape, \
        "shape of slopes({}) differs from in_spectra({})".format(
            slopes.shape, in_spectra.shape)
    # Check number of spectra
    assert in_spectra.shape[0] == in_edges.shape[0], \
        "number of in_spectra({}) differs from number of in_edges({})".format(
            in_spectra.shape, in_edges.shape)
    assert in_spectra.shape[0] == in_edges.shape[0], \
        "number of in_spectra({}) differs from number of in_edges({})".format(
            in_spectra.shape, in_edges.shape)
    # Check len of spectra
    assert in_spectra.shape[1] == in_edges.shape[1] - 1, \
        "`in_spectra`({}) is not 1 channel shorter than `in_edges`({})".format(
            in_spectra.shape, in_edges.shape)
    return _rebin2d(in_spectra, in_edges, out_edges, slopes)
