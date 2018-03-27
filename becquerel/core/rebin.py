import numpy as np
import numba as nb


def _check_ndim(arr, ndim, arr_name='array'):
    """Check the dimensionality of a numpy array

    Check that the array arr has dimension ndim.

    Args:
      arr: numpy array for which dimensionality is checked
      ndim: a scalar number or an iterable
      arr_name: name of array, just for use in the AssertionError message

    Raises:
      AssertionError: if arr does not have dimension ndim
    """
    assert arr.ndim == ndim or arr.ndim in ndim, \
        '{}({}) is not {}D'.format(arr_name, arr.shape, ndim)


def _check_nonneg_monotonic_increasing(arr, arr_name='array'):
    """Check that a numpy array is non-negative and monotonically increasing

    Args:
      arr: numpy array for checking
      arr_name: name of array, just for use in the AssertionError message

    Raises:
      AssertionError: if arr has negative values or
                      is not monotonically increasing
    """
    # Check that elements along the last axis are increasing or
    # neighboring elements are equal
    assert np.all((np.diff(arr) > 0) | np.isclose(np.diff(arr), 0)), \
        "{} is not monotonically increasing: {}".format(arr_name, arr)
    # Check that first bin is nonnegative
    assert np.all((arr[..., 0] > 0) | np.isclose(arr[..., 0], 0)), \
        "{} has negative values: {}".format(arr_name, arr)


@nb.jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _linear_offset(slope, cts, low, high):
    """
    Calculate the offset of the linear aproximation of slope when splitting
    counts between bins.

    Args:
      slope:
      cts:
      low:
      high:

    Returns:
      the offset
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

    Args:
      x:
      m:
      b:
    '''
    return m * x**2 / 2 + b * x


@nb.jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _counts(slope, offset, cts, low, high):
    '''
    Computes the area under a linear approximation of the changing count
    rate in the vincity of relevant bins.  Edges of this integration
    are low and high while offset is provided from _linear_offset and cts
    from the bin being partitioned.

    Args:
      slope:
      offset:
      cts:
      low:
      high:
    '''
    return (_slope_integral(high, slope, offset) -
            _slope_integral(low, slope, offset))


@nb.jit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        locals={'in_idx': nb.u4, 'out_idx': nb.u4, 'counts': nb.f8,
                'slope': nb.f8, 'offset': nb.f8, 'low': nb.f8, 'high': nb.f8},
        nopython=True)
def _rebin_interpolation(in_spectrum, in_edges, out_edges, slopes):
    """
    Rebins a spectrum using linear interpolation.

    Keeps a running counter of two loop indices: in_idx & out_idx

    Args:

    Returns:
      1D numpy array of rebinned spectrum counts in each bin
    """
    out_spectrum = np.zeros(out_edges.shape[0] - 1)  # init output
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


@nb.jit(nb.i8[:](nb.i8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        locals={'energies': nb.f8[:], 'in_idx': nb.u4, 'energy_idx': nb.u4},
        nopython=True)
def _rebin_listmode(in_spectrum, in_edges, out_edges, slopes):
    """
    Stochastic rebinning method: spectrum-histogram to listmode then back

    TODO Assume piecewise constant (ie steps, flat within each bin)
         distribution for in_spectrum (for now). slopes is unused.

    Args:
      in_spectrum: iterable of input spectrum counts in each bin
      in_edges: iterable of input bin edges (len = len(in_spectrum) + 1)
      out_edges: iterable of output bin edges
      slopes: unused, just for keeping number of arguments the same as
              _rebin_interpolation()

    Returns:
      1D numpy array of rebinned spectrum counts in each bin
    """
    energies = np.zeros(np.sum(in_spectrum))
    energy_idx_start = 0
    # loop through input bins:
    # in_idx = index of in_bin or in_left_edge
    for in_idx in range(len(in_spectrum)):
        bin_counts = in_spectrum[in_idx]
        energy_idx_stop = energy_idx_start + bin_counts
        in_left_edge = in_edges[in_idx]
        in_right_edge = in_edges[in_idx + 1]
        energies[energy_idx_start:energy_idx_stop] = in_left_edge + (
            in_right_edge - in_left_edge) * np.random.rand(bin_counts)
        energy_idx_start = energy_idx_stop
    # bin the energies (drops the energies outside of the out-binning-range)
    out_spectrum = np.histogram(energies, bins=out_edges)[0]
    # add the under/flow counts back into the out_spectrum
    out_spectrum[0] += (energies < out_edges[0]).nonzero()[0].size
    out_spectrum[-1] += (energies > out_edges[-1]).nonzero()[0].size
    return out_spectrum


@nb.jit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]),
        locals={'i': nb.u4}, nopython=True)
def _rebin2d_interpolation(in_spectra, in_edges, out_edges, slopes):
    """
    Rebins a 2D array of spectra using linear interpolation

    N.B. Does not keep Poisson statistics

    Wrapper around _rebin_interpolation (1D) for the 2D case

    Args:
      in_spectra: np.2darray, shape (num_spectra, num_channels_in)
        array of input spectrum counts of shape
      in_edges: np.2darray, shape (num_spectra, num_channels_in + 1)
        array of the input bin edges of shape
      out_edges: np.1darray
        array of the output bin edges
      slopes:
    """
    # Init output
    out_spectra = np.zeros((in_spectra.shape[0], out_edges.shape[0] - 1))
    for i in np.arange(in_spectra.shape[0]):
        out_spectra[i, :] = _rebin_interpolation(
            in_spectra[i, :], in_edges[i, :], out_edges, slopes[i, :])
    return out_spectra


@nb.jit(nb.i8[:, :](nb.i8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:, :]),
        locals={'i': nb.u4}, nopython=True)
def _rebin2d_listmode(in_spectra, in_edges, out_edges, slopes):
    """
    Rebins a 2D array of spectra stochastically: histogram to listmode and back

    Wrapper around _rebin_listmode (1D) for the 2D case

    Args:
      in_spectra: np.2darray, shape (num_spectra, num_channels_in)
        array of input spectrum counts of shape
      in_edges: np.2darray, shape (num_spectra, num_channels_in + 1)
        array of the input bin edges of shape
      out_edges: np.1darray
        array of the output bin edges
      slopes: TODO unused, just for keeping number of arguments the same as
              _rebin_interpolation()

    Returns:
      np.2darray, shape (num_spectra, num_channels_in)
      The rebinned spectra
    """
    # Init output
    out_spectra = np.zeros((in_spectra.shape[0], out_edges.shape[0] - 1),
                           np.int64)
    for i in np.arange(in_spectra.shape[0]):
        out_spectra[i, :] = _rebin_listmode(in_spectra[i, :], in_edges[i, :],
                                            out_edges, slopes[i, :])
    return out_spectra


def rebin(in_spectra, in_edges, out_edges, method="interpolation",
          slopes=None):
    """
    Spectra rebining via deterministic or stochastic methods.

    Args:
        in_spectrum (np.ndarray): an array of input spectrum counts (1D or 2D)
            [num_spectra, num_channels_in] or [num_channels_in]
        in_edges (np.ndarray): an array of the input bin edges (1D or 2D)
            [num_spectra, num_channels_in + 1] or [num_channels_in + 1]
        out_edges (np.ndarray): an array of the output bin edges
            [num_channels_out]
        method (str): rebinning method
            "interpolation"
                Deterministic interpolation
            "listmode"
                Stochastic rebinning via conversion to listmode of energies
        slopes (np.ndarray|None): (optional) an array of input bin slopes for
            quadratic interpolation (1D or 2D)
            (only applies for "interpolation" method)
            [num_spectra, num_channels_in + 1] or [num_channels_in + 1]

    Raises:
        AssertionError: for bad input arguments

    Returns:
        The rebinned spectrum/a
    """
    method = method.lower()
    if method == "listmode":
        try:
            in_spectra = np.asarray(in_spectra).astype(
                np.int64, casting="safe", copy=False)
        except TypeError:
            raise ValueError(
                "in_spectrum can only contain ints for method listmode")
    else:
        in_spectra = np.asarray(in_spectra, np.float)
    in_edges = np.asarray(in_edges, np.float)
    out_edges = np.asarray(out_edges, np.float)
    # Check inputs
    # TODO check that in_spectrum are all >= 0
    _check_ndim(in_spectra, {1, 2}, 'in_spectra')
    # broadcast in_edges out to the dimensions of in_spectra
    # specifically for the case 1D -> 2D
    if (in_spectra.ndim == 2) and (in_edges.ndim == 1):
        # copy req'd: the readonly array doesn't work w/ numba
        in_edges = np.copy(
            np.broadcast_to(
                in_edges,
                (in_spectra.shape[0], in_edges.shape[-1])))
    _check_ndim(out_edges, 1, 'out_edges')
    _check_nonneg_monotonic_increasing(in_edges, 'in_edges')
    _check_nonneg_monotonic_increasing(out_edges, 'out_edges')
    # Init slopes
    if slopes is None:
        slopes = np.zeros_like(in_spectra, dtype=np.float)
    else:
        slopes = np.asarray(slopes, np.float)
    # Check slopes
    assert slopes.shape == in_spectra.shape, \
        "shape of slopes({}) differs from in_spectra({})".format(
            slopes.shape, in_spectra.shape)
    # Check len of spectra
    assert in_spectra.shape[-1] == in_edges.shape[-1] - 1, (
        "The last axis of in_spectra({}) is not"
        "1 channel shorter than in_edges({})"
    ).format(in_spectra.shape, in_edges.shape)
    # do rebinning by calling numba JIT-ed functions
    if in_spectra.ndim == 2:
        # Check number of spectra
        assert in_spectra.shape[0] == in_edges.shape[0], (
            "Number of in_spectra({}) differs from number of in_edges({})"
        ).format(in_spectra.shape, in_edges.shape)
    if method == "interpolation":
        if in_spectra.ndim == 2:
            return _rebin2d_interpolation(
                in_spectra, in_edges, out_edges, slopes)
        elif in_spectra.ndim == 1:
            return _rebin_interpolation(
                in_spectra, in_edges, out_edges, slopes)
    elif method == "listmode":
        if in_spectra.ndim == 2:
            return _rebin2d_listmode(in_spectra, in_edges, out_edges, slopes)
        elif in_spectra.ndim == 1:
            return _rebin_listmode(in_spectra, in_edges, out_edges, slopes)
    else:
        raise ValueError("{} is not a valid rebinning method".format(method))
