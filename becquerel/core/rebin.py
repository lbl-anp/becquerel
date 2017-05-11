import numpy as np
import numba as nb


def _check_ndim_and_dtype(arr, ndim, dtype, arr_name='array'):
    assert isinstance(arr, np.ndarray), \
        '{} is not a numpy array: {}'.format(arr_name, arr)
    assert arr.ndim == ndim, \
        '{}({}) is not {}D'.format(arr_name, arr.shape, ndim)
    assert np.issubdtype(arr.dtype, dtype), \
        '{}({}) is not type: {}'.format(arr_name, arr.dtype, dtype)


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


@nb.jit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
        locals={'in_idx': nb.u4, 'out_idx': nb.u4, 'cnts': nb.f8,
                'slope': nb.f8, 'offset': nb.f8, 'low': nb.f8, 'high': nb.f8},
        nopython=True)
def _rebin(in_spectrum, in_edges, out_spectrum, out_edges, slopes):
    # Input bin
    in_idx = 1
    # For each output bin
    for out_idx in range(len(out_spectrum)):
        # Skip output bin if above input edge range
        if out_edges[out_idx] > in_edges[-1]:
            continue
        # If output right edge above last input edge (NOTE needed?)
        # if out_top > in_edges[-1]:
        #     out_top = in_edges[-1]
        # Find index of input edge below or equal to output edge
        while in_edges[in_idx] < out_edges[out_idx]:
            in_idx += 1
        in_idx -= 1
        # For each input bin overlapping output bin
        while (in_idx < len(in_spectrum)) and \
              (in_edges[in_idx] < out_edges[out_idx + 1]):
            # Input bin data
            cts = in_spectrum[in_idx]
            slope = slopes[in_idx]
            # Linear offset
            offset = _linear_offset(slope, cts, in_edges[in_idx],
                                    in_edges[in_idx + 1])
            # High edge for interpolation
            high = in_edges[in_idx + 1]
            if out_edges[out_idx + 1] < high:
                high = out_edges[out_idx + 1]
            # Low edge for interpolation
            low = in_edges[in_idx]
            if out_edges[out_idx] > low:
                low = out_edges[out_idx]
            # Calc counts for this bin
            out_spectrum[out_idx] += _counts(slope, offset, cts, low, high)
            # Increment variables
            low = high
            in_idx += 1
        if in_idx == 0:
            in_idx = 1
    return out_spectrum


def rebin(in_spectrum, in_edges, out_edges, slopes=None):
    """
    Spectrum rebining via interpolation.

    Args:
      in_spectrum (np.ndarray): an array of input spectrum counts
      in_edges (np.ndarray): an array of the input bin edges
                             (len(in_spectrum) + 1)
      out_edges (np.ndarray): an array of the output bin edges
      slopes (np.ndarray): an array of input bin slopes for quadratic
                           interpolation

        If not none, should have length of (len(data) + 1)
      input_file_object: a parser file object (optional)

    Raises:
      SpectrumError: for bad input arguments
    """
    # Init slopes
    if slopes is None:
        slopes = np.zeros_like(in_spectrum, dtype=np.float)
    # Check for inputs
    _check_ndim_and_dtype(in_spectrum, 1, np.float, 'in_spectrum')
    _check_ndim_and_dtype(in_edges, 1, np.float, 'in_edges')
    _check_ndim_and_dtype(out_edges, 1, np.float, 'out_edges')
    _check_ndim_and_dtype(slopes, 1, np.float, 'slopes')
    # Check slopes
    assert slopes.shape == in_spectrum.shape, \
        "shape of slopes({}) differs from in_spectra({})".format(
            slopes.shape, in_spectrum.shape)
    # Init output
    out_spectrum = np.zeros(out_edges.shape[0] - 1, dtype=np.float)
    # Check input spectrum
    assert in_spectrum.shape[0] == in_edges.shape[0] - 1, \
        "in_spectrum({}) is not 1 channel shorter than in_edges({})".format(
            in_spectrum.shape, in_edges.shape)
    return _rebin(in_spectrum, in_edges, out_spectrum, out_edges, slopes)


@nb.jit(nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:, :], nb.f8[:],
                    nb.f8[:, :]),
        locals={'i': nb.u4}, nopython=True)
def _rebin2d(in_spectra, in_edges, out_spectra, out_edges, slopes):
    for i in np.arange(in_spectra.shape[0]):
        out_spectra[i, :] = _rebin(in_spectra[i, :], in_edges[i, :],
                                   out_spectra[i, :], out_edges, slopes[i, :])
    return out_spectra


def rebin2d(in_spectra, in_edges, out_edges, slopes=None):
    """
    Spectrum rebining via interpolation.

    Args:
      in_spectrum (np.ndarray): an array of input spectrum counts
      in_edges (np.ndarray): an array of the input bin edges
                             (len(in_spectrum) + 1)
      out_edges (np.ndarray): an array of the output bin edges
      slopes (np.ndarray): an array of input bin slopes for quadratic
                           interpolation

        If not none, should have length of (len(data) + 1)
      input_file_object: a parser file object (optional)

    Raises:
      SpectrumError: for bad input arguments
    """
    # Init slopes
    if slopes is None:
        slopes = np.zeros_like(in_spectra, dtype=np.float)
    # Check for inputs
    _check_ndim_and_dtype(in_spectra, 2, np.float, 'in_spectrum')
    _check_ndim_and_dtype(in_edges, 2, np.float, 'in_edges')
    _check_ndim_and_dtype(out_edges, 1, np.float, 'out_edges')
    _check_ndim_and_dtype(slopes, 2, np.float, 'slopes')
    # Check slopes
    assert slopes.shape == in_spectra.shape, \
        "shape of slopes({}) differs from in_spectra({})".format(
            slopes.shape, in_spectra.shape)
    # Init output
    out_spectra = np.zeros((in_spectra.shape[0], out_edges.shape[0] - 1),
                           dtype=np.float)
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
    return _rebin2d(in_spectra, in_edges, out_spectra, out_edges, slopes)
