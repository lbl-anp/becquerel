import numpy as np
import numba as nb


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
    in_spectrum = in_spectrum.astype(float)
    # Init output
    out_spectrum = np.zeros(len(out_edges) - 1)
    # Check input spectrum
    assert len(in_spectrum) == len(in_edges) - 1, \
        "`in_spectrum`({}) is not 1 len shorter than `in_edges`({})".format(
            len(in_spectrum), len(in_edges))
    # Init slopes
    if slopes is None:
        slopes = np.zeros(len(in_spectrum))
    else:
        assert len(slopes) == len(in_spectrum)
    return _rebin(in_spectrum, in_edges, out_spectrum, out_edges, slopes)
