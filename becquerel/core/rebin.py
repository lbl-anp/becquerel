"""
Original Author: Tenzing Joshi (thjoshi@lbl.gov)
"""
import numpy as np


def get_offset(slope, cts, low, high):
    '''
    Helper function used by apply_drift()
    Used to calculate the offset of the linear aproximation of slope when
    splitting counts between bins.
    '''
    if slope == 0.:
        b = cts / (high - low)
    else:
        b = (cts - slope / 2. * (high**2 - low**2)) / (high - low)
    return b


def slopeint(x, m, b):
    '''
    Helperfunction for compute_counts, which in tern is a helper for
    apply_drift
    Calculates the integral of our quadratic given some slope and offset.
    '''
    return m * x**2 / 2 + b * x


def compute_counts(slope, offset, cts, low, high):
    '''
    Helper function for apply_drift()
    Computes the area under a linear approximation of the changing count
    rate in the vincity of relevant bins.  Edges of this integration
    are low and high while offset is provided from get_offset and cts
    from the bin being partitioned.
    '''
    return slopeint(high, slope, offset) - slopeint(low, slope, offset)


def rebin(in_spectrum, in_edges, out_edges, slopes=None):
    in_spectrum = in_spectrum.astype(float)
    # Check input spectrum
    assert len(in_spectrum) == len(in_edges) - 1, \
        "`in_spectrum`({}) is not 1 len shorter than `in_edges`({})".format(
            len(in_spectrum), len(in_edges))
    out_spectrum = np.zeros(len(out_edges) - 1)

    # Init slopes
    if slopes is None:
        slopes = np.zeros(len(in_spectrum))
    else:
        assert len(slopes) == len(in_spectrum)

    in_start_idx = 1
    for idx in range(len(out_spectrum)):
        bottom = out_edges[idx]
        top = out_edges[idx + 1]

        if bottom > in_edges[-1]:
            continue

        # init variables before loop
        if top > in_edges[-1]:
            top = in_edges[-1]

        while in_edges[in_start_idx] < bottom:
            in_start_idx += 1
        in_start_idx -= 1

        while (in_start_idx < len(in_edges) + 1) and \
              (in_edges[in_start_idx] < top):

            cts = in_spectrum[in_start_idx]
            slope = slopes[in_start_idx]

            if top < in_edges[in_start_idx + 1]:
                high = top
            else:
                high = in_edges[in_start_idx + 1]

            offset = get_offset(slope,
                                cts,
                                in_edges[in_start_idx],
                                in_edges[in_start_idx + 1])

            out_spectrum[idx] += compute_counts(slope,
                                                offset,
                                                cts,
                                                bottom,
                                                high)
            # Increment variables
            bottom = high
            in_start_idx += 1
        in_start_idx = 1 if in_start_idx == 0 else in_start_idx

    return out_spectrum
