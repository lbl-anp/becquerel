import pytest
import numpy as np
import becquerel as bq


# ----------------------------------------------
#         Test utils
# ----------------------------------------------


def test_sqrt_bins():
    """Test basic functionality of utils.sqrt_bins."""
    edge_min = 0
    edge_max = 3000
    n_bins = 128
    be = bq.utils.sqrt_bins(edge_min, edge_max, n_bins)
    bc = (be[1:] + be[:-1]) / 2
    bw = np.diff(be)
    # compute slope of line
    m = np.diff(bw**2) / np.diff(bc)
    # assert that the square of the bin
    assert np.allclose(m[0], m)
    # negative edge_min
    with pytest.raises(AssertionError):
        be = bq.utils.sqrt_bins(-10, edge_max, n_bins)
    # edge_max < edge_min
    with pytest.raises(AssertionError):
        be = bq.utils.sqrt_bins(100, 50, n_bins)
