import datetime
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


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime.date(year=2023, month=6, day=14),
        datetime.datetime(year=2023, month=6, day=14, hour=0, minute=0, second=0),
        "2023_06_14_00_00_00",
        "2023-06-14T00:00:00.000Z-0000",  # ISO 8601, with timezone
        1686726000.0,  # UNIX timestamp
    ],
)
def test_handle_datetime(timestamp):
    expected = datetime.datetime(year=2023, month=6, day=14, hour=0, minute=0, second=0)
    assert bq.core.utils.handle_datetime(timestamp).replace(tzinfo=None) == expected


def test_handle_datetime_None():
    assert bq.core.utils.handle_datetime(None, allow_none=True) is None
    with pytest.raises(TypeError):
        bq.core.utils.handle_datetime(None, allow_none=False)


@pytest.mark.parametrize(
    "arg,error_type",
    [("2023_06_14-08_01_02", ValueError)],
)
def test_handle_datetime_err(arg, error_type):
    with pytest.raises(error_type):
        bq.core.utils.handle_datetime(arg)
