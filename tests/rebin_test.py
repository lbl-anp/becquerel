import matplotlib.pyplot as plt
import numpy as np
import pytest

import becquerel as bq

# ----------------------------------------------
#         Test Rebinning
# ----------------------------------------------


@pytest.fixture(
    params=[
        np.linspace(0, 30, 31),
        np.linspace(-1, 30, 31),
        np.linspace(0, 30, 11),
        np.linspace(0, 30, 40),
    ],
    ids=[
        "1 keV bins",
        "1 keV bins with negative starting edge",
        "3 keV bins",
        ".75 keV bins",
    ],
)
def old_edges(request):
    return request.param.astype(float)


@pytest.fixture(
    params=[
        np.linspace(0, 30, 31),
        np.linspace(-1, 30, 31),
        np.linspace(0, 30, 7),
        np.linspace(0, 30, 45),
        np.linspace(0, 25, 26),
        np.linspace(5, 30, 26),
        np.linspace(5, 25, 21),
    ],
    ids=[
        "1 keV bins",
        "1 keV bins with negative starting edge",
        "5 keV bins",
        ".66 keV bins",
        "1 keV bins, left edge == 0, right edge < 30",
        "1 keV bins, left edge > 0, right edge == 30",
        "1 keV bins, left edge > 0, right edge < 30",
    ],
)
def new_edges(request):
    return request.param.astype(float)


@pytest.fixture(
    params=[True, False],
    ids=[
        "include overflow events outside of the new binning in the first and last bins",
        "discard any events outside of the new binning",
    ],
)
def include_overflows(request):
    return request.param


@pytest.fixture(params=[1, 5, 25], ids=["sparse counts", "~5 counts", "~25 counts"])
def lam(request):
    """
    NB listmode is too slow, so the original test of 10k+ counts is too big
    """
    return request.param


@pytest.fixture
def old_spec_float(lam, old_edges):
    counts = np.random.poisson(lam=lam, size=len(old_edges) - 1).astype(float)
    return old_edges, counts


@pytest.fixture
def old_spec_int(lam, old_edges):
    counts = np.random.poisson(lam=lam, size=len(old_edges) - 1).astype(int)
    return old_edges, counts


def make_fake_spec_array(lam, size, dtype=float):
    return np.random.poisson(lam=lam, size=size).astype(dtype)


class TestRebin:
    """Tests for core.rebin()"""

    def test_total_counts(self, lam, old_edges, new_edges, include_overflows):
        """Check total counts in spectrum data before and after rebin

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_total_counts(
            lam, old_edges, new_edges, "interpolation", float, include_overflows
        )
        self._test_total_counts(
            lam, old_edges, new_edges, "listmode", int, include_overflows
        )

    def test_total_counts_listmode_float(self, lam, old_edges, new_edges):
        """Check that listmode rebinning raises a warning if counts are floats
        and have non-zero decimal residuals"""
        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=float)
        old_counts += 0.1
        with pytest.warns(bq.RebinWarning):
            bq.core.rebin.rebin(
                old_counts,
                old_edges,
                new_edges,
                method="listmode",
                zero_pad_warnings=False,
            )

    def _test_total_counts(
        self, lam, old_edges, new_edges, method, dtype, include_overflows
    ):
        """Check total counts in spectrum data before and after rebin"""
        old_counts, new_counts = self._gen_old_new_counts(
            lam, old_edges, new_edges, method, dtype, include_overflows
        )
        if include_overflows:
            assert np.isclose(old_counts.sum(), new_counts.sum())
        else:
            assert int(old_counts.sum()) >= int(new_counts.sum())

    def test_rebin2d_counts(self, lam, old_edges, new_edges, include_overflows):
        """Check total counts in spectrum data before and after rebin

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_total_counts_2d(
            lam, old_edges, new_edges, "interpolation", "float", include_overflows
        )
        self._test_total_counts_2d(
            lam, old_edges, new_edges, "listmode", "int", include_overflows
        )

    def _test_total_counts_2d(
        self, lam, old_edges, new_edges, method, dtype, include_overflows
    ):
        """Check total counts in spectra data before and after rebin"""

        nspectra = 20
        old_counts_2d = make_fake_spec_array(
            lam=lam, size=(nspectra, len(old_edges) - 1), dtype=dtype
        )
        old_edges_2d = np.repeat(old_edges[np.newaxis, :], nspectra, axis=0)
        new_counts_2d = bq.core.rebin.rebin(
            old_counts_2d,
            old_edges_2d,
            new_edges,
            method=method,
            zero_pad_warnings=False,
            include_overflows=include_overflows,
        )
        if include_overflows:
            assert np.allclose(old_counts_2d.sum(axis=1), new_counts_2d.sum(axis=1))
        else:
            assert np.all(
                old_counts_2d.sum(axis=1).astype(int)
                >= new_counts_2d.sum(axis=1).astype(int)
            )

    def test_unchanged_binning(self, lam, old_edges, include_overflows):
        """Check counts in spectrum is the same after a identity/null rebin
        (i.e. same bin edges for old and new)

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_unchanged_binning(
            lam, old_edges, "interpolation", "float", include_overflows
        )
        self._test_unchanged_binning(
            lam, old_edges, "listmode", "int", include_overflows
        )

    def _test_unchanged_binning(self, lam, edges, method, dtype, include_overflows):
        old_counts, new_counts = self._gen_old_new_counts(
            lam, edges, edges, method, dtype, include_overflows
        )
        assert np.allclose(old_counts, new_counts)

    def _gen_old_new_counts(
        self, lam, old_edges, new_edges, method, dtype, include_overflows
    ):
        """generate old and new counts (1D)"""
        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=dtype)
        new_counts = bq.core.rebin.rebin(
            old_counts,
            old_edges,
            new_edges,
            method=method,
            zero_pad_warnings=False,
            include_overflows=include_overflows,
        )
        return (old_counts, new_counts)

    def test_subset_bin_edges(self, lam, old_edges, include_overflows):
        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=float)
        new_counts = bq.core.rebin.rebin(
            old_counts,
            old_edges,
            old_edges[1:-1],
            method="interpolation",
            include_overflows=include_overflows,
        )
        if include_overflows:
            assert np.isclose(np.sum(old_counts), np.sum(new_counts))
        else:
            assert int(np.sum(old_counts)) >= int(np.sum(new_counts))
        assert np.allclose(old_counts[2:-2], new_counts[1:-1])

    def test_overlap_warnings(self, old_spec_float):
        old_edges, old_counts = old_spec_float
        new_edges_left = np.linspace(old_edges[0] - 1.0, old_edges[-1], 10)
        new_edges_right = np.linspace(old_edges[0], old_edges[-1] + 1, 10)
        new_edges_both = np.linspace(old_edges[0] - 1.0, old_edges[-1] + 1, 10)
        with pytest.warns(bq.RebinWarning):
            bq.rebin(old_counts, old_edges, new_edges_left, zero_pad_warnings=True)
        with pytest.warns(bq.RebinWarning):
            bq.rebin(old_counts, old_edges, new_edges_right, zero_pad_warnings=True)
        with pytest.warns(bq.RebinWarning):
            bq.rebin(old_counts, old_edges, new_edges_both, zero_pad_warnings=True)

    def test_overlap_errors(self, old_spec_float):
        old_edges, old_counts = old_spec_float
        new_edges_left = np.linspace(
            old_edges[0] - 10, old_edges[0] - 1, 10, dtype=float
        )
        new_edges_right = np.linspace(
            old_edges[-1] + 1, old_edges[-1] + 10, 10, dtype=float
        )
        with pytest.raises(bq.RebinError):
            bq.rebin(old_counts, old_edges, new_edges_left)
        with pytest.raises(bq.RebinError):
            bq.rebin(old_counts, old_edges, new_edges_right)

    def test_negative_input_listmode(self, old_spec_int, new_edges):
        old_edges, old_counts = old_spec_int
        old_counts[0] = -1
        with pytest.raises(bq.RebinError):
            bq.rebin(
                old_counts,
                old_edges,
                new_edges,
                method="listmode",
                zero_pad_warnings=False,
            )

    @pytest.mark.plottest
    def test_uncal_spectrum_counts(self, old_spec_float):
        """Plot the old and new spectrum bins as a sanity check"""

        old_edges, old_counts = old_spec_float
        new_edges = old_edges + 0.3
        new_counts = bq.rebin(old_counts, old_edges, new_edges, zero_pad_warnings=False)
        plt.figure()
        plt.plot(
            *bq.core.plotting.SpectrumPlotter.bin_edges_and_heights_to_steps(
                old_edges, old_counts
            ),
            color="dodgerblue",
            label="original",
        )
        plt.plot(
            *bq.core.plotting.SpectrumPlotter.bin_edges_and_heights_to_steps(
                new_edges, new_counts
            ),
            color="firebrick",
            label="rebinned",
        )
        plt.show()


@pytest.mark.parametrize(
    "edges2, counts2, method, include_overflows",
    [
        (
            np.linspace(2, 9, num=8),
            np.array([3000, 1000, 1000, 1000, 1000, 1000, 2000]),
            "interpolation",
            True,
        ),
        (
            np.linspace(2, 9, num=8),
            np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000]),
            "interpolation",
            False,
        ),
        (
            np.linspace(2, 9, num=8),
            np.array([3000, 1000, 1000, 1000, 1000, 1000, 2000]),
            "listmode",
            True,
        ),
        (
            np.linspace(2, 9, num=8),
            np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000]),
            "listmode",
            False,
        ),
        (
            np.linspace(2.5, 9.5, num=8),
            np.array([3500, 1000, 1000, 1000, 1000, 1000, 1500]),
            "interpolation",
            True,
        ),
        (
            np.linspace(2.5, 9.5, num=8),
            np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000]),
            "interpolation",
            False,
        ),
    ],
)
def test_rebin_include_overflows(edges2, counts2, method, include_overflows):
    """Perform specific numerical rebinning tests."""

    counts = 1000 * np.ones(10)
    spec = bq.Spectrum(counts=counts)
    cal = bq.Calibration("p[0] * x", [1.0])
    spec.apply_calibration(cal)

    spec2 = spec.rebin(edges2, method=method, include_overflows=include_overflows)
    assert np.allclose(counts2, spec2.counts_vals)
