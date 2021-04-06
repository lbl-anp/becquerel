import pytest
import numpy as np
import matplotlib.pyplot as plt
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
    return request.param.astype(np.float)


@pytest.fixture(
    params=[
        np.linspace(0, 30, 31),
        np.linspace(-1, 30, 31),
        np.linspace(0, 30, 7),
        np.linspace(0, 30, 45),
    ],
    ids=[
        "1 keV bins",
        "1 keV bins with negative starting edge",
        "5 keV bins",
        ".66 keV bins",
    ],
)
def new_edges(request):
    return request.param.astype(np.float)


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


def make_fake_spec_array(lam, size, dtype=np.float):
    return np.random.poisson(lam=lam, size=size).astype(dtype)


class TestRebin:
    """Tests for core.rebin()"""

    def test_total_counts(self, lam, old_edges, new_edges):
        """Check total counts in spectrum data before and after rebin

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_total_counts(lam, old_edges, new_edges, "interpolation", float)
        self._test_total_counts(lam, old_edges, new_edges, "listmode", int)

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

    def _test_total_counts(self, lam, old_edges, new_edges, method, dtype):
        """Check total counts in spectrum data before and after rebin"""
        old_counts, new_counts = self._gen_old_new_counts(
            lam, old_edges, new_edges, method, dtype
        )
        assert np.isclose(old_counts.sum(), new_counts.sum())

    def test_rebin2d_counts(self, lam, old_edges, new_edges):
        """Check total counts in spectrum data before and after rebin

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_total_counts_2d(lam, old_edges, new_edges, "interpolation", "float")
        self._test_total_counts_2d(lam, old_edges, new_edges, "listmode", "int")

    def _test_total_counts_2d(self, lam, old_edges, new_edges, method, dtype):
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
        )
        assert np.allclose(old_counts_2d.sum(axis=1), new_counts_2d.sum(axis=1))

    def test_unchanged_binning(self, lam, old_edges):
        """Check counts in spectrum is the same after a identity/null rebin
        (i.e. same bin edges for old and new)

        interpolation and listmode are separate as the latter requires ints
        """
        self._test_unchanged_binning(lam, old_edges, "interpolation", "float")
        self._test_unchanged_binning(lam, old_edges, "listmode", "int")

    def _test_unchanged_binning(self, lam, edges, method, dtype):
        old_counts, new_counts = self._gen_old_new_counts(
            lam, edges, edges, method, dtype
        )
        assert np.allclose(old_counts, new_counts)

    def _gen_old_new_counts(self, lam, old_edges, new_edges, method, dtype):
        """ generate old and new counts (1D)"""
        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=dtype)
        new_counts = bq.core.rebin.rebin(
            old_counts, old_edges, new_edges, method=method, zero_pad_warnings=False
        )
        return (old_counts, new_counts)

    def test_subset_bin_edges(self, lam, old_edges):
        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=float)
        new_counts = bq.core.rebin.rebin(
            old_counts, old_edges, old_edges[1:-1], method="interpolation"
        )
        assert np.isclose(np.sum(old_counts), np.sum(new_counts))
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
