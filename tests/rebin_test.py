from __future__ import print_function
import pytest
import numpy as np
import matplotlib.pyplot as plt
import becquerel as bq


# ----------------------------------------------
#         Test Rebinning
# ----------------------------------------------


@pytest.fixture(params=[np.linspace(0, 3000, 3001),
                        np.linspace(0, 3000, 301),
                        np.linspace(0, 3000, 5000)],
                ids=["1 keV bins",
                     "10 keV bins",
                     "slightly smaller bins"])
def old_edges(request):
    return request.param.astype(np.float)


@pytest.fixture(params=[np.linspace(0, 3000, 3001),
                        np.linspace(0, 3010, 10000),
                        np.linspace(0, 3010, 17)],
                ids=["1 keV bins",
                     "small bins",
                     "large bins"])
def new_edges(request):
    return request.param.astype(np.float)


@pytest.fixture(params=[1, 50, 12555],
                ids=["sparse counts", "medium counts", "high counts"])
def lam(request):
    return request.param


def make_fake_spec_array(lam, size, dtype=np.float):
    return np.random.poisson(
        lam=lam, size=size).astype(dtype)


class TestRebin(object):
    """Tests for core.rebin()"""

    def test_rebin_counts_float(self, lam, old_edges, new_edges):
        """Check total counts in spectrum data before and after rebin"""

        old_counts = make_fake_spec_array(lam, len(old_edges) - 1)
        new_counts = bq.core.rebin.rebin(old_counts, old_edges, new_edges)
        assert np.isclose(old_counts.sum(), new_counts.sum())

    def test_rebin_counts_int(self, lam, old_edges, new_edges):
        """Check that rebin raises an error for counts as integers"""

        old_counts = make_fake_spec_array(lam, len(old_edges) - 1, dtype=int)
        new_counts = bq.core.rebin.rebin(old_counts, old_edges, new_edges)
        assert np.isclose(old_counts.sum(), new_counts.sum())

    def test_rebin2d_counts_float(self, lam, old_edges, new_edges):
        """Check total counts in spectra data before and after rebin"""

        nspectra = 20
        old_counts_2d = make_fake_spec_array(
            lam=lam, size=(nspectra, len(old_edges) - 1))
        old_edges_2d = np.repeat(old_edges[np.newaxis, :], nspectra, axis=0)
        new_counts_2d = bq.core.rebin.rebin(old_counts_2d,
                                            old_edges_2d,
                                            new_edges)
        assert np.allclose(old_counts_2d.sum(axis=1),
                           new_counts_2d.sum(axis=1))

    @pytest.mark.plottest
    def test_uncal_spectrum_counts(self, uncal_spec):
        """Plot the old and new spectrum bins as a sanity check"""

        old_edges = np.concatenate([
            uncal_spec.channels.astype('float') - 0.5,
            np.array([uncal_spec.channels[-1] + 0.5])])
        new_edges = old_edges + 0.3
        new_data = bq.core.rebin(uncal_spec.data, old_edges, new_edges)
        plt.figure()
        plt.plot(*bq.core.bin_edges_and_heights_to_steps(old_edges,
                                                         uncal_spec.data),
                 color='dodgerblue', label='original')
        plt.plot(*bq.core.bin_edges_and_heights_to_steps(new_edges,
                                                         new_data),
                 color='firebrick', label='rebinned')
        plt.show()
