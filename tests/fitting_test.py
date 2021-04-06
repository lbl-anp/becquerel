import os
import glob
import pytest
from copy import deepcopy
import numpy as np
import becquerel as bq

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "samples")

# TODO: use these for fitting actual data
SAMPLES = {}
for extension in [".spe", ".spc", ".cnf"]:
    filenames = glob.glob(os.path.join(SAMPLES_PATH, "*.*"))
    filenames_filtered = []
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        if ext.lower() == extension:
            filenames_filtered.append(filename)
    SAMPLES[extension] = filenames_filtered


def get_model_name(x):
    if isinstance(x, str):
        return x
    else:
        return " ".join(x)


def sim_data(x_min, x_max, y_func, num_x=200, binning="linear", **params):
    if binning == "linear":
        edges = np.linspace(x_min, x_max, num_x, dtype=np.float)
    elif binning == "sqrt":
        edges = np.linspace(np.sqrt(x_min), np.sqrt(x_max), num_x)
        edges = edges ** 2
    x = (edges[1:] + edges[:-1]) * 0.5
    dx = edges[1:] - edges[:-1]

    y_smooth = y_func(x=x, **params) * dx
    np.random.seed(1)
    y = np.random.poisson(y_smooth).astype(np.float)
    y_unc = np.sqrt(y)
    return {"x": x, "y": y, "y_unc": y_unc, "dx": dx}


def compare_params(true_params, fit_params, rtol, fitter):
    for p, v in fit_params.items():
        # TODO: Remove
        # if not np.isclose(v, true_params[p], rtol=rtol):
        #     fitter.custom_plot()
        #     plt.show()
        assert np.isclose(v, true_params[p], rtol=rtol), p


def compare_counts(fitter):
    data_counts = np.sum(fitter.y_roi)
    model_counts = np.sum(
        fitter.eval(fitter.x_roi, **fitter.result.best_values) * fitter.dx_roi
    )
    assert np.allclose(data_counts, model_counts, atol=1e-2)


# -----------------------------------------------------------------------------
# Simulated data generation
# TODO: discuss this 20% tol, maybe smaller tol for gauss?
# -----------------------------------------------------------------------------

HIGH_STAT_SIM_PARAMS = {
    "base_model_params": {
        "gauss": {
            "amp": 1e5,
            "mu": 100.0,
            "sigma": 5.0,
        },
        "gausserf": {
            "ampgauss": 1e5,
            "amperf": 1e4,
            "mu": 100.0,
            "sigma": 5.0,
        },
        "exp": {
            "lam": -5e1,
            "amp": 1e4,
        },
        "erf": {
            "amp": 1e4,
            "mu": 100.0,
            "sigma": 5.0,
        },
        "line": {
            "m": -10.0,
            "b": 1e4,
        },
        "expgauss": {"amp": 1e5, "mu": 100, "sigma": 5.0, "gamma": 0.25},
    },
    "setup": {
        "roi": (25, 175),
        "rtol": 40e-2,
        "sim_data_kwargs": {
            "x_min": 10.0,
            "x_max": 190.0,
            "num_x": 180,
        },
    },
    "models": [
        "gauss",
        ["gauss", "line"],
        "gausserf",
        ["gauss", "exp"],
        ["gausserf", "line"],
        # ['gauss', 'line', 'exp'],
        ["gausserf", "exp"],
        "expgauss",
    ],
    "fixture": {
        "params": [],
        "ids": [],
    },
    "methods": ["lmfit", "lmfit-pml"],
    "binnings": ["linear", "sqrt"],
}

for _e in HIGH_STAT_SIM_PARAMS["methods"]:
    for _m in HIGH_STAT_SIM_PARAMS["models"]:
        for _b in HIGH_STAT_SIM_PARAMS["binnings"]:
            _p = deepcopy(HIGH_STAT_SIM_PARAMS["setup"])
            _p["model"] = deepcopy(_m)
            _p["params"] = {}
            _p["method"] = _e
            _p["binning"] = _b
            if isinstance(_m, str):
                _i = deepcopy(_m)
                _m = [_m]
            else:
                _i = "".join([_bm.capitalize() for _bm in _m])
            for _bm in _m:
                for _bmp, _v in HIGH_STAT_SIM_PARAMS["base_model_params"][_bm].items():
                    _p["params"][f"{_bm}_{_bmp}"] = _v
            HIGH_STAT_SIM_PARAMS["fixture"]["params"].append(_p)
            HIGH_STAT_SIM_PARAMS["fixture"]["ids"].append(_i)


# HIGH_STAT_GAUSS_GAUSS_LINE = dict(
#     params=dict(
#         gauss0_amp=1e5,
#         gauss0_mu=60.,
#         gauss0_sigma=5.,
#         gauss1_amp=1e5,
#         gauss1_mu=120.,
#         gauss1_sigma=7.,
#         line_m=-10.,
#         line_b=1e4),
#     setup=dict(
#         x_min=0.0,
#         x_max=200.0,
#         num_x=200),
#     roi=(25, 175),
#     cls=bq.fitting.FitterGaussGaussLine,
#     rtol=2e-1,
#     name='FitterGaussGaussLine')
#
# HIGH_STAT_GAUSS_GAUSS_EXP = dict(
#     params=dict(
#         gauss0_amp=1e5,
#         gauss0_mu=60.,
#         gauss0_sigma=5.,
#         gauss1_amp=1e5,
#         gauss1_mu=120.,
#         gauss1_sigma=7.,
#         exp_lam=-5e1,
#         exp_amp=1e4),
#     setup=dict(
#         x_min=0.0,
#         x_max=200.0,
#         num_x=200),
#     roi=(25, 175),
#     cls=bq.fitting.FitterGaussGaussExp,
#     rtol=2e-1,
#     name='FitterGaussGaussExp')


@pytest.fixture(**HIGH_STAT_SIM_PARAMS["fixture"])
def sim_high_stat(request):
    """Fake data with high count statistics"""
    out = deepcopy(request.param)
    out["fitter"] = bq.Fitter(out["model"])
    sim_data_kwargs = out["sim_data_kwargs"].copy()
    sim_data_kwargs.update(out["params"])
    out["data"] = sim_data(
        y_func=out["fitter"].eval, binning=out["binning"], **sim_data_kwargs
    )
    return out


# TODO: add fit plotting
# TODO: improve parameter value testing?
class TestFittingHighStatSimData:
    """Test core.fitting.Fitter with high stat generated data"""

    @pytest.mark.filterwarnings("ignore")
    def test_with_init(self, sim_high_stat):
        fitter = bq.Fitter(
            sim_high_stat["model"],
            x=sim_high_stat["data"]["x"],
            y=sim_high_stat["data"]["y"],
            dx=sim_high_stat["data"]["dx"],
            y_unc=sim_high_stat["data"]["y_unc"],
            roi=sim_high_stat["roi"],
        )
        fitter.fit(sim_high_stat["method"])
        compare_params(
            true_params=sim_high_stat["params"],
            fit_params=fitter.result.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] == "lmfit-pml":
            compare_counts(fitter)
        # fitter.custom_plot()
        # plt.show()

    @pytest.mark.filterwarnings("ignore")
    def test_no_roi(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat["model"])
        fitter.set_data(**sim_high_stat["data"])
        fitter.fit(sim_high_stat["method"])
        compare_params(
            true_params=sim_high_stat["params"],
            fit_params=fitter.result.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] == "lmfit-pml":
            compare_counts(fitter)
        # fitter.custom_plot()
        # plt.show()

    @pytest.mark.filterwarnings("ignore")
    def test_with_roi(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat["model"])
        fitter.set_data(**sim_high_stat["data"])
        fitter.set_roi(*sim_high_stat["roi"])
        fitter.fit(sim_high_stat["method"])
        compare_params(
            true_params=sim_high_stat["params"],
            fit_params=fitter.result.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] == "lmfit-pml":
            compare_counts(fitter)
        # fitter.custom_plot()
        # plt.show()
