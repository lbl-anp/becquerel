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
        edges = np.linspace(x_min, x_max, num_x, dtype=float)
    elif binning == "sqrt":
        edges = np.linspace(np.sqrt(x_min), np.sqrt(x_max), num_x)
        edges = edges**2
    x = (edges[1:] + edges[:-1]) * 0.5
    dx = edges[1:] - edges[:-1]

    y_smooth = y_func(x=x, **params) * dx
    np.random.seed(1)
    y = np.random.poisson(y_smooth).astype(float)
    y_unc = np.sqrt(y)
    return {"x": x, "y": y, "y_unc": y_unc, "dx": dx}


def compare_params(true_params, fit_params, rtol, fitter):
    """Test that the fit parameters are close to the true parameters.

    Since minuit can be sensitive to initial values, first try the assert
    normally. If it fails, seed it with lmfit-pml and allow it to try again.
    """
    test = {}
    for p, v in true_params.items():
        test[p] = np.isclose(v, fit_params[p], rtol=rtol)

    if not np.all(list(test.values())) and "minuit" in fitter.backend:
        fitter.fit(backend="lmfit-pml")
        fitter.fit(backend="minuit-pml", guess=fitter.best_values)

        # Just copy the code, since if we recursively called compare_params(),
        # a failed test could raise a potentially-misleading recursion error.
        test = {}
        fit_params = fitter.best_values
        for p, v in true_params.items():
            test[p] = np.isclose(v, fit_params[p], rtol=rtol)
    assert np.all(list(test.values()))


def compare_counts(fitter):
    """Test that the data and model counts match to high accuracy.

    Since minuit can be sensitive to initial values, first try the assert
    normally. If it fails, seed it with lmfit-pml and allow it to try again.
    """

    data_counts = np.sum(fitter.y_roi)
    model_counts = np.sum(
        fitter.eval(fitter.x_roi, **fitter.best_values) * fitter.dx_roi
    )
    test = np.allclose(data_counts, model_counts, atol=1e-2)
    if not test and "minuit" in fitter.backend:
        fitter.fit(backend="lmfit-pml")
        fitter.fit(backend="minuit-pml", guess=fitter.best_values)

        # Just copy the code, since if we recursively called compare_counts(),
        # a failed test could raise a potentially-misleading recursion error.
        data_counts = np.sum(fitter.y_roi)
        model_counts = np.sum(
            fitter.eval(fitter.x_roi, **fitter.best_values) * fitter.dx_roi
        )
        test = np.allclose(data_counts, model_counts, atol=1e-2)
    assert test


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
        ["gausserf", "exp"],
        "expgauss",
    ],
    "fixture": {
        "params": [],
        "ids": [],
    },
    "methods": ["lmfit", "lmfit-pml", "minuit-pml"],
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


def test_method_err(sim_high_stat):
    """Test that unsupported fit methods raise appropriate errors."""
    if sim_high_stat["method"] == "bad-method":
        with pytest.raises(bq.FittingError):
            bq.Fitter(
                sim_high_stat["model"],
                x=sim_high_stat["data"]["x"],
                y=sim_high_stat["data"]["y"],
                dx=sim_high_stat["data"]["dx"],
                y_unc=sim_high_stat["data"]["y_unc"],
            )
    elif sim_high_stat["method"] == "minuit":
        with pytest.raises(NotImplementedError):
            bq.Fitter(
                sim_high_stat["model"],
                x=sim_high_stat["data"]["x"],
                y=sim_high_stat["data"]["y"],
                dx=sim_high_stat["data"]["dx"],
                y_unc=sim_high_stat["data"]["y_unc"],
            )


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
            fit_params=fitter.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] in ["lmfit-pml", "minuit-pml"]:
            compare_counts(fitter)
        # fitter.custom_plot()
        # plt.show()

        # Test some other properties while we're at it
        assert isinstance(str(fitter), str)
        assert fitter.name is None or isinstance(fitter.name, str)
        assert fitter.xmode is None or isinstance(fitter.xmode, str)
        assert fitter.ymode is None or isinstance(fitter.ymode, str)
        assert isinstance(fitter.param_names, list)
        assert len(fitter.param_names) > 0
        assert len(fitter.init_values) > 0
        assert len(fitter.best_values) > 0
        assert fitter.success
        assert bq.fitting._is_count_like(fitter.y_roi)
        assert not bq.fitting._is_count_like(fitter.y_roi * 0.5)

    @pytest.mark.filterwarnings("ignore")
    def test_no_roi(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat["model"])
        fitter.set_data(**sim_high_stat["data"])
        fitter.fit(sim_high_stat["method"])
        compare_params(
            true_params=sim_high_stat["params"],
            fit_params=fitter.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] in ["lmfit-pml", "minuit-pml"]:
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
            fit_params=fitter.best_values,
            rtol=sim_high_stat["rtol"],
            fitter=fitter,
        )
        if sim_high_stat["method"] in ["lmfit-pml", "minuit-pml"]:
            compare_counts(fitter)
        # fitter.custom_plot()
        # plt.show()

    def test_component_areas(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat["model"])
        fitter.set_data(**sim_high_stat["data"])
        fitter.set_roi(*sim_high_stat["roi"])
        fitter.fit(sim_high_stat["method"])

        # Area calculations do not support non-linear bins!
        if sim_high_stat["binning"] != "linear":
            with pytest.raises(NotImplementedError):
                fitter.calc_area_and_unc()
            return

        # Sometimes the fit result does not have a reliable covariance matrix, but alas
        if "minuit" in fitter.backend:
            covariance = np.array(fitter.result.covariance)
        else:
            covariance = np.array(fitter.result.covar)
        # We can at least check that it properly errors, then skip the rest of the tests
        if not covariance.sum():
            with pytest.raises(bq.fitting.FittingError):
                fitter.calc_area_and_unc()
            return

        # Area under the entire curve
        a0 = fitter.calc_area_and_unc()
        assert a0.nominal_value > 0
        assert a0.std_dev > 0
        assert a0.std_dev < a0.nominal_value

        # Area under the ROI
        a1 = fitter.calc_area_and_unc(x=fitter.x_roi)
        # Should be strictly smaller than the area under the entire curve, but we need
        # to allow for a bit of floating point weirdness.
        assert a1.nominal_value / a0.nominal_value < 1 + 1e-9

        # Component-wise areas
        for component in fitter.model.components:
            name = component.prefix.strip("_")
            a2 = fitter.calc_area_and_unc(component=component)
            a3 = fitter.calc_area_and_unc(component=name)
            assert np.isclose(a2.nominal_value, a3.nominal_value)
            assert np.isclose(a2.std_dev, a3.std_dev)
            a4 = fitter.calc_area_and_unc(component=component, x=fitter.x_roi)
            a5 = fitter.calc_area_and_unc(component=name, x=fitter.x_roi)
            assert np.isclose(a4.nominal_value, a5.nominal_value)
            assert np.isclose(a4.std_dev, a5.std_dev)

            if name == "gauss":
                # If the component is a Gaussian, the calculated area should be very
                # close to its given amplitude parameter.
                a_theor = HIGH_STAT_SIM_PARAMS["base_model_params"][name]["amp"]
                assert np.isclose(a2.nominal_value, a_theor, rtol=0.01)

                # Additionally, if there's no background, the uncertainty should be very
                # close to the sqrt of the amplitude (at least with minuit!)
                if len(fitter.model.components) == 1 and "minuit" in fitter.backend:
                    assert np.isclose(a2.std_dev, np.sqrt(a_theor), rtol=0.01)


@pytest.mark.parametrize("method", ["lmfit", "lmfit-pml", "minuit-pml"])
def test_gauss_gauss_gauss_line(method):
    model = (
        bq.fitting.GaussModel(prefix="gauss0_")
        + bq.fitting.GaussModel(prefix="gauss1_")
        + bq.fitting.GaussModel(prefix="gauss2_")
        + bq.fitting.LineModel(prefix="line_")
    )
    params = {
        "gauss0_amp": 1e5,
        "gauss0_mu": 80.0,
        "gauss0_sigma": 5.0,
        "gauss1_amp": 1e5,
        "gauss1_mu": 100.0,
        "gauss1_sigma": 5.0,
        "gauss2_amp": 1e5,
        "gauss2_mu": 120.0,
        "gauss2_sigma": 5.0,
        "line_m": -10.0,
        "line_b": 1e4,
    }
    data = sim_data(y_func=model.eval, x_min=0, x_max=200, **params)

    fitter = bq.Fitter(model, **data)
    for i in range(3):
        n = f"gauss{i}_mu"
        fitter.params[n].set(value=params[n])
    fitter.fit(method)
    compare_params(
        true_params=params,
        fit_params=fitter.best_values,
        rtol=0.05,
        fitter=fitter,
    )
    # fitter.custom_plot()
    # plt.show()
