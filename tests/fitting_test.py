import os
import glob
import pytest
from copy import deepcopy
import numpy as np
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), 'samples')
np.random.seed(1)

# TODO: use these for fitting actual data
SAMPLES = {}
for extension in ['.spe', '.spc', '.cnf']:
    filenames = glob.glob(os.path.join(SAMPLES_PATH, '*.*'))
    filenames_filtered = []
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        if ext.lower() == extension:
            filenames_filtered.append(filename)
    SAMPLES[extension] = filenames_filtered


def get_model_name(x):
    if bq.core.utils.isstring(x):
        return x
    else:
        return ' '.join(x)


def sim_data(x_min, x_max, y_func, num_x=200, **params):
    x = np.linspace(x_min, x_max, num_x, dtype=np.float)
    y_smooth = y_func(x=x, **params)
    y = np.random.poisson(y_smooth).astype(np.float)
    y_unc = np.sqrt(y)
    return {'x': x, 'y': y, 'y_unc': y_unc}


def compare_params(true_params, fit_params, rtol):
    for p, v in fit_params.items():
        assert np.isclose(v, true_params[p], rtol=rtol), p


# -----------------------------------------------------------------------------
# Simulated data generation
# TODO: discuss this 20% tol, maybe smaller tol for gauss?
# -----------------------------------------------------------------------------

HIGH_STAT_SIM_PARAMS = {
    'base_model_params': {
        'gauss': {
            'amp': 1e5,
            'mu': 100.,
            'sigma': 5.,
        },
        'exp': {
            'lam': -5e1,
            'amp': 1e4,
        },
        'erf': {
            'amp': 1e4,
            'mu': 100.,
            'sigma': 5.,
        },
        'line': {
            'm': -10.,
            'b': 1e4,
        },
    },
    'setup': {
        'roi': (25, 175),
        'rtol': 30e-2,
        'sim_data_kwargs': {
            'x_min': 0.0,
            'x_max': 200.0,
            'num_x': 200
        },
    },
    'models': [
        'gauss',
        ['gauss', 'line'],
        ['gauss', 'erf'],
        ['gauss', 'exp'],
        ['gauss', 'line', 'erf'],
        ['gauss', 'line', 'exp'],
        ['gauss', 'exp', 'erf'],
    ],
    'fixture': {
        'params': [],
        'ids': [],
    },
}

for _m in HIGH_STAT_SIM_PARAMS['models']:
    _p = deepcopy(HIGH_STAT_SIM_PARAMS['setup'])
    _p['model'] = deepcopy(_m)
    _p['params'] = {}
    if bq.utils.isstring(_m):
        _i = deepcopy(_m)
        _m = [_m]
    else:
        _i = ''.join([_bm.capitalize() for _bm in _m])
    for _bm in _m:
        for _bmp, _v in HIGH_STAT_SIM_PARAMS['base_model_params'][_bm].items():
            _p['params']['{}_{}'.format(_bm, _bmp)] = _v
    HIGH_STAT_SIM_PARAMS['fixture']['params'].append(_p)
    HIGH_STAT_SIM_PARAMS['fixture']['ids'].append(_i)


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


@pytest.fixture(**HIGH_STAT_SIM_PARAMS['fixture'])
def sim_high_stat(request):
    """Fake data with high count statistics"""
    out = deepcopy(request.param)
    out['fitter'] = bq.Fitter(out['model'])
    sim_data_kwargs = out['sim_data_kwargs'].copy()
    sim_data_kwargs.update(out['params'])
    out['data'] = sim_data(y_func=out['fitter'].eval, **sim_data_kwargs)
    return out


# TODO: add fit plotting
# TODO: improve parameter value testing?
class TestFittingHighStatSimData(object):
    """Test core.fitting.Fitter with high stat generated data"""

    @pytest.mark.filterwarnings("ignore")
    def test_with_init(self, sim_high_stat):
        fitter = bq.Fitter(
            sim_high_stat['model'],
            x=sim_high_stat['data']['x'],
            y=sim_high_stat['data']['y'],
            y_unc=sim_high_stat['data']['y_unc'],
            roi=sim_high_stat['roi'])
        fitter.fit()
        compare_params(true_params=sim_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=sim_high_stat['rtol'])
        fitter.custom_plot()

    @pytest.mark.filterwarnings("ignore")
    def test_no_roi(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat['model'])
        fitter.set_data(**sim_high_stat['data'])
        fitter.fit()
        compare_params(true_params=sim_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=sim_high_stat['rtol'])
        fitter.custom_plot()

    @pytest.mark.filterwarnings("ignore")
    def test_with_roi(self, sim_high_stat):
        fitter = bq.Fitter(sim_high_stat['model'])
        fitter.set_data(**sim_high_stat['data'])
        fitter.set_roi(*sim_high_stat['roi'])
        fitter.fit()
        compare_params(true_params=sim_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=sim_high_stat['rtol'])
        fitter.custom_plot()
