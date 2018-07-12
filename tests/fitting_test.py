import os
import glob
import pytest
from copy import deepcopy
import numpy as np
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), 'samples')


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

# -----------------------------------------------------------------------------
# Fake data generation
# TODO: discuss this 20% tol, maybe smaller tol for gauss?
# -----------------------------------------------------------------------------

HIGH_STAT_GAUSS_EXP = dict(
    params=dict(
        gauss_amp=1e5,
        gauss_mu=100.,
        gauss_sigma=5.,
        exp_lam=-5e1,
        exp_amp=1e4),
    setup=dict(
        x_min=0.0,
        x_max=200.0,
        num_x=200),
    roi=(25, 175),
    cls=bq.fitting.FitterGaussExp,
    rtol=2e-1,
    name='FitterGaussExp')

HIGH_STAT_GAUSS_ERF_LINE = dict(
    params=dict(
        gauss_amp=1e5,
        gauss_mu=100.,
        gauss_sigma=5.,
        erf_amp=1e4,
        erf_mu=100.,
        erf_sigma=5.,
        line_m=-10.,
        line_b=1e4),
    setup=dict(
        x_min=0.0,
        x_max=200.0,
        num_x=200),
    roi=(25, 175),
    cls=bq.fitting.FitterGaussErfLine,
    rtol=2e-1,
    name='FitterGaussErfLine')

HIGH_STAT_GAUSS_ERF_EXP = dict(
    params=dict(
        gauss_amp=1e5,
        gauss_mu=100.,
        gauss_sigma=5.,
        erf_amp=1e4,
        erf_mu=100.,
        erf_sigma=5.,
        exp_lam=-5e1,
        exp_amp=1e4),
    setup=dict(
        x_min=0.0,
        x_max=200.0,
        num_x=200),
    roi=(25, 175),
    cls=bq.fitting.FitterGaussErfExp,
    rtol=2e-1,
    name='FitterGaussErfExp')

HIGH_STAT_GAUSS_GAUSS_LINE = dict(
    params=dict(
        gauss0_amp=1e5,
        gauss0_mu=60.,
        gauss0_sigma=5.,
        gauss1_amp=1e5,
        gauss1_mu=120.,
        gauss1_sigma=7.,
        line_m=-10.,
        line_b=1e4),
    setup=dict(
        x_min=0.0,
        x_max=200.0,
        num_x=200),
    roi=(25, 175),
    cls=bq.fitting.FitterGaussGaussLine,
    rtol=2e-1,
    name='FitterGaussGaussLine')

HIGH_STAT_GAUSS_GAUSS_EXP = dict(
    params=dict(
        gauss0_amp=1e5,
        gauss0_mu=60.,
        gauss0_sigma=5.,
        gauss1_amp=1e5,
        gauss1_mu=120.,
        gauss1_sigma=7.,
        exp_lam=-5e1,
        exp_amp=1e4),
    setup=dict(
        x_min=0.0,
        x_max=200.0,
        num_x=200),
    roi=(25, 175),
    cls=bq.fitting.FitterGaussGaussExp,
    rtol=2e-1,
    name='FitterGaussGaussExp')


def get_fitter_name(x):
    return x['name']


def fake_data(x_min, x_max, y_func, num_x=200, **params):
    x = np.linspace(x_min, x_max, num_x, dtype=np.float)
    y_smooth = y_func(x=x, **params)
    y = np.random.poisson(y_smooth).astype(np.float)
    y_unc = np.sqrt(y)
    return {'x': x, 'y': y, 'y_unc': y_unc}


def compare_params(true_params, fit_params, rtol):
    for p, v in fit_params.items():
        assert np.isclose(v, true_params[p], rtol=rtol), p


@pytest.fixture(
    params=[HIGH_STAT_GAUSS_EXP,
            HIGH_STAT_GAUSS_ERF_LINE,
            HIGH_STAT_GAUSS_ERF_EXP,
            HIGH_STAT_GAUSS_GAUSS_LINE,
            HIGH_STAT_GAUSS_GAUSS_EXP],
    ids=get_fitter_name)
def fake_high_stat(request):
    """Fake data with high count statistics"""
    out = deepcopy(request.param)
    out['fitter'] = out['cls']()
    out['data'] = fake_data(y_func=out['fitter'].eval,
                            **out['setup'],
                            **out['params'])
    return out


# TODO: add fit plotting
# TODO: improve parameter value testing?
class TestFittingFakeData(object):
    """Test core.fitting Fitters with generated data"""

    def test_fake_high_stat_with_init(self, fake_high_stat):
        fitter = fake_high_stat['cls'](x=fake_high_stat['data']['x'],
                                       y=fake_high_stat['data']['y'],
                                       y_unc=fake_high_stat['data']['y_unc'],
                                       roi=fake_high_stat['roi'])
        fitter.fit()
        compare_params(true_params=fake_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=fake_high_stat['rtol'])
        fitter.custom_plot()

    def test_fake_high_stat_no_roi(self, fake_high_stat):
        fitter = fake_high_stat['cls']()
        fitter.set_data(**fake_high_stat['data'])
        fitter.fit()
        compare_params(true_params=fake_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=fake_high_stat['rtol'])
        fitter.custom_plot()

    def test_fake_high_stat_with_roi(self, fake_high_stat):
        fitter = fake_high_stat['cls']()
        fitter.set_data(**fake_high_stat['data'])
        fitter.set_roi(*fake_high_stat['roi'])
        fitter.fit()
        compare_params(true_params=fake_high_stat['params'],
                       fit_params=fitter.result.best_values,
                       rtol=fake_high_stat['rtol'])
        fitter.custom_plot()
