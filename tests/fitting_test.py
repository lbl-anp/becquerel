import os
import glob
import numpy as np
import becquerel as bq


SAMPLES_PATH = os.path.join(os.path.dirname(__file__), 'samples')


# TODO:use these for fitting actual data
SAMPLES = {}
for extension in ['.spe', '.spc', '.cnf']:
    filenames = glob.glob(os.path.join(SAMPLES_PATH, '*.*'))
    filenames_filtered = []
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        if ext.lower() == extension:
            filenames_filtered.append(filename)
    SAMPLES[extension] = filenames_filtered


# TODO: add fit plotting
# TODO: improve parameter value testing?
class TestFittingFakeData(object):
    """Test core.fitting Fitters with generated data"""

    def test_gauss_exp(self):
        fitter = bq.fitting.FitterGaussExp()
        params_true = dict(
            gauss_amp=1e5,
            gauss_mu=50.,
            gauss_sigma=5.,
            exp_lam=-5e1,
            exp_amp=1e4)
        x = np.linspace(0, params_true['gauss_mu'] * 2.0, 200)
        y_smooth = fitter.eval(x=x, **params_true)
        y = np.random.poisson(y_smooth)
        y_unc = np.sqrt(y)
        fitter.set_data(x=x, y=y, y_unc=y_unc)
        fitter.fit()
        for p, v in fitter.result.best_values.items():
            # TODO: discuss this 20% tol, maybe smaller tol for gauss?
            assert np.isclose(v, params_true[p], rtol=2e-1), p

    def test_gauss_erf_line(self):
        fitter = bq.fitting.FitterGaussErfLine()
        params_true = dict(
            gauss_amp=1e5,
            gauss_mu=50.,
            gauss_sigma=5.,
            erf_amp=1e4,
            erf_mu=50.,
            erf_sigma=5.,
            line_m=-10.,
            line_b=1e4)
        x = np.linspace(0, params_true['gauss_mu'] * 2.0, 200)
        y_smooth = fitter.eval(x=x, **params_true)
        y = np.random.poisson(y_smooth)
        y_unc = np.sqrt(y)
        fitter.set_data(x=x, y=y, y_unc=y_unc)
        fitter.fit()
        for p, v in fitter.result.best_values.items():
            # TODO: discuss this 20% tol, maybe smaller tol for gauss?
            assert np.isclose(v, params_true[p], rtol=2e-1), p

    def test_gauss_erf_exp(self):
        fitter = bq.fitting.FitterGaussErfExp()
        params_true = dict(
            gauss_amp=1e5,
            gauss_mu=50.,
            gauss_sigma=5.,
            erf_amp=1e4,
            erf_mu=50.,
            erf_sigma=5.,
            exp_lam=-5e1,
            exp_amp=1e4)
        x = np.linspace(0, params_true['gauss_mu'] * 2.0, 200)
        y_smooth = fitter.eval(x=x, **params_true)
        y = np.random.poisson(y_smooth)
        y_unc = np.sqrt(y)
        fitter.set_data(x=x, y=y, y_unc=y_unc)
        fitter.fit()
        for p, v in fitter.result.best_values.items():
            # TODO: discuss this 20% tol, maybe smaller tol for gauss?
            assert np.isclose(v, params_true[p], rtol=2e-1), p

    def test_gauss_gauss_line(self):
        fitter = bq.fitting.FitterGaussGaussLine()
        params_true = dict(
            gauss0_amp=1e5,
            gauss0_mu=30.,
            gauss0_sigma=3.,
            gauss1_amp=1e5,
            gauss1_mu=70.,
            gauss1_sigma=4.,
            line_m=-10.,
            line_b=1e4)
        x = np.linspace(0, 100., 200)
        y_smooth = fitter.eval(x=x, **params_true)
        y = np.random.poisson(y_smooth)
        y_unc = np.sqrt(y)
        fitter.set_data(x=x, y=y, y_unc=y_unc)
        fitter.fit()
        for p, v in fitter.result.best_values.items():
            # TODO: discuss this 20% tol, maybe smaller tol for gauss?
            assert np.isclose(v, params_true[p], rtol=2e-1), p

    def test_gauss_gauss_exp(self):
        fitter = bq.fitting.FitterGaussGaussExp()
        params_true = dict(
            gauss0_amp=1e5,
            gauss0_mu=30.,
            gauss0_sigma=3.,
            gauss1_amp=1e5,
            gauss1_mu=70.,
            gauss1_sigma=4.,
            exp_lam=-5e1,
            exp_amp=1e4)
        x = np.linspace(0, 100., 200)
        y_smooth = fitter.eval(x=x, **params_true)
        y = np.random.poisson(y_smooth)
        y_unc = np.sqrt(y)
        fitter.set_data(x=x, y=y, y_unc=y_unc)
        fitter.fit()
        for p, v in fitter.result.best_values.items():
            # TODO: discuss this 20% tol, maybe smaller tol for gauss?
            assert np.isclose(v, params_true[p], rtol=2e-1), p
