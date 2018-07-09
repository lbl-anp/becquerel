from __future__ import print_function
import numpy as np
import scipy.special
from lmfit.model import Model

# TODO: add fitting exception
# -----------------------------------------------------------------------------
# Base functions
# -----------------------------------------------------------------------------


def constant(x, c):
    return np.ones_like(x) * c


def line(x, m, b):
    return m * x + b


def gauss(x, amp, mu, sigma):
    amp = np.abs(amp)
    mu = np.abs(mu)
    sigma = np.abs(sigma)
    return amp / np.sqrt(2. * sigma ** 2. * np.pi) * \
        np.exp(-(x - mu) ** 2. / (2. * sigma ** 2.))


def erf(x, amp, mu, sigma):
    return amp * 0.5 * (1. - scipy.special.erf((x - mu) / (1.414214 * sigma)))


def exp(x, amp, lam):
    return amp * np.exp(x / lam)


# -----------------------------------------------------------------------------
# Fitting models
# TODO: Add def guess to each class like here:
# https://github.com/lmfit/lmfit-py/blob/master/lmfit/models.py
# -----------------------------------------------------------------------------


class ConstantModel(Model):

    def __init__(self, *args, **kwargs):
        super(ConstantModel, self).__init__(constant, *args, **kwargs)
        # TODO: remove this min setting?
        self.set_param_hint('{}c'.format(self.prefix), min=0.)


class LineModel(Model):

    def __init__(self, *args, **kwargs):
        super(LineModel, self).__init__(line, *args, **kwargs)


class GaussModel(Model):

    def __init__(self, *args, **kwargs):
        super(GaussModel, self).__init__(gauss, *args, **kwargs)
        self.set_param_hint(
            '{}fwhm'.format(self.prefix),
            expr='2.354820 * {}sigma'.format(self.prefix))


class ErfModel(Model):

    def __init__(self, *args, **kwargs):
        super(ErfModel, self).__init__(erf, *args, **kwargs)


class ExpModel(Model):

    def __init__(self, *args, **kwargs):
        super(ExpModel, self).__init__(exp, *args, **kwargs)
        self.set_param_hint('{}amp'.format(self.prefix), min=0.)
        self.set_param_hint('{}lam'.format(self.prefix), max=0.)


# -----------------------------------------------------------------------------
# Fitters
# TODO: add docs
# TODO: add ability to override defaults
# TODO: add ability to initialize and fit with Fitter.__init__
# TODO: change to use ABC
# TODO: include x_edges?
# TODO: handle y normalization (i.e. cps vs cps/keV), needs x_edges
# TODO: use set_param_hint to set global model defaults
# -----------------------------------------------------------------------------


class Fitter(object):

    def __init__(self, x=None, y=None, y_unc=None, roi=None):
        if y is None:
            self._x = None
            self._y = None
            self._y_unc = None
        else:
            self.set_data(y, x, y_unc)
        if roi is None:
            self._roi_low = None
            self._roi_high = None
            self._roi_msk = None
        else:
            self.set_roi(*roi)
        self.make_model()
        self.params = self.model.make_params()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def y_unc(self):
        return self._y_unc

    @property
    def x_roi(self):
        return self.x[self.roi_msk]

    @property
    def y_roi(self):
        return self.y[self.roi_msk]

    @property
    def y_unc_roi(self):
        return self.y_unc[self.roi_msk]

    @property
    def roi(self):
        return self._roi

    @property
    def roi_msk(self):
        if self._roi_msk is None:
            return np.ones_like(self.x, dtype=bool)
        else:
            return self._roi_msk

    def set_data(self, y, x=None, y_unc=None):
        # Set y data
        self._y = np.asarray(y)
        # Set x data
        if x is None:
            self._x = np.arange(len(self.y))
        else:
            self._x = np.asarray(x)
            assert len(self.x) == len(self.y), \
                'Fitting x (len {}) does not match y (len {})'.format(
                    len(self.x), len(self.y))
        # Handle y uncertainties
        if y_unc is None:
            # TODO: add warning
            self._y_unc = None
        else:
            self._y_unc = np.asarray(y_unc, dtype=np.float)
            assert len(self.x) == len(self._y_unc), \
                'Fitting x (len {}) does not match y_unc (len {})'.format(
                    len(self.x), len(self._y_unc))
            # TODO: revisit this non zero unc option
            self._y_unc[self._y_unc <= 0.] = np.min(self._y_unc > 0.)
        # Set param defaults based on guess
        defaults = self.guess_param_defaults()
        if defaults is not None:
            for dp in defaults:
                self.set_param(*dp)

    def set_roi(self, low, high):
        self._roi_low = float(low)
        self._roi_high = float(high)
        self._roi_msk = ((self.x >= self.roi_low) &
                         (self.x <= self.roi_high))

    def set_param(self, pname, ptype, pvalue):
        self.params[pname].set(**{ptype: pvalue})

    def make_model(self):
        # TODO: change to ABC
        raise NotImplementedError()

    def guess_param_defaults(self):
        # TODO: change to ABC
        raise NotImplementedError()

    def fit(self, backend='lmfit'):
        assert self.y is not None, \
            'No data initialized, did you call set_data?'
        if backend.lower().strip() == 'lmfit':
            if self.y_unc is None:
                weights = None
            else:
                # TODO: check this
                weights = self.y_unc_roi ** -1.0
            self.result = self.model.fit(self.y_roi,
                                         self.params,
                                         x=self.x_roi,
                                         weights=weights)
        else:
            raise ValueError('Unknown fitting backend: {}'.format(backend))

    def eval(self, x, **params):
        return self.model.eval(x=x, **params)

    # Parameter guessing
    def _xy_left(self, num=4):
        return np.mean(self.x_roi[:num]), np.mean(self.y_roi[:num])

    def _xy_right(self, num=4):
        return np.mean(self.x_roi[-num:]), np.mean(self.y_roi[-num:])

    def _guess_gauss_params(self, prefix='gauss_', center_ratio=0.5,
                            width_ratio=0.5):
        assert center_ratio < 1, \
            'Center mask ratio cannot exceed 1: {}'.format(center_ratio)
        assert width_ratio < 1, \
            'Width mask ratio cannot exceed 1: {}'.format(width_ratio)
        xspan = self.x_roi[-1] - self.x_roi[0]
        mu = self.x_roi[0] + xspan * center_ratio
        msk = ((self.x_roi >= (mu - xspan * width_ratio)) &
               (self.x_roi <= mu + xspan * width_ratio))
        # NOTE: this integration assumes y is NOT normalized to dx
        amp = np.sum(self.y_roi[msk])
        # TODO: update this, minimizer creates NaN's if default sigma used (0)
        sigma = xspan * width_ratio / 10.
        return [
            ('{}amp'.format(prefix), 'value', amp),
            ('{}amp'.format(prefix), 'min', 0.0),
            ('{}mu'.format(prefix), 'value', mu),
            ('{}mu'.format(prefix), 'min', self.x_roi[0]),
            ('{}mu'.format(prefix), 'max', self.x_roi[-1]),
            ('{}sigma'.format(prefix), 'value', sigma),
            ('{}sigma'.format(prefix), 'min', 0.0),
        ]

    def _guess_erf_params(self, prefix='erf_'):
        return [
            ('{}amp'.format(prefix), 'value', self.y_roi[0] - self.y_roi[-1]),
            ('{}amp'.format(prefix), 'min', 0.),
            ('{}mu'.format(prefix), 'expr', 'gauss_mu'),
            ('{}sigma'.format(prefix), 'expr', 'gauss_sigma'),
        ]

    def _guess_line_params(self, prefix='line_', num=2):
        _, b = self._xy_left(num=num)
        return [
            ('{}m'.format(prefix), 'value', 0.),
            ('{}b'.format(prefix), 'value', b),
        ]

    def _guess_exp_params(self, num=1, prefix='exp_'):
        (xl, yl), (xr, yr) = self._xy_left(num), self._xy_right(num)
        # TODO: update this hardcoded zero offset
        lam = (xl - xr) / np.log(yl / (yr + 0.0001))
        amp = yl / np.exp(xl / lam)
        return [
            ('{}lam'.format(prefix), 'value', lam),
            ('{}lam'.format(prefix), 'max', -1e-3),
            ('{}amp'.format(prefix), 'value', amp),
            ('{}amp'.format(prefix), 'min', 0.0),
        ]


class FitterGaussExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ExpModel(prefix='exp_')

    def guess_param_defaults(self):
        params = []
        params += self._guess_gauss_params(prefix='gauss_')
        params += self._guess_exp_params(prefix='exp_')
        return params


class FitterGaussErfLine(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ErfModel(prefix='erf_') + \
            LineModel(prefix='line_')

    def guess_param_defaults(self):
        params = []
        params += self._guess_gauss_params(prefix='gauss_')
        params += self._guess_erf_params(prefix='erf_')
        params += self._guess_line_params(prefix='line_')
        return params


class FitterGaussErfExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ErfModel(prefix='erf_') + \
            ExpModel(prefix='exp_')

    def guess_param_defaults(self):
        params = []
        params += self._guess_gauss_params(prefix='gauss_')
        params += self._guess_erf_params(prefix='erf_')
        params += self._guess_exp_params(prefix='exp_')
        return params


class FitterGaussGaussLine(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss0_') + \
            GaussModel(prefix='gauss1_') + \
            LineModel(prefix='line_')

    def guess_param_defaults(self):
        params = []
        params += self._guess_gauss_params(prefix='gauss0_',
                                           center_ratio=0.33,
                                           width_ratio=0.5)
        params += self._guess_gauss_params(prefix='gauss1_',
                                           center_ratio=0.66,
                                           width_ratio=0.5)
        params += self._guess_line_params(prefix='line_')
        return params


class FitterGaussGaussExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss0_') + \
            GaussModel(prefix='gauss1_') + \
            ExpModel(prefix='exp_')

    def guess_param_defaults(self):
        params = []
        params += self._guess_gauss_params(prefix='gauss0_',
                                           center_ratio=0.33,
                                           width_ratio=0.5)
        params += self._guess_gauss_params(prefix='gauss1_',
                                           center_ratio=0.66,
                                           width_ratio=0.5)
        params += self._guess_exp_params(prefix='exp_')
        return params
