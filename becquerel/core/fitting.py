from __future__ import print_function
import numpy as np
import scipy.special
from lmfit.model import Model


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


class FitterGaussErfLine(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ErfModel(prefix='erf_') + \
            LineModel(prefix='line_')

    def guess_param_defaults(self):
        params = []
        # Defaults
        cen50_msk = (
            (np.arange(len(self.x_roi)) >= float(len(self.x_roi) - 1) * 0.25) &
            (np.arange(len(self.x_roi)) <= float(len(self.x_roi) - 1) * 0.75))
        # Integrate and clip last channel of center 50%
        params.append((
            'gauss_amp',
            'value',
            np.sum(
                np.diff(self.x_roi[cen50_msk]) * self.y_roi[cen50_msk][:-1])))
        params.append(('gauss_amp', 'min', 0.))
        params.append((
            'gauss_mu',
            'value',
            (self.x_roi[0] + self.x_roi[-1]) / 2.))
        params.append(('gauss_mu', 'min', self.x_roi[0]))
        params.append(('gauss_mu', 'max', self.x_roi[-1]))
        params.append(('gauss_sigma', 'min', 0.))
        # TODO: update this, minimizer creates NaN's if default sigma used (0)
        params.append((
            'gauss_sigma',
            'value',
            (self.x_roi[-1] - self.x_roi[0]) / 10.))
        params.append(('erf_amp', 'value', self.y_roi[0] - self.y_roi[-1]))
        params.append(('erf_amp', 'min', 0.))
        params.append(('erf_mu', 'expr', 'gauss_mu'))
        params.append(('erf_sigma', 'expr', 'gauss_sigma'))
        params.append(('line_m', 'value', 0.))
        params.append(('line_b', 'value', self.y_roi[0]))
        return params
