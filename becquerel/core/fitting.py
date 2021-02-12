from __future__ import print_function
import warnings
import inspect
import numpy as np
import pandas as pd
import scipy.special
from lmfit.model import Model
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from .utils import isstring, bin_centers_from_edges

FWHM_SIG_RATIO = np.sqrt(8*np.log(2))  # 2.35482
SQRT_TWO = np.sqrt(2) #1.414213562

class FittingError(Exception):
    """Exception raised by Fitters."""
    pass


class FittingWarning(UserWarning):
    """Warning raised by Fitters."""


# -----------------------------------------------------------------------------
# Base functions
# TODO: Should these be replaced with lmfit.models???
# -----------------------------------------------------------------------------


def constant(x, c):
    return np.ones_like(x) * c


def line(x, m, b):
    return m * x + b


def gauss(x, amp, mu, sigma):
    return amp / sigma / np.sqrt(2. * np.pi) * \
        np.exp(-(x - mu) ** 2. / (2. * sigma ** 2.))


def erf(x, amp, mu, sigma):
    return amp * 0.5 * (1. - scipy.special.erf((x - mu) / (SQRT_TWO * sigma)))


def exp(x, amp, lam):
    return amp * np.exp(x / lam)


def gausserf(x, ampgauss, amperf, mu, sigma):
    return gauss(x, ampgauss, mu, sigma) + erf(x, amperf, mu, sigma)


def expgauss(x, amp=1, mu=0, sigma=1.0, gamma=1.0):
    gss = gamma*sigma*sigma
    arg1 = gamma*(mu + gss/2.0 - x)
    arg2 = (mu + gss - x)/(SQRT_TWO*sigma)
    return amp*(gamma/2) * np.exp(arg1) * scipy.special.erfc(arg2)

# -----------------------------------------------------------------------------
# Fitting models
# -----------------------------------------------------------------------------


# Helper functions for guessing
def _xy_right(y, x=None, num=4):
    """Compute mean x and y in the last `num` points of a dataset (x, y).

    Parameters
    ----------
    y : array-like
        Y-data
    x : array-like, optional
        X-data. If not specified, return len(y)/2.
    num : int, optional
        Number of points to include in the averaging; default 4.

    Returns
    -------
    (float, float)
        Tuple of (xmean, ymean).
    """
    if x is not None:
        return np.mean(x[-num:]), np.mean(y[-num:])
    return len(y)*0.5, np.mean(y[-num:])


def _xy_left(y, x=None, num=4):
    """Compute mean x and y in the first `num` points of a dataset (x, y).

    Parameters
    ----------
    y : array-like
        Y-data
    x : array-like, optional
        X-data. If not specified, return len(y)/2.
    num : int, optional
        Number of points to include in the averaging; default 4.

    Returns
    -------
    (float, float)
        Tuple of (xmean, ymean).
    """
    if x is not None:
        return np.mean(x[:num]), np.mean(y[:num])
    return len(y)*0.5, np.mean(y[:num])


class ConstantModel(Model):

    def __init__(self, *args, **kwargs):
        super(ConstantModel, self).__init__(constant, *args, **kwargs)
        # TODO: remove this min setting?
        self.set_param_hint('{}c'.format(self.prefix), min=0.)

    def guess(self, y, x=None, dx=None, num=2):
        if dx is None:
            dx = np.ones_like(x)
        c = (y[-1]/dx[-1] + y[0]/dx[0]) * 0.5
        return [
            ('{}c'.format(self.prefix), 'value', c),
        ]


class LineModel(Model):

    def __init__(self, *args, **kwargs):
        super(LineModel, self).__init__(line, *args, **kwargs)

    def guess(self, y, x=None, dx=None, num=2):
        if dx is None:
            dx = np.ones_like(x)
        m = (y[-1]/dx[-1] - y[0]/dx[0])/(x[-1] - x[0])
        b = ((y[-1]/dx[-1] + y[0]/dx[0]) - m*(x[1] + x[0])) * 0.5
        return [
            ('{}m'.format(self.prefix), 'value', m),
            ('{}b'.format(self.prefix), 'value', b),
        ]


class GaussModel(Model):

    def __init__(self, *args, **kwargs):
        super(GaussModel, self).__init__(gauss, *args, **kwargs)
        self.set_param_hint(
            '{}fwhm'.format(self.prefix),
            expr='{} * {}sigma'.format(FWHM_SIG_RATIO, self.prefix))

    def guess(self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.5):
        assert center_ratio < 1, \
            'Center mask ratio cannot exceed 1: {}'.format(center_ratio)
        assert width_ratio < 1, \
            'Width mask ratio cannot exceed 1: {}'.format(width_ratio)

        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)

        # counts = (y - (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0]) - y[0])
        # max_idx = np.argmax(counts)
        # counts[counts < counts[max_idx] * 0.5] *= 0
        # amp = np.sum(counts) / 0.8
        # mu = x[max_idx]
        # sigma = np.sqrt(np.sum(counts * (x - mu)**2) / amp) * 2
        # if x is None:
        #     x = np.arange(0, len(y))

        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = ((x >= (mu - xspan * width_ratio)) &
               (x <= mu + xspan * width_ratio))
        # NOTE: this integration assumes y is NOT normalized to dx (NOW IT IS)
        amp = np.sum(y[msk]/dx[msk])
        # c = y[msk]
        # amp = np.sum(c) - (c[0] + c[-1]) * np.sum(msk) * 0.5
        # TODO: update this, minimizer creates NaN's if default sigma used (0)
        sigma = xspan * width_ratio / 10.
        return [
            ('{}amp'.format(self.prefix), 'value', amp),
            ('{}amp'.format(self.prefix), 'min', 0.0),
            ('{}mu'.format(self.prefix), 'value', mu),
            ('{}mu'.format(self.prefix), 'min', x[0]),
            ('{}mu'.format(self.prefix), 'max', x[-1]),
            ('{}sigma'.format(self.prefix), 'value', sigma),
            ('{}sigma'.format(self.prefix), 'min', 0.0),
        ]


class ErfModel(Model):

    def __init__(self, *args, **kwargs):
        super(ErfModel, self).__init__(erf, *args, **kwargs)

    def guess(self, y, x=None, dx=None, center_ratio=0):
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        return [
            ('{}amp'.format(self.prefix), 'value', y[0] - y[-1]),
            ('{}mu'.format(self.prefix), 'value', mu),
            ('{}sigma'.format(self.prefix), 'expr', 'gauss_sigma'),
        ]


class GaussErfModel(Model):

    def __init__(self, *args, **kwargs):
        super(GaussErfModel, self).__init__(gausserf, *args, **kwargs)
        self.set_param_hint(
            '{}fwhm'.format(self.prefix),
            expr='{} * {}sigma'.format(FWHM_SIG_RATIO, self.prefix))

    def guess(self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.5,
              amp_ratio=0.9):
        assert center_ratio < 1, \
            'Center mask ratio cannot exceed 1: {}'.format(center_ratio)
        assert width_ratio < 1, \
            'Width mask ratio cannot exceed 1: {}'.format(width_ratio)
        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = ((x >= (mu - xspan * width_ratio)) &
               (x <= mu + xspan * width_ratio))
        amp = np.sum(y[msk]/dx[msk])
        sigma = xspan * width_ratio / 10.
        return [
            ('{}ampgauss'.format(self.prefix), 'value', amp*amp_ratio),
            ('{}ampgauss'.format(self.prefix), 'min', 0.0),
            ('{}amperf'.format(self.prefix), 'value', amp*(1.0 - amp_ratio)),
            ('{}amperf'.format(self.prefix), 'min', 0.0),
            ('{}mu'.format(self.prefix), 'value', mu),
            ('{}mu'.format(self.prefix), 'min', x[0]),
            ('{}mu'.format(self.prefix), 'max', x[-1]),
            ('{}sigma'.format(self.prefix), 'value', sigma),
            ('{}sigma'.format(self.prefix), 'min', 0.0),
        ]


class ExpModel(Model):

    def __init__(self, *args, **kwargs):
        super(ExpModel, self).__init__(exp, *args, **kwargs)
        self.set_param_hint('{}amp'.format(self.prefix), min=0.)
        self.set_param_hint('{}lam'.format(self.prefix), max=0.)

    def guess(self, y, x=None, dx=None, num=1):
        xl, yl = _xy_left(y, x=x, num=num)
        xr, yr = _xy_right(y, x=x, num=num)
        # TODO: update this hardcoded zero offset
        lam = (xl - xr) / np.log(yl / (yr + 0.0001))
        amp = yl / np.exp(xl / lam)
        return [
            ('{}lam'.format(self.prefix), 'value', lam),
            ('{}lam'.format(self.prefix), 'max', -1e-3),
            ('{}amp'.format(self.prefix), 'value', amp),
            ('{}amp'.format(self.prefix), 'min', 0.0),
        ]


class ExpGaussModel(Model):
    """A model of an Exponentially modified Gaussian distribution
    (see https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
    """

    def __init__(self, *args, **kwargs):
        super(ExpGaussModel, self).__init__(expgauss, **kwargs)
        self.set_param_hint('{}sigma'.format(self.prefix), min=0)
        self.set_param_hint('{}gamma'.format(self.prefix), min=0, max=1)
        # TODO: This is obviously wrong
        self.set_param_hint(
            '{}fwhm'.format(self.prefix),
            expr='{} * {}sigma'.format(FWHM_SIG_RATIO, self.prefix))

    def guess(self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.5):
        assert center_ratio < 1, \
            'Center mask ratio cannot exceed 1: {}'.format(center_ratio)
        assert width_ratio < 1, \
            'Width mask ratio cannot exceed 1: {}'.format(width_ratio)
        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = ((x >= (mu - xspan * width_ratio)) &
               (x <= mu + xspan * width_ratio))
        # NOTE: this integration assumes y is NOT normalized to dx
        amp = np.sum(y[msk]/dx[msk])
        # TODO: update this, minimizer creates NaN's if default sigma used (0)
        sigma = xspan * width_ratio / 10.
        # TODO: We miss gamma here
        return [
            ('{}amp'.format(self.prefix), 'value', amp),
            ('{}amp'.format(self.prefix), 'min', 0.0),
            ('{}mu'.format(self.prefix), 'value', mu),
            ('{}mu'.format(self.prefix), 'min', x[0]),
            ('{}mu'.format(self.prefix), 'max', x[-1]),
            ('{}sigma'.format(self.prefix), 'value', sigma),
            ('{}sigma'.format(self.prefix), 'min', 0.0),
            ('{}gamma'.format(self.prefix), 'value', 0.95),
            ('{}gamma'.format(self.prefix), 'min', 0.0),
        ]


MODEL_STR_TO_CLS = {
    'constant': ConstantModel,
    'line': LineModel,
    'gauss': GaussModel,
    'gausserf': GaussErfModel,
    'erf': ErfModel,
    'exp': ExpModel,
    'expgauss': ExpGaussModel
}


# -----------------------------------------------------------------------------
# Fitters
# TODO: add docs
# TODO: add ability to override defaults
# TODO: add ability to initialize and fit with Fitter.__init__
# TODO: include x_edges?
# TODO: handle y normalization (i.e. cps vs cps/keV), needs x_edges
# TODO: use set_param_hint to set global model defaults
# -----------------------------------------------------------------------------


class Fitter(object):
    '''Base class for more specialized fit objects.

    A note on interpreting fit results: ascribing meaning to histogram fit
    parameters is notoriously tricky, since the y-scale has units of counts
    per bin width, not just counts. The user may need to divide by the
    histogram bin width for area- or height-like parameters if the histogram is
    not already normalized by bin width. See, e.g., p. 171 of Bevington and
    Robinson, "Data reduction and error analysis for the physical sciences".
    '''

    def __init__(self, model, x=None, y=None, y_unc=None, dx=None, roi=None):
        # Initialize
        self._model = None
        self._name = None
        self._x = None
        self._y = None
        self._y_unc = None
        self._roi = None
        self._roi_msk = None
        self._xmode = None
        self._ymode = None
        self.result = None
        self.dx = None
        # Model and parameters
        self._make_model(model)
        self.params = self.model.make_params()
        # Set data
        self.set_data(x=x, y=y, y_unc=y_unc, dx=dx, roi=roi,
                      update_defaults=True)

    def __str__(self):
        return (
            'bq.Fitter instance\n' +
            '     name: {}\n'.format(self.name) +
            '    model: {}\n'.format(self.model) +
            '        x: {}\n'.format(self.x) +
            '        y: {}\n'.format(self.y) +
            '    y_unc: {}\n'.format(self.y_unc) +
            '    xmode: {}\n'.format(self.xmode) +
            '    ymode: {}\n'.format(self.ymode) +
            '       dx: {}\n'.format(self.dx) +
            '      roi: {}'.format(self.roi)
        )

    __repr__ = __str__

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def y_unc(self):
        if self._y_unc is None:
            warnings.warn(
                'No y uncertainties (y_unc) provided. The fit will not be ' +
                'weighted causing in poor results at low counting statistics.',
                FittingWarning)
        return self._y_unc

    @y_unc.setter
    def y_unc(self, y_unc):
        if y_unc is not None:
            self._y_unc = np.asarray(y_unc, dtype=np.float)
            assert len(self.x) == len(self._y_unc), \
                'Fitting x (len {}) does not match y_unc (len {})'.format(
                    len(self.x), len(self._y_unc))
            if np.any(self._y_unc <= 0.):
                min_v = np.min(self._y_unc[self._y_unc > 0.])
                warnings.warn(
                    'Negative or zero uncertainty not supported. Changing ' +
                    'them to {}. If you have Poisson data, '.format(min_v) +
                    'this should be 1.')
                self._y_unc[self._y_unc <= 0.] = min_v
        else:
            self._y_unc = None

    @property
    def x_roi(self):
        return self.x[self.roi_msk]

    @property
    def y_roi(self):
        return self.y[self.roi_msk]

    @property
    def y_unc_roi(self):
        if self.y_unc is None:
            return None
        return self.y_unc[self.roi_msk]

    @property
    def dx_roi(self):
        if self.dx is None:
            return None
        return self.dx[self.roi_msk]

    @property
    def roi(self):
        return self._roi

    @property
    def roi_msk(self):
        if self._roi_msk is None:
            return np.ones_like(self.x, dtype=bool)
        else:
            return self._roi_msk

    @property
    def xmode(self):
        return self._xmode

    @property
    def ymode(self):
        return self._ymode

    @property
    def param_names(self):
        return list(self.params.keys())

    def set_data(self, y, x=None, y_unc=None, dx=None, roi=None,
                 update_defaults=True):
        # Set y data (skip if y is None)
        if y is None:
            return
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
        self.y_unc = y_unc
        # set deltax (bin width)
        self.dx = dx

        if roi is not None:
            self.set_roi(*roi)
        if update_defaults:
            self.guess_param_defaults(update=True)

    def set_roi(self, low, high, update_defaults=True):
        """Set the region of interest (ROI) of x-values for the fit.

        Parameters
        ----------
        low : float
            Lower x-value of the ROI
        high : float
            Upper x-value of the ROI
        update_defaults : bool, optional
            If True, recompute default params based on new ROI.
        """

        self._roi = (float(low), float(high))
        self._roi_msk = ((self.x >= self.roi[0]) &
                         (self.x <= self.roi[1]))
        if update_defaults:
            self.guess_param_defaults(update=True)

    def set_param(self, pname, ptype, pvalue):
        self.params[pname].set(**{ptype: pvalue})

    def _translate_model(self, m):
        if inspect.isclass(m):
            if not issubclass(m, Model):
                raise FittingError(
                    'Input model is not a subclass of Model: {}'.format(m))
            self._model_cls_cnt[m] = self._model_cls_cnt.get(m, 0) + 1
            return m
        elif isinstance(m, Model):
            cls = m.__class__
            self._model_cls_cnt[cls] = self._model_cls_cnt.get(cls, 0) + 1
            return m
        elif isstring(m):
            cls = MODEL_STR_TO_CLS.get(m.lower(), None)
            if cls is not None:
                self._model_cls_cnt[cls] = self._model_cls_cnt.get(cls, 0) + 1
                return cls
        raise FittingError('Unknown model type: {}'.format(m))

    def _make_model(self, model):
        if isstring(model) or isinstance(model, Model):
            model = [model]
        # Convert the model(s) to a list of Model classes / Model instancess
        self._model_cls_cnt = {}
        model_translated = [self._translate_model(m) for m in model]
        # Build complete model with appropriate prefixes
        model_prefixes = set()
        models = []
        name = ''
        for m in model_translated:
            if inspect.isclass(m):
                prefix_base = m.__name__.lower()
                if prefix_base.endswith('model'):
                    prefix_base = prefix_base[:-5]
                if self._model_cls_cnt[m] == 1:
                    prefix = '{}_'.format(prefix_base)
                else:
                    for i in range(self._model_cls_cnt[m]):
                        prefix = '{}{}_'.format(prefix_base, i)
                        if prefix not in model_prefixes:
                            break
                m_instance = m(prefix=prefix)
            else:
                m_instance = m
            if m_instance.prefix in model_prefixes:
                raise FittingError(
                    'A model prefix is not unique: ' +
                    '{} '.format(m_instance.prefix) +
                    'All models: {}'.format(model_translated)
                )
            model_prefixes.add(m_instance.prefix)
            models.append(m_instance)
            name += m_instance._name.capitalize()
        # Construct final model
        self._name = name
        self._model = models[0]
        for m in models[1:]:
            self._model += m

    def _guess_param_defaults(self, **kwargs):
        params = []
        for comp in self.model.components:
            params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi)
        return params

    def guess_param_defaults(self, update=False, **kwargs):
        defaults = self._guess_param_defaults(**kwargs)
        if update:
            # TODO: check this logic
            if defaults is not None:
                for dp in defaults:
                    self.set_param(*dp)
        return defaults

    def fit(self, backend='lmfit'):
        """Perform the weighted fit to data.

        Parameters
        ----------
        backend : {'lmfit', 'lmfit-pml'}
            Backend fitting module to use.

        Raises
        ------
        FittingError
            If `backend` is not supported.
        AssertionError
            If self.y is None.
        """

        assert self.y is not None, \
            'No data initialized, did you call set_data?'
        self.result = None
        y_roi_norm = self.y_roi
        if self.dx is not None:
            y_roi_norm = self.y_roi / self.dx_roi
        if backend.lower().strip() == 'lmfit':
            # Perform the fit, weighted by 1/uncertainties.
            weights = self.y_unc_roi ** -1.0
            self.result = self.model.fit(
                y_roi_norm, self.params, x=self.x_roi, weights=weights
            )
        elif backend.lower().strip() == 'lmfit-pml':
            self._set_likelihood_residual()
            # Perform the fit. PML automatically applies 1/sqrt(y) weights, so
            # additional weights here just convert back to counts.
            self.result = self.model.fit(
                self.y_roi, self.params, #self.result.params,
                x=self.x_roi, weights=self.dx_roi,
                fit_kws={'reduce_fcn': lambda r: np.sum(r)},
                method='Nelder-Mead', calc_covar=False
            )  # no, bounds, default would be L-BFGS-B'
            # TODO: Calculate errors breaks minimization right now
        else:
            raise FittingError('Unknown fitting backend: {}'.format(backend))

    def _set_likelihood_residual(self):
        def _likelihood_residual(self, params, data, weights, **kwargs):
            """same as model._residual of lmfit"""
            model = self.eval(params, **kwargs)
            if weights is not None:
                model *= weights
            if self.nan_policy == 'raise' and not np.all(np.isfinite(model)):
                msg = ('The model function generated NaN values and the fit '
                       'aborted! Please check your model function and/or set '
                       'boundaries on parameters where applicable. In cases '
                       'like this, using "nan_policy=\'omit\'" will probably '
                       'not work.')
                raise ValueError(msg)
            mask = model <= 0  # This should not be necessary
            diff = model - scipy.special.xlogy(data, model)
            diff[mask] = 1e32
            if diff.dtype == np.complex:
                # data/model are complex
                diff = diff.ravel().view(np.float)
            return np.asarray(diff).ravel()  # for compatibility with pandas.Series
        # This overwrites the  model residual method, is an ugly hack to make
        # poisson fitting possible. This is not undone for now.
        self.model._residual = _likelihood_residual.__get__(self.model, Model)

    def eval(self, x, params=None, **kwargs):
        return self.model.eval(x=x, params=params, **kwargs)

    def param_val(self, param):
        """
        Value of fit parameter `param`
        """
        if self.result is None:
            return None
        if param in self.result.params:
            return self.result.params[param].value
        elif param in self.fit.best_values:
            return self.result.best_values[param]
        else:
            raise FittingError('Unknown param: {}', param)

    def param_unc(self, param):
        """
        Fit error of fit parameter `param`
        """
        if self.result is None:
            return None
        if param in self.result.params:
            return self.result.params[param].stderr
        elif param in self.result.best_values:
            # This is the case for the `erf_form` key
            return np.nan
        else:
            raise FittingError('Unknown param: {}', param)

    def param_dataframe(self, sort_by_model=False):
        """
        Dataframe of all fit parameters value and fit error
        """
        if self.result is None:
            return None
        df = pd.DataFrame(columns=['val', 'unc'], dtype=np.float)
        for k in self.param_names:
            df.loc[k, 'val'] = self.param_val(k)
            df.loc[k, 'unc'] = self.param_unc(k)
        if sort_by_model:
            df.set_index(
                pd.MultiIndex.from_tuples(
                    [tuple(p.split('_')) for p in df.index],
                    names=['model', 'param']),
                inplace=True)
        return df

    def compute_residuals(self, residual_type='abs'):
        """Compute residuals between the data and the fit.

        Parameters
        ----------
        residual_type : {'abs', 'rel', 'sigma'}, optional
            Residual type to calculate (default: 'abs')
                'abs' : data - fit
                'rel' : (data - fit) / |fit|
                'sigma' : (data - fit) / (data_uncertainty)

        Returns
        -------
        np.ndarray
            Array of residuals
        """
        dx_roi = np.ones_like(self.x_roi) if self.dx_roi is None else self.dx_roi
        y_eval = self.eval(self.x_roi, **self.result.best_values) * dx_roi
        y_res = self.y_roi - y_eval

        if residual_type == 'rel':
            # Residuals relative to the model evaluation
            return y_res / np.abs(y_eval)
        elif residual_type == 'sigma':
            # Residuals relative to the data uncertainty
            return y_res / self.y_unc_roi
        elif residual_type == 'abs':
            # Absolute residuals
            return y_res
        else:
            raise ValueError(
                'Unknown residuals type: {0:s}'.format(residual_type)
            )

    def plot(self, npts=1000, **kwargs):
        """Plot the fit result on the current axis.

        Parameters
        ----------
        npts : int (optional)
            Number of points in x to generate.
        kwargs : dict (optional)
            Additional kwargs passed to plt.plot().

        Returns
        -------
        int
            Description of anonymous integer return value.
        """

        x_plot = np.linspace(self.x_roi[0], self.x_roi[-1], npts)
        y = self.eval(x_plot, **self.result.best_values)
        plt.plot(x_plot, y, **kwargs)

    def custom_plot(self, title=None, savefname=None, title_fontsize=24,
                    title_fontweight='bold', residual_type='abs', **kwargs):
        """Three-panel figure showing fit results.

        Top-left panel shows the data and the fit. Bottom-left shows the fit
        residuals. Right prints fit statistics and correlations.

        Parameters
        ----------
        title : str, optional
            Title of the figure (default: no title)
        savefname : str, optional
            Filename to save the figure as (default: not saved)
        title_fontsize : int, optional
            Title font size (default: 24)
        title_fontweight : str, optional
            Title font weight (default: 'bold')
        residual_type : {'abs', 'rel', 'sigma'}, optional
            Residual type to calculate (default: 'abs')
                'abs' : data - fit
                'rel' : (data - fit) / |fit|
                'sigma' : (data - fit) / (data_uncertainty)
        **kwargs
            Additional kwargs. Currently unused.

        Returns
        -------
        matplotlib figure
            Returned only if savefname is None
        """

        ymin, ymax = self.y_roi.min(), self.y_roi.max()
        # Prepare plots
        dx, dx_roi = self.dx, self.dx_roi
        if dx is None:
            dx = np.ones_like(self.x)
        if dx_roi is None:
            dx_roi = np.ones_like(self.x_roi)
        gs = GridSpec(2, 2, height_ratios=(4, 1))
        gs.update(left=0.05, right=0.99, wspace=0.03, top=0.94, bottom=0.06,
                  hspace=0.06)
        fig = plt.figure(figsize=(18, 9))
        fit_ax = fig.add_subplot(gs[0, 0])
        res_ax = fig.add_subplot(gs[1, 0], sharex=fit_ax)
        txt_ax = fig.add_subplot(gs[:, 1])
        # Set fig title
        if title is not None:
            fig.suptitle(str(title), fontweight=title_fontweight,
                         fontsize=title_fontsize)

        # ---------------------------------------
        # Fit plot (keep track of min/max in roi)
        # ---------------------------------------
        # Smooth roi x values
        x_plot = np.linspace(self.x_roi[0], self.x_roi[-1], 1000)
        # All data (not only roi)
        fit_ax.errorbar(self.x, self.y/dx, yerr=self.y_unc,
                        c='k', fmt='o', markersize=5, alpha=0.1, label='data')
        # Init fit
        y = self.eval(x_plot, **self.result.init_values)
        ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
        fit_ax.plot(x_plot, y, 'k--', label='init')
        # Best fit
        y = self.eval(x_plot, **self.result.best_values)
        ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
        fit_ax.plot(x_plot, y, color='#e31a1c', label='best fit')
        # Components (currently will work for <=3 component)
        colors = ['#1f78b4', '#33a02c', '#6a3d9a']
        for i, m in enumerate(self.result.model.components):
            y = m.eval(x=x_plot, **self.result.best_values)
            if isinstance(y, float):
                y = np.ones(x_plot.shape) * y
            ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
            fit_ax.plot(x_plot, y, label=m.prefix, color=colors[i])
        # Plot Peak center and FWHM
        peak_centers = [self.param_val(p) for p in self.param_names if
                        (p.startswith('gauss') and p.endswith('mu'))]
        peak_fwhms = [self.param_val(p) * FWHM_SIG_RATIO
                      for p in self.param_names if
                      (p.startswith('gauss') and p.endswith('sigma'))]
        for i, (c, f) in enumerate(zip(peak_centers, peak_fwhms)):
            if i == 0:
                label = 'Centroid and FWHM'
            else:
                label = None
            fit_ax.axvline(c, color='#ff7f00')
            fit_ax.axvspan(c - f / 2.0, c + f / 2.0, color='#ff7f00',
                           alpha=0.2, label=label)
        # Misc
        fit_ax.legend(loc='upper right')
        # Set viewing window to only include the roi (not entire spectrum)
        xpad = (self.x_roi[-1] - self.x_roi[0]) * 0.05
        ypad = (ymax - ymin) * 0.05
        fit_ax.set_xlim([self.x_roi[0] - xpad, self.x_roi[-1] + xpad])
        fit_ax.set_ylim([ymin - ypad, ymax + ypad])
        fit_ax.set_ylabel(self.ymode)

        # ---------
        # Residuals
        # ---------
        y_eval = self.eval(self.x_roi, **self.result.best_values) * dx_roi
        res_kwargs = dict(fmt='o', color='k', markersize=5, label='residuals')

        # Y-values of the residual plot, depending on residual_type
        y_plot = self.compute_residuals(residual_type)

        # Error bars and ylabel of the residual plot
        if residual_type == 'rel':
            yerr_plot = self.y_unc_roi / np.abs(y_eval)
            ylabel = 'Relative residuals'
        elif residual_type == 'sigma':
            yerr_plot = np.zeros_like(y_plot)
            ylabel = r'Residuals $(\sigma)$'
        elif residual_type == 'abs':
            yerr_plot = self.y_unc_roi
            ylabel = 'Residuals'
        else:
            raise ValueError(
                'Unknown residuals option: {0:s}'.format(residual_type)
            )
        res_ax.errorbar(x=self.x_roi, y=y_plot, yerr=yerr_plot, **res_kwargs)
        res_ax.axhline(0.0, linestyle='dashed', c='k', linewidth=1.0)
        res_ax.set_ylabel(ylabel)
        res_ax.set_xlabel(self.xmode)

        # -------------------
        # Fit report (txt_ax)
        # -------------------
        txt_ax.get_xaxis().set_visible(False)
        txt_ax.get_yaxis().set_visible(False)
        best_fit_values = ''
        op = self.result.params
        for p in self.result.params:
            if op[p].stderr is None:
                pass
                # TODO: Calculate errors breaks minimization right now
                # warnings.warn(
                #     "Package numdifftools is required to have "
                #     "stderr calculated.", FittingWarning)
            else:
                best_fit_values += '{:15} {: .6e} +/- {:.5e} ({:6.1%})\n'.format(
                   p, op[p].value, op[p].stderr, abs(op[p].stderr / op[p].value))
        best_fit_values += '{:15} {: .6e}\n'.format('Chi Squared:',
                                                    self.result.chisqr)
        best_fit_values += '{:15} {: .6e}'.format('Reduced Chi Sq:',
                                                  self.result.redchi)
        props = dict(boxstyle='round', facecolor='white', edgecolor='black',
                     alpha=1)
        props = dict(facecolor='white', edgecolor='none', alpha=0)
        fp = FontProperties(family='monospace', size=8)
        # Remove first 2 lines of fit report (long model description)
        s = '\n'.join(self.result.fit_report().split('\n')[2:])
        # Add some more parameter details
        s += '\n'
        param_df = self.param_dataframe(sort_by_model=True)
        for model_name, sdf in param_df.groupby(level='model'):
            s += model_name + '\n'
            for (_, param_name), param_data in sdf.iterrows():
                v = param_data['val']
                e = param_data['unc']
                s += '    {:24}: {: .6e} +/- {:.5e} ({:6.1%})\n'.format(
                    param_name, v, e, np.abs(e / v))
        # Add info about the ROI and units
        s += 'ROI: [{0:.3f}, {1:.3f}]\n'.format(*self.roi)
        s += 'X units: {:s}\n'.format(self.xmode if self.xmode else 'None')
        s += 'Y units: {:s}\n'.format(self.ymode if self.ymode else 'None')
        # Add to empty axis
        txt_ax.text(x=0.01, y=0.99, s=s, fontproperties=fp,
                    ha='left', va='top', transform=txt_ax.transAxes,
                    bbox=props)
        if savefname is not None:
            fig.savefig(savefname)
            plt.close(fig)
        else:
            return fig


class FitterGaussGauss(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss0_') + \
            GaussModel(prefix='gauss1_')

    def _guess_param_defaults(self):
        params = []
        for comp in self.model.components:
            if comp.prefix == 'gauss0_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi,
                                     center_ratio=0.33, width_ratio=0.5)
            elif comp.prefix == 'gauss1_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi,
                                     center_ratio=0.66, width_ratio=0.5)
        return params


class FitterGaussGaussLine(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss0_') + \
            GaussModel(prefix='gauss1_') + \
            LineModel(prefix='line_')

    def _guess_param_defaults(self):
        params = []
        for comp in self.model.components:
            if comp.prefix == 'gauss0_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi,
                                     center_ratio=0.33, width_ratio=0.5)
            elif comp.prefix == 'gauss1_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi,
                                     center_ratio=0.66, width_ratio=0.5)
            else:
                params += comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi)
        return params


class FitterGaussGaussExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss0_') + \
            GaussModel(prefix='gauss1_') + \
            ExpModel(prefix='exp_')

    def _guess_param_defaults(self):
        params = []
        for comp in self.model.components:
            if comp.prefix == 'gauss0_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=dx_roi,
                                     center_ratio=0.33, width_ratio=0.5)
            elif comp.prefix == 'gauss1_':
                params += comp.guess(self.y_roi, x=self.x_roi, dx=dx_roi,
                                     center_ratio=0.66, width_ratio=0.5)
            else:
                params += comp.guess(self.y_roi, x=self.x_roi, dx=dx_roi)
        return params
