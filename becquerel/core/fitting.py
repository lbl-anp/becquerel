from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import scipy.special
from lmfit.model import Model
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties


FWHM_SIG_RATIO = np.sqrt(8*np.log(2))  # 2.35482


class FittingError(Exception):
    """Exception raised by Fitters."""
    pass


# -----------------------------------------------------------------------------
# Base functions
# TODO: Should these be replaced with lmfit.models???
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
# -----------------------------------------------------------------------------

# Helper functions for guessing
def _xy_right(y, x=None, num=4):
    if x is not None:
        return np.mean(x[-num:]), np.mean(y[-num:])
    return len(y)*0.5, np.mean(y[-num:])


def _xy_left(y, x=None, num=4):
    if x is not None:
        return np.mean(x[:num]), np.mean(y[:num])
    return len(y)*0.5, np.mean(y[:num])


class ConstantModel(Model):

    def __init__(self, *args, **kwargs):
        super(ConstantModel, self).__init__(constant, *args, **kwargs)
        # TODO: remove this min setting?
        self.set_param_hint('{}c'.format(self.prefix), min=0.)

    def guess(self, y, x=None, num=2):
        return []


class LineModel(Model):

    def __init__(self, *args, **kwargs):
        super(LineModel, self).__init__(line, *args, **kwargs)

    def guess(self, y, x=None, num=2):
        _, b = _xy_left(y, x=x, num=num)
        return [
            ('{}m'.format(self.prefix), 'value', 0.),
            ('{}b'.format(self.prefix), 'value', b),
        ]


class GaussModel(Model):

    def __init__(self, *args, **kwargs):
        super(GaussModel, self).__init__(gauss, *args, **kwargs)
        self.set_param_hint(
            '{}fwhm'.format(self.prefix),
            expr='{} * {}sigma'.format(FWHM_SIG_RATIO, self.prefix))

    def guess(self, y, x=None, center_ratio=0.5, width_ratio=0.5):
        assert center_ratio < 1, \
            'Center mask ratio cannot exceed 1: {}'.format(center_ratio)
        assert width_ratio < 1, \
            'Width mask ratio cannot exceed 1: {}'.format(width_ratio)
        if x is None:
            x = np.arange(0, len(y))
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = ((x >= (mu - xspan * width_ratio)) &
               (x <= mu + xspan * width_ratio))
        # NOTE: this integration assumes y is NOT normalized to dx
        amp = np.sum(y[msk])
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

    def guess(self, y, x=None):
        return [
            ('{}amp'.format(self.prefix), 'value', y[0] - y[-1]),
            ('{}amp'.format(self.prefix), 'min', 0.),
            ('{}mu'.format(self.prefix), 'expr', 'gauss_mu'),
            ('{}sigma'.format(self.prefix), 'expr', 'gauss_sigma'),
        ]


class ExpModel(Model):

    def __init__(self, *args, **kwargs):
        super(ExpModel, self).__init__(exp, *args, **kwargs)
        self.set_param_hint('{}amp'.format(self.prefix), min=0.)
        self.set_param_hint('{}lam'.format(self.prefix), max=0.)

    def guess(self, y, x=None, num=1):
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
    __metaclass__ = ABCMeta

    def __init__(self, x=None, y=None, y_unc=None, roi=None):
        # Model and parameters
        self.make_model()
        self.params = self.model.make_params()
        # Empty fit result
        self.result = None
        # Set data vectors
        if y is None:
            self._x = None
            self._y = None
            self._y_unc = None
        else:
            self.set_data(x=x, y=y, y_unc=y_unc, update_defaults=False)
        # Set roi and roi_msk
        if roi is None:
            self._roi = None
            self._roi_msk = None
        else:
            self.set_roi(*roi, update_defaults=False)
        # Update parameter defaults if possible
        if self.y is not None:
            self.guess_param_defaults(update=True)

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

    @property
    def param_names(self):
        return list(self.params.keys())

    def set_data(self, y, x=None, y_unc=None, update_defaults=True):
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
        if update_defaults:
            self.guess_param_defaults(update=True)

    def set_roi(self, low, high, update_defaults=True):
        self._roi = (float(low), float(high))
        self._roi_msk = ((self.x >= self.roi[0]) &
                         (self.x <= self.roi[1]))
        if update_defaults:
            self.guess_param_defaults(update=True)

    def set_param(self, pname, ptype, pvalue):
        self.params[pname].set(**{ptype: pvalue})

    @abstractmethod
    def make_model(self):
        return

    def _guess_param_defaults(self, **kwargs):
        params = []
        for comp in self.model.components:
            params += comp.guess(self.y_roi, x=self.x_roi)
        return params

    def guess_param_defaults(self, update=False, **kwargs):
        defaults = self._guess_param_defaults(**kwargs)
        if update:
            if defaults is not None:
                for dp in defaults:
                    self.set_param(*dp)
        return defaults

    def fit(self, backend='lmfit'):
        assert self.y is not None, \
            'No data initialized, did you call set_data?'
        self.result = None
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
            raise FittingError('Unknown fitting backend: {}'.format(backend))

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

    def custom_plot(self, title=None, savefname=None, title_fontsize=24,
                    title_fontweight='bold', **kwargs):
        ymin, ymax = self.y_roi.min(), self.y_roi.max()
        # Prepare plots
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
        fit_ax.errorbar(self.x, self.y, yerr=self.y_unc,
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
        # TODO: add ylabel based on units of y
        # Set viewing window to only include the roi (not entire spectrum)
        xpad = (self.x_roi[-1] - self.x_roi[0]) * 0.05
        ypad = (ymax - ymin) * 0.05
        fit_ax.set_xlim([self.x_roi[0] - xpad, self.x_roi[-1] + xpad])
        fit_ax.set_ylim([ymin - ypad, ymax + ypad])
        # ---------
        # Residuals
        # ---------
        res_ax.errorbar(
            self.x_roi,
            self.eval(self.x_roi, **self.result.best_values) - self.y_roi,
            yerr=self.y_unc_roi, fmt='o', color='k',
            markersize=5, label='residuals')
        res_ax.set_ylabel('Residuals')
        # TODO: add xlabel based on units of x
        # -------------------
        # Fit report (txt_ax)
        # -------------------
        txt_ax.get_xaxis().set_visible(False)
        txt_ax.get_yaxis().set_visible(False)
        best_fit_values = ''
        op = self.result.params
        for p in self.result.params:
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
        # Add some more details
        s += '\n'
        param_df = self.param_dataframe(sort_by_model=True)
        for model_name, sdf in param_df.groupby(level='model'):
            s += model_name + '\n'
            for (_, param_name), param_data in sdf.iterrows():
                v = param_data['val']
                e = param_data['unc']
                s += '    {:24}: {: .6e} +/- {:.5e} ({:6.1%})\n'.format(
                    param_name, v, e, e / v)
        # Add to empty axis
        txt_ax.text(x=0.01, y=0.99, s=s, fontproperties=fp,
                    ha='left', va='top', transform=txt_ax.transAxes,
                    bbox=props)
        if savefname is not None:
            fig.savefig(savefname)
            plt.close(fig)
        else:
            return fig


class FitterGaussExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ExpModel(prefix='exp_')


class FitterGaussErfLine(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ErfModel(prefix='erf_') + \
            LineModel(prefix='line_')


class FitterGaussErfExp(Fitter):

    def make_model(self):
        self.model = \
            GaussModel(prefix='gauss_') + \
            ErfModel(prefix='erf_') + \
            ExpModel(prefix='exp_')


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
                params += comp.guess(self.y_roi, x=self.x_roi,
                                     center_ratio=0.33, width_ratio=0.5)
            elif comp.prefix == 'gauss1_':
                params += comp.guess(self.y_roi, x=self.x_roi,
                                     center_ratio=0.66, width_ratio=0.5)
            else:
                params += comp.guess(self.y_roi, x=self.x_roi)
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
                params += comp.guess(self.y_roi, x=self.x_roi,
                                     center_ratio=0.33, width_ratio=0.5)
            elif comp.prefix == 'gauss1_':
                params += comp.guess(self.y_roi, x=self.x_roi,
                                     center_ratio=0.66, width_ratio=0.5)
            else:
                params += comp.guess(self.y_roi, x=self.x_roi)
        return params
