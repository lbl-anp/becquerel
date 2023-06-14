import warnings
import inspect
import numpy as np
import pandas as pd
import scipy.special
from lmfit.model import Model
from lmfit.parameter import Parameters
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from iminuit import Minuit
from uncertainties import ufloat
import numdifftools as nd

FWHM_SIG_RATIO = np.sqrt(8 * np.log(2))  # 2.35482
SQRT_TWO = np.sqrt(2)  # 1.414213562
COLORS = [
    matplotlib.colors.to_rgb(c)
    for c in ["C0", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
]


class FittingError(Exception):
    """Exception raised by Fitters."""


class FittingWarning(UserWarning):
    """Warning raised by Fitters."""


# -----------------------------------------------------------------------------
# Base functions
# -----------------------------------------------------------------------------


def constant(x, c):
    return np.ones_like(x) * c


def line(x, m, b):
    return m * x + b


def gauss(x, amp, mu, sigma):
    return (
        amp
        / sigma
        / np.sqrt(2.0 * np.pi)
        * np.exp(-((x - mu) ** 2.0) / (2.0 * sigma**2.0))
    )


def erf(x, amp, mu, sigma):
    return amp * 0.5 * (1.0 - scipy.special.erf((x - mu) / (SQRT_TWO * sigma)))


def exp(x, amp, lam):
    return amp * np.exp(x / lam)


def gausserf(x, ampgauss, amperf, mu, sigma):
    return gauss(x, ampgauss, mu, sigma) + erf(x, amperf, mu, sigma)


def expgauss(x, amp=1, mu=0, sigma=1.0, gamma=1.0):
    gss = gamma * sigma * sigma
    arg1 = gamma * (mu + gss / 2.0 - x)
    arg2 = (mu + gss - x) / (SQRT_TWO * sigma)
    return amp * (gamma / 2) * np.exp(arg1) * scipy.special.erfc(arg2)


def gauss_dbl_exp(
    x: np.ndarray,
    amp: float,
    mu: float,
    sigma: float,
    ltail_ratio: float,
    ltail_slope: float,
    ltail_cutoff: float,
    rtail_ratio: float,
    rtail_slope: float,
    rtail_cutoff: float,
):
    """A Gaussian with exponential tails added to either side of the peak. This
    is an extension of the Gaussian with a low-side exponential tail described
    by Namboodiri et al here: https://www.osti.gov/biblio/392720

    The function is suitable for spectra with asymmetric peaks, such as those
    from CZT crystals. The exponential tails are characterized by 3 parameters: a
    tail-to-peak amplitude ratio, a slope, and a cutoff parameter.

    Parameters
    ----------
    x : numpy.ndarray
        X-data
    amp : float
        Amplitude parameter
    mu: float
        Mean parameter
    sigma: float
        Width parameter
    ltail_ratio: float
        Left-side exponential tail amplitude / Amplitude
    ltail_slope: float
        Left-side exponential tail slope, affects "steepness" of the tail
    ltail_cutoff: float
        Left-side exponential tail cutoff, affects "length" of the tail
    rtail_ratio: float
        Right-side exponential tail amplitude / Amplitude
    rtail_slope: float
        Right-side exponential tail slope, affects "steepness" of the tail
    rtail_cutoff: float
        Right-side exponential tail cutoff, affects "length" of the tail

    Returns
    -------
    numpy.ndarray
        Y-data
    """
    alpha = -1.0 / (2.0 * sigma**2)
    ltail_func = np.zeros(shape=(len(x),))
    rtail_func = np.zeros(shape=(len(x),))
    # l and r tail_func are 0 when we're below/above the Gaussian mean
    # "heavyside convolution"
    mask = x < mu
    ltail_func[mask] = amp * ltail_ratio * np.exp(ltail_slope * ((x[mask] - mu)))
    ltail_func[mask] *= -np.expm1(ltail_cutoff * alpha * ((x[mask] - mu) ** 2))
    mask = x > mu
    rtail_func[mask] = amp * rtail_ratio * np.exp(rtail_slope * ((x[mask] - mu)))
    rtail_func[mask] *= -np.expm1(rtail_cutoff * alpha * ((x[mask] - mu) ** 2))
    return amp * np.exp(alpha * ((x - mu) ** 2)) + ltail_func + rtail_func


# -----------------------------------------------------------------------------
# Custom loss functions
# -----------------------------------------------------------------------------


def poisson_loss(y_eval, y_data):
    return np.sum(y_eval - scipy.special.xlogy(y_data, y_eval))


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
    return len(y) * 0.5, np.mean(y[-num:])


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
    return len(y) * 0.5, np.mean(y[:num])


def _is_count_like(y):
    """Determine whether data 'looks like' it could be counts.

    Parameters
    ----------
    y : array-like
        Y-data
    """
    if np.any(y < 0):
        return False
    if not np.allclose(np.rint(y).astype(int), y):
        return False
    return True


class ConstantModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(constant, *args, **kwargs)
        self.set_param_hint(f"{self.prefix}c", min=0.0)

    def guess(self, y, x=None, dx=None, num=2):
        if dx is None:
            dx = np.ones_like(x)
        c = (y[-1] / dx[-1] + y[0] / dx[0]) * 0.5
        return [
            (f"{self.prefix}c", "value", c),
            (f"{self.prefix}c", "min", 0.0),
        ]


class LineModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(line, *args, **kwargs)

    def guess(self, y, x=None, dx=None, num=2):
        if dx is None:
            dx = np.ones_like(x)
        m = (y[-1] / dx[-1] - y[0] / dx[0]) / (x[-1] - x[0])
        b = ((y[-1] / dx[-1] + y[0] / dx[0]) - m * (x[1] + x[0])) * 0.5
        return [
            (f"{self.prefix}m", "value", m),
            (f"{self.prefix}b", "value", b),
        ]


class GaussModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(gauss, *args, **kwargs)
        self.set_param_hint(
            f"{self.prefix}fwhm",
            expr=f"{FWHM_SIG_RATIO} * {self.prefix}sigma",
        )

    def guess(self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.05):
        assert center_ratio < 1, f"Center mask ratio cannot exceed 1: {center_ratio}"
        assert (
            width_ratio < 1.0 and width_ratio > 0.0
        ), f"Width ratio must be between 0.0 and 1.0: {width_ratio}"

        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)

        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = (x >= (mu - xspan * width_ratio)) & (x <= mu + xspan * width_ratio)

        sigma = xspan * width_ratio
        amp = np.max(y[msk]) * np.sqrt(2 * np.pi) * sigma  # new amplitude guess
        return [
            (f"{self.prefix}amp", "value", amp),
            (f"{self.prefix}amp", "min", 0.0),
            (f"{self.prefix}mu", "value", mu),
            (f"{self.prefix}mu", "min", x[0]),
            (f"{self.prefix}mu", "max", x[-1]),
            (f"{self.prefix}sigma", "value", sigma),
            (f"{self.prefix}sigma", "min", 0.0),
        ]


class ErfModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(erf, *args, **kwargs)

    def guess(self, y, x=None, dx=None, center_ratio=0, width_ratio=0.05):
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        sigma = xspan * width_ratio
        return [
            (f"{self.prefix}amp", "value", y[0] - y[-1]),
            (f"{self.prefix}mu", "value", mu),
            (f"{self.prefix}sigma", "value", sigma),
        ]


class GaussErfModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(gausserf, *args, **kwargs)
        self.set_param_hint(
            f"{self.prefix}fwhm",
            expr=f"{FWHM_SIG_RATIO} * {self.prefix}sigma",
        )

    def guess(
        self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.05, amp_ratio=0.9
    ):
        assert center_ratio < 1, f"Center mask ratio cannot exceed 1: {center_ratio}"
        assert (
            width_ratio < 1.0 and width_ratio > 0.0
        ), f"Width ratio must be between 0.0 and 1.0: {width_ratio}"
        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = (x >= (mu - xspan * width_ratio)) & (x <= mu + xspan * width_ratio)

        sigma = xspan * width_ratio
        amp = np.max(y[msk]) * np.sqrt(2 * np.pi) * sigma
        amp_gauss = amp * amp_ratio
        amp_erf = amp * (1.0 - amp_ratio) / dx[0] / (np.sqrt(2 * np.pi) * sigma)
        return [
            (f"{self.prefix}ampgauss", "value", amp_gauss),
            (f"{self.prefix}ampgauss", "min", 0.0),
            (f"{self.prefix}amperf", "value", amp_erf),
            (f"{self.prefix}amperf", "min", 0.0),
            (f"{self.prefix}mu", "value", mu),
            (f"{self.prefix}mu", "min", x[0]),
            (f"{self.prefix}mu", "max", x[-1]),
            (f"{self.prefix}sigma", "value", sigma),
            (f"{self.prefix}sigma", "min", 0.0),
        ]


class ExpModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(exp, *args, **kwargs)
        self.set_param_hint(f"{self.prefix}amp", min=0.0)
        self.set_param_hint(f"{self.prefix}lam", max=0.0)

    def guess(self, y, x=None, dx=None, num=1):
        xl, yl = _xy_left(y, x=x, num=num)
        xr, yr = _xy_right(y, x=x, num=num)
        lam = (xl - xr) / np.log(yl / (yr + 0.0001))
        amp = yl / np.exp(xl / lam)
        return [
            (f"{self.prefix}lam", "value", lam),
            (f"{self.prefix}lam", "max", -1e-3),
            (f"{self.prefix}amp", "value", amp),
            (f"{self.prefix}amp", "min", 0.0),
        ]


class ExpGaussModel(Model):
    """A model of an Exponentially modified Gaussian distribution
    (see https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(expgauss, **kwargs)
        self.set_param_hint(f"{self.prefix}sigma", min=0)
        self.set_param_hint(f"{self.prefix}gamma", min=0, max=1)
        # TODO: This is obviously wrong but best I can think of
        self.set_param_hint(
            f"{self.prefix}fwhm",
            expr=f"{FWHM_SIG_RATIO} * {self.prefix}sigma",
        )

    def guess(self, y, x=None, dx=None, center_ratio=0.5, width_ratio=0.05):
        assert center_ratio < 1, f"Center mask ratio cannot exceed 1: {center_ratio}"
        assert (
            width_ratio < 1.0 and width_ratio > 0.0
        ), f"Width ratio must be between 0.0 and 1.0:: {width_ratio}"
        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)
        xspan = x[-1] - x[0]
        mu = x[0] + xspan * center_ratio
        msk = (x >= (mu - xspan * width_ratio)) & (x <= mu + xspan * width_ratio)

        sigma = xspan * width_ratio
        amp = np.max(y[msk]) * np.sqrt(2 * np.pi) * sigma
        return [
            (f"{self.prefix}amp", "value", amp),
            (f"{self.prefix}amp", "min", 0.0),
            (f"{self.prefix}mu", "value", mu),
            (f"{self.prefix}mu", "min", x[0]),
            (f"{self.prefix}mu", "max", x[-1]),
            (f"{self.prefix}sigma", "value", sigma),
            (f"{self.prefix}sigma", "min", 0.0),
            (f"{self.prefix}gamma", "value", 0.95),
            (f"{self.prefix}gamma", "min", 0.0),
        ]


class GaussDblExpModel(Model):
    """This model is a Gaussian combined with exponential tails on either side
    of the center of the distribution. Each tail is characterized by 3 parameters:
    an amplitude ratio, a slope, and a cutoff. This is an extension of the peak
    shape described by Namboodiri et al here: https://www.osti.gov/biblio/392720
    to fit CZT energy spectra. In their case, the exponential tail was only on
    the low side of the peak.

    A distribution with the exponential tail on only one side may be obtained by
    fixing the undesired side's parameters to 0.

    The tail amplitudes are expressed as ratios with respect to the peak
    amplitude (tail_amplitude / amplitude).

    The slope parameter impacts how steep the tail is, and the cutoff parameter
    impacts how far out it goes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(gauss_dbl_exp, **kwargs)
        self.set_param_hint(f"{self.prefix}amp", min=0)
        self.set_param_hint(f"{self.prefix}mu", min=0)
        self.set_param_hint(f"{self.prefix}sigma", min=0)
        self.set_param_hint(f"{self.prefix}ltail_ratio", min=0)
        self.set_param_hint(f"{self.prefix}rtail_ratio", min=0)

    def guess(self, y, x=None, dx=None):
        if x is None:
            x = np.arange(0, len(y))
        if dx is None:
            dx = np.ones_like(x)
        mu = np.mean(x)
        amp = np.max(y)
        return [
            (f"{self.prefix}amp", "value", amp),
            (f"{self.prefix}amp", "min", 0.0),
            (f"{self.prefix}mu", "value", mu),
            (f"{self.prefix}mu", "min", x[0]),
            (f"{self.prefix}mu", "max", x[-1]),
            (f"{self.prefix}sigma", "value", 1.0),
            (f"{self.prefix}sigma", "min", 0.0),
            (f"{self.prefix}ltail_ratio", "value", 0.1),
            (f"{self.prefix}ltail_ratio", "min", 0.0),
            (f"{self.prefix}ltail_slope", "value", 0.05),
            (f"{self.prefix}ltail_cutoff", "value", 1.0),
            (f"{self.prefix}rtail_ratio", "value", 0.1),
            (f"{self.prefix}rtail_ratio", "min", 0.0),
            (f"{self.prefix}rtail_slope", "value", -0.05),
            (f"{self.prefix}rtail_cutoff", "value", 1.0),
        ]


MODEL_STR_TO_CLS = {
    "constant": ConstantModel,
    "line": LineModel,
    "gauss": GaussModel,
    "gausserf": GaussErfModel,
    "erf": ErfModel,
    "exp": ExpModel,
    "expgauss": ExpGaussModel,
    "gaussdblexp": GaussDblExpModel,
}


def _parameters_to_bq_guess(params: Parameters):
    """Convert a Parameters object to the tuple format becquerel expects for guess().

    Parameters
    ----------
    params : lmfit.parameter.Parameters
        Parameters object to convert

    Returns
    -------
    list of tuples
        Guess, min, and max values for each parameter.
    """
    s = []
    for k, v in params.items():
        s.append((k, "value", v.value))
        s.append((k, "min", v.min))
        s.append((k, "max", v.max))
    return s


# -----------------------------------------------------------------------------
# Fitters
# -----------------------------------------------------------------------------


class Fitter:
    """Base class for more specialized fit objects.

    A note on interpreting fit results: ascribing meaning to histogram fit
    parameters is notoriously tricky, since the y-scale has units of counts
    per bin width, not just counts. The user may need to divide by the
    histogram bin width for area- or height-like parameters if the histogram is
    not already normalized by bin width. See, e.g., p. 171 of Bevington and
    Robinson, "Data reduction and error analysis for the physical sciences".
    """

    def __init__(self, model, x=None, y=None, y_unc=None, dx=None, roi=None, mask=None):
        # Initialize
        self._model = None
        self._name = None
        self._x = None
        self._y = None
        self._y_unc = None
        self._roi = None
        self._roi_msk = None
        self._mask = None
        self._xmode = None
        self._ymode = None
        self.result = None
        self.dx = None
        # Model and parameters
        self._make_model(model)
        self.params = self.model.make_params()
        # Set data
        self.set_data(
            x=x, y=y, y_unc=y_unc, dx=dx, roi=roi, mask=mask, update_defaults=True
        )

    def __str__(self):
        return (
            "bq.Fitter instance\n"
            + f"     name: {self.name}\n"
            + f"    model: {self.model}\n"
            + f"        x: {self.x}\n"
            + f"        y: {self.y}\n"
            + f"    y_unc: {self.y_unc}\n"
            + f"    xmode: {self.xmode}\n"
            + f"    ymode: {self.ymode}\n"
            + f"       dx: {self.dx}\n"
            + f"      roi: {self.roi}"
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
                "No y uncertainties (y_unc) provided. The fit will not be "
                + "weighted causing in poor results at low counting statistics.",
                FittingWarning,
            )
        return self._y_unc

    @y_unc.setter
    def y_unc(self, y_unc):
        if y_unc is not None:
            self._y_unc = np.asarray(y_unc, dtype=float)
            assert len(self.x) == len(self._y_unc), (
                f"Fitting x (len {len(self.x)}) does not match y_unc "
                "(len {len(self._y_unc)})"
            )
            if np.any(self._y_unc <= 0.0):
                min_v = np.min(self._y_unc[self._y_unc > 0.0])
                warnings.warn(
                    "Negative or zero uncertainty not supported. Changing "
                    + f"them to {min_v}. If you have Poisson data, "
                    + "this should be 1."
                )
                self._y_unc[self._y_unc <= 0.0] = min_v
        else:
            self._y_unc = None

    @property
    def x_roi(self):
        return self.x[self.roi_msk & self.mask]

    @property
    def y_roi(self):
        return self.y[self.roi_msk & self.mask]

    @property
    def y_unc_roi(self):
        if self.y_unc is None:
            return None
        return self.y_unc[self.roi_msk & self.mask]

    @property
    def dx_roi(self):
        if self.dx is None:
            return None
        return self.dx[self.roi_msk & self.mask]

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
    def mask(self):
        if self._mask is None:
            return np.ones_like(self.x, dtype=bool)
        else:
            return self._mask

    @property
    def xmode(self):
        return self._xmode

    @property
    def ymode(self):
        return self._ymode

    @property
    def param_names(self):
        return list(self.params.keys())

    def set_data(
        self, y, x=None, y_unc=None, dx=None, roi=None, update_defaults=True, mask=None
    ):
        # Set y data (skip if y is None)
        if y is None:
            return
        self._y = np.asarray(y)
        # Set x data
        if x is None:
            self._x = np.arange(len(self.y))
        else:
            self._x = np.asarray(x)
            assert len(self.x) == len(
                self.y
            ), f"Fitting x (len {len(self.x)}) does not match y (len {len(self.y)})"
        # Handle y uncertainties
        self.y_unc = y_unc
        # set deltax (bin width)
        self.dx = dx
        # fit mask
        self._mask = mask

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
        self._roi_msk = (self.x >= self.roi[0]) & (self.x <= self.roi[1])
        if update_defaults:
            self.guess_param_defaults(update=True)

    def set_param(self, pname, ptype, pvalue):
        self.params[pname].set(**{ptype: pvalue})  # FIXME set(ptype, pvalue) ?

    def _translate_model(self, m):
        if inspect.isclass(m):
            if not issubclass(m, Model):
                raise FittingError(f"Input model is not a subclass of Model: {m}")
            self._model_cls_cnt[m] = self._model_cls_cnt.get(m, 0) + 1
            return m
        elif isinstance(m, Model):
            cls = m.__class__
            self._model_cls_cnt[cls] = self._model_cls_cnt.get(cls, 0) + 1
            return m
        elif isinstance(m, str):
            cls = MODEL_STR_TO_CLS.get(m.lower(), None)
            if cls is not None:
                self._model_cls_cnt[cls] = self._model_cls_cnt.get(cls, 0) + 1
                return cls
        raise FittingError(f"Unknown model type: {m}")

    def _make_model(self, model):
        if isinstance(model, str) or isinstance(model, Model):
            model = [model]
        # Convert the model(s) to a list of Model classes / Model instancess
        self._model_cls_cnt = {}
        model_translated = [self._translate_model(m) for m in model]
        # Build complete model with appropriate prefixes
        model_prefixes = set()
        models = []
        name = ""
        for m in model_translated:
            if inspect.isclass(m):
                prefix_base = m.__name__.lower()
                if prefix_base.endswith("model"):
                    prefix_base = prefix_base[:-5]
                if self._model_cls_cnt[m] == 1:
                    prefix = f"{prefix_base}_"
                else:
                    for i in range(self._model_cls_cnt[m]):
                        prefix = f"{prefix_base}{i}_"
                        if prefix not in model_prefixes:
                            break
                m_instance = m(prefix=prefix)
            else:
                m_instance = m
            if m_instance.prefix in model_prefixes:
                raise FittingError(
                    "A model prefix is not unique: "
                    + f"{m_instance.prefix} "
                    + f"All models: {model_translated}"
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
            p = comp.guess(self.y_roi, x=self.x_roi, dx=self.dx_roi)
            if isinstance(p, Parameters):
                p = _parameters_to_bq_guess(p)
            elif p is None:
                raise TypeError()
            params += p
        return params

    def guess_param_defaults(self, update=False, **kwargs):
        defaults = self._guess_param_defaults(**kwargs)
        if update:
            if defaults is not None:
                for dp in defaults:
                    self.set_param(*dp)
        return defaults

    def fit(self, backend="lmfit", guess=None, limits=None):
        """Perform the weighted fit to data.

        Parameters
        ----------
        backend : {'lmfit', 'lmfit-pml', '[i]minuit-pml'}
            Backend fitting module to use.
        guess : dict of {str: numeric}, optional
            User-specified parameter guesses that override guesses provided by
            the Model.guess() or Fitter.guess_param_defaults() methods.
            Currently only implemented for backend="lmfit" and "iminuit".
        limits : dict of {str: tuple}, optional
            User-specified parameter limits that override limits provided by
            the Model.guess() or Fitter.guess_param_defaults() methods.
            Currently only implemented for backend="lmfit" and "iminuit".

        Raises
        ------
        FittingError
            If `backend` is not supported.
        AssertionError
            If self.y is None.
        """

        assert self.y is not None, "No data initialized, did you call set_data?"
        self.result = None
        y_roi_norm = self.y_roi
        if self.dx is not None:
            y_roi_norm = self.y_roi / self.dx_roi

        self.backend = backend.lower().strip()

        if self.backend == "lmfit":
            # Perform the fit, weighted by 1/uncertainties.
            weights = self.y_unc_roi**-1.0
            self.result = self.model.fit(
                y_roi_norm,
                self.params,
                x=self.x_roi,
                weights=weights,
            )

        elif self.backend == "lmfit-pml":
            self._set_likelihood_residual()
            # Perform the fit. PML automatically applies 1/sqrt(y) weights, so
            # additional weights here just convert back to counts.
            self.result = self.model.fit(
                self.y_roi,
                self.params,  # self.result.params,
                x=self.x_roi,
                weights=self.dx_roi,
                fit_kws={"reduce_fcn": lambda r: np.sum(r)},
                method="Nelder-Mead",
                calc_covar=False,
            )  # no, bounds, default would be L-BFGS-B'
            # NOTE: Calculating errors in lmfit-pml breaks minimization

        elif self.backend in ["iminuit", "minuit"]:
            raise NotImplementedError(
                f"Backend {self.backend} with least-squares loss not yet "
                + f"supported. Use {self.backend}-pml for Poisson loss or "
                + "lmfit for least-squares."
            )

        elif self.backend in ["iminuit-pml", "minuit-pml"]:
            # Translate a model from lmfit to minuit

            if not _is_count_like(self.y_roi):
                warnings.warn(
                    "Passing non-count-like data to a Poisson loss fit", FittingWarning
                )

            def model_loss(*args):
                """Poisson loss given the model specified by args."""
                # Convert args to kwargs as lmfit.model.eval _requires_ kwargs,
                # while the cost function passed to the Minuit object seems to
                # _require_ args.
                kwargs = {self.model.param_names[i]: arg for i, arg in enumerate(args)}
                y_eval = self.model.eval(x=self.x_roi, **kwargs)
                if self.dx_roi is not None:
                    y_eval *= self.dx_roi
                return poisson_loss(y_eval, self.y_roi)

            # Filter out fixed params and have one consistent variable name
            # instead of all the params / param_names / parameters / etc.
            free_vars = [p for p in self.params if self.params[p].vary]

            # Parameter guesses
            try:
                # First try parsing the lmfit guess into iminuit
                guess_l = self.guess_param_defaults()
                guess_i = {
                    g[0]: g[2]
                    for g in guess_l
                    if (g[1] == "value" and g[0] in free_vars)
                }
            except NotImplementedError:
                # If the model/component does not have a guess() method
                guess_i = {}
            # Then override with user input
            if guess is not None:
                guess_i.update(guess)
                del guess  # eliminate subscript confusion

            # Handle user parameter limits
            # Supplying guesses and/or bounds seems more important for minuit
            try:
                # First try parsing the lmfit limits into iminuit
                limits_l = self.guess_param_defaults()
                min_vals, max_vals = {}, {}
                for lim in limits_l:
                    if lim[1] == "min":
                        min_vals[lim[0]] = lim[2]
                    elif lim[1] == "max":
                        max_vals[lim[0]] = lim[2]
                limits_i = {
                    p: (min_vals.get(p, None), max_vals.get(p, None)) for p in free_vars
                }
            except NotImplementedError:
                # If the model/component does not have a guess() method
                limits_i = {}
            # Then override with user input
            if limits is not None:
                limits_i.update(limits)
                del limits  # eliminate subscript confusion

            # Since Minuit requires guesses for every parameter, if we don't
            # have a guess, use limits_i midpoint, or zero barring that.
            for p in free_vars:
                if p not in guess_i:
                    warn_str = f"No guess provided for parameter {p}. "
                    if (
                        p in limits_i
                        and limits_i[p][0] is not None
                        and limits_i[p][1] is not None
                    ):
                        warnings.warn(warn_str + "Setting to limits midpoint.")
                        guess_i[p] = 0.5 * (limits_i[p][0] + limits_i[p][1])
                    else:
                        warnings.warn(warn_str + "Setting to 0.")
                        guess_i[p] = 0.0

            # Set up the Minuit minimizer with initial guess
            self.result = Minuit(model_loss, name=free_vars, **guess_i)

            # Set user parameter limits
            for k, v in limits_i.items():
                self.result.limits[k] = v

            # Specify proper error definition for likelihood model
            self.result.errordef = Minuit.LIKELIHOOD

            # Specify extra checks
            self.result.strategy = 2

            # Run the minimizer
            self.result.migrad()

            # Compute errors
            self.result.hesse()
            try:
                self.result.minos()
            except RuntimeError as exc:
                warnings.warn(f"Could not compute Minos errors: {exc}")

            # Update the interfaces to best_values and init_values
            self._best_values, self._init_values = {}, {}
            for i in range(self.result.npar):
                p = self.result.parameters[i]
                self._best_values[p] = self.result.values[i]
                self._init_values[p] = self.result.init_params[p].value

            # Arg order sanity checks
            assert list(self._best_values.keys()) == free_vars
            assert list(self._init_values.keys()) == free_vars
            assert list(self.result.parameters) == free_vars

        else:
            raise FittingError(f"Unknown fitting backend: {backend}")

    def _set_likelihood_residual(self):
        def _likelihood_residual(self, params, data, weights, **kwargs):
            """same as model._residual of lmfit"""
            model = self.eval(params, **kwargs)
            if weights is not None:
                model *= weights
            if self.nan_policy == "raise" and not np.all(np.isfinite(model)):
                msg = (
                    "The model function generated NaN values and the fit "
                    "aborted! Please check your model function and/or set "
                    "boundaries on parameters where applicable. In cases "
                    "like this, using \"nan_policy='omit'\" will probably "
                    "not work."
                )
                raise ValueError(msg)
            mask = model <= 0  # This should not be necessary
            diff = model - scipy.special.xlogy(data, model)
            diff[mask] = 1e32
            if np.issubdtype(diff.dtype, complex):
                # data/model are complex
                diff = diff.ravel().view(float)
            return np.asarray(diff).ravel()  # for compatibility with pandas.Series

        # This overwrites the  model residual method, is an ugly hack to make
        # poisson fitting possible. This is not undone for now.
        self.model._residual = _likelihood_residual.__get__(self.model, Model)

    def eval(self, x, params=None, **kwargs):
        return self.model.eval(x=x, params=params, **kwargs)

    def calc_area_and_unc(self, component=None, x=None):
        """Calculate the area (and uncertainty) under the fit (or component thereof).

        Parameters
        ----------
        component : Model or str, optional
            Model component (or name thereof) for which to calculate the area, by
            default None, in which case the entire model is used.
        x : array-like, optional
            x values to use for function evaluation, by default `self.x`. Note that
            changing x can easily lead to incorrect results if done carelessly, as the
            spacing of x values needs to be the same as used to compute the fit. Thus
            one should only pass in `self.x`, `self.x_roi`, or slices thereof.

        Returns
        -------
        uncertainties.ufloat
            Area under the fit and its 1-sigma uncertainty.

        References
        ----------
        .. Tellinghuisen, J. (2001). Statistical error propagation. The Journal of
           Physical Chemistry A, 105(15), 3917-3921.

        """

        if "lmfit" in self.backend:
            warnings.warn(
                "`lmfit` error estimates are unreliable. "
                "`minuit` is recommended where possible",
                FittingWarning,
            )

        def _calc_area(param_vec, **kwargs):
            """Internal function to compute the area given the fit values."""
            param_dict = {name: val for (name, val) in zip(kwargs["names"], param_vec)}
            return kwargs["model"].eval(x=kwargs["xvals"], **param_dict).sum()

        # Handle input defaults
        xvals = self.x if x is None else x
        dxvals = np.diff(xvals)
        if not np.allclose(dxvals[0], dxvals):
            raise NotImplementedError("Non-linear x spacing is not supported.")
        if not np.allclose(dxvals[0], np.diff(self.x)):
            raise FittingError("dx values must match those used in the fit")

        if component is None:
            # Use the entire model (i.e., all components)
            model = self.model
        elif isinstance(component, str):
            # Look up the component based on its name
            idx = [
                component.prefix.strip("_") for component in self.model.components
            ].index(component)
            model = self.model.components[idx]
        else:
            # Use the Model object itself
            model = component

        # Format param names/values so the gradient calculation can handle _calc_area
        names = [name for name in self.param_names if self.params[name].vary]
        values = [self.param_val(name) for name in names]

        # Compute the area under the curve
        area = _calc_area(values, xvals=xvals, model=model, names=names)

        # Compute the gradient with respect to the best fit parameters
        grad = nd.Gradient(_calc_area)
        g = np.atleast_2d(grad(values, xvals=xvals, model=model, names=names)).T

        # Compute the gradient with respect to the best fit parameters
        grad = nd.Gradient(_calc_area)
        g = np.atleast_2d(grad(values, xvals=xvals, model=model, names=names)).T

        # Compute the variance in the area estimate: Tellinghuisen Eq. 1
        if self.covariance is None or np.allclose(self.covariance, 0.0):
            warnings.warn(
                "The covariance could not be estimated. Returning 0 for error estimate",
                FittingWarning,
            )
            covariance = np.zeros((len(g), len(g)))
        else:
            covariance = self.covariance
        area_variance = g.T @ covariance @ g
        area_variance = area_variance[0, 0]
        # We don't divide by the binwidth here because we are summing bins: if we double
        # the binwidth, we double the counts per bin but halve the number of bins.
        return ufloat(area, np.sqrt(area_variance))

    def param_val(self, param):
        """
        Value of fit parameter `param`
        """
        if self.result is None:
            return None
        if "lmfit" in self.backend:
            if param in self.result.params:
                return self.result.params[param].value
            elif param in self.result.best_values:
                return self.result.best_values[param]
            else:
                raise FittingError(f"Unknown param: {param}")
        elif "minuit" in self.backend:
            if param in self.result.parameters:
                return self.result.params[param].value
            raise FittingError(f"Unknown param: {param}")
        raise FittingError(f"Unknown backend: {self.backend}")

    def param_unc(self, param):
        """
        Fit error of fit parameter `param`
        """
        if self.result is None:
            return None
        if "lmfit" in self.backend:
            if param in self.result.params:
                return self.result.params[param].stderr
            elif param in self.result.best_values:
                # This is the case for the `erf_form` key
                return np.nan
            else:
                raise FittingError(f"Unknown param: {param}")
        elif "minuit" in self.backend:
            if param in self.result.parameters:
                return self.result.params[param].error  # TODO minos vs hesse?
            raise FittingError(f"Unknown param: {param}")
        raise FittingError(f"Unknown backend: {self.backend}")

    def param_rel_unc(self, param):
        """
        Relative error of fit parameter `param`
        """
        if self.param_unc(param):
            return self.param_unc(param) / self.param_val(param)
        return None

    @property
    def best_values(self):
        """Wrapper for dictionary of best_values."""
        if "lmfit" in self.backend:
            return self.result.best_values
        elif "minuit" in self.backend:
            return self._best_values
        raise FittingError(f"Unknown backend: {self.backend}")

    @property
    def init_values(self):
        """Wrapper for dictionary of init_values."""
        if "lmfit" in self.backend:
            return self.result.init_values
        elif "minuit" in self.backend:
            return self._init_values
        raise FittingError(f"Unknown backend: {self.backend}")

    @property
    def success(self):
        if "lmfit" in self.backend:
            return self.result.success
        elif "minuit" in self.backend:
            return self.result.valid
        raise FittingError(f"Unknown backend: {self.backend}")

    @property
    def covariance(self):
        """Wrapper for fit covariance matrix."""
        if "lmfit" in self.backend:
            return self.result.covar
        elif "minuit" in self.backend:
            return self.result.covariance
        raise FittingError(f"Unknown backend: {self.backend}")

    def param_dataframe(self, sort_by_model=False):
        """
        Dataframe of all fit parameters value and fit error
        """
        if self.result is None:
            return None
        df = pd.DataFrame(columns=["val", "unc"], dtype=float)
        for k in self.param_names:
            df.loc[k, "val"] = self.param_val(k)
            df.loc[k, "unc"] = self.param_unc(k)
        if sort_by_model:
            df.set_index(
                pd.MultiIndex.from_tuples(
                    # Only split on the first underscore, in case the param name
                    # has an underscore in it
                    [tuple(p.split("_", maxsplit=1)) for p in df.index],
                    names=["model", "param"],
                ),
                inplace=True,
            )
        return df

    def compute_residuals(self, residual_type="abs"):
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
        y_eval = self.eval(self.x_roi, **self.best_values) * dx_roi
        y_res = self.y_roi - y_eval

        if residual_type == "rel":
            # Residuals relative to the model evaluation
            return y_res / np.abs(y_eval)
        elif residual_type == "sigma":
            # Residuals relative to the data uncertainty
            return y_res / self.y_unc_roi
        elif residual_type == "abs":
            # Absolute residuals
            return y_res
        else:
            raise ValueError(f"Unknown residuals type: {residual_type:s}")

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
        y = self.eval(x_plot, **self.best_values)
        plt.plot(x_plot, y, **kwargs)

    def custom_plot(
        self,
        title=None,
        savefname=None,
        title_fontsize=24,
        title_fontweight="bold",
        residual_type="abs",
        **kwargs,
    ):
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
        gs.update(
            left=0.05, right=0.99, wspace=0.03, top=0.94, bottom=0.06, hspace=0.06
        )
        fig = plt.figure(figsize=(18, 9))
        fit_ax = fig.add_subplot(gs[0, 0])
        res_ax = fig.add_subplot(gs[1, 0], sharex=fit_ax)
        txt_ax = fig.add_subplot(gs[:, 1])
        txt_ax.get_xaxis().set_visible(False)
        txt_ax.get_yaxis().set_visible(False)
        # Set fig title
        if title is not None:
            fig.suptitle(
                str(title), fontweight=title_fontweight, fontsize=title_fontsize
            )

        # ---------------------------------------
        # Fit plot (keep track of min/max in roi)
        # ---------------------------------------
        # Smooth roi x values
        x_plot = np.linspace(self.x_roi[0], self.x_roi[-1], 1000)
        # All data (not only roi)
        fit_ax.errorbar(
            self.x,
            self.y / dx,
            yerr=self.y_unc,
            c="k",
            fmt="o",
            markersize=5,
            alpha=0.1,
            label="data",
        )
        # Init fit
        y = self.eval(x_plot, **self.init_values)
        ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
        fit_ax.plot(x_plot, y, "k--", label="init")
        # Best fit
        y = self.eval(x_plot, **self.best_values)
        ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
        fit_ax.plot(x_plot, y, color="#e31a1c", label="best fit", zorder=10)
        # Plot 1 sigma uncertainty bands
        # TODO: for minuit, see how_to_draw_error_bands iminuit notebook
        if self.success and "lmfit" in self.backend:
            yunc = self.result.eval_uncertainty(x=x_plot, sigma=1)
            fit_ax.fill_between(
                x_plot,
                y - yunc,
                y + yunc,
                label=r"$\pm 1\, \sigma$",
                color="#e31a1c",
                alpha=0.2,
                zorder=9,
            )
        # Components
        fit_ax.set_prop_cycle(color=COLORS)
        for i, m in enumerate(self.model.components):
            y = m.eval(x=x_plot, **self.best_values)
            if isinstance(y, float):
                y = np.ones(x_plot.shape) * y
            ymin, ymax = min(y.min(), ymin), max(y.max(), ymax)
            fit_ax.plot(x_plot, y, label=m.prefix)
        # Plot Peak center and FWHM
        peak_centers = [
            self.param_val(p)
            for p in self.param_names
            if (p.startswith("gauss") and p.endswith("mu"))
        ]
        peak_fwhms = [
            self.param_val(p) * FWHM_SIG_RATIO
            for p in self.param_names
            if (p.startswith("gauss") and p.endswith("sigma"))
        ]
        for i, (c, f) in enumerate(zip(peak_centers, peak_fwhms)):
            if i == 0:
                label = "Centroid and FWHM"
            else:
                label = None
            fit_ax.axvline(c, color="#ff7f00")
            fit_ax.axvspan(
                c - f / 2.0, c + f / 2.0, color="#ff7f00", alpha=0.2, label=label
            )
        # Misc
        fit_ax.legend(loc="upper right")
        # Set viewing window to only include the roi (not entire spectrum)
        xpad = (self.x_roi[-1] - self.x_roi[0]) * 0.05
        ypad = (ymax - ymin) * 0.05
        fit_ax.set_xlim([self.x_roi[0] - xpad, self.x_roi[-1] + xpad])
        fit_ax.set_ylim([ymin - ypad, ymax + ypad])
        fit_ax.set_ylabel(self.ymode)

        # ---------
        # Residuals
        # ---------
        y_eval = self.eval(self.x_roi, **self.best_values) * dx_roi
        res_kwargs = dict(fmt="o", color="k", markersize=5, label="residuals")

        # Y-values of the residual plot, depending on residual_type
        y_plot = self.compute_residuals(residual_type)

        # Error bars and ylabel of the residual plot
        if residual_type == "rel":
            yerr_plot = self.y_unc_roi / np.abs(y_eval)
            ylabel = "Relative residuals"
        elif residual_type == "sigma":
            yerr_plot = np.zeros_like(y_plot)
            ylabel = r"Residuals $(\sigma)$"
        elif residual_type == "abs":
            yerr_plot = self.y_unc_roi
            ylabel = "Residuals"
        else:
            raise ValueError(f"Unknown residuals option: {residual_type:s}")
        res_ax.errorbar(x=self.x_roi, y=y_plot, yerr=yerr_plot, **res_kwargs)
        res_ax.axhline(0.0, linestyle="dashed", c="k", linewidth=1.0)
        res_ax.set_ylabel(ylabel)
        res_ax.set_xlabel(self.xmode)

        # -------------------
        # Fit report (txt_ax)
        # -------------------
        props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=1)
        props = dict(facecolor="white", edgecolor="none", alpha=0)
        fp = FontProperties(family="monospace", size=8)
        if "lmfit" in self.backend:
            best_fit_values = ""
            op = self.result.params
            for p in self.result.params:
                if op[p].stderr is None:
                    pass
                    # TODO: Calculate errors breaks minimization right now
                    # warnings.warn(
                    #     "Package numdifftools is required to have "
                    #     "stderr calculated.", FittingWarning)
                else:
                    best_fit_values += "{:15} {: .6e} +/- {:.5e} ({:6.1%})\n".format(
                        p, op[p].value, op[p].stderr, abs(op[p].stderr / op[p].value)
                    )
            best_fit_values += "{:15} {: .6e}\n".format(
                "Chi Squared:", self.result.chisqr
            )
            best_fit_values += "{:15} {: .6e}".format(
                "Reduced Chi Sq:", self.result.redchi
            )
            # Remove first 2 lines of fit report (long model description)
            s = "\n".join(self.result.fit_report().split("\n")[2:])
            # Add some more parameter details
            s += "\n"
            param_df = self.param_dataframe(sort_by_model=True)
            for model_name, sdf in param_df.groupby(level="model"):
                s += model_name + "\n"
                for (_, param_name), param_data in sdf.iterrows():
                    v = param_data["val"]
                    e = param_data["unc"]
                    s += "    {:24}: {: .6e} +/- {:.5e} ({:6.1%})\n".format(
                        param_name, v, e, np.abs(e / v)
                    )
        elif "minuit" in self.backend:
            s = str(self.result) + "\n"

        # Add info about the ROI and units
        if self.roi:
            s += "ROI: [{:.3f}, {:.3f}]\n".format(*self.roi)
        s += "X units: {:s}\n".format(self.xmode if self.xmode else "None")
        s += "Y units: {:s}\n".format(self.ymode if self.ymode else "None")
        # Add to empty axis
        txt_ax.text(
            x=0.01,
            y=0.99,
            s=s,
            fontproperties=fp,
            ha="left",
            va="top",
            transform=txt_ax.transAxes,
            bbox=props,
        )

        if savefname is not None:
            fig.savefig(savefname)
            plt.close(fig)
        else:
            return fig
