""""Energy calibration classes"""

from abc import ABCMeta, abstractmethod, abstractproperty
from future.builtins import dict, super, zip
from future.utils import viewitems
import numpy as np
from lmfit.model import Model
from ..tools.nai_saturation import NaISaturation


# TODO: Move models to central location once peak fitting merged


def constant(x, c):
    return np.ones_like(x) * c


def line(x, m, b):
    return m * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


###############################################################################


def quadratic_guess_coeffs(x, data):
    a, b, c = 0., 0., 0.
    if x is not None:
        assert len(x) == len(data), \
            'Cannot guess quad coeffs from different len x and data'
        if len(x) > 2:
            a, b, c = np.polyfit(x, data, 2)
        elif len(x) > 1:
            b, c = np.polyfit(x, data, 1)
        elif len(x) == 1:
            if not np.isclose(x, 0.):
                b, = data / x
    return a, b, c


def nai_saturation_guess_coeffs(x, data):
    g, s, c = 0.6, 2.E-5, 0.
    return g, s, c


###############################################################################


def _update_param_vals(pars, prefix, **kwargs):
    """
    Convenience function to update parameter values with keyword arguments
    """
    for key, val in kwargs.items():
        pname = "{}{}".format(prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars


###############################################################################


class ConstantModel(Model):

    def __init__(self, *args, **kwargs):
        super(ConstantModel, self).__init__(constant, *args, **kwargs)
        self.set_param_hint('{}c'.format(self.prefix), min=0.)


class LineModel(Model):

    def __init__(self, *args, **kwargs):
        super(LineModel, self).__init__(line, *args, **kwargs)


class QuadraticModel(Model):

    def __init__(self, *args, **kwargs):
        super(QuadraticModel, self).__init__(quadratic, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        a, b, c = quadratic_guess_coeffs(x, data)
        pars = self.make_params(a=a, b=b, c=c)
        return _update_param_vals(pars, self.prefix, **kwargs)


class NaISaturationModel(Model):

    def __init__(self, nai_saturation, *args, **kwargs):
        if not isinstance(nai_saturation, NaISaturation):
            raise TypeError(
                'Invalid NaISaturation object: {}'.format(nai_saturation))
        self.nai_saturation = nai_saturation
        super().__init__(nai_saturation.ch2kev, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        g, s, c = nai_saturation_guess_coeffs(x, data)
        pars = self.make_params(g=g, s=s, c=c)
        return _update_param_vals(pars, self.prefix, **kwargs)



###############################################################################


class EnergyCalError(Exception):
    """Base class for errors in energycal.py"""

    pass


class BadInput(EnergyCalError):
    """Error related to energy cal input"""

    pass


###############################################################################


class EnergyCalBase(object):
    """Abstract base class for energy calibration.

    A note on nomenclature: for historic reasons, 'channels' is used in
    energycal.py for generic uncalibrated x-axis values. A 'channel' is no
    longer necessarily an integer channel number (i.e., bin) from a
    multi-channel analyzer, but could for instance be a float-type fC of charge
    collected.

    Subclasses must implement:
        _ch2kev (method)
        kev2ch (method)
        valid_coeffs (property)
        _perform_fit (method)
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """Create an empty calibration instance.

        Normally you should use from_points or from_coeffs classmethods.

        Args:
            none
        """

        self._calpoints = dict()
        self._coeffs = dict()
        # initialize fit constraints?

    @classmethod
    def from_points(cls, chlist, kevlist, include_origin=False):
        """Construct EnergyCal from calibration points.

        Args:
            chlist: list/tuple/array of the channel values of calibration points
            kevlist: list/tuple/array of the corresponding energy values [keV]
            include_origin: Default is False, if set to True will add a point
                            at zero into the fit.

        Raises:
          BadInput: for bad chlist and/or kevlist.
        """

        if chlist is None or kevlist is None:
            raise BadInput('Channel list and energy list are required')

        try:
            cond = len(chlist) != len(kevlist)
        except TypeError:
            raise BadInput('Inputs must be one dimensional iterables')
        if cond:
            raise BadInput('Channels and energies must be same length')

        cal = cls()

        if include_origin:
            cal.new_calpoint(0, 0)

        for ch, kev in zip(chlist, kevlist):
            try:
                cal.new_calpoint(ch, kev)
            except (ValueError, TypeError):
                raise BadInput('Inputs must be one dimensional iterables')
        cal.update_fit()
        return cal

    @classmethod
    def from_coeffs(cls, coeffs):
        """Construct EnergyCal from equation coefficients dict.

        Args:
            coeffs: a dict with keys equal to elements in valid_coeffs,
                    and values specifying the value of the coefficient
        """

        cal = cls()

        for coeff, val in viewitems(coeffs):
            cal._set_coeff(coeff, val)

        # TODO make sure all coefficients are specified

        return cal

    @property
    def channels(self):
        """The channel values of calibration points.

        Returns:
            an np.ndarray of channel values (floats)
        """

        return np.array(list(self._calpoints.values()), dtype=float)

    @property
    def energies(self):
        """The energy values of calibration points.

        Returns:
            an np.ndarray of energy values [keV]
        """

        return np.array(list(self._calpoints), dtype=float)

    @property
    def calpoints(self):
        """The calibration points, in (ch, kev) pairs.

        Returns:
            a list of 2-element tuples of (channel, energy[keV])
        """

        return list(zip(self.channels, self.energies))

    @property
    def coeffs(self):
        """The coefficients of the current calibration curve.

        Returns:
            a dict of {coeff: value}
        """

        # TODO: if there are no coeffs, error?
        return self._coeffs

    def add_calpoint(self, ch, kev):
        """Add a calibration point (ch, kev) pair. May be new or existing.

        Args:
            ch: the channel value of the calibration point
            kev: the energy value of the calibration point [keV]
        """

        self._calpoints[float(kev)] = float(ch)

    def new_calpoint(self, ch, kev):
        """Add a new calibration point. Error if energy matches existing point.

        Args:
            ch: the channel value of the calibration point
            kev: the energy value of the calibration point [keV]

        Raises:
            EnergyCalError: if energy value already exists in calibration
        """

        if kev in self._calpoints:
            raise EnergyCalError('Calibration energy already exists')
        self.add_calpoint(ch, kev)

    def rm_calpoint(self, kev):
        """Remove a calibration point, if it exists.

        Args:
            the energy value of the point to remove [keV]
        """

        if kev in self._calpoints:
            del self._calpoints[kev]
        # TODO erroring version?

    def ch2kev(self, ch):
        """Convert channel(s) to energy value(s).

        Args:
            ch: a scalar, np.array, list or tuple of channel values

        Returns:
            the energy value(s) corresponding to the channel value(s) [keV].
            a float if input is scalar. an np.array if input is iterable
        """

        if isinstance(ch, (list, tuple)):
            ch = np.array(ch)

        return self._ch2kev(ch)

    @abstractmethod
    def _ch2kev(self, ch):
        """Convert scalar OR np.array of channel(s) to energies.

        Should use numpy ufuncs so that the input dtype doesn't matter.

        Args:
            ch: an np.array, float, or int of channel values

        Returns:
            energy values, the same size/type as ch [keV]
        """

        pass

    def kev2ch(self, kev):
        """Convert energy value(s) to channel(s).

        Args:
            kev: a scalar, np.array, list or tuple of energy values [keV]

        Returns:
            the channel value(s) corresponding to the input energies.
            a float if input is scalar. an np.array if input is iterable
        """

        if isinstance(kev, (list, tuple)):
            kev = np.array(kev)

        return self._kev2ch(kev)

    def _kev2ch(self, kev):
        """Convert energy value(s) to channel(s).

        Should use numpy ufuncs so that the input dtype doesn't matter.

        Args:
            kev: an np.array, float, or int of energy values [keV]

        Returns:
            the channel value(s) corresponding to the input energies.
            a float if input is scalar. an np.array if input is iterable
        """

        kev = np.asarray(kev, dtype=np.float)
        # Should we allow negative inputs? (This will change the bounds below)
        assert (kev >= 0).all()
        # Find bounds (assumes self._kev2ch(kev) > 0))
        _ch_min = 0.0
        _ch_max = 1.0
        _num_interp_pts = 1e4
        while True:
            _ch_max_kev = self.ch2keV(_ch_max)
            if _ch_max_kev < kev.max():
                if _ch_max_kev < kev.min():
                    _ch_min = _ch_max
                _ch_max *= 2.0
        # Create at vectors of length _num_interp_pts to interpolate from
        if (_ch_max - _ch_min) > _num_interp_pts:
            _ch = np.arange(int(round(_ch_min)), int(round(_ch_max)), dtype=np.float)
        else:
            _ch = np.linspace(_ch_min, _ch_max, _num_interp_pts, dtype=np.float)
        _kev = self.ch2kev(_ch)
        # calibration must be monotonically increasing
        assert (np.diff(_kev) > 0).all()
        # make sure we can interpolate
        assert (_kev.min() <= kev).all() and (kev <= _kev.max())
        ch = np.asarray(np.interp(kev, kev_interp, _ch))
        # Removing because this won't be allowed by the bound search
        # assert (ch >= 0).all()
        return ch

    @abstractproperty
    def valid_coeffs(self):
        """A list of valid coefficients for the calibration curve.

        Returns:
            a tuple of strings, the names of the coefficients for this curve
        """

        pass

    def _set_coeff(self, name, val):
        """Set a coefficient for the calibration curve.

        Args:
            name: a string, the name of the coefficient to set
            val: the value to set the coefficient to

        Raises:
            EnergyCalError: if name is not in valid_coeffs
        """

        if name in self.valid_coeffs:
            self._coeffs[name] = val
        else:
            raise EnergyCalError('Invalid coefficient name: {}'.format(name))

    def update_fit(self):
        """Compute the calibration curve from the current points.

        Raises:
            EnergyCalError: if there are too few calibration points to fit
        """

        num_coeffs = len(self.valid_coeffs)
        # TODO: free coefficients, not all coefficients
        num_points = len(self._calpoints)

        if num_points == 0:
            raise EnergyCalError('No calibration points; cannot calibrate')
        elif num_points < num_coeffs:
            raise EnergyCalError('Not enough calibration points to fit curve')
        else:
            self._perform_fit()

    @abstractmethod
    def _perform_fit(self):
        """Do the actual curve fitting."""

        pass


# TODO: dummy class for testing?


class LinearEnergyCal(EnergyCalBase):
    """
    kev = b*ch + c
    """

    @classmethod
    def from_coeffs(cls, coeffs):
        """Construct LinearEnergyCal from equation coefficients dict.

        Valid coefficient names (slope, offset):
            ('b', 'c')
            ('p1', 'p0')
            ('slope', 'offset')
            ('m', 'b')

        Args:
            coeffs: a dict with keys equal to valid coeff names,
                    and values specifying the value of the coefficient
        """

        new_coeffs = {}
        if 'p0' in coeffs and 'p1' in coeffs:
            new_coeffs['b'] = coeffs['p1']
            new_coeffs['c'] = coeffs['p0']
        elif 'slope' in coeffs and 'offset' in coeffs:
            new_coeffs['b'] = coeffs['slope']
            new_coeffs['c'] = coeffs['offset']
        elif 'm' in coeffs and 'b' in coeffs:
            new_coeffs['b'] = coeffs['m']
            new_coeffs['c'] = coeffs['b']
        else:
            new_coeffs = coeffs.copy()
        cal = super().from_coeffs(new_coeffs)
        return cal

    @property
    def valid_coeffs(self):
        """A list of valid coefficients for the calibration curve.

        Returns:
          a tuple of strings, the names of the coefficients for this curve
        """

        return ('b', 'c')

    @property
    def slope(self):
        """Return the slope coefficient value."""

        try:
            return self._coeffs['b']
        except KeyError:
            raise EnergyCalError(
                'Slope coefficient not yet supplied or calculated.')

    @property
    def offset(self):
        """Return the offset coefficient value."""

        try:
            return self._coeffs['c']
        except KeyError:
            raise EnergyCalError(
                'Offset coefficient not yet supplied or calculated.')

    def _ch2kev(self, ch):
        """Convert scalar OR np.array of channel(s) to energies.

        Should use numpy ufuncs so that the input dtype doesn't matter.

        Args:
            ch: an np.array, float, or int of channel values

        Returns:
            energy values, the same size/type as ch [keV]
        """

        return self.slope * ch + self.offset

    def _kev2ch(self, kev):
        """Convert energy value(s) to channel(s).

        Args:
            kev: an np.array, float, or int of energy values [keV]

        Returns:
            the channel value(s) corresponding to the input energies.
            a float if input is scalar. an np.array if input is iterable
        """

        return (kev - self.offset) / self.slope

    def _perform_fit(self):
        """Do the actual curve fitting."""

        b, c = np.polyfit(self.channels, self.energies, 1)
        self._set_coeff('b', b)
        self._set_coeff('c', c)


class QuadraticEnergyCal(EnergyCalBase):
    """
    kev = a*ch^2 + b*ch + c
    """

    @property
    def valid_coeffs(self):
        """A list of valid coefficients for the calibration curve.

        Returns:
            a tuple of strings, the names of the coefficients for this curve
        """
        return ('a', 'b', 'c')

    def _ch2kev(self, ch):
        """Convert scalar OR np.array of channel(s) to energies.
        Should use numpy ufuncs so that the input dtype doesn't matter.
        Args:
            ch: an np.array, float, or int of channel values
        Returns:
            energy values, the same size/type as ch [keV]
        """
        return (self._coeffs['a'] * ch ** 2 +
                self._coeffs['b'] * ch +
                self._coeffs['c'])

    def _kev2ch(self, kev):
        """Convert energy value(s) to channel(s).
        Args:
            kev: an np.array, float, or int of energy values [keV]
        Returns:
            the channel value(s) corresponding to the input energies.
            a float if input is scalar. an np.array if input is iterable
        """
        # TODO: address with the quadratic inverse formula
        raise NotImplementedError('Sorry')

    def _perform_fit(self):
        """Do the actual curve fitting."""
        a, b, c = np.polyfit(self.channels, self.energies, 2)
        self._set_coeff('a', a)
        self._set_coeff('b', b)
        self._set_coeff('c', c)

















class NaISaturationEnergyCal(EnergyCalBase):
    """
    Energy calibration in the form of:
        E = finv[ (ch - c) / (g - s * (ch - c)) ]
    Where:
        ch : channel
        c : offset
        g : pmt gain
        s : pmt saturation
        E : energy (keV)
        finv : transformation from light to energy
    """

    def __init__(self, nai_saturation, **kwargs):
        super(NaISaturationEnergyCal, self).__init__(**kwargs)
        if not isinstance(nai_saturation, NaISaturation):
            raise TypeError(
                'Invalid NaISaturation object: {}'.format(nai_saturation))
        self.nai_saturation = nai_saturation

    @classmethod
    def from_detector_model(cls, detector_model='4x4', **kwargs):
        return cls(
            nai_saturation=NaISaturation.from_detector_model(detector_model),
            **kwargs)

    @property
    def valid_coeffs(self):
        return ('g', 's', 'c')

    def _ch2kev(self, ch):
        """Convert channel to energy (keV).

        Parameters
        ----------
        ch : int or float or array-like
            Input adc channels

        Returns
        -------
        float or np.ndarray
            Energies in keV
        """
        return nai_saturation_forward(ch, nai_saturation=self.nai_saturation,
                                      **self._coeffs)

    def _kev2ch(self, kev):
        """Convert energy (keV) to channel.

        Parameters
        ----------
        kev : int or float or array-like
            Input energy in keV

        Returns
        -------
        float or np.ndarray
            Channels
        """
        return nai_saturation_inverse(kev, nai_saturation=self.nai_saturation,
                                      **self._coeffs)

    def _perform_fit(self):
        """Do the actual curve fitting."""
        pass
        # a, b, c = np.polyfit(self.channels, self.energies, 2)
        # self._set_coeff('a', a)
        # self._set_coeff('b', b)
        # self._set_coeff('c', c)
