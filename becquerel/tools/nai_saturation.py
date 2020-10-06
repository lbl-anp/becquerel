import numpy as np
from scipy.interpolate import CubicSpline


DETECTOR_MODELS = {
    '2.5x2': {
        # 2.5 x 2 inch
        # 2615 point is made up!
        'energies': np.array([
            22.09, 35.849, 59.835, 86.31, 144.671, 276.39, 361.164,
            406.322, 503.526, 653.533, 764.413, 1255.126, 1477.289, 2615.]),
        'values': np.array([
            1.468, 1.422, 1.458, 1.424, 1.381, 1.327, 1.315, 1.304, 1.309,
            1.302, 1.293, 1.271, 1.267, 1.262]),
    },
    '4x4alt': {
        # 4x4 inch
        # Last Point is made up.
        'energies': np.array([
            22.390, 35.917, 59.992, 88.146, 133.327, 192.335, 277.460,
            362.690, 407.884, 509.269, 653.023, 886.742, 1320.840, 2615.]),
        'values': np.array([
            1.605, 1.532, 1.580, 1.578, 1.507, 1.477, 1.447, 1.434, 1.433,
            1.425, 1.405, 1.417, 1.400, 1.400]),
    },
    '4x4': {
        # 4x4 inch curve
        'energies': np.array([
            22.390, 35.917, 59.992, 88.146, 133.327, 192.335, 277.460,
            362.690, 509.310, 657.003, 928.561, 1320.840, 2615.]),
        'values': np.array([
            1.605, 1.532, 1.580, 1.578, 1.507, 1.477, 1.447, 1.434, 1.424,
            1.416, 1.407, 1.400, 1.400]),
    },
}


class NaISaturation(object):

    def __init__(self, energies, values):
        self._energies = np.asfarray(energies)
        self._values = np.asfarray(values)
        self._energy2light = CubicSpline(
            x=self.energies,
            y=self.values * self.energies,
            extrapolate=True)
        self._light2energy = CubicSpline(
            x=self.values * self.energies,
            y=self.energies,
            extrapolate=True)

    @property
    def energies(self):
        return self._energies

    @property
    def values(self):
        return self._values

    def energy2light(self, energy):
        energy = np.asfarray(energy)
        light = self._energy2light(energy)
        light[np.array(energy) < 30.] = 0.
        return light

    def light2energy(self, light):
        light = np.asfarray(light)
        energy = self._light2energy(light)
        energy[np.array(light) < 1.] = 0.
        return energy

    def ch2kev(self, x, g, s, c):
        """Convert channel to energy (keV).

        Parameters
        ----------
        x : int or float or array-like
            Input adc channels
        g : float
            Coefficient `g`
        s : float
            Coefficient `s`
        c : float
            Coefficient `c`

        Returns
        -------
        float or np.ndarray
            Energies in keV
        """
        light = (x - c) / (g - s * (x - c))
        return self.light2energy(light)

    def kev2ch(self, x, g, s, c):
        """Convert energy (keV) to channel.

        Parameters
        ----------
        x : int or float or array-like
            Input energy in keV
        g : float
            Coefficient `g`
        s : float
            Coefficient `s`
        c : float
            Coefficient `c`

        Returns
        -------
        float or np.ndarray
            Channels
        """
        light = self.energy2light(x)
        return (g * light) / (1 + s * light) + c

    @staticmethod
    def built_in_detector_models():
        return list(DETECTOR_MODELS.keys())

    @classmethod
    def from_detector_model(cls, detector_model='4x4'):
        try:
            return cls(*DETECTOR_MODELS[detector_model])
        except KeyError:
            raise ValueError(
                'Unknown detector_model: {}'.format(detector_model))
