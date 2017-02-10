"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import numpy as np
import becquerel.parsers as parsers
# from ..parsers import SpeFile, SpcFile, CnfFile


class SpectrumError(Exception):
    """Exception raised by Spectrum."""

    pass


class UncalibratedError(SpectrumError):
    """Exception raised when an uncalibrated spectrum is treated as calibrated.
    """

    pass


class Spectrum(object):
    """
    Spectrum class.

    ....
    Basic operation is:
        spec = CalSpectrum(array of counts, bin_energies)
        or
        spec = CalSpectrum.from_file("filename.extension")

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.bin_energies -- subject to convention!

    """

    def __init__(self, data, bin_edges_kev=None):
        """Initialize the spectrum."""

        if len(data) == 0:
            raise SpectrumError('Empty spectrum data')
        self.data = np.array(data, dtype=float)

        if bin_edges_kev is None:
            self.bin_edges_kev = None
        elif len(bin_edges_kev) != len(data) + 1:
            raise SpectrumError('Bad length of bin edges vector')
        else:
            self.bin_edges_kev = np.array(bin_edges_kev, dtype=float)

        self.infilename = None
        self.infileobject = None

    @property
    def channels(self):
        return np.arange(len(self.data), dtype=int)

    @property
    def energies_kev(self):
        """Convenience function for accessing the energies of bin centers."""

        if self.bin_edges_kev is None:
            raise UncalibratedError('Spectrum is not calibrated')
        else:
            return self.bin_centers_from_edges(self.bin_edges_kev)

    @property
    def is_calibrated(self):
        return bool(self.bin_edges_kev)

    @classmethod
    def from_file(cls, infilename):
        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data,
                        bin_edges_kev=spect_file_obj.energy_bin_edges)
        spect_obj.infileobject = spect_file_obj

        # TODO Get more attributes from self.infileobj

        return spect_obj

    @staticmethod
    def bin_centers_from_edges(edges_kev):
        edges_kev = np.array(edges_kev)
        centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
        return centers_kev


def _get_file_object(infilename):
    '''
    Input:
        infilename
    Output:
        SpectrumFile
    '''
    _, extension = os.path.splitext(infilename)
    if extension.lower() == '.spe':
        spect_file_obj = parsers.SpeFile(infilename)
    elif extension.lower() == '.spc':
        spect_file_obj = parsers.SpcFile(infilename)
    elif extension.lower() == '.cnf':
        spect_file_obj = parsers.CnfFile(infilename)
    else:
        raise NotImplementedError(
            'File type {} can not be read'.format(extension))
    return spect_file_obj
