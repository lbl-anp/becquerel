"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import numpy as np
import becquerel.parsers as parsers
# from ..parsers import SpeFile, SpcFile, CnfFile


class RawSpectrumError(Exception):
    """Exception raised by RawSpectrum."""

    pass


class RawSpectrum(object):
    """Raw spectrum class.

    Basic operation is:
        spec = RawSpectrum(list or array)
        or
        spec = RawSpectrum.from_file("filename.extension")

    Then the data are in
        spec.data [counts]
        spec.channels
    """

    def __init__(self, data):
        """Initialize the spectrum."""
        self.data = np.array(data, dtype=float)
        assert len(data) > 0

        # TODO should channels be integers?
        # TODO what convention is used for channels?
        # bin centers, lower edge etc...?
        self.channels = np.arange(len(self.data), dtype=float)
        self.infilename = None
        self.infileobject = None

    @classmethod
    def from_file(cls, infilename):
        # Read
        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data)
        spect_obj.infileobject = spect_file_obj
        spect_obj.channels = spect_obj.infileobject.channels
        return spect_obj

    # def __str__(self):
    #     """String form of the spectrum."""
    #     s = ''
    #     s += 'Filename:              {:s}\n'.format(self.filename)
    #     s += 'Spectrum ID:           {:s}\n'.format(self.spectrum_id)

    #     return s


class CalSpectrumError(RawSpectrumError):
    """Exception raised by CalSpectrum."""

    pass


class CalSpectrum(RawSpectrum):
    """Cal spectrum class.

    Basic operation is:
        spec = CalSpectrum(array of counts, bin_energies)
        or
        spec = CalSpectrum.from_file("filename.extension")
        or
        spec = CalSpectrum.from_raw(raw_spectrum, energycal)

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.bin_energies -- subject to convention!

    """

    def __init__(self, data, bin_energies):
        """Initialize the spectrum."""
        assert(len(bin_energies) == len(data))
        assert len(data) > 0
        self.data = np.array(data, dtype=float)
        # TODO should channels be integers?
        # TODO what convention is used for channels?
        # bin centers, lower edge etc...?
        self.channels = np.arange(len(self.data), dtype=float)
        self.bin_energies = np.array(bin_energies, dtype=float)

        self.infilename = None
        self.infileobject = None

    @classmethod
    def from_file(cls, infilename):
        """Generate CalSpectrum from a file."""
        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data, spect_file_obj.energies)
        spect_obj.infileobject = spect_file_obj
        spect_obj.channels = spect_obj.infileobject.channels

        # TODO Get more attributes from self.infileobj

        return spect_obj

    @classmethod
    def from_raw(cls, raw_spectrum, energycal):
        """Generate CalSpectrum from a RawSpectrum plus energy calibration."""
        data = raw_spectrum.data
        bin_energies = energycal.channels_to_energy(np.arange(len(data)))
        spec = cls(data, bin_energies)
        return spec


    # def __str__(self):
    #     """String form of the spectrum."""
    #     s = ''
    #     s += 'Filename:              {:s}\n'.format(self.filename)
    #     s += 'Spectrum ID:           {:s}\n'.format(self.spectrum_id)

    #     return s

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
