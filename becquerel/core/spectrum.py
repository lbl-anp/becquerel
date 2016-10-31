"""Base class for spectrum file parsers."""

from __future__ import print_function
import os
import numpy as np


class RawSpectrum(object):
    """Raw spectrum class.

    Basic operation is:
        spec = RawSpectrum(list or array)
        or
        spec = RawSpectrum.from_file("filename.extension")
        spec.apply_calibration()

    Then the data are in
        spec.data [counts]
        spec.channels

    """

    def __init__(self, data):
        """Initialize the spectrum."""
        self.data = np.array(data, dtype=float)

        # TODO should channels be integers? 
        # TODO what convention is used for channels? bin centers, lower edge etc...?
        self.channels = np.arange(len(self.data), dtype=float)
        self.infilename = None
        self.infileobject = None

    @classmethod
    def from_file(cls, infilename):
        # Read 
        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls.__init__(spect_file_obj.data)
        spect_obj.infileobject = spect_file_obj
        spect_obj.channels = self.infileobject.channels
        return spect_obj

    # def __str__(self):
    #     """String form of the spectrum."""
    #     s = ''
    #     s += 'Filename:              {:s}\n'.format(self.filename)
    #     s += 'Spectrum ID:           {:s}\n'.format(self.spectrum_id)

    #     return s



class CalSpectrum(RawSpectrum):
    """Cal spectrum class.

    Basic operation is:
        spec = CalSpectrum(array of counts, bin_energies)
        or
        spec = CalSpectrum.from_file("filename.extension")

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.bin_energies -- subject to convention!

    """

    def __init__(self, data, bin_energies):
        """Initialize the spectrum."""
        assert(len(bin_energies) == len(data)+1) 
        self.data = np.array(data, dtype=float)
        # TODO should channels be integers? 
        # TODO what convention is used for channels? bin centers, lower edge etc...?
        self.channels = np.arange(len(self.data), dtype=float)
        self.bin_energies = np.array(bin_energies, dtype=float)

        self.infilename = None
        self.infileobject = None

    @classmethod
    def from_file(cls, infilename):
        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls.__init__(spect_file_obj.data, spect_file_obj.energies)
        spect_obj.infileobject = spect_file_obj
        spect_obj.channels = self.infileobject.channels
        
        # TODO Get more attributes from self.infileobj

        return spect_obj


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
    extension = self.infilename.split('.')[-1]
    if extension is 'spe':
        spect_file_obj = parsers.spe_file(infilename)
    elif extension is 'spc':
        spect_file_obj = parsers.spc_file(infilename)
    elif extension is 'cnf':
        spect_file_obj = parsers.cnf_file(infilename)
    else:
        raise NotImplementedError('File type {} can not be read'.format(extensions))
    return spect_file_obj
