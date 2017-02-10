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
    Represents an energy spectrum.

    Initialize a Spectrum directly, or with Spectrum.from_file(filename).

    Attributes:
      data: np.array of counts in each channel
      channels: [Read-only] np.array of channel index as integers
      is_calibrated: [Read-only] bool
      energies_kev: [Read-only] np.array of energy bin centers, if calibrated
      bin_edges_kev: np.array of energy bin edges, if calibrated
    """

    def __init__(self, data, bin_edges_kev=None):
        """Initialize the spectrum.

        Args:
          data: an iterable of counts per channel
          bin_edges_kev: an iterable of bin edge energies.
            Defaults to None for an uncalibrated spectrum.
            If not none, should have length of (len(data) + 1).

        Raises:
          SpectrumError: for bad input arguments
        """

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
        """Channel index.

        Returns:
          np.array of int's from 0 to (len(self.data) - 1)
        """

        return np.arange(len(self.data), dtype=int)

    @property
    def energies_kev(self):
        """Convenience function for accessing the energies of bin centers.

        Returns:
          np.array of floats, same length as self.data

        Raises:
          UncalibratedError: if spectrum is not calibrated
        """

        if self.bin_edges_kev is None:
            raise UncalibratedError('Spectrum is not calibrated')
        else:
            return self.bin_centers_from_edges(self.bin_edges_kev)

    @property
    def is_calibrated(self):
        """Is the spectrum calibrated?

        Returns:
          A bool.
          True if spectrum has defined energy bin edges. False otherwise.
        """

        return bool(self.bin_edges_kev)

    @classmethod
    def from_file(cls, infilename):
        """Construct a Spectrum object from a filename.

        Args:
          infilename: a string representing the path to a parsable file.

        Returns:
          A Spectrum object.

        Raises:
          IOError: for a bad filename.
        """

        spect_file_obj = _get_file_object(infilename)

        spect_obj = cls(spect_file_obj.data,
                        bin_edges_kev=spect_file_obj.energy_bin_edges)
        spect_obj.infileobject = spect_file_obj

        # TODO Get more attributes from self.infileobj

        return spect_obj

    @staticmethod
    def bin_centers_from_edges(edges_kev):
        """Calculate bin centers from bin edges.

        Args:
          edges_kev: an iterable representing bin edge energies in keV.

        Returns:
          np.array of length (len(edges_kev) - 1),
          representing bin center energies.
        """

        edges_kev = np.array(edges_kev)
        centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
        return centers_kev


def _get_file_object(infilename):
    """
    Parse a file and return an object according to its extension.

    Args:
      infilename: a string representing a path to a parsable file.

    Raises:
      AssertionError: for a bad filename.  # TODO let this be an IOError
      NotImplementedError: for an unparsable file extension.
      ...?

    Returns:
      a file object of type SpeFile, SpcFile, or CnfFile
    """

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
