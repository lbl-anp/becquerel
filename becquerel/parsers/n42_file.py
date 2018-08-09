"""Create and parse N42 radiation measurement files."""

# pylint: disable=no-member

from __future__ import print_function
from .spectrum_file import SpectrumFile, SpectrumFileParsingError
import os
import sys
import re
import numpy as np
# import xml.etree.ElementTree as ET
import dateutil.parser
# import requests
from lxml import etree
import matplotlib.pyplot as plt


scriptdir = os.path.dirname(os.path.realpath(__file__))

# download N42 schema
NIST_N42_URL = 'http://physics.nist.gov/N42/2011/'
# N42_XSD_URL = NIST_N42_URL + 'n42.xsd'
# req = requests.get(N42_XSD_URL)
schema_text = etree.parse(os.path.join(scriptdir, 'n42/n42.xsd'))
# schema_root = etree.XML(schema_text)
N42_SCHEMA = etree.XMLSchema(schema_text)
N42_NAMESPACE = '{{{}}}'.format(NIST_N42_URL + 'N42')


SAMPLES = [
    os.path.join(scriptdir, 'n42/Annex_B_n42.xml'),
    # 'n42/Annex_B_alternate_energy_calibration_n42.xml',
    os.path.join(scriptdir, 'n42/Annex_C_n42.xml'),
    os.path.join(scriptdir, 'n42/Annex_E_n42.xml'),
    os.path.join(scriptdir, 'n42/Annex_G_n42.xml'),
    os.path.join(scriptdir, 'n42/Annex_I_n42.xml'),
]


class N42FileParsingError(SpectrumFileParsingError):
    """Failed while parsing an SPE file."""

    pass


class N42FileWritingError(SpectrumFileParsingError):
    """Failed while writing an SPE file."""

    pass


class N42File(SpectrumFile):
    """N42 XML file parser.

    Just instantiate a class with a filename:
        spec = N42File(filename)

    Then the data are in
        spec.data [counts]
        spec.channels
        spec.energies
        spec.energy_bin_widths

    http://physics.nist.gov/N42/2011/

    """

    def __init__(self, filename):
        """Initialize the N42 file."""
        super(N42File, self).__init__(filename)
        _, ext = os.path.splitext(self.filename)
        if not ((ext.lower() == '.n42') or (ext.lower() == '.xml')):
            raise N42FileParsingError('File extension is incorrect: ' + ext)

        # read in the data
        self.read()
        self.apply_calibration()

    def read(self, verbose=False):
        """Read in the file."""
        print('N42File: Reading file ' + self.filename)
        self.realtime = 0.0
        self.livetime = 0.0
        self.channels = np.array([], dtype=np.float)
        self.data = np.array([], dtype=np.float)
        self.cal_coeff = []

        xml_text = etree_parse_clean(self.filename)
        self._parse_xml(xml_text)

    def _parse_xml(self, xml_text):
        tree = xml_text
        root = tree.getroot()
        # root should be a RadInstrumentData
        assert root.tag == N42_NAMESPACE + 'RadInstrumentData'

        # read instrument information
        instrument_info = {}
        for info in root.findall(N42_NAMESPACE + 'RadInstrumentInformation'):
            for thing in info:
                tag = thing.tag.split(N42_NAMESPACE)[-1]
                instrument_info[tag] = thing.text
        self.detector_description = instrument_info

        # read detector information
        detector_info = {}
        for info in root.findall(N42_NAMESPACE + 'RadDetectorInformation'):
            for thing in info:
                tag = thing.tag.split(N42_NAMESPACE)[-1]
                detector_info[tag] = thing.text

        # read energy calibrations
        energy_cals = {}
        for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
            for thing in cal:
                tag = thing.tag.split(N42_NAMESPACE)[-1]
                if tag == 'CoefficientValues':
                    coefs = [float(x) for x in thing.text.split(' ')]
                    # TODO This was introduced based on a Fulcrum n42, but
                    # looks like it isn't needed.  Should be removed on merge.
                    # if 'PHD' in self.detector_description['RadInstrumentManufacturerName']:
                    #     coefs[1] *= 2.
                    energy_cals[tag] = np.array(coefs)
                    self.cal_coeff = energy_cals[tag]
                else:
                    energy_cals[tag] = thing.text

        # read FWHM calibrations
        fwhm_cals = {}
        for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
            for thing in cal:
                tag = thing.tag.split(N42_NAMESPACE)[-1]
                fwhm_cals[tag] = thing.text

        # read measurements
        measurements = []
        for idx, measurement in enumerate(root.findall(
                N42_NAMESPACE + 'RadMeasurement')):
            if idx > 0:
                print('WARNING: N42 parser ignoring additional measurements.')

            real_time = None
            measurements.append(measurement)

            # read real time duration
            real_times = measurement.findall(N42_NAMESPACE + 'RealTimeDuration')
            if len(real_times) > 0:
                real_time = real_times[0]
                real_time = parse_duration(real_time.text)

            spectra = {}
            for spectrum in measurement.findall(N42_NAMESPACE + 'Spectrum'):
                spect = dict(spectrum.items())
                # read live time duration
                live_times = spectrum.findall(
                    N42_NAMESPACE + 'LiveTimeDuration')
                if len(live_times) > 0:
                    live_time = parse_duration(live_times[0].text)
                    spect['live_time'] = live_time
                    if idx == 0:
                        self.livetime = live_time

                real_times = spectrum.findall(
                    N42_NAMESPACE + 'RealTimeDuration')
                if len(real_times) > 0:
                    real_time = real_times[0]
                    real_time = parse_duration(real_time.text)
                if real_time is not None:
                    spect['real_time'] = real_time
                    if idx == 0:
                        self.realtime = real_time

                spect['channel_data'] = []
                for cd in spectrum.findall(N42_NAMESPACE + 'ChannelData'):
                    comp = cd.get('compressionCode', None)
                    d = parse_channel_data(cd.text, compression=comp)
                    spect['channel_data'].append(d)
                    if idx == 0:
                        self.data = d
                        self.channels = np.arange(len(d))
                        self.apply_calibration()
                        self.spectrum_id = spect['id']

                spectra[spect['id']] = spect

        # self._instrument_info = instrument_info
        # self._detector_info = detector_info
        # self._energy_cals = energy_cals
        # self._fwhm_cals = fwhm_cals
        # self._spectra = spectra
        return


def etree_parse_clean(filename):
    with open(filename, 'r') as f:
        text = f.read()
    text = re.sub(u"[^\x01-\x7f]+", u"", text)
    with open('.temp.xml', 'w') as f:
        f.write(text)
    xml_text = etree.parse('.temp.xml')
    os.remove('.temp.xml')
    return xml_text


def parse_duration(text):
    """Parse ISO 8601 time duration into seconds.

    Only covers case where text is "PTXS", where X is the number of seconds.
    https://en.wikipedia.org/wiki/ISO_8601#Durations

    """
    assert text.startswith('PT')
    assert text.endswith('S')
    return float(text[2:-1])


def parse_channel_data(text, compression=None):
    """Parse ChannelData text into a list of integer channel data.

    Keywords:
        compression: None or 'CountedZeroes'.

    """
    text = text.strip().replace('\n', ' ')
    tokens = text.split()
    data = [int(token) for token in tokens]
    if compression == 'CountedZeroes':
        new_data = []
        k = 0
        while k < len(data):
            if data[k] != 0:
                new_data.append(data[k])
                k += 1
            else:
                new_data.extend([0] * data[k + 1])
                k += 2
        data = new_data
    return data


def compress_channel_data(channel_data):
    """Compress a list of integers using the CountedZeroes algorithm."""
    compressed_data = []
    k = 0
    while k < len(channel_data):
        compressed_data.append(channel_data[k])
        if channel_data[k] == 0:
            n_zeros = 0
            while k < len(channel_data) and channel_data[k] == 0:
                n_zeros += 1
                k += 1
            compressed_data.append(n_zeros)
        else:
            k += 1
    return compressed_data


def valid_xml(text):
    """True if XML conforms to its schema.

    Uses the solution from:
    http://stackoverflow.com/questions/17819884/xml-xsd-feed-validation-against-a-schema

    """
    N42_SCHEMA.validate(text)
    return True
    # xml_parser = etree.XMLParser(schema=N42_SCHEMA)
    # try:
    #     etree.fromstring(text, xml_parser)
    #     return True
    # except etree.XMLSchemaError:
    #     return False


# def parse_n42(text):
#     """Parse an N42 file."""
#     tree = text
#     root = tree.getroot()
#     # root should be a RadInstrumentData
#     assert root.tag == N42_NAMESPACE + 'RadInstrumentData'
#
#     # read instrument information
#     instrument_info = {}
#     for info in root.findall(N42_NAMESPACE + 'RadInstrumentInformation'):
#         instrument_info[info.attrib['id']] = info
#
#     # read detector information
#     detector_info = {}
#     for info in root.findall(N42_NAMESPACE + 'RadDetectorInformation'):
#         detector_info[info.attrib['id']] = info
#
#     # read energy calibrations
#     energy_cals = {}
#     for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
#         energy_cals[cal.attrib['id']] = cal
#
#     # read FWHM calibrations
#     fwhm_cals = {}
#     for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
#         fwhm_cals[cal.attrib['id']] = cal
#
#     # read measurements
#     for measurement in root.findall(
#             N42_NAMESPACE + 'RadMeasurement'):
#         print('    ', measurement.tag, measurement.attrib)
#         class_codes = measurement.findall(
#             N42_NAMESPACE + 'MeasurementClassCode')
#         # read start time
#         start_times = measurement.findall(N42_NAMESPACE + 'StartDateTime')
#         # assert len(start_times) == 1
#         if len(start_times) == 1:
#             start_time = start_times[0]
#             print('        Start time:', start_time.text)
#             start_time = dateutil.parser.parse(start_time.text)
#             print('        Start time:', start_time)
#         else:
#             print('        Start times:', start_times)
#         # read real time duration
#         real_times = measurement.findall(N42_NAMESPACE + 'RealTimeDuration')
#         assert len(real_times) == 1
#         real_time = real_times[0]
#         print('        Real time: ', real_time.text)
#         real_time = parse_duration(real_time.text)
#         print('        Real time: ', real_time)
#         plt.figure()
#         for spectrum in measurement.findall(N42_NAMESPACE + 'Spectrum'):
#             print('        ', spectrum.tag, spectrum.attrib, spectrum.text)
#             # read live time duration
#             live_times = spectrum.findall(
#                 N42_NAMESPACE + 'LiveTimeDuration')
#             assert len(live_times) == 1
#             live_time = live_times[0]
#             print('            Live time: ', live_time.text)
#             live_time = parse_duration(live_time.text)
#             print('            Live time: ', live_time)
#             for cd in spectrum.findall(N42_NAMESPACE + 'ChannelData'):
#                 print('            ', cd.tag, cd.attrib)
#                 comp = cd.get('compressionCode', None)
#                 d = parse_channel_data(cd.text, compression=comp)
#                 plt.plot(d, label=spectrum.attrib['id'])
#                 plt.xlim(0, len(d))
#         plt.legend(prop={'size': 8})
#     return tree


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse N42 samples')
    # parser.add_argument(
    #     'filename',
    #     metavar='filename',
    #     help='The N42 filename',
    # )
    args = parser.parse_args()
    # assert args.filename.lower().endswith('n42')
    # assert validate(args.filename), 'N42 file does not validate'

    for filename in SAMPLES:
        print('')
        print(filename)
        test = N42File(filename)


        # xml_text = etree_parse_clean(filename)
        # os.remove('.temp.xml')

        assert valid_xml(xml_text), 'N42 file is not valid'
        tree = parse_n42(xml_text)
        # tree.write('tests/' + filename)
        plt.show()
