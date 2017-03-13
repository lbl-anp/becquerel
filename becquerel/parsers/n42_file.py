"""Create and parse N42 radiation measurement files."""

# pylint: disable=no-member

from __future__ import print_function
import xml.etree.ElementTree as ET
import dateutil.parser
import requests
from lxml import etree
import matplotlib.pyplot as plt


# download N42 schema
NIST_N42_URL = 'http://physics.nist.gov/N42/2011/'
N42_XSD_URL = NIST_N42_URL + 'n42.xsd'
req = requests.get(N42_XSD_URL)
schema_text = req.text.encode('ascii')
schema_root = etree.XML(schema_text)
N42_SCHEMA = etree.XMLSchema(schema_root)
N42_NAMESPACE = '{{{}}}'.format(NIST_N42_URL + 'N42')


SAMPLES = [
    'AnnexB.n42',
    'AnnexB_alternate_energy_calibration.n42',
    'AnnexC.n42',
    'AnnexE.n42',
    'AnnexG.n42',
    'AnnexI.n42',
]


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
    print('length of data', len(data))
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
    xml_parser = etree.XMLParser(schema=N42_SCHEMA)
    try:
        etree.fromstring(text, xml_parser)
        return True
    except etree.XMLSchemaError:
        return False


def parse_n42(text):
    """Parse an N42 file."""
    tree = ET.ElementTree(ET.fromstring(text))
    root = tree.getroot()
    # root should be a RadInstrumentData
    assert root.tag == N42_NAMESPACE + 'RadInstrumentData'
    # read instrument information
    instrument_info = {}
    for info in root.findall(N42_NAMESPACE + 'RadInstrumentInformation'):
        instrument_info[info.attrib['id']] = info
    # read detector information
    detector_info = {}
    for info in root.findall(N42_NAMESPACE + 'RadDetectorInformation'):
        detector_info[info.attrib['id']] = info
    # read energy calibrations
    energy_cals = {}
    for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
        energy_cals[cal.attrib['id']] = cal
    # read FWHM calibrations
    fwhm_cals = {}
    for cal in root.findall(N42_NAMESPACE + 'EnergyCalibration'):
        fwhm_cals[cal.attrib['id']] = cal
    # read measurements
    for measurement in root.findall(
            N42_NAMESPACE + 'RadMeasurement'):
        print('    ', measurement.tag, measurement.attrib)
        class_codes = measurement.findall(
            N42_NAMESPACE + 'MeasurementClassCode')
        # read start time
        start_times = measurement.findall(N42_NAMESPACE + 'StartDateTime')
        assert len(start_times) == 1
        start_time = start_times[0]
        print('        Start time:', start_time.text)
        start_time = dateutil.parser.parse(start_time.text)
        print('        Start time:', start_time)
        # read real time duration
        real_times = measurement.findall(N42_NAMESPACE + 'RealTimeDuration')
        assert len(real_times) == 1
        real_time = real_times[0]
        print('        Real time: ', real_time.text)
        real_time = parse_duration(real_time.text)
        print('        Real time: ', real_time)
        plt.figure()
        for spectrum in measurement.findall(N42_NAMESPACE + 'Spectrum'):
            print('        ', spectrum.tag, spectrum.attrib, spectrum.text)
            # read live time duration
            live_times = spectrum.findall(
                N42_NAMESPACE + 'LiveTimeDuration')
            assert len(live_times) == 1
            live_time = live_times[0]
            print('            Live time: ', live_time.text)
            live_time = parse_duration(live_time.text)
            print('            Live time: ', live_time)
            for cd in spectrum.findall(N42_NAMESPACE + 'ChannelData'):
                print('            ', cd.tag, cd.attrib)
                comp = cd.get('compressionCode', None)
                d = parse_channel_data(cd.text, compression=comp)
                plt.plot(d, label=spectrum.attrib['id'])
                plt.xlim(0, len(d))
        plt.legend(prop={'size': 8})
    return tree


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
        req = requests.get(NIST_N42_URL + filename)
        xml_text = req.text.encode('ascii').decode('ascii')
        assert valid_xml(xml_text), 'N42 file is not valid'
        tree = parse_n42(xml_text)
        tree.write('tests/' + filename)
        plt.show()
