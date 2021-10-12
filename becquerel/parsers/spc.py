"""Read in an Ortec SPC file."""

import os
import struct
import dateutil.parser
import numpy as np
from ..core import calibration
from .parsers import BecquerelParserError


SPC_FORMAT_BEGINNING = [
    # Record 1
    [
        ["INFTYP", "H"],
        ["FILTYP", "H"],
        ["3", "H"],
        ["4", "H"],
        ["ACQIRP", "H"],
        ["SAMDRP", "H"],
        ["DETDRP", "H"],
        ["EBRDESC", "H"],
        ["ANARP1", "H"],
        ["ANARP2", "H"],
        ["ANARP3", "H"],
        ["ANARP4", "H"],
        ["SRPDES", "H"],
        ["IEQDESC", "H"],
        ["GEODES", "H"],
        ["MPCDESC", "H"],
        ["CALDES", "H"],
        ["CALRP1", "H"],
        ["CALRP2", "H"],
        ["EFFPRP", "H"],
        ["ROIRP1", "H"],
        ["22", "H"],
        ["23", "H"],
        ["24", "H"],
        ["25", "H"],
        ["26", "H"],
        ["PERPTR", "H"],
        ["MAXRCS", "H"],
        ["LSTREC", "H"],
        ["EFFPNM", "H"],
        ["SPCTRP", "H"],
        ["SPCRCN", "H"],
        ["SPCCHN", "H"],
        ["ABSTCH", "H"],
        ["ACQTIM", "f"],
        ["ACQTI8", "d"],
        ["SEQNUM", "H"],
        ["MCANU", "H"],
        ["SEGNUM", "H"],
        ["MCADVT", "H"],
        ["CHNSRT", "H"],
        ["RLTMDT", "f"],
        ["LVTMDT", "f"],
        ["50", "H"],
        ["51", "H"],
        ["52", "H"],
        ["53", "H"],
        ["54", "H"],
        ["55", "H"],
        ["56", "H"],
        ["57", "H"],
        ["58", "H"],
        ["59", "H"],
        ["60", "H"],
        ["61", "H"],
        ["62", "H"],
        ["RRSFCT", "f"],
    ],
    # Unknown Record
    [
        ["Unknown Record 1", "128B"],
    ],
    # Acquisition Information Record
    [
        ["Default spectrum file name", "16s"],
        ["Date", "12s"],
        ["Time", "10s"],
        ["Live Time", "10s"],
        ["Real Time", "10s"],
        ["59--90", "B33x"],
        ["Start date of sample collection", "10s"],
        ["Start time of sample collection", "8s"],
        ["Stop date of sample collection", "10s"],
        ["Stop time of sample collection", "8s"],
    ],
    # Sample Description Record
    [
        ["Sample Description", "128s"],
    ],
    # Detector Description Record
    [
        ["Detector Description", "128s"],
    ],
    # First Analysis Parameter
    [
        ["Calibration ?", "16f"],
        ["Testing", "64s"],
    ],
    # Unknown Record
    [
        ["Unknown Record 2", "128B"],
    ],
    # Calibration Description Record
    [
        ["Calibration Description", "128s"],
    ],
    # Description Record 1
    [
        ["Location Description Record 1", "x127s"],
    ],
    # Description Record 2
    [
        ["Location Description Record 2", "128s"],
    ],
    # Unknown Record
    [
        ["Unknown Record 3", "128B"],
    ],
    # Unknown Record
    [
        ["Unknown Record 4", "128B"],
    ],
    # Unknown Record
    [
        ["Unknown Record 5", "128B"],
    ],
    # Unknown Record
    [
        ["Unknown Record 6", "128B"],
    ],
    # Empty Record
    [
        ["Empty Record 1", "128B"],
    ],
    # Empty Record
    [
        ["Empty Record 2", "128B"],
    ],
    # Empty Record
    [
        ["Empty Record 3", "128B"],
    ],
    # Empty Record
    [
        ["Empty Record 4", "128B"],
    ],
    # Empty Record
    [
        ["Empty Record 5", "128B"],
    ],
    # Hardware Parameters Record 1
    [
        ["Hardware Parameters Record 1", "128s"],
    ],
    # Hardware Parameters Record 2
    [
        ["Hardware Parameters Record 2", "128s"],
    ],
]

SPC_FORMAT_END = [
    # Unknown Record
    [
        ["Unknown Record 1", "128B"],
    ],
    # Unknown Record
    [
        ["Unknown Record 2", "128B"],
    ],
    # Calibration parameters
    [
        ["Calibration parameter 0", "f"],
        ["Calibration parameter 1", "f"],
        ["Calibration parameter 2", "f116x"],
    ],
]


def read(filename, verbose=False):
    """Parse the binary SPC file and return a dictionary of data.

    ORTEC's SPC file format is divided into records of 128 bytes each. The
    specifications for what each record should contain can be found on pages
    29--44 of this document:
        http://www.ortec-online.com/download/ortec-software-file-structure-manual.pdf

    In the example file, not all of the records that are supposed to be in
    the files seem to be there, and so this code may depart in some places
    from the specification.

    Parameters
    ----------
    filename : str
        The filename of the CNF file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.

    Returns
    -------
    data : dict
        Dictionary of data that can be used to instantiate a Spectrum.
    cal : Calibration
        Energy calibration stored in the file.
    """
    print("SpcFile: Reading file " + filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".spc":
        raise BecquerelParserError("File extension is incorrect: " + ext)

    # initialize a dictionary of spectrum data to populate as we parse
    data = {}

    with open(filename, "rb") as f:
        # read the file in chunks of 128 bytes
        data_records = []
        binary_data = None
        while True:
            if binary_data is not None:
                data_records.append(binary_data)
            try:
                binary_data = f.read(128)
            except OSError:
                raise BecquerelParserError("Unable to read 128 bytes from file")
            if len(binary_data) < 128:
                break
        if verbose:
            print("Done reading in SPC file.  Number of records: ", len(data_records))
        if len(data_records) not in (279, 280):
            raise BecquerelParserError(
                f"Number of data records incorrect: {len(data_records)}"
            )

        # read record data
        i_rec = 0
        for record_format in SPC_FORMAT_BEGINNING:
            if not (
                len(data_records) == 279
                and record_format[0][0] == "Location Description Record 2"
            ):
                binary_data = data_records[i_rec]
                i_rec += 1
                fmt = "<"
                for data_format in record_format:
                    fmt += data_format[1]
                if verbose:
                    print("")
                    print("")
                    print("-" * 60)
                    print("")
                    print(record_format)
                    print(fmt)
                    print("")
                binary_data = struct.unpack(fmt, binary_data)
                if verbose:
                    print("")
                    print(binary_data)
                    print("")
                for j, data_format in enumerate(record_format):
                    if isinstance(binary_data[j], bytes):
                        data[data_format[0]] = binary_data[j].decode("ascii")
                    else:
                        data[data_format[0]] = binary_data[j]
                    if verbose:
                        print(data_format[0], ": ", data[data_format[0]])

        # read spectrum records
        # These records are the spectrum data stored as INTEGER*4
        # numbers beginning with the channel number given and going
        # through the number of channels in the file. They are stored
        # as 64-word records, which gives 32 data channels per record.
        # They are stored sequentially, beginning with the record
        # pointer given.
        i_channel = 0
        channels = []
        counts = []
        for j in range(256):
            binary_data = data_records[i_rec]
            i_rec += 1
            N = struct.unpack("<32I", binary_data)
            # print(': ', N)
            for j, N_j in enumerate(N):
                channels = np.append(channels, i_channel)
                counts = np.append(counts, N_j)
                i_channel += 1

        # read record data
        for record_format in SPC_FORMAT_END:
            binary_data = data_records[i_rec]
            i_rec += 1
            fmt = "<"
            for data_format in record_format:
                fmt += data_format[1]
            if verbose:
                print("")
                print("")
                print("-" * 60)
                print("")
                print(record_format)
                print(fmt)
                print("")
            binary_data = struct.unpack(fmt, binary_data)
            if verbose:
                print("")
                print(binary_data)
                print("")
            for j, data_format in enumerate(record_format):
                if isinstance(binary_data[j], bytes):
                    data[data_format[0]] = binary_data[j].decode("ascii")
                else:
                    data[data_format[0]] = binary_data[j]
                if verbose:
                    print(data_format[0], ": ", data[data_format[0]])

    # finish populating data dict
    data["counts"] = counts

    data["sample_description"] = data["Sample Description"]
    data["detector_description"] = data["Detector Description"]
    if verbose:
        print(data["Start date of sample collection"])
    data["Start date of sample collection"] = data["Start date of sample collection"][
        :-1
    ]
    if verbose:
        print(data["Start date of sample collection"])
        print(data["Start time of sample collection"])
    data["start_time"] = dateutil.parser.parse(
        data["Start date of sample collection"]
        + " "
        + data["Start time of sample collection"]
    )
    if verbose:
        print(data["start_time"])
        print(data["Stop date of sample collection"])
    data["Stop date of sample collection"] = data["Stop date of sample collection"][:-1]
    if verbose:
        print(data["Stop date of sample collection"])
        print(data["Stop time of sample collection"])
    data["collection_stop"] = dateutil.parser.parse(
        data["Stop date of sample collection"]
        + " "
        + data["Stop time of sample collection"]
    )
    data["collection_stop"] = f"{data['collection_stop']:%Y-%m-%dT%H:%M:%S.%f%z}"
    if verbose:
        print(data["collection_stop"])
    data["location_description"] = data["Location Description Record 1"][3:]
    data["location_description"] = (
        data["location_description"].split("\x00\x00\x00")[0].replace("\x00", "\n")
    )
    if len(data_records) > 279:
        data["location_description"] += (
            data["Location Description Record 2"]
            .split("\x00\x00\x00")[0]
            .replace("\x00", "\n")
        )
    data["hardware_status"] = (
        (data["Hardware Parameters Record 1"] + data["Hardware Parameters Record 2"])
        .split("\x00\x00\x00")[0]
        .replace("\x00", "\n")
    )
    data["livetime"] = float(data["Live Time"])
    data["realtime"] = float(data["Real Time"])
    if data["realtime"] <= 0.0:
        raise BecquerelParserError(
            "Realtime not parsed correctly: {}".format(data["realtime"])
        )
    if data["livetime"] <= 0.0:
        raise BecquerelParserError(
            "Livetime not parsed correctly: {}".format(data["livetime"])
        )
    if data["livetime"] > data["realtime"]:
        raise BecquerelParserError(
            "Livetime > realtime: {} > {}".format(data["livetime"], data["realtime"])
        )
    try:
        cal_coeff = [
            float(data["Calibration parameter 0"]),
            float(data["Calibration parameter 1"]),
            float(data["Calibration parameter 2"]),
        ]
    except KeyError:
        raise BecquerelParserError("Calibration parameters not found")

    # clean up null characters in any strings
    for key in data.keys():
        if isinstance(data[key], str):
            data[key] = data[key].replace("\x00", " ")
            data[key] = data[key].replace("\x01", " ")
            data[key] = data[key].strip()

    # create an energy calibration object
    cal = calibration.Calibration.from_polynomial(cal_coeff)

    return data, cal
