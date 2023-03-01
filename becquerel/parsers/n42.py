"""Read in an N42 xml file."""

from pathlib import Path
from typing import Union
from dataclasses import dataclass

# import struct
# import dateutil.parser
import numpy as np
from lxml import etree

from ..core.calibration import Calibration
from .parsers import BecquerelParserError


_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA = etree.XMLSchema(etree.parse(str(_THIS_DIR / "n42.xsd")))
_NAMESPACE = "{http://physics.nist.gov/N42/2011/N42}"


def _ns(suffix: str) -> str:
    return f"{_NAMESPACE}{suffix}"


def _strip_ns(item: str) -> str:
    return item.split(_NAMESPACE)[-1]


def make_element_dict(element: etree.Element, dic: dict = None) -> dict:
    """Recurse down element's children and build a nested dictionary containing
    items (properties), children (subelements) and text (value).

    Parameters
    ----------
    element : etree.Element
        lxml element to read
    dic : dict, optional
        Dictionary to populate (will be returned), by default None (create a new
        dictionary).

    Returns
    -------
    dictionary
        The contents of the element in the form:
            item name: item value
            children tag: children dict (recursion)
            value: text for this element (between the tags: <tag>VALUE</tag>)

    Raises
    ------
    NotImplementedError
        If children tags or item names are not unique (i.e. multiple children
        of the same type.) This can be implemented at a later time.
    """
    dic = {} if dic is None else dic
    # Value (text)
    if element.text.strip() != "":
        dic["value"] = element.text
    # Attributes (items)
    for k, v in element.items():
        dic[k] = v
    # Children (subitems)
    for subelement in element.getchildren():
        if subelement.tag in dic:
            raise NotImplementedError(
                f"Multiple subelements/items with same tag: {subelement.tag}"
            )
        make_element_dict(
            subelement, dic.setdefault(etree.QName(subelement).localname, {})
        )
    return dic


def _find_all(val: str, element: etree.Element) -> list:
    """Find all element tags matching `val` and build a nested dict from
    `make_element_dict`.

    Parameters
    ----------
    val : str
        Search term
    element : etree.Element
        lxml element to search within.

    Returns
    -------
    dict
        Dictionary of nested XML elements and attributes. The top level will
        contain the key 'id', which holds the name of the top level element or
        attribute requested. The other keys are element or attribute names. Values
        are stored as dictionaries of the form {'value': str}.
    """
    results = []
    for e in element.findall(val, namespaces=element.nsmap):
        results.append(make_element_dict(e))
    return results


def _parse_iso8601_duration(text: str) -> float:
    """Parse ISO 8601 time duration into seconds.

    Only covers case where text is "PTXS", where X is the number of seconds.
    https://en.wikipedia.org/wiki/ISO_8601#Durations

    """
    if not text.startswith("PT") or not text.endswith("S"):
        raise BecquerelParserError("Invalid ISO 8601 duration provided")
    return float(text[2:-1])


def _parse_int_or_float(s) -> Union[int, float]:
    """Parse `s` to an int if possible or a float otherwise.

    Returns
    -------
    int or float

    Examples
    --------
    _parse_int_or_float(1)        # 1
    _parse_int_or_float("1")      # 1
    _parse_int_or_float(1.1)      # 1
    _parse_int_or_float("1.1")    # 1.1
    _parse_int_or_float(1.0e3)    # 1000
    _parse_int_or_float("1.0e3")  # 1000.0

    References
    ----------
    * https://stackoverflow.com/a/5609191
    """
    try:
        return int(s)
    except ValueError:
        return float(s)


def _parse_channel_data(text: str, compression: str = "None") -> np.ndarray:
    """Parse N42 ChannelData text into a numpy array of integer channel data.

    Arguments
    ---------
    text : str
        Raw ChannelData string read from an N42 file
    compression : str
        compression: 'None' or 'CountedZeroes'.
    """
    text = text.strip().replace("\n", " ")
    tokens = text.split()
    data = [_parse_int_or_float(token) for token in tokens]
    if compression == "CountedZeroes":
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
    return np.array(data, dtype=int)


@dataclass
class N42RadMeasurement:
    startime: str
    realtime: float
    livetime: float
    counts: np.ndarray
    # Corresponding name in the XML file, e.g. "EnergyCalibration-1"
    # TODO: keep the becquerel Calibration?
    calibration: str


class N42File:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path if isinstance(path, Path) else Path(path)
        _ext = self.path.suffix.lower()
        if _ext not in (".n42", ".xml"):
            raise BecquerelParserError(f"File extension is incorrect: {_ext}")
        # Open
        self.tree = etree.parse(str(self.path))
        self.root = self.tree.getroot()
        # Validate
        try:
            _SCHEMA.validate(self.tree)
        except Exception:
            raise BecquerelParserError("TODO")
        # N42 requires that this be the top level key
        if self.root.tag != _ns("RadInstrumentData"):
            raise BecquerelParserError(f"Invalid N42 root tag: {self.root.tag}")
        # Read
        self._parse()

    def find_all(self, val: str, element: etree.Element = None) -> list:
        """Find all element tags matching `val` and build a list of dicts from
        `make_element_dict`.

        Parameters
        ----------
        val : str
            Search term
        element : etree.Element, optional
            lxml element to search within, by default None (uses the root element)

        Returns
        -------
        list
            Iterable of element dictionaries from `make_element_dict`.
        """
        element = self.root if element is None else element
        return _find_all(val, element)

    def _parse(self) -> None:
        # Populate basic top level properties
        self.rad_inst_info = self.find_all("RadInstrumentInformation")
        self.rad_det_info = self.find_all("RadDetectorInformation")

        # Grab energy calibrations
        calibrations = self.find_all("EnergyCalibration")
        self.calibrations = {}
        for cal in calibrations:
            txt = cal["CoefficientValues"]["value"]
            coefs = np.array([float(x) for x in txt.split(" ")], dtype=float)
            # TODO: are these always 3 params?
            self.calibrations[cal["id"]] = Calibration(
                "p[0] + p[1] * x + p[2] * x**2", coefs
            )
        print(self.calibrations)

        # Grab measurements
        measurements = self.find_all("RadMeasurement")
        self.measurements = {}
        for meas in measurements:
            starttime = meas["StartDateTime"]["value"]
            realtime = _parse_iso8601_duration(meas["RealTimeDuration"]["value"])
            livetime = _parse_iso8601_duration(
                meas["Spectrum"]["LiveTimeDuration"]["value"]
            )
            calib = meas["Spectrum"]["energyCalibrationReference"]
            compression = meas["Spectrum"]["ChannelData"]["compressionCode"]
            # TODO: can you have multiple spectra per measurement?
            counts = _parse_channel_data(
                meas["Spectrum"]["ChannelData"]["value"], compression
            )
            n42_meas = N42RadMeasurement(starttime, realtime, livetime, counts, calib)
            self.measurements[meas["id"]] = n42_meas
        print(self.measurements)
