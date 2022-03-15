"""Read in an N42 xml file."""

from pathlib import Path
from typing import Union
from dataclasses import dataclass

# import struct
# import dateutil.parser
import numpy as np
from lxml import etree

# from ..core import calibration
from .parsers import BecquerelParserError


_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA = etree.XMLSchema(etree.parse(str(_THIS_DIR / "n42.xsd")))
_NAMESPACE = "{http://physics.nist.gov/N42/2011/N42}"


def _ns(suffix: str) -> str:
    return f"{_NAMESPACE}{suffix}"


def _strip_ns(item: str) -> str:
    return item.split(_NAMESPACE)[-1]


def make_element_dict(element: etree.Element, dic: dict = {}) -> dict:
    """Recurse down element's children and build a nested dictionary containing
    items (properties), children (subelements) and text (value).

    Parameters
    ----------
    element : etree.Element
        lxml element to read
    dic : dict, optional
        Dictionary to populate (will be returned), by default {}

    Returns
    -------
    dictionary
        _description_

    Raises
    ------
    NotImplementedError
        If children tags or item names are not unique (i.e. multiple children
        of the same type.) This can be implemented at a later time.
    """
    # Value (text)
    dic["value"] = "" if element.text.strip() == "" else element.text
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
    """Find all element tags matching `val` and build a list of dicts from
    `make_element_dict`.

    Parameters
    ----------
    val : str
        Search term
    element : etree.Element
        lxml element to search within.

    Returns
    -------
    list
        Iterable of element dictionaries from `make_element_dict`.
    """
    results = []
    for e in element.findall(val, namespaces=element.nsmap):
        results.append(make_element_dict(e))
    return results


@dataclass
class N42Spectrum:
    counts: np.ndarray
    realtime: float
    livetime: float


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

    def get_compression_code(self) -> str:
        pass

    def _parse(self) -> None:
        # Crawl through all of the possible levels and extract
        self.info = {}
        children = self.root.getchildren()
        for child in children:
            if _strip_ns(child.tag) == "RadInstrumentInformation":
                print(f"RadInstrumentInformation: {child.text}")
                if len(child.getchildren()) > 0:
                    gchildren = child.getchildren()
                    for gchild in gchildren:
                        self.info[_strip_ns(gchild.tag)] = gchild.text
                else:
                    self.info[_strip_ns(child.tag)] = child.text
            if _strip_ns(child.tag) == "RadDetectorInformation":
                print(f"RadDetectorInformation: {child.text}")
                self.info[_strip_ns(child.tag)] = child.text
            if _strip_ns(child.tag) == "EnergyCalibration":
                print(f"ENERGY CAL: {child.text}")
                self.info[_strip_ns(child.tag)] = child.text
            if _strip_ns(child.tag) == "RadMeasurement":
                for gchild in child.getchildren():
                    print(f"{gchild.tag=}")
                    print(f"{gchild.text=}")
                    # Get an attribute
                    if len(gchild.items()) > 0:
                        print(gchild.items())
                    if len(gchild.getchildren()) > 0:
                        for ggchild in gchild.getchildren():
                            print(f"{ggchild.tag=}")
                            print(f"{ggchild.text=}")
        print(self.info)
