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
        self._parse()

    def get_compression_code(self) -> str:
        pass

    def _parse(self) -> None:
        tree = etree.parse(str(self.path))
        try:
            _SCHEMA.validate(tree)
        except Exception:
            raise BecquerelParserError("TODO")

        # N42 requires that this be the top level key
        root = tree.getroot()
        if root.tag != _ns("RadInstrumentData"):
            raise BecquerelParserError(f"Invalid N42 root tag: {root.tag}")

        # Crawl through all of the possible levels and extract
        self.info = {}
        children = root.getchildren()
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
        print(self.info)
