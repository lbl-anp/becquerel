"""Read in an N42 xml file."""

from pathlib import Path
from typing import Union
from dataclasses import dataclass

import struct
import dateutil.parser
import numpy as np
from lxml import etree
from ..core import calibration
from .parsers import BecquerelParserError


_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA = etree.XMLSchema(etree.parse(_THIS_DIR / "n42.xsd"))
_NAMESPACE = "http://physics.nist.gov/N42/2011/N42"


def _ns(suffix: str) -> str:
    return f"{_NAMESPACE}/{suffix}"


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
        self._validate()
        self._parse()

    def get_compression_code(self) -> str:
        pass

    def _validate(self) -> None:
        try:
            _SCHEMA.validate(self.path.read())
        except Exception:
            pass

    def _parse(self) -> None:
        tree = etree.parse(self.path.read())
        root = tree.getroot()
        if root.tag != _ns("RadInstrumentData"):
            raise BecquerelParserError(f"Invalid N42 root tag: {root.tag}")
