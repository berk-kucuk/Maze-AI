"""XML from outside, parsed without its own entity definitions (audit MA6).

defusedxml's default in a few lines, without adding a package to Maze's
shared Python: a document that declares entities is refused before the parser
sees it. That rules out entity expansion (billion laughs) and external
entities whatever the expat version underneath; ordinary RSS, Atom and
Thunar's uca.xml never declare any. A DOCTYPE without an internal subset
(some old RSS has one) is still fine.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ParseError = ET.ParseError

_ENTITY_DECL = re.compile(rb"<!ENTITY", re.IGNORECASE)


def _check(data: bytes) -> None:
    if _ENTITY_DECL.search(data):
        raise ET.ParseError("XML that declares entities is not accepted")


def fromstring(text: str | bytes) -> ET.Element:
    data = text.encode("utf-8") if isinstance(text, str) else text
    _check(data)
    return ET.fromstring(text)


def parse(path: str | Path) -> ET.ElementTree:
    data = Path(path).read_bytes()
    _check(data)
    return ET.ElementTree(ET.fromstring(data))
