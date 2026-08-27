"""Helpers for attaching local images to a chat message.

A user turn may carry an ``images`` key (a list of file paths). Each backend
encodes those into its own wire format; this module does the shared work of
reading the file and guessing its MIME type.
"""

from __future__ import annotations

import base64
import os

_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def mime_for(path: str) -> str:
    return _EXT_MIME.get(os.path.splitext(path)[1].lower(), "image/png")


def encode_image(path: str) -> tuple[str, str]:
    """Return ``(mime_type, base64_data)`` for an image file.

    Raises ``OSError`` if the file can't be read.
    """
    with open(os.path.expanduser(path), "rb") as fh:
        data = fh.read()
    return mime_for(path), base64.b64encode(data).decode("ascii")


def data_uri(path: str) -> str:
    """Return a ``data:<mime>;base64,<data>`` URI for an image file."""
    mime, b64 = encode_image(path)
    return f"data:{mime};base64,{b64}"
