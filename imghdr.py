"""Minimal imghdr shim for Python 3.13 compatibility.

Streamlit 1.19 imports the stdlib `imghdr` module, which was removed
in Python 3.13. This lightweight replacement implements `what()` for
common image signatures. It is intentionally small and safe.
"""

from __future__ import annotations


def _what_from_header(header: bytes | bytearray | memoryview) -> str | None:
    if not header:
        return None
    if isinstance(header, memoryview):
        header = header.tobytes()
    if isinstance(header, bytearray):
        header = bytes(header)

    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if header.startswith(b"BM"):
        return "bmp"
    if header[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def what(file, h: bytes | None = None) -> str | None:
    """Return image type based on a file path, file object, or header bytes."""
    header = h
    if header is None:
        if hasattr(file, "read"):
            header = file.read(32)
            if hasattr(file, "seek"):
                try:
                    file.seek(0)
                except Exception:
                    pass
        else:
            with open(file, "rb") as f:
                header = f.read(32)
    return _what_from_header(header)
