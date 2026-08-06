"""Read EXIF out of the image bytes, using only the standard library.

Claude Code re-encodes what you paste (a HEIC from a phone arrives as a much
smaller JPEG), but it keeps the APP1 EXIF segment, so the capture time and GPS
fix survive and no filesystem access or third-party library is needed.

Every offset here comes from the image, which is untrusted, so each read is
bounds-checked and any malformed structure yields None rather than an exception.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JPEG_SOI = b"\xff\xd8"
EXIF_HEADER = b"Exif\x00\x00"

# IFD0
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_EXIF_IFD = 0x8769
TAG_GPS_IFD = 0x8825
# Exif sub-IFD
TAG_DATETIME_ORIGINAL = 0x9003
TAG_OFFSET_TIME_ORIGINAL = 0x9011
# GPS sub-IFD
TAG_GPS_LAT_REF = 0x0001
TAG_GPS_LAT = 0x0002
TAG_GPS_LON_REF = 0x0003
TAG_GPS_LON = 0x0004
TAG_GPS_ALT_REF = 0x0005
TAG_GPS_ALT = 0x0006

#: bytes per EXIF type code
TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

MAX_IFD_ENTRIES = 512


@dataclass(frozen=True)
class ExifData:
    """The fields worth putting in front of a model."""

    taken: Optional[str] = None
    utc_offset: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None


def extract_exif(data: bytes) -> Optional[ExifData]:
    """Parse EXIF from JPEG bytes, or None if absent or unreadable."""
    try:
        segment = _find_exif_segment(data)
        if segment is None:
            return None
        return _parse_tiff(segment)
    except Exception as exc:  # noqa: BLE001 - untrusted input, never propagate
        logger.debug("could not parse EXIF: %s", exc)
        return None


def _find_exif_segment(data: bytes) -> Optional[bytes]:
    if not data.startswith(JPEG_SOI):
        return None
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image, or start of scan
            return None
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if length < 2:
            return None
        if marker == 0xE1 and data[offset + 4 : offset + 10] == EXIF_HEADER:
            return data[offset + 10 : offset + 2 + length]
        offset += 2 + length
    return None


class _Tiff:
    """Bounds-checked reader over the TIFF block inside an EXIF segment."""

    def __init__(self, raw: bytes) -> None:
        if len(raw) < 8:
            raise ValueError("TIFF header truncated")
        if raw[:2] == b"II":
            self.byte_order = "<"
        elif raw[:2] == b"MM":
            self.byte_order = ">"
        else:
            raise ValueError("unknown TIFF byte order")
        self.raw = raw

    def u16(self, offset: int) -> int:
        self._check(offset, 2)
        return struct.unpack(self.byte_order + "H", self.raw[offset : offset + 2])[0]

    def u32(self, offset: int) -> int:
        self._check(offset, 4)
        return struct.unpack(self.byte_order + "I", self.raw[offset : offset + 4])[0]

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or offset + size > len(self.raw):
            raise ValueError(f"read of {size} at {offset} is out of bounds")

    def entries(self, offset: int) -> Dict[int, Tuple[int, int, int]]:
        count = self.u16(offset)
        if count > MAX_IFD_ENTRIES:
            raise ValueError(f"implausible IFD entry count {count}")
        found = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            tag = self.u16(entry)
            type_code = self.u16(entry + 2)
            values = self.u32(entry + 4)
            found[tag] = (type_code, values, entry + 8)
        return found

    def value_offset(self, type_code: int, count: int, slot: int) -> int:
        size = TYPE_SIZES.get(type_code, 1) * count
        return self.u32(slot) if size > 4 else slot

    def ascii(self, entry: Optional[Tuple[int, int, int]]) -> Optional[str]:
        if entry is None:
            return None
        type_code, count, slot = entry
        if count <= 0 or count > 4096:
            return None
        offset = self.value_offset(type_code, count, slot)
        self._check(offset, count)
        text = self.raw[offset : offset + count].split(b"\x00", 1)[0]
        cleaned = text.decode("ascii", "replace").strip()
        return cleaned or None

    def rationals(self, entry: Optional[Tuple[int, int, int]]) -> List[float]:
        if entry is None:
            return []
        type_code, count, slot = entry
        if type_code not in (5, 10) or count <= 0 or count > 16:
            return []
        offset = self.value_offset(type_code, count, slot)
        values = []
        for index in range(count):
            numerator = self.u32(offset + index * 8)
            denominator = self.u32(offset + index * 8 + 4)
            values.append(numerator / denominator if denominator else 0.0)
        return values


def _parse_tiff(segment: bytes) -> Optional[ExifData]:
    tiff = _Tiff(segment)
    ifd0 = tiff.entries(tiff.u32(4))

    taken = utc_offset = None
    if TAG_EXIF_IFD in ifd0:
        sub = tiff.entries(tiff.u32(ifd0[TAG_EXIF_IFD][2]))
        taken = tiff.ascii(sub.get(TAG_DATETIME_ORIGINAL))
        utc_offset = tiff.ascii(sub.get(TAG_OFFSET_TIME_ORIGINAL))

    latitude = longitude = altitude = None
    if TAG_GPS_IFD in ifd0:
        gps = tiff.entries(tiff.u32(ifd0[TAG_GPS_IFD][2]))
        latitude = _coordinate(
            tiff.rationals(gps.get(TAG_GPS_LAT)), tiff.ascii(gps.get(TAG_GPS_LAT_REF))
        )
        longitude = _coordinate(
            tiff.rationals(gps.get(TAG_GPS_LON)), tiff.ascii(gps.get(TAG_GPS_LON_REF))
        )
        altitude = _altitude(tiff, gps)

    return ExifData(
        taken=taken,
        utc_offset=utc_offset,
        make=tiff.ascii(ifd0.get(TAG_MAKE)),
        model=tiff.ascii(ifd0.get(TAG_MODEL)),
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )


def _coordinate(parts: List[float], reference: Optional[str]) -> Optional[float]:
    """Degrees/minutes/seconds plus a hemisphere letter into signed degrees."""
    if len(parts) < 3:
        return None
    degrees = parts[0] + parts[1] / 60 + parts[2] / 3600
    if degrees > 180:
        return None
    if reference and reference.upper().startswith(("S", "W")):
        degrees = -degrees
    return degrees


def _altitude(tiff: _Tiff, gps: Dict[int, Tuple[int, int, int]]) -> Optional[float]:
    values = tiff.rationals(gps.get(TAG_GPS_ALT))
    if not values:
        return None
    altitude = values[0]
    reference = gps.get(TAG_GPS_ALT_REF)
    if reference is not None:
        type_code, count, slot = reference
        # BYTE 1 means the value is metres below sea level.
        if type_code == 1 and count == 1 and tiff.raw[slot] == 1:
            altitude = -altitude
    return altitude
