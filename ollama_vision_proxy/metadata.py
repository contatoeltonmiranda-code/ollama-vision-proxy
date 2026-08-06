"""Render the <metadata> block that follows an image description.

Shown only when the image carries a GPS fix. A screenshot has nothing useful
here, so it gets no block at all rather than a near-empty one.

Values originate in the image's own EXIF and in a geocoding response, neither of
which is trustworthy, so every value is sanitised before it goes anywhere near
the prompt.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .exif import ExifData
from .geocode import Address

METADATA_OPEN = "<metadata>"
METADATA_CLOSE = "</metadata>"

#: Label column width, so values line up in the rendered block.
LABEL_WIDTH = 8

MAX_VALUE_LENGTH = 120


def render_metadata(
    exif: Optional[ExifData], address: Optional[Address] = None
) -> Optional[str]:
    """The metadata block, or None when there is no GPS fix to report."""
    if exif is None or not exif.has_gps:
        return None

    rows: List[Tuple[str, Optional[str]]] = [
        ("taken", _timestamp(exif)),
        ("make", exif.make),
        ("device", exif.model),
        ("lat", _latitude(exif.latitude)),
        ("long", _longitude(exif.longitude)),
        ("alt", _altitude(exif.altitude)),
    ]
    if address is not None:
        rows += [
            ("road", address.road),
            ("zipcode", address.postcode),
            ("suburb", address.suburb),
            ("state", address.state),
            ("city", address.city),
            ("country", address.country),
        ]

    lines = [
        f"{label.ljust(LABEL_WIDTH)} {_sanitise(value)}"
        for label, value in rows
        if value
    ]
    if not lines:
        return None
    return "\n".join([METADATA_OPEN, *lines, METADATA_CLOSE])


def _timestamp(exif: ExifData) -> Optional[str]:
    """EXIF writes 'YYYY:MM:DD HH:MM:SS'; show a normal date and the offset."""
    if not exif.taken:
        return None
    stamp = exif.taken.strip()
    date, separator, clock = stamp.partition(" ")
    if separator:
        stamp = f"{date.replace(':', '-')} {clock}"
    if exif.utc_offset:
        stamp = f"{stamp} {exif.utc_offset.strip()}"
    return stamp


def _latitude(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{abs(value):.8f} {'S' if value < 0 else 'N'}"


def _longitude(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{abs(value):.8f} {'W' if value < 0 else 'E'}"


def _altitude(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.2f} m"


def _sanitise(value: str) -> str:
    """Keep a value on one line and unable to forge a block delimiter."""
    flattened = " ".join(str(value).split())
    flattened = flattened.replace("<", "(").replace(">", ")")
    if len(flattened) > MAX_VALUE_LENGTH:
        flattened = flattened[:MAX_VALUE_LENGTH] + "..."
    return flattened
