"""Tests for stdlib EXIF extraction, including hostile and truncated input."""

import struct

from ollama_vision_proxy.exif import extract_exif

TYPE_BYTE, TYPE_ASCII, TYPE_LONG, TYPE_RATIONAL = 1, 2, 4, 5


def _rational(values):
    payload = b""
    for value in values:
        payload += struct.pack("<II", int(round(value * 10000)), 10000)
    return payload


def _build_ifd(entries, base, trailing_pool_base):
    """Lay out one IFD. `entries` maps tag -> (type, count, raw value bytes).

    Values of four bytes or fewer are inlined; longer ones go into a pool whose
    absolute position the caller controls, which is how real EXIF is laid out.
    """
    tags = sorted(entries)
    body = struct.pack("<H", len(tags))
    pool = b""
    pool_offset = trailing_pool_base
    for tag in tags:
        type_code, count, raw = entries[tag]
        if len(raw) <= 4:
            slot = raw.ljust(4, b"\x00")
        else:
            slot = struct.pack("<I", pool_offset)
            pool += raw
            pool_offset += len(raw)
        body += struct.pack("<HHI", tag, type_code, count) + slot
    body += struct.pack("<I", 0)  # no next IFD
    return body, pool


def build_exif_jpeg(
    make="Apple",
    model="iPhone",
    taken="2026:05:04 14:30:00",
    utc_offset="-04:00",
    latitude=(43, 38, 33.0),
    latitude_ref=b"N",
    longitude=(79, 23, 14.0),
    longitude_ref=b"W",
    altitude=76.50,
    altitude_ref=0,
    with_gps=True,
):
    """A minimal but structurally real JPEG carrying an APP1 EXIF segment."""
    # Sub-IFDs are placed after IFD0 and its pool; sizes are known up front.
    exif_entries = {}
    if taken is not None:
        exif_entries[0x9003] = (TYPE_ASCII, len(taken) + 1, taken.encode() + b"\x00")
    if utc_offset is not None:
        exif_entries[0x9011] = (
            TYPE_ASCII,
            len(utc_offset) + 1,
            utc_offset.encode() + b"\x00",
        )

    gps_entries = {}
    if with_gps:
        gps_entries = {
            0x0001: (TYPE_ASCII, 2, latitude_ref + b"\x00"),
            0x0002: (TYPE_RATIONAL, 3, _rational(latitude)),
            0x0003: (TYPE_ASCII, 2, longitude_ref + b"\x00"),
            0x0004: (TYPE_RATIONAL, 3, _rational(longitude)),
            0x0005: (TYPE_BYTE, 1, bytes([altitude_ref])),
            0x0006: (TYPE_RATIONAL, 1, _rational([altitude])),
        }

    ifd0_entries = {
        0x010F: (TYPE_ASCII, len(make) + 1, make.encode() + b"\x00"),
        0x0110: (TYPE_ASCII, len(model) + 1, model.encode() + b"\x00"),
    }
    n_ifd0 = len(ifd0_entries) + (1 if exif_entries else 0) + (1 if gps_entries else 0)
    ifd0_size = 2 + 12 * n_ifd0 + 4
    ifd0_base = 8
    ifd0_pool_base = ifd0_base + ifd0_size
    ifd0_pool_size = sum(
        len(raw) for _, _, raw in ifd0_entries.values() if len(raw) > 4
    )

    exif_base = ifd0_pool_base + ifd0_pool_size
    exif_size = (2 + 12 * len(exif_entries) + 4) if exif_entries else 0
    exif_body, exif_pool = (
        _build_ifd(exif_entries, exif_base, exif_base + exif_size)
        if exif_entries
        else (b"", b"")
    )

    gps_base = exif_base + exif_size + len(exif_pool)
    gps_size = (2 + 12 * len(gps_entries) + 4) if gps_entries else 0
    gps_body, gps_pool = (
        _build_ifd(gps_entries, gps_base, gps_base + gps_size)
        if gps_entries
        else (b"", b"")
    )

    if exif_entries:
        ifd0_entries[0x8769] = (TYPE_LONG, 1, struct.pack("<I", exif_base))
    if gps_entries:
        ifd0_entries[0x8825] = (TYPE_LONG, 1, struct.pack("<I", gps_base))
    ifd0_body, ifd0_pool = _build_ifd(ifd0_entries, ifd0_base, ifd0_pool_base)

    tiff = (
        b"II"
        + struct.pack("<H", 42)
        + struct.pack("<I", ifd0_base)
        + ifd0_body
        + ifd0_pool
        + exif_body
        + exif_pool
        + gps_body
        + gps_pool
    )
    segment = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(segment) + 2) + segment
    return b"\xff\xd8" + app1 + b"\xff\xd9"


class TestHappyPath:
    def test_reads_camera_make_and_model(self):
        data = extract_exif(build_exif_jpeg())
        assert data.make == "Apple"
        assert data.model == "iPhone"

    def test_reads_capture_time_and_offset(self):
        data = extract_exif(build_exif_jpeg())
        assert data.taken == "2026:05:04 14:30:00"
        assert data.utc_offset == "-04:00"

    def test_converts_dms_to_signed_degrees(self):
        data = extract_exif(build_exif_jpeg())
        assert data.latitude == pytest_approx(43.642500, 1e-5)
        # West is negative.
        assert data.longitude == pytest_approx(-79.387222, 1e-5)

    def test_reads_altitude(self):
        data = extract_exif(build_exif_jpeg())
        assert data.altitude == pytest_approx(76.50, 1e-3)

    def test_has_gps_is_true(self):
        assert extract_exif(build_exif_jpeg()).has_gps is True

    def test_southern_and_eastern_hemispheres_are_negative_and_positive(self):
        data = extract_exif(
            build_exif_jpeg(
                latitude=(33, 51, 54.0),
                latitude_ref=b"S",
                longitude=(151, 12, 36.0),
                longitude_ref=b"E",
            )
        )
        assert data.latitude < 0
        assert data.longitude > 0

    def test_below_sea_level_altitude_is_negative(self):
        data = extract_exif(build_exif_jpeg(altitude=12.5, altitude_ref=1))
        assert data.altitude == pytest_approx(-12.5, 1e-3)


class TestMissingData:
    def test_no_gps_means_has_gps_is_false(self):
        data = extract_exif(build_exif_jpeg(with_gps=False))
        assert data is not None
        assert data.has_gps is False
        assert data.make == "Apple"  # the rest still parses

    def test_missing_timestamp_is_none(self):
        data = extract_exif(build_exif_jpeg(taken=None, utc_offset=None))
        assert data.taken is None
        assert data.utc_offset is None


class TestUnparseableInput:
    """Image bytes are untrusted, so nothing here may raise."""

    def test_empty_bytes(self):
        assert extract_exif(b"") is None

    def test_not_a_jpeg(self):
        assert extract_exif(b"just some text, not an image at all") is None

    def test_png_returns_none(self):
        assert extract_exif(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64) is None

    def test_jpeg_without_exif(self):
        assert extract_exif(b"\xff\xd8\xff\xdb\x00\x04\x00\x00\xff\xd9") is None

    def test_truncated_after_exif_header(self):
        assert extract_exif(b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00II") is None

    def test_unknown_byte_order(self):
        payload = b"Exif\x00\x00ZZ\x00\x00\x00\x00\x00\x00"
        data = (
            b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"
        )
        assert extract_exif(data) is None

    def test_offsets_pointing_outside_the_segment(self):
        good = build_exif_jpeg()
        # Corrupt the IFD0 offset so every read runs off the end.
        index = good.index(b"Exif\x00\x00") + 6
        broken = bytearray(good)
        broken[index + 4 : index + 8] = struct.pack("<I", 0xFFFFFF)
        assert extract_exif(bytes(broken)) is None

    def test_absurd_entry_count_is_rejected(self):
        good = build_exif_jpeg()
        index = good.index(b"Exif\x00\x00") + 6
        broken = bytearray(good)
        broken[index + 8 : index + 10] = struct.pack("<H", 60000)
        assert extract_exif(bytes(broken)) is None

    def test_truncated_mid_segment_does_not_raise(self):
        good = build_exif_jpeg()
        for cut in range(12, len(good), 7):
            extract_exif(good[:cut])  # must not raise


def pytest_approx(value, tolerance):
    class _Approx:
        def __eq__(self, other):
            return other is not None and abs(other - value) < tolerance

        def __repr__(self):
            return f"~{value}"

    return _Approx()
