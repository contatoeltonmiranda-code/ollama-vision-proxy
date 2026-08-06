"""Turn GPS coordinates into a postal address.

This is the one part that leaves the machine. Only two rounded numbers are sent,
never the image, and the result is cached so a photo sitting in conversation
history does not re-query on every turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from . import __version__
from .cache import TranscriptionCache

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

#: Nominatim asks callers to identify themselves.
USER_AGENT = f"ollama-vision-proxy/{__version__}"

DEFAULT_TIMEOUT = 10.0

#: 4 decimals is about 11 m, finer than a phone's own GPS error, so rounding
#: costs no useful precision while cutting the number of distinct queries.
COORDINATE_PRECISION = 4

#: Nominatim zoom 18 is building level.
ADDRESS_ZOOM = 18

_CITY_KEYS = ("city", "town", "village", "municipality", "hamlet")
_SUBURB_KEYS = ("suburb", "neighbourhood", "quarter", "city_district")
_STATE_KEYS = ("state", "province", "region")


@dataclass(frozen=True)
class Address:
    road: Optional[str] = None
    postcode: Optional[str] = None
    suburb: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.road, self.postcode, self.suburb, self.state, self.city, self.country)
        )


class ReverseGeocoder:
    """Look up an address for a coordinate pair. Never raises."""

    def __init__(
        self,
        url: str = NOMINATIM_URL,
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache: Optional[TranscriptionCache] = None,
        enabled: bool = True,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.enabled = enabled
        self.cache = cache if cache is not None else TranscriptionCache()
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def __call__(self, latitude: float, longitude: float) -> Optional[Address]:
        return self.lookup(latitude, longitude)

    def lookup(self, latitude: float, longitude: float) -> Optional[Address]:
        if not self.enabled:
            return None
        key = self._key(latitude, longitude)
        try:
            return self.cache.get_or_compute(key, lambda: self._request(key))
        except Exception as exc:  # noqa: BLE001 - a lookup must never break a request
            logger.warning("reverse geocoding failed: %s", exc)
            return None

    @staticmethod
    def _key(latitude: float, longitude: float) -> str:
        return (
            f"{round(latitude, COORDINATE_PRECISION)},"
            f"{round(longitude, COORDINATE_PRECISION)}"
        )

    def _request(self, key: str) -> Optional[Address]:
        latitude, longitude = key.split(",")
        response = self._client.get(
            self.url,
            params={
                "format": "jsonv2",
                "lat": latitude,
                "lon": longitude,
                "zoom": ADDRESS_ZOOM,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        fields = payload.get("address")
        if not isinstance(fields, dict):
            return None
        address = Address(
            road=_pick(fields, ("road", "pedestrian", "footway")),
            postcode=_pick(fields, ("postcode",)),
            suburb=_pick(fields, _SUBURB_KEYS),
            state=_pick(fields, _STATE_KEYS),
            city=_pick(fields, _CITY_KEYS),
            country=_pick(fields, ("country",)),
        )
        return None if address.is_empty else address

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ReverseGeocoder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _pick(fields: dict, keys) -> Optional[str]:
    for key in keys:
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
