"""Tests for reverse geocoding: request shape, caching, and fail-soft."""

import httpx

from ollama_vision_proxy.geocode import Address, ReverseGeocoder

# Shaped like a real Nominatim jsonv2 reply for the coordinates in the README.
TORONTO = {
    "display_name": "Bremner Boulevard, Entertainment District, Toronto, Ontario, M5V 2T6, Canada",
    "address": {
        "road": "Bremner Boulevard",
        "suburb": "Entertainment District",
        "city": "Toronto",
        "county": "Toronto Division",
        "state": "Ontario",
        "postcode": "M5V 2T6",
        "country": "Canada",
    },
}


def _geocoder(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ReverseGeocoder(client=client, **kwargs)


class TestRequestShape:
    def test_sends_rounded_coordinates(self):
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json=TORONTO)

        _geocoder(handler).lookup(43.6425, -79.38722222222223)
        assert seen["params"]["lat"] == "43.6425"
        assert seen["params"]["lon"] == "-79.3872"

    def test_requests_address_details_at_building_zoom(self):
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json=TORONTO)

        _geocoder(handler).lookup(1.0, 2.0)
        assert seen["params"]["format"] == "jsonv2"
        assert seen["params"]["addressdetails"] == "1"
        assert seen["params"]["zoom"] == "18"

    def test_identifies_itself_with_a_user_agent(self):
        """Nominatim's usage policy requires it."""
        seen = {}

        def handler(request):
            seen["ua"] = request.headers.get("user-agent")
            return httpx.Response(200, json=TORONTO)

        _geocoder(handler).lookup(1.0, 2.0)
        assert "ollama-vision-proxy" in seen["ua"]

    def test_custom_url_is_used(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json=TORONTO)

        _geocoder(handler, url="http://127.0.0.1:8080/reverse").lookup(1.0, 2.0)
        assert seen["url"].startswith("http://127.0.0.1:8080/reverse")


class TestParsing:
    def test_maps_the_fields(self):
        def handler(request):
            return httpx.Response(200, json=TORONTO)

        address = _geocoder(handler).lookup(43.6425, -79.3872)
        assert address == Address(
            road="Bremner Boulevard",
            postcode="M5V 2T6",
            suburb="Entertainment District",
            state="Ontario",
            city="Toronto",
            country="Canada",
        )

    def test_town_stands_in_for_city(self):
        def handler(request):
            return httpx.Response(
                200, json={"address": {"town": "Elora", "country": "Canada"}}
            )

        address = _geocoder(handler).lookup(1.0, 2.0)
        assert address.city == "Elora"

    def test_neighbourhood_stands_in_for_suburb(self):
        def handler(request):
            return httpx.Response(
                200, json={"address": {"neighbourhood": "Old Town", "city": "X"}}
            )

        assert _geocoder(handler).lookup(1.0, 2.0).suburb == "Old Town"

    def test_partial_address_is_kept(self):
        def handler(request):
            return httpx.Response(200, json={"address": {"country": "Canada"}})

        address = _geocoder(handler).lookup(1.0, 2.0)
        assert address.country == "Canada"
        assert address.road is None

    def test_empty_address_becomes_none(self):
        def handler(request):
            return httpx.Response(200, json={"address": {}})

        assert _geocoder(handler).lookup(1.0, 2.0) is None

    def test_missing_address_key_becomes_none(self):
        def handler(request):
            return httpx.Response(200, json={"error": "Unable to geocode"})

        assert _geocoder(handler).lookup(1.0, 2.0) is None


class TestFailSoft:
    """A lookup failure must never break the request it is decorating."""

    def test_http_error_returns_none(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        assert _geocoder(handler).lookup(1.0, 2.0) is None

    def test_rate_limited_returns_none(self):
        def handler(request):
            return httpx.Response(429, text="Too Many Requests")

        assert _geocoder(handler).lookup(1.0, 2.0) is None

    def test_connection_error_returns_none(self):
        def handler(request):
            raise httpx.ConnectError("offline")

        assert _geocoder(handler).lookup(1.0, 2.0) is None

    def test_timeout_returns_none(self):
        def handler(request):
            raise httpx.ReadTimeout("slow")

        assert _geocoder(handler).lookup(1.0, 2.0) is None

    def test_malformed_json_returns_none(self):
        def handler(request):
            return httpx.Response(200, text="<html>not json</html>")

        assert _geocoder(handler).lookup(1.0, 2.0) is None


class TestDisabled:
    def test_disabled_never_calls_out(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=TORONTO)

        assert _geocoder(handler, enabled=False).lookup(1.0, 2.0) is None
        assert calls == []


class TestCaching:
    def test_same_coordinates_query_once(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=TORONTO)

        geocoder = _geocoder(handler)
        for _ in range(5):
            geocoder.lookup(43.6425, -79.3872)
        assert len(calls) == 1

    def test_nearby_coordinates_share_a_cache_entry(self):
        """Rounding means a jittering fix does not re-query."""
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=TORONTO)

        geocoder = _geocoder(handler)
        geocoder.lookup(43.642500, -79.387222)
        geocoder.lookup(43.642549, -79.387201)
        assert len(calls) == 1

    def test_distant_coordinates_query_separately(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=TORONTO)

        geocoder = _geocoder(handler)
        geocoder.lookup(43.6425, -79.3872)
        geocoder.lookup(48.8584, 2.2945)
        assert len(calls) == 2

    def test_failures_are_not_cached(self):
        state = {"n": 0}

        def handler(request):
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json=TORONTO)

        geocoder = _geocoder(handler)
        assert geocoder.lookup(1.0, 2.0) is None
        assert geocoder.lookup(1.0, 2.0) is not None
