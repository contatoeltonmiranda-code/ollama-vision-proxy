"""Tests for the SHA256-keyed transcription cache."""

import threading

from ollama_vision_proxy.cache import TranscriptionCache


class TestBasics:
    def test_miss_then_hit(self):
        cache = TranscriptionCache()
        calls = []

        def produce():
            calls.append(1)
            return "described"

        assert cache.get_or_compute("DATA", produce) == "described"
        assert cache.get_or_compute("DATA", produce) == "described"
        assert len(calls) == 1

    def test_same_image_transcribed_only_once(self):
        """The whole point: conversation history resends the same image."""
        cache = TranscriptionCache()
        calls = []

        def produce():
            calls.append(1)
            return "x"

        for _ in range(10):
            cache.get_or_compute("SAME", produce)
        assert len(calls) == 1

    def test_different_images_computed_separately(self):
        cache = TranscriptionCache()
        results = [
            cache.get_or_compute("A", lambda: "ra"),
            cache.get_or_compute("B", lambda: "rb"),
        ]
        assert results == ["ra", "rb"]
        assert len(cache) == 2

    def test_keys_are_sha256_of_the_payload(self):
        import hashlib

        cache = TranscriptionCache()
        cache.get_or_compute("payload", lambda: "v")
        expected = hashlib.sha256(b"payload").hexdigest()
        assert expected in cache.keys()

    def test_distinct_payloads_do_not_collide(self):
        cache = TranscriptionCache()
        cache.get_or_compute("a" * 5000, lambda: "big-a")
        cache.get_or_compute("b" * 5000, lambda: "big-b")
        assert cache.get_or_compute("a" * 5000, lambda: "wrong") == "big-a"
        assert cache.get_or_compute("b" * 5000, lambda: "wrong") == "big-b"

    def test_stats_report_hits_and_misses(self):
        cache = TranscriptionCache()
        cache.get_or_compute("A", lambda: "1")
        cache.get_or_compute("A", lambda: "1")
        cache.get_or_compute("B", lambda: "2")
        assert cache.misses == 2
        assert cache.hits == 1


class TestFailureHandling:
    def test_failures_are_not_cached(self):
        """A transient vision timeout must not poison the entry forever."""
        cache = TranscriptionCache()
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("timeout")
            return "recovered"

        try:
            cache.get_or_compute("K", flaky)
        except RuntimeError:
            pass
        assert cache.get_or_compute("K", flaky) == "recovered"
        assert len(attempts) == 2

    def test_exception_propagates_to_caller(self):
        cache = TranscriptionCache()
        raised = False
        try:
            cache.get_or_compute("K", lambda: (_ for _ in ()).throw(ValueError("nope")))
        except ValueError:
            raised = True
        assert raised


class TestThreadSafety:
    def test_concurrent_identical_requests_compute_once(self):
        """Single-flight: parallel subagent calls resend the same history."""
        cache = TranscriptionCache()
        calls = []
        release = threading.Event()

        def slow():
            calls.append(1)
            release.wait(timeout=5)
            return "value"

        threads = [
            threading.Thread(target=cache.get_or_compute, args=("SHARED", slow))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=10)

        assert len(calls) == 1
        assert cache.get_or_compute("SHARED", lambda: "wrong") == "value"

    def test_concurrent_distinct_requests_all_complete(self):
        cache = TranscriptionCache()
        results = {}

        def work(key):
            results[key] = cache.get_or_compute(key, lambda: f"v-{key}")

        threads = [threading.Thread(target=work, args=(str(i),)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert len(results) == 16
        assert all(results[str(i)] == f"v-{i}" for i in range(16))

    def test_distinct_keys_do_not_block_each_other(self):
        """A slow transcription must not serialise unrelated ones."""
        cache = TranscriptionCache()
        started = threading.Event()
        blocked = threading.Event()

        def slow():
            started.set()
            blocked.wait(timeout=5)
            return "slow"

        slow_thread = threading.Thread(
            target=cache.get_or_compute, args=("SLOW", slow)
        )
        slow_thread.start()
        assert started.wait(timeout=5)

        # Must not block behind the in-flight "SLOW" entry.
        assert cache.get_or_compute("FAST", lambda: "fast") == "fast"

        blocked.set()
        slow_thread.join(timeout=10)
