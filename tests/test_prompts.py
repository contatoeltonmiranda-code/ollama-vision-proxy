"""Tests for image-kind classification parsing and the per-kind prompts."""

from ollama_vision_proxy.prompts import (
    CLASSIFY_PROMPT,
    ImageKind,
    classify_prompt,
    describe_prompt,
    parse_kind,
)


class TestParseKind:
    def test_exact_replies(self):
        assert parse_kind("SCREENSHOT") is ImageKind.SCREENSHOT
        assert parse_kind("DOCUMENT") is ImageKind.DOCUMENT
        assert parse_kind("DIAGRAM") is ImageKind.DIAGRAM
        assert parse_kind("PHOTO") is ImageKind.PHOTO
        assert parse_kind("OTHER") is ImageKind.OTHER

    def test_case_and_whitespace_are_tolerated(self):
        assert parse_kind("  screenshot\n") is ImageKind.SCREENSHOT
        assert parse_kind("Photo") is ImageKind.PHOTO

    def test_punctuation_is_tolerated(self):
        assert parse_kind("SCREENSHOT.") is ImageKind.SCREENSHOT
        assert parse_kind("**PHOTO**") is ImageKind.PHOTO

    def test_a_chatty_reply_still_classifies(self):
        """Small models add a sentence however firmly you ask them not to."""
        assert parse_kind("This is a SCREENSHOT of a terminal.") is ImageKind.SCREENSHOT

    def test_unrecognised_reply_falls_back_to_other(self):
        assert parse_kind("a lovely picture") is ImageKind.OTHER

    def test_empty_reply_falls_back_to_other(self):
        assert parse_kind("") is ImageKind.OTHER
        assert parse_kind("   ") is ImageKind.OTHER


class TestClassifyPrompt:
    def test_lists_every_category(self):
        prompt = classify_prompt()
        for kind in ImageKind:
            assert kind.value.upper() in prompt

    def test_asks_for_a_single_word(self):
        assert "one word" in CLASSIFY_PROMPT.lower()


class TestDescribePrompts:
    def test_every_kind_has_a_prompt(self):
        for kind in ImageKind:
            assert describe_prompt(kind).strip()

    def test_screenshot_prompt_prioritises_text_and_apps(self):
        prompt = describe_prompt(ImageKind.SCREENSHOT).lower()
        assert "transcribe" in prompt
        assert "exactly" in prompt
        assert "application" in prompt
        assert any(word in prompt for word in ("error", "stack trace"))

    def test_photo_prompt_prioritises_subjects_then_background(self):
        prompt = describe_prompt(ImageKind.PHOTO).lower()
        assert "subject" in prompt
        assert "background" in prompt
        assert prompt.index("subject") < prompt.index("background")

    def test_diagram_prompt_asks_for_structure_and_labels(self):
        prompt = describe_prompt(ImageKind.DIAGRAM).lower()
        assert "label" in prompt
        assert "connect" in prompt or "structure" in prompt

    def test_document_prompt_asks_for_complete_transcription(self):
        prompt = describe_prompt(ImageKind.DOCUMENT).lower()
        assert "transcribe" in prompt
        assert "reading order" in prompt or "layout" in prompt

    def test_all_prompts_ban_preamble(self):
        """The generic prompt wasted tokens on openers and closing offers."""
        for kind in ImageKind:
            prompt = describe_prompt(kind).lower()
            assert "no preamble" in prompt

    def test_all_prompts_forbid_guessing_proper_nouns(self):
        """It confidently named the wrong buildings on a skyline photo."""
        for kind in ImageKind:
            prompt = describe_prompt(kind).lower()
            assert "do not guess proper nouns" in prompt

    def test_unknown_kind_falls_back_to_the_generic_prompt(self):
        assert describe_prompt(ImageKind.OTHER) == describe_prompt(ImageKind.OTHER)
        assert "describe this image" in describe_prompt(ImageKind.OTHER).lower()
