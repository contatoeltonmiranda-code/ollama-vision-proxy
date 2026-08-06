"""Tests for image-block detection and replacement."""

import copy

import pytest

from ollama_vision_proxy.transform import (
    URL_SOURCE_PLACEHOLDER,
    ImageBlock,
    transform_request,
)


def _describe(block: ImageBlock) -> str:
    """Deterministic stand-in for the vision model."""
    return f"desc({block.data})"


def _text_block(text):
    return {"type": "text", "text": text}


def _image_block(data="AAAA", media_type="image/png"):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


class TestNoImages:
    def test_string_content_is_untouched(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
        result = transform_request(body, _describe)
        assert result.body == body
        assert result.images_transcribed == 0

    def test_text_blocks_are_untouched(self):
        body = {"messages": [{"role": "user", "content": [_text_block("hi")]}]}
        result = transform_request(body, _describe)
        assert result.body["messages"][0]["content"] == [_text_block("hi")]
        assert result.images_transcribed == 0

    def test_no_messages_key_does_not_raise(self):
        body = {"model": "m"}
        result = transform_request(body, _describe)
        assert result.body == {"model": "m"}
        assert result.images_transcribed == 0

    def test_empty_body_does_not_raise(self):
        result = transform_request({}, _describe)
        assert result.body == {}
        assert result.images_transcribed == 0


class TestSingleImage:
    def test_image_block_becomes_text_block(self):
        body = {"messages": [{"role": "user", "content": [_image_block("XYZ")]}]}
        result = transform_request(body, _describe)
        blocks = result.body["messages"][0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "[Image: desc(XYZ)]"
        assert result.images_transcribed == 1

    def test_no_image_blocks_survive(self):
        body = {
            "messages": [
                {"role": "user", "content": [_text_block("look:"), _image_block()]}
            ]
        }
        result = transform_request(body, _describe)
        blocks = result.body["messages"][0]["content"]
        assert all(b["type"] != "image" for b in blocks)

    def test_surrounding_blocks_keep_their_order(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _text_block("before"),
                        _image_block("IMG"),
                        _text_block("after"),
                    ],
                }
            ]
        }
        result = transform_request(body, _describe)
        texts = [b["text"] for b in result.body["messages"][0]["content"]]
        assert texts == ["before", "[Image: desc(IMG)]", "after"]

    def test_media_type_is_passed_to_the_transcriber(self):
        seen = []

        def describe(block):
            seen.append(block.media_type)
            return "ok"

        body = {
            "messages": [
                {"role": "user", "content": [_image_block(media_type="image/jpeg")]}
            ]
        }
        transform_request(body, describe)
        assert seen == ["image/jpeg"]


class TestMultipleImages:
    def test_multiple_images_in_one_message(self):
        body = {
            "messages": [
                {"role": "user", "content": [_image_block("A"), _image_block("B")]}
            ]
        }
        result = transform_request(body, _describe)
        texts = [b["text"] for b in result.body["messages"][0]["content"]]
        assert texts == ["[Image: desc(A)]", "[Image: desc(B)]"]
        assert result.images_transcribed == 2

    def test_images_across_multiple_messages(self):
        body = {
            "messages": [
                {"role": "user", "content": [_image_block("A")]},
                {"role": "assistant", "content": [_text_block("I see")]},
                {"role": "user", "content": [_image_block("B"), _image_block("C")]},
            ]
        }
        result = transform_request(body, _describe)
        assert result.images_transcribed == 3
        for message in result.body["messages"]:
            for block in message["content"]:
                assert block["type"] == "text"


class TestNestedImages:
    """Claude Code returns MCP/tool screenshots as images inside tool_result."""

    def test_image_inside_tool_result_is_transcribed(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [_text_block("screenshot:"), _image_block("NEST")],
                        }
                    ],
                }
            ]
        }
        result = transform_request(body, _describe)
        inner = result.body["messages"][0]["content"][0]["content"]
        assert inner[0] == _text_block("screenshot:")
        assert inner[1] == {"type": "text", "text": "[Image: desc(NEST)]"}
        assert result.images_transcribed == 1

    def test_tool_result_wrapper_is_preserved(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_9",
                            "is_error": False,
                            "content": [_image_block()],
                        }
                    ],
                }
            ]
        }
        result = transform_request(body, _describe)
        wrapper = result.body["messages"][0]["content"][0]
        assert wrapper["type"] == "tool_result"
        assert wrapper["tool_use_id"] == "toolu_9"
        assert wrapper["is_error"] is False

    def test_image_in_system_blocks_is_transcribed(self):
        body = {
            "system": [_text_block("you are x"), _image_block("SYS")],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = transform_request(body, _describe)
        assert result.body["system"][1] == {"type": "text", "text": "[Image: desc(SYS)]"}
        assert result.images_transcribed == 1


class TestOtherBlockTypesUntouched:
    def test_document_block_passes_through(self):
        doc = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"},
        }
        body = {"messages": [{"role": "user", "content": [doc]}]}
        result = transform_request(body, _describe)
        assert result.body["messages"][0]["content"][0] == doc
        assert result.images_transcribed == 0

    def test_tool_use_block_passes_through(self):
        tool_use = {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "Read",
            "input": {"file_path": "/tmp/a"},
        }
        body = {"messages": [{"role": "assistant", "content": [tool_use]}]}
        result = transform_request(body, _describe)
        assert result.body["messages"][0]["content"][0] == tool_use

    def test_thinking_block_passes_through(self):
        thinking = {"type": "thinking", "thinking": "hmm", "signature": "sig"}
        body = {"messages": [{"role": "assistant", "content": [thinking]}]}
        result = transform_request(body, _describe)
        assert result.body["messages"][0]["content"][0] == thinking

    def test_top_level_keys_are_preserved(self):
        body = {
            "model": "glm-5.2:cloud",
            "max_tokens": 4096,
            "stream": True,
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [_image_block()]}],
        }
        result = transform_request(body, _describe)
        assert result.body["model"] == "glm-5.2:cloud"
        assert result.body["max_tokens"] == 4096
        assert result.body["stream"] is True
        assert result.body["tools"] == body["tools"]


class TestUrlSources:
    def test_url_image_is_replaced_without_calling_the_transcriber(self):
        calls = []

        def describe(block):
            calls.append(block)
            return "should not happen"

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/y.png"},
                        }
                    ],
                }
            ]
        }
        result = transform_request(body, describe)
        assert calls == []
        assert result.body["messages"][0]["content"][0] == {
            "type": "text",
            "text": URL_SOURCE_PLACEHOLDER,
        }
        assert result.images_transcribed == 0
        assert result.images_skipped == 1

    def test_malformed_image_block_is_still_removed(self):
        """A block we cannot read must not be forwarded, or upstream 400s."""
        body = {"messages": [{"role": "user", "content": [{"type": "image"}]}]}
        result = transform_request(body, _describe)
        block = result.body["messages"][0]["content"][0]
        assert block["type"] == "text"
        assert result.images_skipped == 1


class TestPurity:
    def test_input_body_is_not_mutated(self):
        body = {
            "messages": [
                {"role": "user", "content": [_text_block("hi"), _image_block("KEEP")]}
            ]
        }
        original = copy.deepcopy(body)
        transform_request(body, _describe)
        assert body == original

    def test_transcriber_exceptions_propagate(self):
        """Fail-soft belongs in the vision client, not here."""

        def boom(block):
            raise RuntimeError("vision down")

        body = {"messages": [{"role": "user", "content": [_image_block()]}]}
        with pytest.raises(RuntimeError):
            transform_request(body, boom)


class TestHasImages:
    def test_detects_images_without_transcribing(self):
        from ollama_vision_proxy.transform import has_images

        assert has_images({"messages": [{"role": "user", "content": [_image_block()]}]})
        assert not has_images({"messages": [{"role": "user", "content": "hi"}]})

    def test_detects_nested_images(self):
        from ollama_vision_proxy.transform import has_images

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": [_image_block()]}],
                }
            ]
        }
        assert has_images(body)
