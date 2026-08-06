"""Classify an image first, then ask for the description that kind deserves.

A single generic prompt treats a stack trace and a landscape the same way. It
produced padded prose with a "Here's a detailed description" preamble, and it
invented proper nouns: on a Hong Kong skyline it confidently named the wrong
buildings. Each prompt below therefore states what matters for that kind, bans
the preamble, and forbids guessing names that are not written in the image.
"""

from __future__ import annotations

import enum


class ImageKind(enum.Enum):
    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    DIAGRAM = "diagram"
    PHOTO = "photo"
    OTHER = "other"


CLASSIFY_PROMPT = (
    "Classify this image into exactly one category. Reply with one word only, "
    "no punctuation or explanation.\n"
    "SCREENSHOT: a computer or phone interface, terminal, code editor, web page, "
    "chat window, or dashboard.\n"
    "DOCUMENT: a scan or photograph of printed or handwritten text, such as a "
    "page, form, receipt, or whiteboard of writing.\n"
    "DIAGRAM: a chart, graph, plot, map, architecture or flow diagram.\n"
    "PHOTO: a real-world scene, people, animals, objects, food, or places.\n"
    "OTHER: anything that fits none of the above.\n"
    "Answer with one of: SCREENSHOT, DOCUMENT, DIAGRAM, PHOTO, OTHER"
)

#: Appended to every description prompt. The preamble ban is worth real tokens:
#: the generic prompt spent them on "Here's a detailed description" openers and
#: "let me know if you would like more detail" closers.
_COMMON_RULES = (
    "\nRules: Start directly with the content, with no preamble, no restating of "
    "this instruction, and no offer of further help. Do not use markdown headings. "
    "Do not guess proper nouns: never name a person, company, building, city, or "
    "landmark unless that name is written in the image. If you are unsure, "
    "describe what you see instead of naming it."
)

_PROMPTS = {
    ImageKind.SCREENSHOT: (
        "This is a screenshot of a user interface. Transcribe every piece of "
        "visible text exactly, preserving line breaks, indentation, and reading "
        "order. Name the applications, tools, or websites visible, but only from "
        "text or labels actually shown. Report any error message, stack trace, "
        "warning, command, file path, or URL verbatim and in full. Note which "
        "element appears focused or selected, and describe the layout only "
        "briefly, since the text is what matters."
    ),
    ImageKind.DOCUMENT: (
        "This is a document or a photograph of text. Transcribe all text exactly "
        "and completely, preserving reading order, line breaks, headings, lists, "
        "and table structure. Use plain text alignment for tables. Mark anything "
        "genuinely illegible as [illegible] rather than guessing at it."
    ),
    ImageKind.DIAGRAM: (
        "This is a diagram, chart, or map. Transcribe every label, axis, legend, "
        "and data value exactly. Then state the structure: what the nodes or "
        "series are, how they connect, and in which direction, so the diagram "
        "could be rebuilt from your description alone. For charts, give the "
        "actual values or the trend if values are not printed."
    ),
    ImageKind.PHOTO: (
        "This is a photograph of a real scene. Describe the main subjects in "
        "detail first: what they are, their appearance, position, and what they "
        "are doing. Then describe the setting and background in less detail. "
        "Note lighting, time of day, and weather if evident. Transcribe any text "
        "that is visible, such as signs or labels, exactly."
    ),
    ImageKind.OTHER: (
        "Describe this image in detail. Transcribe any visible text exactly."
    ),
}


def classify_prompt() -> str:
    return CLASSIFY_PROMPT


def describe_prompt(kind: ImageKind) -> str:
    """The description prompt for an image kind, including the shared rules."""
    return _PROMPTS.get(kind, _PROMPTS[ImageKind.OTHER]) + _COMMON_RULES


def parse_kind(reply: str) -> ImageKind:
    """Map a classifier reply onto a kind, defaulting to OTHER.

    Models add stray punctuation, casing, and the occasional sentence, so match
    on the first recognised keyword rather than demanding an exact answer.
    """
    if not reply:
        return ImageKind.OTHER
    text = reply.strip().upper()
    for kind in (
        ImageKind.SCREENSHOT,
        ImageKind.DOCUMENT,
        ImageKind.DIAGRAM,
        ImageKind.PHOTO,
    ):
        if kind.value.upper() in text:
            return kind
    return ImageKind.OTHER
