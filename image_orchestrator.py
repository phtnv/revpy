"""
Chat-side image request detection.

The trigger is user-written: a <IMAGE_REQUEST_TAG> block inside a user message, which
this module parses, validates and removes before the conversation reaches the text
model. Assistant output is never scanned -- the model cannot ask for an image, so it
cannot spend money by talking about one.

The one rule that matters here: chat clients resend the whole history on every turn, so
a block written three turns ago arrives again with every request. Blocks are therefore
stripped from *every* user message (the text model must never see the control syntax)
but only *trigger* generation when they appear in the last one. Without that split,
every turn would regenerate every image the conversation has ever asked for.
"""

import json5
import re

from dataclasses import dataclass, field
from typing      import Any, Dict, List, Optional, Tuple

from common import cfg
from v1_images_generation import ImageRequest, ImageRequestError, build_request


@dataclass
class Extraction:
    messages  : List[Dict[str, Any]] = field(default_factory=list)
    requests  : List[ImageRequest]   = field(default_factory=list)
    errors    : List[str]            = field(default_factory=list)
    # False when the user's turn was nothing but image requests, which is the case that
    # skips the text model entirely.
    needs_text : bool                = True
    found      : bool                = False


def block_pattern() -> re.Pattern:
    """The control block regex for the configured tag. Built per call because the tag is reloadable."""
    tag = re.escape(cfg.image_request_tag)
    return re.compile(rf"<{tag}\s*>(?P<body>.*?)</{tag}\s*>", re.IGNORECASE | re.DOTALL)


def parse_block_body(body: str) -> Dict[str, Any]:
    """
    One control block's fields.

    The documented syntax is json5 without the surrounding braces, so they are added back
    before parsing. Two conveniences on top: a body that already carries its own braces is
    taken as-is, and a body with no field syntax at all is taken as a bare prompt -- the
    common case of a user typing the tag around a sentence. A body that looks like fields
    but does not parse is an error rather than a prompt, so a typo is reported instead of
    being generated verbatim.
    """
    text = (body or "").strip()
    if not text:
        raise ImageRequestError("the image request block is empty; a prompt is required.")

    if not (text.startswith("{") and text.endswith("}")):
        if ":" not in text:
            return {"prompt": text}
        text = "{" + text + "}"

    try:
        parsed = json5.loads(text)
    except Exception as exc:
        raise ImageRequestError(f"could not parse the image request block ({exc}). Expected json5 fields like: prompt: \"...\", quality: \"high\".")

    if not isinstance(parsed, dict):
        raise ImageRequestError("the image request block must be a set of key: value fields.")

    return parsed


def strip_blocks(content: Any, pattern: re.Pattern) -> Tuple[Any, List[str]]:
    """
    Removes every control block from one message's content, returning the cleaned content
    and the raw block bodies found. List-form content is walked so that a client sending
    OpenAI content parts is handled without being flattened to a string.
    """
    bodies: List[str] = []

    def clean(text: str) -> str:
        matches = list(pattern.finditer(text))
        if not matches:
            return text
        bodies.extend(match.group("body") for match in matches)
        # Collapse the whitespace the removed block leaves behind, so a message that was
        # only a block becomes genuinely empty rather than a run of newlines.
        return re.sub(r"\n{3,}", "\n\n", pattern.sub("", text)).strip()

    if isinstance(content, str):
        return clean(content), bodies

    if isinstance(content, list):
        cleaned_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                cleaned_parts.append({**item, "text": clean(str(item.get("text", "")))})
            else:
                cleaned_parts.append(item)
        return cleaned_parts, bodies

    return content, bodies


def content_is_empty(content: Any) -> bool:
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for item in content:
            # Any non-text part (an image, an audio clip) is content in its own right.
            if not isinstance(item, dict) or item.get("type") != "text":
                return False
            if str(item.get("text", "")).strip():
                return False
        return True
    return not str(content or "").strip()


def last_user_index(messages: List[Any]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            return index
    return -1


def extract(payload: Dict[str, Any]) -> Extraction:
    """
    Splits an incoming chat payload into the conversation the text model should see and
    the image requests the proxy should run.

    Returns an Extraction whose `messages` replaces payload["messages"]. When nothing was
    found the original list is returned untouched, so the ordinary chat path is unchanged.
    """
    messages = payload.get("messages")
    if not cfg.image_enabled or not cfg.image_chat_enabled or not isinstance(messages, list):
        return Extraction(messages=messages if isinstance(messages, list) else [], needs_text=True)

    pattern   = block_pattern()
    trigger   = last_user_index(messages)
    result    = Extraction(messages=[], needs_text=True)
    triggered : List[str] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            result.messages.append(message)
            continue

        cleaned, bodies = strip_blocks(message.get("content", ""), pattern)
        if not bodies:
            result.messages.append(message)
            continue

        result.found = True
        result.messages.append({**message, "content": cleaned})

        # Older turns are cleaned but never re-run; see the module docstring.
        if index == trigger:
            triggered = bodies
            result.needs_text = not content_is_empty(cleaned)

    if not result.found:
        return Extraction(messages=messages, needs_text=True)

    for body in triggered:
        try:
            result.requests.append(build_request(parse_block_body(body), source="chat"))
        except ImageRequestError as exc:
            result.errors.append(str(exc))
        except Exception as exc:
            result.errors.append(f"{exc.__class__.__name__}: {exc}")

    # A block that only produced errors still consumed the user's turn. Sending the now
    # empty message to the text model would ask it to reply to nothing, so the turn ends
    # with the error text instead.
    if not result.requests and not result.errors:
        result.needs_text = True

    return result
