"""
Chat-side image request detection.

User-written <IMAGE_REQUEST_TAG> blocks are stripped from every user message, but only
the last user message can trigger generation. Chat clients resend history; old image
blocks must not spend again.
"""

import json5
import re

from typing      import Any, Dict, List, Optional, Tuple

from common import cfg
from v1_images import ImageRequest, ImageRequestError, build_request


class Extraction:
    """What one incoming payload turned into."""
    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None, needs_text: bool = True):
        self.messages = messages or []
        # False when the user's turn was only image requests.
        self.needs_text = needs_text
        self.found      = False

        self.requests : List[ImageRequest] = []
        self.errors   : List[str]          = []


def block_pattern() -> re.Pattern:
    """The control block regex for the configured tag. Built per call because the tag is reloadable."""
    tag = re.escape(cfg.image_request_tag)
    return re.compile(rf"<{tag}\s*>(?P<body>.*?)</{tag}\s*>", re.IGNORECASE | re.DOTALL)


def parse_block_body(body: str) -> Dict[str, Any]:
    """
    Parse one control block. Bodies may be json5 fields, a braced object, or a bare prompt.
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
    Remove control blocks from one message and return the cleaned content plus raw bodies.
    """
    bodies: List[str] = []

    def clean(text: str) -> str:
        matches = list(pattern.finditer(text))
        if not matches:
            return text
        bodies.extend(match.group("body") for match in matches)
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
    Split a chat payload into cleaned messages and image requests to run.
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

    # If no current block existed, leave the normal chat path alone.
    if not result.requests and not result.errors:
        result.needs_text = True

    return result
