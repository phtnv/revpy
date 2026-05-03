import json
import os
import re
import time
import traceback
import threading
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request, stream_with_context
from flask_cors import CORS
from waitress import serve

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
def getenv_float(name: str, default: float) -> float:
    """
    Reads a float from the environment while keeping startup resilient if a
    value is missing or malformed.
    """
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name} must be a number. Defaulting to {default}.")
        return default

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "5001"))

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")

# Route-specific model aliases. If unset, they fall back to DEFAULT_MODEL.
HAIKU_MODEL    = os.getenv("HAIKU_MODEL", DEFAULT_MODEL)
SONNET_MODEL   = os.getenv("SONNET_MODEL", DEFAULT_MODEL)
SONNET35_MODEL = os.getenv("SONNET35_MODEL", DEFAULT_MODEL)
OPUS_MODEL     = os.getenv("OPUS_MODEL", DEFAULT_MODEL)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
PROXY_KEY         = os.getenv("PROXY_KEY", "").strip()

# Safer default for a public Cloudflare Tunnel:
# JanitorAI sends PROXY_KEY, while your real Anthropic key stays local in .env.
REQUIRE_PROXY_KEY = os.getenv("REQUIRE_PROXY_KEY", "true").lower() == "true"

# Compatibility fallback: if true and ANTHROPIC_API_KEY is not set,
# the proxy will use the Bearer token from the incoming request as the Anthropic key.
# For a public tunnel, I recommend leaving this false.
ALLOW_KEY_PASSTHROUGH = os.getenv("ALLOW_KEY_PASSTHROUGH", "false").lower() == "true"

DEBUG_LOG = os.getenv("DEBUG_LOG", "true").lower() == "true"
AUTO_TRIM = os.getenv("AUTO_TRIM", "true").lower() == "true"

# Leave empty by default. The original notebook used a strong assistant prefill.
# For safety and reliability, keep this blank unless you have a benign reason to use it.
ASSISTANT_PREFILL = os.getenv("ASSISTANT_PREFILL", "")

# assistant: old behavior, sends ASSISTANT_PREFILL as an assistant message/prefill.
# instruction: appends an OOC instruction containing ASSISTANT_PREFILL to the last user message.
# PREFILL_MODE is accepted as a shorter backwards-compatible alias.
ASSISTANT_PREFILL_MODE = os.getenv("ASSISTANT_PREFILL_MODE", os.getenv("PREFILL_MODE", "assistant")).strip().lower()

VALID_ASSISTANT_PREFILL_MODES = {"assistant", "instruction"}
if ASSISTANT_PREFILL_MODE not in VALID_ASSISTANT_PREFILL_MODES:
    print(
        "WARNING: ASSISTANT_PREFILL_MODE must be 'assistant' or 'instruction'. "
        "Defaulting to 'assistant'."
    )
    ASSISTANT_PREFILL_MODE = "assistant"

# Generation defaults
TEMPERATURE_OVERRIDE = float(os.getenv("TEMPERATURE_OVERRIDE", "-1"))
DEFAULT_TEMPERATURE  = float(os.getenv("DEFAULT_TEMPERATURE", "0.9"))
SEND_TOP_P           = os.getenv("SEND_TOP_P", "false").lower() == "true"
TOP_P                = float(os.getenv("TOP_P", "0.9"))
TOP_K                = int(os.getenv("TOP_K", "75"))
DEFAULT_MAX_TOKENS   = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))

# Cost tracking.
# Values are USD per 1 million tokens. Defaults are Anthropic's Claude Sonnet
# 4.5 API prices: input $3/MTok, output $15/MTok, 5m cache write $3.75/MTok,
# 1h cache write $6/MTok, and cache read $0.30/MTok.
INPUT_TOKEN_COST_USD      = getenv_float("INPUT_TOKEN_COST_USD"   ,  3.00)
OUTPUT_TOKEN_COST_USD     = getenv_float("OUTPUT_TOKEN_COST_USD"  , 15.00)
CACHE_WRITE_5M_COST_USD   = getenv_float("CACHE_WRITE_5M_COST_USD",  3.75)
CACHE_WRITE_1H_COST_USD   = getenv_float("CACHE_WRITE_1H_COST_USD",  6.00)
CACHE_READ_COST_USD       = getenv_float("CACHE_READ_COST_USD"    ,  0.30)

# Prompt caching.
# Anthropic supports automatic top-level caching and explicit block-level caching.
# This script uses explicit block-level caching by default because assistant prefill
# can otherwise become the final cacheable block.
PROMPT_CACHE         = os.getenv("PROMPT_CACHE", "true").lower() == "true"
CACHE_TTL            = os.getenv("CACHE_TTL", "5m").strip() # "5m" or "1h"
CACHE_SYSTEM_PROMPT  = os.getenv("CACHE_SYSTEM_PROMPT", "true").lower() == "true"
CACHE_FIRST_MESSAGES = max(0, int(os.getenv("CACHE_FIRST_MESSAGES", "0")))

ERROR_LOG_PATH = os.getenv("ERROR_LOG_PATH", "claude_error_log.txt")


# Session cost totals since this process started.
# SESSION_CACHE_NET_COST_USD is positive when caching has cost extra money and
# negative when caching has saved money.
SESSION_TOTAL_SPENT_USD    = 0.0
SESSION_CACHE_NET_COST_USD = 0.0
SESSION_COST_LOCK          = threading.Lock()

# =============================================================================
# Flask app
# =============================================================================
app = Flask(__name__)
CORS(app)

# =============================================================================
# Utility helpers
# =============================================================================

def log_debug(*args: Any) -> None:
    if DEBUG_LOG:
        print(*args)


def write_error_log(body: Any) -> None:
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(body) + "\n\n")
    except Exception:
        print("Failed to write error log:")
        traceback.print_exc()


def get_bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Recommended public-tunnel mode:
      .env contains ANTHROPIC_API_KEY and PROXY_KEY.
      JanitorAI uses PROXY_KEY as the reverse proxy key.

    Optional compatibility mode:
      ALLOW_KEY_PASSTHROUGH=true lets incoming Bearer token act as Anthropic key.
    """
    provided_key = get_bearer_token()

    if ANTHROPIC_API_KEY:
        if REQUIRE_PROXY_KEY:
            if not PROXY_KEY:
                abort(
                    500,
                    description=(
                        "Server is configured with REQUIRE_PROXY_KEY=true, "
                        "but PROXY_KEY is missing from .env."
                    ),
                )

            if provided_key != PROXY_KEY:
                abort(401, description="Invalid proxy key.")

        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if ALLOW_KEY_PASSTHROUGH:
        if not provided_key:
            abort(401, description="Missing Authorization bearer token.")
        return anthropic.Anthropic(api_key=provided_key)

    abort(
        500,
        description=(
            "ANTHROPIC_API_KEY is not configured. Either set ANTHROPIC_API_KEY "
            "and PROXY_KEY in .env, or set ALLOW_KEY_PASSTHROUGH=true."
        ),
    )


def make_cache_control() -> Dict[str, str]:
    """
    5-minute cache is default. 1-hour cache is more expensive but useful
    if there are longer pauses between messages.
    """
    cache_control = {"type": "ephemeral"}
    if CACHE_TTL == "1h":
        cache_control["ttl"] = "1h"
    return cache_control


def add_cache_control_to_content(content: Any) -> Any:
    """
    Adds explicit Anthropic cache_control to the last non-empty text block.

    Anthropic prompt caching is enabled by adding cache_control either at the
    request level or on content blocks. This script uses explicit block-level
    caching to avoid caching the assistant prefill as the final block.
    """
    if not PROMPT_CACHE:
        return content

    cache_control = make_cache_control()

    if isinstance(content, str):
        if not content.strip():
            return content
        return [
            {
                "type": "text",
                "text": content,
                "cache_control": cache_control,
            }
        ]

    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(dict(block))
            else:
                blocks.append({"type": "text", "text": str(block)})

        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i].get("type") == "text" and blocks[i].get("text", "").strip():
                blocks[i]["cache_control"] = cache_control
                return blocks

        return blocks

    text = str(content)
    if not text.strip():
        return content

    return [
        {
            "type": "text",
            "text": text,
            "cache_control": cache_control,
        }
    ]


def append_text_to_content(content: Any, text: str) -> Any:
    """
    Appends text while preserving Anthropic list-form content blocks.
    """
    if text is None:
        text = ""

    if isinstance(content, str):
        return content + "\n" + text

    if isinstance(content, list):
        return content + [{"type": "text", "text": "\n" + text}]

    return str(content) + "\n" + text


def make_prefill_instruction(prefix_text: str) -> str:
    """
    Creates the instruction-mode version of ASSISTANT_PREFILL.

    This avoids Anthropic assistant prefill by telling Claude, inside the
    last user message, to continue as though the prefix was already present.
    """
    return (
        "\n<OOC>\n"
        "The assistant will begin its reply with the following prefix:\n"
        f"<prefix>{prefix_text}</prefix>\n"
        "Continue immediately after that prefix. Do not display the prefix in your answer.\n"
        "</OOC>"
    )


def append_prefill_instruction_to_last_user_message(formatted: List[Dict[str, Any]], prefix_text: str) -> None:
    """
    Appends instruction-mode prefill text to the last user message in-place.
    """
    instruction = make_prefill_instruction(prefix_text)

    for i in range(len(formatted) - 1, -1, -1):
        if formatted[i].get("role") == "user":
            formatted[i]["content"] = append_text_to_content(formatted[i].get("content", ""), instruction)
            return

    # Defensive fallback. The current formatter always creates an initial user
    # message, but keep this here in case that changes later.
    formatted.append({"role": "user", "content": instruction})


def content_to_plain_text(content: Any) -> str:
    """
    The proxy primarily expects text-only OpenAI-style messages.

    If a client sends a list of text parts, this joins text parts.
    Non-text parts are serialized. This is intentionally conservative;
    it does not implement OpenAI-image-to-Anthropic-image conversion.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


PERSONA_END_RE = re.compile(r"</[^<>]*\bPersona>", re.IGNORECASE)

def split_system_prompt_into_text_blocks(system_prompt: str) -> List[Dict[str, str]]:
    """
    Always returns Anthropic system content as list-form text blocks.

    Split priority:
      1. After </Scenario>, if present.
      2. Otherwise after the last </* Persona> marker, if present.
      3. Otherwise keep the whole system prompt as one block.

    The split marker stays in the first block.
    Anything after it becomes the second block, usually lorebook / extra context.
    """
    if not system_prompt or not system_prompt.strip():
        return []

    text = system_prompt.strip()

    # Priority 1: split after </Scenario>
    scenario_marker = "</Scenario>"
    scenario_idx = text.find(scenario_marker)
    if scenario_idx != -1:
        split_at = scenario_idx + len(scenario_marker)
    else:
        # Priority 2: split after the last </* Persona> closing tag.
        persona_matches = list(PERSONA_END_RE.finditer(text))
        if persona_matches:
            split_at = persona_matches[-1].end()
        else:
            split_at = -1

    if split_at == -1 or split_at >= len(text):
        return [{"type": "text", "text": text}]

    before = text[:split_at].rstrip()
    after = text[split_at:].strip()

    blocks = [{"type": "text", "text": before}]

    if after:
        # Keep a clean visual/semantic separator between Scenario/Persona and suffix.
        blocks.append({"type": "text", "text": "\n\n" + after})

    return blocks


def format_system_for_claude(system_prompt: Optional[str]) -> Optional[Any]:
    """
    Optionally cache the system prompt separately.

    This helps when JanitorAI sends large character cards, scenario text,
    behavior rules, or examples as system content.
    """
    if system_prompt is None:
        return None

    blocks = split_system_prompt_into_text_blocks(system_prompt)
    if not blocks:
        return None

    system_prompt: List[Dict[str, Any]] = []

    for block in blocks:
        new_block: Dict[str, Any] = dict(block)
        if (PROMPT_CACHE and CACHE_SYSTEM_PROMPT and new_block.get("type") == "text" and new_block.get("text", "").strip()):
            new_block["cache_control"] = make_cache_control()
        system_prompt.append(new_block)

    return system_prompt


def format_to_claude_messages(mlist: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Converts OpenAI-style chat messages to Anthropic Messages format.

    Consecutive same-role messages are merged because Anthropic expects
    alternating user/assistant turns.

    The cache breakpoint is placed on the last real conversation block before
    ASSISTANT_PREFILL is applied. In assistant mode, the prefill is sent as an
    assistant message. In instruction mode, the prefill instruction is appended
    to the last user message instead.
    """

    cache_target_index = 0
    if PROMPT_CACHE and CACHE_FIRST_MESSAGES > 0 and mlist:
        # If fewer than N messages exist, cache through the last available message.
        cache_target_index = min(CACHE_FIRST_MESSAGES, len(mlist))

    formatted = [{"role": "user", "content": "<OOC>\nBegin the scenario.\n</OOC>"}]
    old_role  = "user"

    for idx, msg in enumerate(mlist, start=1):
        incoming_role = msg.get("role", "user")
        content       = msg.get("content", "")

        claude_role = "assistant" if incoming_role == "assistant" else "user"

        if claude_role == old_role:
            formatted[-1]["content"] = append_text_to_content(formatted[-1]["content"], content)
        else:
            formatted.append({"role" : claude_role, "content" : content})

        old_role = claude_role

        # Mark only the configured first-N-message prefix.
        # Later messages are still sent to Claude, but are not part of the
        # explicit conversation cache breakpoint.
        if idx == cache_target_index:
            formatted[-1]["content"] = add_cache_control_to_content(formatted[-1]["content"])

    # Optional Claude prefill.
    # assistant mode preserves the original assistant-message/prefill behavior.
    # instruction mode avoids assistant prefill and appends an OOC instruction
    # to the last user message instead.
    if ASSISTANT_PREFILL.strip():
        if ASSISTANT_PREFILL_MODE == "instruction":
            append_prefill_instruction_to_last_user_message(formatted, ASSISTANT_PREFILL)
        else:
            if formatted[-1]["role"] == "user":
                formatted.append({"role" : "assistant", "content" : ASSISTANT_PREFILL})
            else:
                formatted[-1]["content"] = append_text_to_content(formatted[-1]["content"], ASSISTANT_PREFILL)

    return formatted


def trim_to_end_sentence(input_str: str, include_newline: bool = False) -> str:
    punctuation = set([".", "!", "?", "*", '"', ")", "}", "`", "]", "$", "。", "！", "？", "”", "）", "】", "’", "」"])

    last = -1
    for i in range(len(input_str) - 1, -1, -1):
        char = input_str[i]

        if char in punctuation:
            if i > 0 and input_str[i - 1] in [" ", "\n"]:
                last = i - 1
            else:
                last = i
            break

        if include_newline and char == "\n":
            last = i
            break

    if last == -1:
        return input_str.rstrip()

    return input_str[: last + 1].rstrip()


def auto_trim_text(text: str) -> str:
    return trim_to_end_sentence(text)


def extract_text_from_anthropic_message(message: Any) -> str:
    """
    Collects text blocks from an Anthropic response.
    """
    chunks = []

    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            chunks.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text", ""))

    return "".join(chunks)

def token_cost_usd(tokens: int, usd_per_million_tokens: float) -> float:
    return (tokens * usd_per_million_tokens) / 1_000_000.0
def format_usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.6f}"
def cache_net_label(net_cost_usd: float) -> str:
    if net_cost_usd < 0:
        return f"{format_usd(abs(net_cost_usd))} saved"
    if net_cost_usd > 0:
        return f"{format_usd(net_cost_usd)} lost"
    return "$0.000000 break-even"
def print_openai_usage(usage: Any) -> None:
    global SESSION_TOTAL_SPENT_USD, SESSION_CACHE_NET_COST_USD

    cache_creation       = getattr(usage, "cache_creation", {}) or {}
    ephemeral_1h         = int(getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
    ephemeral_5m         = int(getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
    input_tokens         = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens        = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read           = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_input = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    ttl_tokens = input_tokens + cache_read + cache_creation_input;

    # Older SDK responses may expose only cache_creation_input_tokens without
    # the 5m/1h split. Attribute that unknown split to the configured TTL so
    # the cost row still matches the printed cache-write token total.
    known_cache_write = ephemeral_1h + ephemeral_5m
    if cache_creation_input > known_cache_write:
        unknown_cache_write = cache_creation_input - known_cache_write
        if CACHE_TTL == "1h":
            ephemeral_1h += unknown_cache_write
        else:
            ephemeral_5m += unknown_cache_write

    cache_creation_input = ephemeral_1h + ephemeral_5m
    ttl_tokens           = input_tokens + cache_read + cache_creation_input

    input_cost          = token_cost_usd(input_tokens, INPUT_TOKEN_COST_USD)
    cache_read_cost     = token_cost_usd(cache_read, CACHE_READ_COST_USD)
    cache_write_1h_cost = token_cost_usd(ephemeral_1h, CACHE_WRITE_1H_COST_USD)
    cache_write_5m_cost = token_cost_usd(ephemeral_5m, CACHE_WRITE_5M_COST_USD)
    cache_write_cost    = cache_write_1h_cost + cache_write_5m_cost
    total_input_cost    = input_cost + cache_read_cost + cache_write_cost

    output_cost        = token_cost_usd(output_tokens, OUTPUT_TOKEN_COST_USD)
    request_total_cost = total_input_cost + output_cost

    cache_write_extra_cost = (
        token_cost_usd(ephemeral_1h, CACHE_WRITE_1H_COST_USD - INPUT_TOKEN_COST_USD)
        +
        token_cost_usd(ephemeral_5m, CACHE_WRITE_5M_COST_USD - INPUT_TOKEN_COST_USD)
    )
    cache_read_saved_cost  = token_cost_usd(cache_read, INPUT_TOKEN_COST_USD - CACHE_READ_COST_USD)
    request_cache_net_cost = cache_write_extra_cost - cache_read_saved_cost

    with SESSION_COST_LOCK:
        SESSION_TOTAL_SPENT_USD    += request_total_cost
        SESSION_CACHE_NET_COST_USD += request_cache_net_cost
        session_total_spent    = SESSION_TOTAL_SPENT_USD
        session_cache_net_cost = SESSION_CACHE_NET_COST_USD

    log_debug("=== Claude usage ===")
    print("Total input tokens =   uncached + cache read + cache write (        1h +         5m)")
    print("{:18d} = {:10d} + {:10d} + {:11d} ({:10d} + {:10d})".format(ttl_tokens, input_tokens, cache_read, cache_creation_input, ephemeral_1h, ephemeral_5m))
    print("{:>18s} = {:>10s} + {:>10s} + {:>11s} ({:>10s} + {:>10s})".format(format_usd(total_input_cost), format_usd(input_cost), format_usd(cache_read_cost), format_usd(cache_write_cost), format_usd(cache_write_1h_cost), format_usd(cache_write_5m_cost)))
    print("Output tokens = {:d} ({})".format(output_tokens, format_usd(output_cost)))
    print("Request cache cost = {} ({})".format(format_usd(request_cache_net_cost), cache_net_label(request_cache_net_cost)))
    print("Session cache cost = {} ({})".format(format_usd(session_cache_net_cost), cache_net_label(session_cache_net_cost)))
    print("Request total cost = {}".format(format_usd(request_total_cost)))
    print("Session total cost = {}".format(format_usd(session_total_spent)))
    log_debug("=== End Claude usage ===")


def usage_to_openai_dict(usage: Any) -> Dict[str, int]:
    """
    Anthropic separates cache usage into:
      input_tokens
      cache_creation_input_tokens
      cache_read_input_tokens
      output_tokens

    OpenAI-compatible clients usually expect:
      prompt_tokens
      completion_tokens
      total_tokens

    This preserves both.
    """
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    prompt_tokens = input_tokens + cache_read + cache_creation

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "input_tokens_uncached": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


def get_temperature(payload: Dict[str, Any]) -> float:
    if TEMPERATURE_OVERRIDE != -1:
        return TEMPERATURE_OVERRIDE
    return float(payload.get("temperature", DEFAULT_TEMPERATURE))


def split_system_and_messages(mlist: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Claude wants system prompt separately from messages.

    Empty system messages are discarded.
    Non-system messages are preserved as user/assistant text messages.
    """
    system_parts  = []
    chat_messages = []

    for msg in mlist:
        role = msg.get("role", "user")
        content = content_to_plain_text(msg.get("content", ""))

        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
        else:
            if role not in ("user", "assistant"):
                role = "user"

            chat_messages.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, chat_messages


def build_claude_kwargs(payload: Dict[str, Any], route_model: str) -> Dict[str, Any]:
    messages = payload.get("messages")

    if not isinstance(messages, list):
        abort(400, description="Request body must include a messages list.")

    system_prompt, chat_messages = split_system_and_messages(messages)
    if chat_messages and chat_messages[0].get("role") == "user" and chat_messages[0].get("content", "").strip() == ".":
        chat_messages = chat_messages[1:]
    formatted_messages = format_to_claude_messages(chat_messages)

    # Route model wins by default.
    # If you want JanitorAI/client to choose model from JSON, set ALLOW_CLIENT_MODEL=true.
    allow_client_model = os.getenv("ALLOW_CLIENT_MODEL", "false").lower() == "true"
    selected_model = payload.get("model") if allow_client_model else route_model
    selected_model = selected_model or route_model

    kwargs: Dict[str, Any] = {
        "model"       : selected_model,
        "max_tokens"  : int(payload.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "temperature" : get_temperature(payload),
        "top_k"       : TOP_K,
    }
    if SEND_TOP_P:
        kwargs["top_p"] = TOP_P

    formatted_system = format_system_for_claude(system_prompt)
    if formatted_system is not None:
        kwargs["system"] = formatted_system
    kwargs["messages"] = formatted_messages

    return kwargs


def make_openai_nonstream_response(message: Any, model: str) -> Dict[str, Any]:
    output_text = extract_text_from_anthropic_message(message)

    if AUTO_TRIM:
        output_text = auto_trim_text(output_text)

    usage = getattr(message, "usage", None)

    return {
        "id": getattr(message, "id", "claude"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"anthropic/{model}",
        "choices": [
            {
                "index": 0,
                "finish_reason": getattr(message, "stop_reason", "stop"),
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
            }
        ],
        "usage": usage,
    }


def make_error_response(exc: Exception, payload: Optional[Dict[str, Any]] = None) -> Response:
    status_code = 500
    message = str(exc)
    error_type = exc.__class__.__name__

    # Flask abort errors
    if hasattr(exc, "code"):
        status_code = getattr(exc, "code", 500)
        message = getattr(exc, "description", message)

    # Anthropic SDK errors often expose status_code/body.
    if hasattr(exc, "status_code"):
        status_code = getattr(exc, "status_code", status_code)

    if hasattr(exc, "body"):
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            message = error_obj.get("message", message)
            error_type = error_obj.get("type", error_type)

    error_body = {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
        }
    }

    log_body = {
        "error": error_body,
        "request": payload,
        "traceback": traceback.format_exc(),
    }
    write_error_log(log_body)

    return Response(
        json.dumps(error_body, ensure_ascii=False),
        status=status_code,
        content_type="application/json",
    )


# =============================================================================
# Generation
# =============================================================================

def generate_nonstream(payload: Dict[str, Any], route_model: str) -> Dict[str, Any]:
    client = get_anthropic_client()
    kwargs = build_claude_kwargs(payload, route_model)

    log_debug("=== Payload sent to Claude ===")
    log_debug(json.dumps(kwargs, indent=2, ensure_ascii=False))
    log_debug("=== End Claude payload ===")

    message = client.messages.create(**kwargs)

    usage = getattr(message, "usage", None)
    print_openai_usage(usage)

    model_used = kwargs.get("model", route_model)
    return make_openai_nonstream_response(message, model_used)


def generate_stream(payload: Dict[str, Any], route_model: str):
    client = get_anthropic_client()
    kwargs = build_claude_kwargs(payload, route_model)

    log_debug("=== Payload sent to Claude ===")
    log_debug(json.dumps(kwargs, indent=2, ensure_ascii=False))
    log_debug("=== End Claude payload ===")

    model_used = kwargs.get("model", route_model)

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            event = {
                "id": "claude",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": f"anthropic/{model_used}",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "role": "assistant",
                            "content": text,
                        },
                    }
                ],
            }

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            time.sleep(0.01)

        final_message = stream.get_final_message()

        usage = getattr(final_message, "usage", None)
        print_openai_usage(usage)

        final_event = {
            "id": getattr(final_message, "id", "claude"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": f"anthropic/{model_used}",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": getattr(final_message, "stop_reason", "stop"),
                    "delta": {},
                }
            ],
            "usage": usage_to_openai_dict(getattr(final_message, "usage", None)),
        }

        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def handle_chat_completion(route_model: str):
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return Response(
            json.dumps({"error": {"message": "Invalid JSON body."}}),
            status=400,
            content_type="application/json",
        )

    try:
        stream = bool(payload.get("stream", False))

        if stream:
            return Response(
                stream_with_context(generate_stream(payload, route_model)),
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = generate_nonstream(payload, route_model)
        return jsonify(response)

    except Exception as exc:
        return make_error_response(exc, payload)


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET"])
def running():
    base_url = request.base_url.rstrip("/")

    with SESSION_COST_LOCK:
        session_total_spent    = SESSION_TOTAL_SPENT_USD
        session_cache_net_cost = SESSION_CACHE_NET_COST_USD

    return jsonify(
        {
            "status"        : "ok",
            "default_model" : DEFAULT_MODEL,
            "prompt_cache"  : PROMPT_CACHE,
            "cache_ttl"     : CACHE_TTL,
            "cost_tracking" : {
                "input_token_cost_usd"       : INPUT_TOKEN_COST_USD,
                "output_token_cost_usd"      : OUTPUT_TOKEN_COST_USD,
                "cache_write_5m_cost_usd"    : CACHE_WRITE_5M_COST_USD,
                "cache_write_1h_cost_usd"    : CACHE_WRITE_1H_COST_USD,
                "cache_read_cost_usd"        : CACHE_READ_COST_USD,
                "session_total_spent_usd"    : session_total_spent,
                "session_cache_net_cost_usd" : session_cache_net_cost,
                "session_cache_net_meaning"  : "positive = lost money, negative = saved money",
            },
            "routes": {
                "chat_completions" : base_url + "/chat/completions",
                "short_post"       : base_url + "/",
                "haiku"            : base_url + "/haiku",
                "sonnet"           : base_url + "/sonnet",
                "sonnet35"         : base_url + "/sonnet35",
                "opus"             : base_url + "/opus",
            },
        }
    )


@app.route("/", methods=["POST"])
def short_baseurl():
    return handle_chat_completion(DEFAULT_MODEL)


@app.route("/chat/completions", methods=["POST"])
def baseurl():
    return handle_chat_completion(DEFAULT_MODEL)


@app.route("/v1/chat/completions", methods=["POST"])
def v1_baseurl():
    return handle_chat_completion(DEFAULT_MODEL)


@app.route("/haiku", methods=["POST"])
@app.route("/haiku/chat/completions", methods=["POST"])
def haiku():
    return handle_chat_completion(HAIKU_MODEL)


@app.route("/haiku35", methods=["POST"])
@app.route("/haiku35/chat/completions", methods=["POST"])
def haiku35():
    return handle_chat_completion(HAIKU_MODEL)


@app.route("/sonnet", methods=["POST"])
@app.route("/sonnet/chat/completions", methods=["POST"])
def sonnet():
    return handle_chat_completion(SONNET_MODEL)


@app.route("/sonnet35", methods=["POST"])
@app.route("/sonnet35/chat/completions", methods=["POST"])
def sonnet35():
    return handle_chat_completion(SONNET35_MODEL)


@app.route("/opus", methods=["POST"])
@app.route("/opus/chat/completions", methods=["POST"])
def opus():
    return handle_chat_completion(OPUS_MODEL)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting Claude reverse proxy")
    print(f"Local URL: http://{HOST}:{PORT}")
    print(f"Chat completions: http://{HOST}:{PORT}/chat/completions")
    print("Cloudflare Tunnel service URL should point to this local address:")
    print(f"  http://{HOST}:{PORT}")
    print()

    if REQUIRE_PROXY_KEY and not PROXY_KEY:
        print("WARNING: REQUIRE_PROXY_KEY=true but PROXY_KEY is missing.")
        print("Set PROXY_KEY in .env before exposing this through Cloudflare Tunnel.")
        print()

    if not ANTHROPIC_API_KEY and not ALLOW_KEY_PASSTHROUGH:
        print("WARNING: ANTHROPIC_API_KEY is missing and ALLOW_KEY_PASSTHROUGH=false.")
        print("Requests will fail until you configure one of these modes.")
        print()

    serve(app, host=HOST, port=PORT)
