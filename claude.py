import anthropic
import base64
import json
import re
import threading

from flask             import abort, request
from packaging.version import Version
from typing            import Any, Dict, List, Optional, Tuple

from common import (
    add_session_cost,
    append_text_to_content,
    cfg,
)


ANTHROPIC_TIMEOUT_ERROR                      = getattr(anthropic, "APITimeoutError", TimeoutError)
ANTHROPIC_MODELS      : List[Dict[str, Any]] = []
MODEL_LIST_LAST_ERROR : Optional[str]        = None
MODEL_LIST_LAST_TIMEOUT                      = False
MODEL_LOCK                                   = threading.Lock()


def anthropic_object_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Converts Anthropic SDK model objects into JSON-printable dictionaries.
    """
    if isinstance(obj, dict)      : return dict(obj)
    if hasattr(obj, "model_dump") : return obj.model_dump(mode="json")
    if hasattr(obj, "dict")       : return obj.dict()

    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }

    return {"value": str(obj)}


def anthropic_error_body(exc: Exception) -> Optional[Dict[str, Any]]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_body = response.json()
            if isinstance(response_body, dict):
                return response_body
        except Exception:
            pass

    return None


def anthropic_error_message(body: Optional[Dict[str, Any]], fallback: str) -> str:
    if isinstance(body, dict):
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if message:
                return str(message)

        return json.dumps(body, ensure_ascii=False, default=str)

    return fallback


def print_anthropic_error(exc: Exception) -> bool:
    ANSI_RED   : str = "\033[31m"
    ANSI_RESET : str = "\033[0m"
    body   = anthropic_error_body(exc)
    module = exc.__class__.__module__.split(".", 1)[0]
    if module != "anthropic" and body is None:
        return False

    fallback = str(exc) or exc.__class__.__name__

    if body is None:
        body = {
            "type"  : "error",
            "error" : {
                "type"    : exc.__class__.__name__,
                "message" : fallback,
            },
        }

    message = anthropic_error_message(body, fallback)
    print(json.dumps(body, indent=2, ensure_ascii=False, default=str))
    print(f"{ANSI_RED}{message}{ANSI_RESET}")
    return True


def model_id_from_info(model_info: Dict[str, Any]) -> str:
    return str(model_info.get("id") or model_info.get("model") or "").strip()


def refresh_anthropic_models(key: str, timeout_s: float) -> bool:
    """
    Fetches the available Anthropic models and stores them for CLI use.
    """
    global ANTHROPIC_MODELS, MODEL_LIST_LAST_ERROR, MODEL_LIST_LAST_TIMEOUT

    try:
        if not key: raise RuntimeError("ANTHROPIC_API_KEY is not configured; model list cannot be retrieved at startup.")

        client = anthropic.Anthropic(api_key=key)
        page   = client.models.list(limit=100, timeout=timeout_s)

        raw_models = getattr(page, "data", None)
        if raw_models is None:
            raw_models = list(page)

        models = [anthropic_object_to_dict(model) for model in raw_models]

        with MODEL_LOCK:
            ANTHROPIC_MODELS        = models
            MODEL_LIST_LAST_ERROR   = None
            MODEL_LIST_LAST_TIMEOUT = False

        if models : print(f"Retrieved {len(models)} Anthropic model(s).")
        else      : print(f"Anthropic returned an empty model list.")

        return True

    except (ANTHROPIC_TIMEOUT_ERROR, TimeoutError) as exc:
        with MODEL_LOCK:
            ANTHROPIC_MODELS        = []
            MODEL_LIST_LAST_ERROR   = None
            MODEL_LIST_LAST_TIMEOUT = True
        print("WARNING: Could not retrieve a model list from Anthropic. Timeout.")
        return False

    except Exception as exc:
        anthropic_exception_msg : str = anthropic_error_message(anthropic_error_body(exc), str(exc) or exc.__class__.__name__)
        with MODEL_LOCK:
            ANTHROPIC_MODELS        = []
            MODEL_LIST_LAST_ERROR   = anthropic_exception_msg
            MODEL_LIST_LAST_TIMEOUT = False
        print(f"WARNING: Could not retrieve a model list from Anthropic. {anthropic_exception_msg}.")
        return False


def print_no_model_list_available() -> None:
    print("No Anthropic model list is available.")

    with MODEL_LOCK:
        last_timeout = MODEL_LIST_LAST_TIMEOUT
        last_error   = MODEL_LIST_LAST_ERROR

    if   last_timeout : print("Last retrieval failed: timeout")
    elif last_error   : print(f"Last retrieval failed: error: {last_error}")


def print_model_list() -> None:
    with MODEL_LOCK:
        models            = list(ANTHROPIC_MODELS)
        selected_model_id = cfg.model
    if not models:
        print_no_model_list_available()
        return

    number_width = len(str(len(models)))

    for index, model_info in enumerate(models, start=1):
        model_id     = model_id_from_info(model_info)
        display_name = str(model_info.get("display_name") or model_info.get("name") or "")
        created_at   = str(model_info.get("created_at") or "")

        number      = str(index).rjust(number_width)
        number_cell = f"[{number}]" if model_id == selected_model_id else f" {number} "

        extra_parts = []
        if display_name : extra_parts.append(display_name)
        if created_at   : extra_parts.append(created_at)

        extra  = "  ".join(extra_parts)
        suffix = f"  {extra}" if extra else ""

        print(f"{number_cell}  {model_id:<42}{suffix}")


def select_model_by_number(index: int) -> None:
    with MODEL_LOCK:
        if not ANTHROPIC_MODELS:
            print_no_model_list_available()
            return
        if index < 1 or index > len(ANTHROPIC_MODELS):
            print(f"Model number out of range [1:{len(ANTHROPIC_MODELS)}].")
            return
        model_info = ANTHROPIC_MODELS[index - 1]
        model_id   = model_id_from_info(model_info)
        if not model_id:
            print(f"Model {index} does not have an id and cannot be selected.")
            return
        cfg.apply_model(model_info)


def print_model_info(index: int) -> None:
    with MODEL_LOCK:
        if not ANTHROPIC_MODELS:
            print_no_model_list_available()
            return
        if index < 1 or index > len(ANTHROPIC_MODELS):
            print(f"Model number out of range. Use 1 through {len(ANTHROPIC_MODELS)}.")
            return
        model_info = dict(ANTHROPIC_MODELS[index - 1])

    print(json.dumps(model_info, indent=2, ensure_ascii=False, default=str))


def get_bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Recommended public-tunnel mode:
        .env contains ANTHROPIC_API_KEY and PROXY_KEY. JanitorAI uses PROXY_KEY as the proxy key.

    Optional compatibility mode:
        ALLOW_KEY_PASSTHROUGH=true lets incoming Bearer token act as Anthropic key.
    """
    provided_key = get_bearer_token()

    if cfg.anthropic_api_key:
        if cfg.require_proxy_key:
            if not cfg.proxy_key             : abort(500, description=("Server is configured with REQUIRE_PROXY_KEY=true, but PROXY_KEY is missing from .env."))
            if provided_key != cfg.proxy_key : abort(401, description="Invalid proxy key.")
        return anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    if cfg.allow_key_passthrough:
        if not provided_key : abort(401, description="Missing Authorization bearer token.")
        return anthropic.Anthropic(api_key=provided_key)
    abort(500, description=("ANTHROPIC_API_KEY is not configured. Either set ANTHROPIC_API_KEY and PROXY_KEY in .env, or set ALLOW_KEY_PASSTHROUGH=true."))


def make_cache_control(ttl: str) -> Dict[str, str]:
    """
    Builds Anthropic cache_control metadata for a specific marker TTL.

    5-minute cache is the API default. 1-hour cache is more expensive but useful for longer pauses.
    """
    cache_control = {"type": "ephemeral"}
    if ttl == "1h":
        cache_control["ttl"] = "1h"
    return cache_control


def add_cache_control_to_content(content: Any, ttl: str) -> Any:
    """
    Adds explicit Anthropic cache_control to the last non-empty text block.

    Anthropic prompt caching is enabled by adding cache_control either at the
    request level or on content blocks. This script uses explicit block-level
    caching to avoid caching the assistant prefill as the final block.
    """
    if not cfg.cache_en:
        return content

    cache_control = make_cache_control(ttl)

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
            if isinstance(block, dict) : blocks.append(dict(block))
            else                       : blocks.append({"type": "text", "text": str(block)})

        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i].get("type") == "text" and blocks[i].get("text", "").strip():
                blocks[i]["cache_control"] = cache_control
                return blocks

        return blocks

    text = str(content)
    if not text.strip():
        return content

    return [{"type": "text", "text": text, "cache_control": cache_control}]


def make_prefill_instruction(prefix_text: str) -> str:
    """
    Creates the instruction-mode version of ASSISTANT_PREFILL.

    This avoids Anthropic assistant prefill by telling Claude, inside the
    last user message, to continue as though the prefix was already present.
    """
    return (
        "\n<OOC>\n"
        f"{prefix_text}"
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

    # Defensive fallback. The current formatter always creates an initial user message, but keep this here in case that changes later.
    formatted.append({"role": "user", "content": instruction})


# Anthropic thinking block round-tripping
# Plain-text envelope lets Janitor carry signed Anthropic thinking blocks across turns.
THINKING_ENVELOPE_TAG   = "thinking_preservation_block_v1"
THINKING_ENVELOPE_START = f"~~~<{THINKING_ENVELOPE_TAG}>"
THINKING_ENVELOPE_END   = f"~~~</{THINKING_ENVELOPE_TAG}>"

# Accept the old marker for existing chats, but emit only the proxy-owned marker going forward.
THINKING_ENVELOPE_TAG_RE = r"(?:thinking_preservation_block_v1|anthropic_thinking_v1)"
THINKING_ENVELOPE_RE    = re.compile(
    rf"(?:^|\n)~~~<(?P<tag>{THINKING_ENVELOPE_TAG_RE})>\s*\n(?P<body>.*?)(?:\n)?~~~</(?P=tag)>\s*",
    re.DOTALL,
)
VISIBLE_THINK_RE = re.compile(r"\s*<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def thinking_preservation_enabled() -> bool:
    return cfg.preserve_thinking_blocks > 0


def extract_preservable_thinking_blocks(blocks: Any) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    preserved: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "thinking" and isinstance(block.get("thinking"), str) and isinstance(block.get("signature"), str):
            # Keep only Anthropic-signed thinking blocks; never reconstruct them from <think> text.
            preserved.append(dict(block))
        elif block_type == "redacted_thinking" and isinstance(block.get("data"), str):
            # Redacted thinking is opaque; pass it back exactly as received.
            preserved.append(dict(block))

    return preserved


def make_hidden_thinking_envelope(blocks: List[Dict[str, Any]]) -> str:
    if not blocks:
        return ""

    payload = {
        "version" : 1,
        "kind"    : "anthropic_thinking_blocks",
        "blocks"  : blocks,
    }

    # Keep the envelope ASCII-safe and line-wrapped for text-only clients.
    raw     = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    wrapped = [encoded[i:i + 120] for i in range(0, len(encoded), 120)]
    body    = "\n".join(f"~~~{line}" for line in wrapped)

    return f"\n{THINKING_ENVELOPE_START}\n{body}\n{THINKING_ENVELOPE_END}"


def extract_hidden_thinking_envelopes(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    all_blocks: List[Dict[str, Any]] = []

    def replace(match: re.Match) -> str:
        try:
            # Decode only the matched preservation envelope, not arbitrary ~~~ lines.
            encoded = "".join(
                line[3:].strip()
                for line in match.group("body").splitlines()
                if line.startswith("~~~")
            )
            if encoded:
                decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
                payload = json.loads(decoded.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("version") == 1:
                    all_blocks.extend(extract_preservable_thinking_blocks(payload.get("blocks", [])))
        except Exception as exc:
            if cfg.debug_log:
                print(f"WARNING: Failed to decode hidden Anthropic thinking envelope: {exc}")

        # Always strip matched envelopes so malformed metadata does not leak into Claude.
        return ""

    cleaned = THINKING_ENVELOPE_RE.sub(replace, text or "")
    return cleaned.rstrip(), all_blocks


PERSONA_END_RE  = re.compile(r"</[^<>]*\bPersona>", re.IGNORECASE)
LOREBOOK_XML_RE = re.compile(r"<lorebook\b[^>]*>.*?</lorebook>", re.IGNORECASE | re.DOTALL)
def format_system_for_claude(system_prompt: str, system_summary_text: str = "") -> Tuple[List[Dict[str, Any]], str]:
    """
    Formats the top-level Anthropic system prompt and optionally extracts the dynamic lorebook suffix.

    When SPLIT_LOREBOOK=true, the prompt is split into:
        1. stable core definition
        2. dynamic lorebook / user-script suffix

    Split priority:
        1. After </summary>
        2. After </example_dialogs>
        3. After </UserPersona>
        4. After </Scenario>
        5. After the last </* Persona> marker
        6. Otherwise keep the whole system prompt as one block

    When LOREBOOK_AT_END=true, the suffix is returned as plain text instead of being kept as a
    top-level system block. That moved suffix deliberately does not receive the old system/lorebook
    cache marker; it is handled later as an ordinary end-of-conversation item.

    When LOREBOOK_XML_AT_END=true, every <lorebook>...</lorebook> block is removed from the
    top-level system prompt and appended after any other moved end-of-chat lorebook text.

    Returns:
        - formatted_system: top-level Anthropic system blocks, or an empty list
        - lorebook_at_end_text: moved lorebook/suffix text, or an empty string
    """
    text         = system_prompt.strip()
    summary_text = system_summary_text.strip()
    if (not text) and (not summary_text):
        return [], ""

    blocks: List[Dict[str, Any]] = []
    lorebook_at_end_text     = ""
    lorebook_xml_at_end_text = ""

    if text and cfg.lorebook_xml_at_end:
        matches = [match.group(0).strip() for match in LOREBOOK_XML_RE.finditer(text) if match.group(0).strip()]
        if matches:
            text = LOREBOOK_XML_RE.sub("", text).strip()
            lorebook_xml_at_end_text = "\n\n".join(matches)

    if text:
        if cfg.split_lorebook:
            split_at = -1

            for split_marker in ("</summary>", "</example_dialogs>", "</UserPersona>", "</Scenario>"):
                split_idx = text.find(split_marker)
                if split_idx != -1:
                    split_at = split_idx + len(split_marker)
                    break

            if split_at == -1:
                persona_matches = list(PERSONA_END_RE.finditer(text))
                if persona_matches:
                    split_at = persona_matches[-1].end()

            if split_at == -1 or split_at >= len(text):
                blocks.append({"type": "text", "text": text})
            else:
                before = text[:split_at].rstrip()
                after  = text[split_at:].strip()

                if before:
                    blocks.append({"type": "text", "text": before})

                if after:
                    if cfg.lorebook_at_end:
                        existing_text = lorebook_at_end_text.strip()
                        lorebook_at_end_text = f"{existing_text}\n\n{after}" if existing_text else after
                    else:
                        # Keep a clean visual/semantic separator between Scenario/Persona and suffix.
                        blocks.append({"type": "text", "text": "\n\n" + after})
        else:
            blocks.append({"type": "text", "text": text})

    if lorebook_xml_at_end_text:
        existing_text = lorebook_at_end_text.strip()
        lorebook_at_end_text = f"{existing_text}\n\n{lorebook_xml_at_end_text}" if existing_text else lorebook_xml_at_end_text

    formatted_system: List[Dict[str, Any]] = []

    for block in blocks:
        new_block: Dict[str, Any] = dict(block)
        if cfg.cache_en and cfg.cache_system and new_block.get("text", "").strip():
            new_block["cache_control"] = make_cache_control(cfg.cache_system_ttl)
        formatted_system.append(new_block)

    if summary_text:
        formatted_system.append({"type": "text", "text": summary_text})

    return formatted_system, lorebook_at_end_text


def format_to_claude_messages(mlist: List[Dict[str, Any]], lorebook_at_end_text: str = "") -> List[Dict[str, Any]]:
    """
    Converts OpenAI-style chat messages to Anthropic Messages format.

    Consecutive same-role user/assistant messages are merged because Anthropic expects alternating user/assistant turns.
    Internal mid-conversation system messages are inserted only for Claude 4.8+ when LOREBOOK_AT_END moves
    the split lorebook out of the top-level system prompt.

    Manual caching marks the configured first-N-message prefix. Automatic caching marks an
    end-relative conversation point after any lorebook relocation and before optional prefill,
    so moved lorebook content is treated like any other end-of-conversation item.
    """

    formatted: List[Dict[str, Any]] = []
    old_role: Optional[str] = None

    # Maps each incoming OpenAI-style chat message index to the Anthropic message index
    # that contains it after same-role merging. Cache markers are applied after the final
    # message shape is known instead of checking targets on every loop iteration.
    incoming_to_formatted_index: List[int] = []

    for msg in mlist:
        incoming_role = msg.get("role", "user")
        content       = msg.get("content", "")

        if msg.get("role") == "assistant" and msg.get("send_anthropic_thinking_blocks"):
            thinking_blocks = extract_preservable_thinking_blocks(msg.get("anthropic_thinking_blocks") or [])
            if thinking_blocks:
                # Remove the display-only <think> copy before sending signed blocks back to Claude.
                visible_text = VISIBLE_THINK_RE.sub("", content or "").strip()
                content = list(thinking_blocks)
                if visible_text:
                    content.append({"type": "text", "text": visible_text})

        claude_role = "assistant" if incoming_role == "assistant" else "user"

        if formatted and claude_role == old_role:
            if isinstance(content, list):
                # Preserve block form when a same-role assistant turn carries thinking blocks.
                merged_blocks: List[Dict[str, Any]] = []
                existing = formatted[-1].get("content", "")
                if isinstance(existing, list):
                    merged_blocks.extend(dict(block) if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in existing)
                elif isinstance(existing, str) and existing:
                    merged_blocks.append({"type": "text", "text": existing})
                elif existing not in (None, ""):
                    merged_blocks.append({"type": "text", "text": str(existing)})
                merged_blocks.extend(dict(block) if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in content)
                formatted[-1]["content"] = merged_blocks
            else:
                formatted[-1]["content"] = append_text_to_content(formatted[-1]["content"], str(content))
        else:
            formatted.append({"role" : claude_role, "content" : content})

        old_role = claude_role
        incoming_to_formatted_index.append(len(formatted) - 1)

    if lorebook_at_end_text:
        if cfg.version >= Version("4.8"):
            formatted.append({"role": "system", "content": lorebook_at_end_text.strip()})
        else:
            scenario_update = f"\n<OOC>\nGameMaster lore update:\n\n{lorebook_at_end_text.strip()}\n</OOC>"
            for i in range(len(formatted) - 1, -1, -1):
                if formatted[i].get("role") == "user":
                    formatted[i]["content"] = append_text_to_content(formatted[i].get("content", ""), scenario_update)
                    break

    if cfg.cache_en and cfg.cache_manual_msg > 0 and incoming_to_formatted_index:
        target_incoming_index = min(cfg.cache_manual_msg, len(incoming_to_formatted_index)) - 1
        target_index = incoming_to_formatted_index[target_incoming_index]
        formatted[target_index]["content"] = add_cache_control_to_content(formatted[target_index].get("content", ""), cfg.cache_manual_ttl)

    if cfg.cache_en and cfg.cache_auto_msg > 0 and formatted:
        target_index = max(0, len(formatted) - cfg.cache_auto_msg)
        formatted[target_index]["content"] = add_cache_control_to_content(formatted[target_index].get("content", ""), cfg.cache_auto_ttl)

    # Optional Claude prefill.
    # assistant mode preserves the original assistant-message/prefill behavior.
    # instruction mode avoids assistant prefill and appends an OOC instruction to the last user message instead.
    if cfg.assistant_prefill.strip() and cfg.assistant_prefill_mode != "none":
        if cfg.assistant_prefill_mode == "instruction":
            append_prefill_instruction_to_last_user_message(formatted, cfg.assistant_prefill)
        elif cfg.assistant_prefill_mode == "assistant":
            if not formatted                   : formatted.append({"role" : "user", "content" : ""})
            if formatted[-1]["role"] == "user" : formatted.append({"role" : "assistant", "content" : cfg.assistant_prefill})
            else                               : formatted[-1]["content"] = append_text_to_content(formatted[-1]["content"], cfg.assistant_prefill)

    return formatted


def extract_text_from_anthropic_message(message: Any) -> str:
    """
    Collects text blocks from an Anthropic response.
    """
    chunks = []
    for block in getattr(message, "content", []) or []:
        if   getattr(block, "type", None) == "text"                  : chunks.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text" : chunks.append(block.get("text", ""))

    return "".join(chunks)


def anthropic_blocks_to_dicts(message: Any) -> List[Dict[str, Any]]:
    blocks = []
    for block in getattr(message, "content", []) or []:
        if hasattr(block, "model_dump") : blocks.append(block.model_dump(mode="json"))
        elif isinstance(block, dict)    : blocks.append(block)
        else                            : blocks.append({"type": getattr(block, "type", "unknown"), "value": str(block)})
    return blocks


def fallback_cache_write_ttl() -> str:
    """
    Older SDK usage payloads may not split cache creation by 5m/1h.
    If any active marker is configured for 1h, assume 1h for unknown write tokens to avoid under-counting cost.
    """
    if not cfg.cache_en:
        return "5m"
    active_ttl: List[str] = []
    if cfg.cache_system         : active_ttl.append(cfg.cache_system_ttl)
    if cfg.cache_manual_msg > 0 : active_ttl.append(cfg.cache_manual_ttl)
    if cfg.cache_auto_msg   > 0 : active_ttl.append(cfg.cache_auto_ttl)
    return "1h" if "1h" in active_ttl else "5m"


def print_usage(usage: Any) -> None:
    def tok_usd(tokens: int, usd_per_million_tokens: float) -> float:
        return (tokens*usd_per_million_tokens)/1_000_000.0
    def fmt_usd(amount: float) -> str:
        sign = "-" if amount < 0 else ""
        return f"{sign}${abs(amount):,.6f}"
    def cache_lbl(net_cost_usd: float) -> str:
        if net_cost_usd < 0: return f"{fmt_usd(abs(net_cost_usd))} saved"
        if net_cost_usd > 0: return f"{fmt_usd(net_cost_usd)} lost"
        return "$0.000000 break-even"

    cache_creation       = getattr(usage, "cache_creation", {}) or {}
    ephemeral_1h         = int(getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
    ephemeral_5m         = int(getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
    input_tok            = int(getattr(usage, "input_tokens", 0) or 0)
    output_tok           = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read           = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_input = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    ttl_tokens = input_tok + cache_read + cache_creation_input;

    # Older SDK responses may expose only cache_creation_input_tokens without the 5m/1h split.
    # Mixed-TTL requests cannot be reconstructed from that legacy shape, so use a conservative fallback.
    known_cache_write = ephemeral_1h + ephemeral_5m
    if cache_creation_input > known_cache_write:
        unknown_cache_write = cache_creation_input - known_cache_write
        if fallback_cache_write_ttl() == "1h" : ephemeral_1h += unknown_cache_write
        else                                  : ephemeral_5m += unknown_cache_write

    cache_creation_input = ephemeral_1h + ephemeral_5m
    ttl_tokens           = input_tok + cache_read + cache_creation_input

    input_cost          = tok_usd(input_tok, cfg.input_token_cost_usd)
    cache_read_cost     = tok_usd(cache_read, cfg.cache_read_cost_usd)
    cache_write_1h_cost = tok_usd(ephemeral_1h, cfg.cache_write_1h_cost_usd)
    cache_write_5m_cost = tok_usd(ephemeral_5m, cfg.cache_write_5m_cost_usd)
    cache_write_cost    = cache_write_1h_cost + cache_write_5m_cost
    total_input_cost    = input_cost + cache_read_cost + cache_write_cost

    output_cost        = tok_usd(output_tok, cfg.output_token_cost_usd)
    request_total_cost = total_input_cost + output_cost

    cache_write_extra_cost = (
        tok_usd(ephemeral_1h, cfg.cache_write_1h_cost_usd - cfg.input_token_cost_usd)
        +
        tok_usd(ephemeral_5m, cfg.cache_write_5m_cost_usd - cfg.input_token_cost_usd)
    )
    cache_read_saved_cost  = tok_usd(cache_read, cfg.input_token_cost_usd - cfg.cache_read_cost_usd)
    request_cache_net_cost = cache_write_extra_cost - cache_read_saved_cost

    session = add_session_cost(
        request_total_cost = request_total_cost,
        total_input_cost   = total_input_cost,
        output_cost        = output_cost,
        input_tokens       = ttl_tokens,
        output_tokens      = output_tok,
        cache_net_cost     = request_cache_net_cost,
    )

    if not cfg.debug_log:
        return

    print("=== Claude usage start ===")
    print("Request:")
    print("    Input tokens       =   uncached + cache read + cache write (        1h +         5m)")
    print("    {:18d} = {:10d} + {:10d} + {:11d} ({:10d} + {:10d})".format(ttl_tokens, input_tok, cache_read, cache_creation_input, ephemeral_1h, ephemeral_5m))
    print("    {:>18s} = {:>10s} + {:>10s} + {:>11s} ({:>10s} + {:>10s})".format(fmt_usd(total_input_cost), fmt_usd(input_cost), fmt_usd(cache_read_cost), fmt_usd(cache_write_cost), fmt_usd(cache_write_1h_cost), fmt_usd(cache_write_5m_cost)))
    print("    Output tokens      = {:d} ({})".format(output_tok, fmt_usd(output_cost)))
    print("    Cache cost         = {} ({})".format(fmt_usd(request_cache_net_cost), cache_lbl(request_cache_net_cost)))
    print("    Total cost         = {}".format(fmt_usd(request_total_cost)))
    print("Session:")
    print("    Input tokens       = {:d} ({})".format(session["input_tokens"], fmt_usd(session["input_cost_usd"])))
    print("    Output tokens      = {:d} ({})".format(session["output_tokens"], fmt_usd(session["output_cost_usd"])))
    print("    Cache cost         = {} ({})".format(fmt_usd(session["cache_net_cost_usd"]), cache_lbl(session["cache_net_cost_usd"])))
    print("    Average input cost = {} / MTok.".format(fmt_usd(session["average_input_cost_usd"]*1_000_000)))
    print("    Total cost         = {} ({} input / {} output)".format(fmt_usd(session["total_spent_usd"]), fmt_usd(session["input_cost_usd"]), fmt_usd(session["output_cost_usd"])))
    print("=== Claude usage end ===")
    print("> ", end="", flush=True)


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
    input_tokens   = int(getattr(usage, "input_tokens"               , 0) or 0)
    output_tokens  = int(getattr(usage, "output_tokens"              , 0) or 0)
    cache_read     = int(getattr(usage, "cache_read_input_tokens"    , 0) or 0)
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    prompt_tokens = input_tokens + cache_read + cache_creation

    return {
        "prompt_tokens"               : prompt_tokens,
        "completion_tokens"           : output_tokens,
        "total_tokens"                : prompt_tokens + output_tokens,
        "input_tokens_uncached"       : input_tokens,
        "cache_creation_input_tokens" : cache_creation,
        "cache_read_input_tokens"     : cache_read,
    }
