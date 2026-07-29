"""
The /v1/messages backend: Anthropic's own protocol, spoken through the Anthropic SDK.

This is where the proxy's Anthropic-only features live -- explicit cache markers with
a TTL each, signed thinking-block preservation, and assistant prefill. None of them
have an equivalent on the other two endpoints, so nothing here is shared with them.
Providers are registered, listed, selected and priced in providers.py like any other.
"""

import anthropic
import base64
import json
import re
import time

from packaging.version import Version
from typing            import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    UINT64_MAX,
    append_prefill_instruction_to_last_user_message,
    append_text_to_content,
    cfg,
    deep_get,
    print_payload,
    print_usage,
    resolve_api_key,
    trim_to_end_sentence,
    usage_to_openai_dict,
)


def resolve_thinking() -> None:
    """
    Anthropic thinking resolution: capability checks against the selected model record and the Anthropic parameter constraints.
    """
    if not cfg.thinking_enabled:
        return

    name = deep_get(cfg.info, "id")

    if not deep_get(cfg.info, "capabilities.thinking.supported"):
        print(f"Model {name} does not support thinking. Disabling.")
        cfg.thinking_enabled = False
        return
    if deep_get(cfg.info, "capabilities.thinking.types.adaptive.supported"):
        print(f"Models supports adaptive thinking. Using with effort '{cfg.thinking_effort}'.")
        cfg.use_adaptive = True
    elif deep_get(cfg.info, "capabilities.thinking.types.enabled.supported"):
        print(f"Using thinking with a budget of {cfg.thinking_budget} tokens")
        cfg.use_adaptive = False
    else:
        print("Neither adaptive nor budget thinking are supported. Disabling.")
        cfg.thinking_enabled = False
        return

    if (cfg.assistant_prefill_mode == "assistant") and (cfg.assistant_prefill != ""):
        print("When thinking is enabled, only instruction mode prefill is supported. Switching.")
        cfg.assistant_prefill_mode = "instruction"
    if cfg.send_temperature:
        print("Temperature is not compatible with thinking. Disabling.")
        cfg.send_temperature = False
    if cfg.send_top_k:
        print("top_k is not compatible with thinking. Disabling.")
        cfg.send_top_k = False
    if cfg.send_top_p:
        if (cfg.top_p < 0.95) or (cfg.top_p > 1.00) :
            print("Thinking supports top_p in the range [0.95:1]. Clamping.")
        if   cfg.top_p < 0.95 : cfg.top_p = 0.95
        elif cfg.top_p > 1.00 : cfg.top_p = 1.00


def print_think_status() -> None:
    """
    CLI 'think' status for the Anthropic backend.
    """
    if cfg.preserve_thinking_blocks == UINT64_MAX : preserve_str = "inf"
    else                                          : preserve_str = str(cfg.preserve_thinking_blocks)

    if cfg.thinking_enabled              : print( "  Thinking enabled    ✅")
    else                                 : print( "  Thinking enabled    ❌")
    if cfg.use_adaptive                  :
                                           print(f"  Thinking effort     ✅  {cfg.thinking_effort}")
                                           print(f"  Thinking budget     ❌  {cfg.thinking_budget}")
    else                                 :
                                           print(f"  Thinking effort     ❌  {cfg.thinking_effort}")
                                           print(f"  Thinking budget     ✅  {cfg.thinking_budget}")
    if cfg.preserve_thinking_blocks <= 0 : print( "  Thinking preserved  ❌")
    else                                 : print(f"  Thinking preserved  {preserve_str}")


def after_model_switch() -> None:
    """
    Post-switch hook for this backend. Runs after providers.apply_model has pointed cfg at the new model.

    Prefill is validated here rather than in resolve_thinking() because the rules turn
    on the model version, so the check has to run whether or not thinking is enabled.
    Do not call cfg.set_prefill() here: that would overwrite ASSISTANT_PREFILL with the
    mode string (for example "none", "assistant", or "instruction").
    """
    cfg.set_prefill_mode(cfg.assistant_prefill_mode)
    resolve_thinking()


# The SDK appends /v1/messages to its base_url itself, while a provider is configured
# with the /v1 root like every other one, so the suffix has to come back off.
V1_SUFFIX_RE = re.compile(r"/v1/?$")


def get_anthropic_client() -> anthropic.Anthropic:
    provider = cfg.providers[cfg.backend]
    return anthropic.Anthropic(
        api_key  = resolve_api_key(provider["api_key"], provider["api_key_name"]),
        base_url = V1_SUFFIX_RE.sub("", provider["base_url"]),
    )


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

    Anthropic prompt caching is enabled by adding cache_control either at the request level or on content blocks.
    This script uses explicit block-level caching to avoid caching the assistant prefill as the final block.
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


def format_system(system_segments: List[str], system_summary_text: str = "") -> List[Dict[str, Any]]:
    """
    Turns pre-split system prompt segments into top-level Anthropic system blocks.

    The model-agnostic lorebook splitting happens in server.split_system_text().
    This only decides the Anthropic representation: one text block per segment, with the explicit system cache marker applied to each non-empty block.
    """
    summary_text = system_summary_text.strip()

    formatted_system: List[Dict[str, Any]] = []

    for segment in system_segments:
        new_block: Dict[str, Any] = {"type": "text", "text": segment}
        if cfg.cache_en and cfg.cache_system and segment.strip():
            new_block["cache_control"] = make_cache_control(cfg.cache_system_ttl)
        formatted_system.append(new_block)

    if summary_text:
        formatted_system.append({"type": "text", "text": summary_text})

    return formatted_system


def format_messages(mlist: List[Dict[str, Any]], lorebook_at_end_text: str = "") -> List[Dict[str, Any]]:
    """
    Converts OpenAI-style chat messages to Anthropic Messages format.

    Consecutive same-role user/assistant messages are merged because Anthropic expects alternating user/assistant turns.
    Internal mid-conversation system messages are inserted only for Claude 4.8+ when LOREBOOK_AT_END moves the split lorebook out of the top-level system prompt.

    Manual caching marks the configured first-N-message prefix.
    Automatic caching marks an end-relative conversation point after any lorebook relocation and before optional prefill,
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


def parse_usage(usage: Any) -> Dict[str, Any]:
    """
    Pulls the token counts the proxy tracks out of an Anthropic usage payload.

    Anthropic reports input_tokens net of caching -- cache reads and cache writes are
    counted separately rather than included -- so the normalized 'prompt' total has to
    be summed back up. Anthropic never reports a reasoning count, so 'reasoning' stays
    None rather than claiming zero for thinking that demonstrably happened.
    """
    cache_creation       = getattr(usage, "cache_creation", {}) or {}
    ephemeral_1h         = int(getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
    ephemeral_5m         = int(getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
    input_tok            = int(getattr(usage, "input_tokens", 0) or 0)
    output_tok           = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read           = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_input = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    # Older SDK responses may expose only cache_creation_input_tokens without the 5m/1h split.
    # Mixed-TTL requests cannot be reconstructed from that legacy shape, so use a conservative fallback.
    known_cache_write = ephemeral_1h + ephemeral_5m
    if cache_creation_input > known_cache_write:
        unknown_cache_write = cache_creation_input - known_cache_write
        if fallback_cache_write_ttl() == "1h" : ephemeral_1h += unknown_cache_write
        else                                  : ephemeral_5m += unknown_cache_write

    prompt_tokens = input_tok + cache_read + ephemeral_1h + ephemeral_5m

    return {
        "prompt"     : prompt_tokens,
        "completion" : output_tok,
        "total"      : prompt_tokens + output_tok,
        "cached"     : cache_read,
        "write_1h"   : ephemeral_1h,
        "write_5m"   : ephemeral_5m,
        "uncached"   : input_tok,
        "reasoning"  : None,
    }


# Generation


def build_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the Anthropic Messages API request from a prepared chat request
    (see server.prepare_chat_request for the dict shape).
    """
    provider           = cfg.providers[cfg.backend]
    formatted_system   = format_system(prepared["system_segments"], prepared["system_summary_text"])
    formatted_messages = format_messages(prepared["messages"], prepared["lorebook_at_end_text"])

    kwargs: Dict[str, Any] = {
        "model"      : cfg.model,
        "max_tokens" : prepared["max_tokens"],
    }

    # Anthropic request-level automatic prompt caching.
    # This is separate from the script's existing explicit block-level markers.
    # Guard it with cfg.cache_en so USE_CACHE=false disables all cache behavior.
    if cfg.cache_en and cfg.cache_anthropic_auto:
        kwargs["cache_control"] = make_cache_control(cfg.cache_anthropic_ttl)
    if cfg.send_temperature : kwargs["temperature"] = cfg.temperature
    if cfg.send_top_k       : kwargs["top_k"      ] = cfg.top_k
    if cfg.send_top_p       : kwargs["top_p"      ] = cfg.top_p
    if cfg.thinking_enabled:
        if   cfg.use_adaptive:
            kwargs["thinking"]      = { "type": "adaptive", "display": "summarized" }
            kwargs["output_config"] = { "effort": cfg.thinking_effort }
        else:
            kwargs["thinking"] = { "type": "enabled", "budget_tokens": cfg.thinking_budget }

    if formatted_system:
        kwargs["system"] = formatted_system
    kwargs["messages"] = formatted_messages

    # The SDK rejects unknown keyword arguments, so <NAME>_EXTRA_BODY cannot simply be
    # merged into the body the way the OpenAI-style backends do it. It is still merged
    # last upstream, so it can override anything the proxy sends.
    if provider["extra_body"]:
        kwargs["extra_body"] = provider["extra_body"]

    return kwargs


def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming completion.

    Returns the backend-neutral result dict consumed by the server core:
        id, stop_reason, text, usage, message_extra
    """
    client = get_anthropic_client()
    kwargs = build_body(prepared)

    print_payload(kwargs)

    message = client.messages.create(**kwargs)

    counts = parse_usage(getattr(message, "usage", None))
    print_usage(counts)

    output_text = extract_text_from_anthropic_message(message)
    if cfg.auto_trim:
        output_text = trim_to_end_sentence(output_text)

    anthropic_content = anthropic_blocks_to_dicts(message)
    thinking_blocks   = extract_preservable_thinking_blocks(anthropic_content)

    # Keep ordinary <think> output for Janitor/client compatibility.
    thinking_text = "\n".join(
        block.get("thinking", "")
        for block in thinking_blocks
        if block.get("type") == "thinking" and block.get("thinking", "")
    ).strip()
    if thinking_text:
        output_text = f"<think>\n{thinking_text}\n</think>\n\n" + output_text

    # Add a second, hidden-ish signed-block envelope only when preservation is enabled.
    if thinking_preservation_enabled() and thinking_blocks:
        output_text += make_hidden_thinking_envelope(thinking_blocks)

    return {
        "id"            : getattr(message, "id", "claude"),
        "stop_reason"   : getattr(message, "stop_reason", "stop"),
        "text"          : output_text,
        "usage"         : usage_to_openai_dict(counts),
        "message_extra" : {
            "anthropic_content"            : anthropic_content,
            "anthropic_thinking_preserved" : bool(thinking_preservation_enabled() and thinking_blocks),
        },
    }


def generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming completion, yielding backend-neutral events:
        ("reasoning", text)  incremental reasoning text
        ("text", text)       incremental visible content
        ("final", dict)      id, stop_reason, usage, snapshot_text, snapshot_reasoning

    Errors propagate to the caller, which owns SSE error formatting and logging.
    """
    client = get_anthropic_client()
    kwargs = build_body(prepared)

    print_payload(kwargs)

    response_parts  : List[str] = []
    reasoning_parts : List[str] = []

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    reasoning_parts.append(event.delta.thinking)
                    yield ("reasoning", event.delta.thinking)
                elif event.delta.type == "text_delta":
                    response_parts.append(event.delta.text)
                    yield ("text", event.delta.text)
            time.sleep(0.01)

        final_message = stream.get_final_message()
        if not response_parts:
            response_parts.append(extract_text_from_anthropic_message(final_message))

        # The visible <think> text has already gone through the reasoning events above.
        thinking_blocks = extract_preservable_thinking_blocks(anthropic_blocks_to_dicts(final_message))
        if thinking_preservation_enabled() and thinking_blocks:
            # Send only the hidden signed-block envelope for next-turn rehydration.
            thinking_envelope = make_hidden_thinking_envelope(thinking_blocks)
            response_parts.append(thinking_envelope)
            yield ("text", thinking_envelope)

        counts = parse_usage(getattr(final_message, "usage", None))
        print_usage(counts)

        yield ("final", {
            "id"                 : getattr(final_message, "id", "claude"),
            "stop_reason"        : getattr(final_message, "stop_reason", "stop"),
            "usage"              : usage_to_openai_dict(counts),
            "snapshot_text"      : "".join(response_parts),
            "snapshot_reasoning" : "".join(reasoning_parts),
        })
