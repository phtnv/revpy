"""
The /responses backend: OpenAI's own endpoint, and the only one that returns the
model's reasoning. Ask for a summary and the response carries reasoning items whose
text the proxy wraps in a <think> block, which /chat/completions cannot do at all.

Everything here assumes OpenAI's catalogue and OpenAI's rules -- one effort ladder per
model rather than one per model per endpoint -- since OpenAI is the only vendor serving
this protocol. A model it does not recognize simply gets no reasoning parameter, and
<NAME>_EXTRA_BODY is the escape hatch for a gateway with its own dialect.
"""

import httpx
import json
import re

from packaging.version import Version
from typing            import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    cfg,
    deep_get,
    print_payload,
    print_usage,
    trim_to_end_sentence,
    usage_to_openai_dict,
)
from providers import (
    OFF_EFFORTS,
    ProviderError,
    build_message_list,
    error_from_response,
    fold_effort,
    is_openai_model,
    reported_reasoning,
    request_headers,
    request_timeout,
    warn_truncated_by_reasoning,
    wrap_think,
)


CHAT_LATEST_RE = re.compile(r"(?:^|-)chat-latest$")
GPT_RE         = re.compile(r"^gpt-(\d+(?:\.\d+)?)")
O_SERIES_RE    = re.compile(r"^o(\d+)(?:-|$)")

# Reasoning effort levels each model accepts on this endpoint, weakest first.
# Established by sending each level to each model: the endpoint's own "supported
# values" error lists the union across models, not what the addressed model takes,
# so it cannot be trusted on its own.
EFFORTS_GPT56 = ("none", "low", "medium", "high", "xhigh", "max")  # gpt-5.6+
EFFORTS_GPT52 = ("none", "low", "medium", "high", "xhigh")         # gpt-5.2..5.5
EFFORTS_GPT51 = ("none", "low", "medium", "high")                  # gpt-5.1
EFFORTS_GPT50 = ("minimal", "low", "medium", "high")               # gpt-5
EFFORTS_O     = ("low", "medium", "high")                          # o-series


def effort_ladder(model_id: str) -> Optional[Tuple[str, ...]]:
    """
    The reasoning effort ladder for an OpenAI model, weakest first. Returns None when
    reasoning effort is not a parameter of the model at all (gpt-4 and older reject it
    as an unknown argument), and an empty tuple for the *-chat-latest snapshots, which
    accept only the default and never reason.
    """
    if CHAT_LATEST_RE.search(model_id):
        return ()
    if O_SERIES_RE.match(model_id) is not None:
        return EFFORTS_O

    match = GPT_RE.match(model_id)
    if match is None:
        return None

    version = Version(match.group(1))
    if version < Version("5")   : return None
    if version < Version("5.1") : return EFFORTS_GPT50
    if version < Version("5.2") : return EFFORTS_GPT51
    if version < Version("5.6") : return EFFORTS_GPT52
    return EFFORTS_GPT56


def effort_for(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[str]:
    """
    The effort level to send for one model, or None when the model takes no effort
    parameter at all.

    How far thinking can be turned down is model-dependent: gpt-5.1+ take 'none' and
    stop reasoning outright, gpt-5 bottoms out at 'minimal', and the o-series has no
    off switch at all, so a disable request sends the weakest level it does offer.
    """
    ladder = effort_ladder(model_id)
    if not ladder:
        return None
    if not thinking_enabled:
        return next((off for off in OFF_EFFORTS if off in ladder), ladder[0])
    return fold_effort(thinking_effort, tuple(e for e in ladder if e not in OFF_EFFORTS))


# OpenAI gates reasoning summaries behind organization verification and refuses the whole
# request when an unverified account asks for one. Since the proxy is what adds the
# summary, it remembers the refusal and drops it rather than failing every turn.
# The gate is per model, not per account -- the gpt-5 family answers unverified while the
# older o-series does not -- so one model's refusal must not mute the rest.
SUMMARY_BLOCKED_MARKER = "verified to generate reasoning summaries"
SUMMARY_BLOCKED_MODELS : set = set()


def note_summary_blocked(exc: Exception) -> bool:
    """
    True when the error is the unverified-organization refusal, in which case summaries
    are disabled for that model so the caller can retry without one.
    """
    if SUMMARY_BLOCKED_MARKER not in str(exc):
        return False
    if cfg.model not in SUMMARY_BLOCKED_MODELS:
        print(f"WARNING: this OpenAI organization is not verified for reasoning summaries on '{cfg.model}', "
              "so it cannot show its thinking.")
        print("         Retrying without the summary. Verify the org at "
              "https://platform.openai.com/settings/organization/general to see reasoning.")
    SUMMARY_BLOCKED_MODELS.add(cfg.model)
    return True


def reasoning_param(model_id: str, thinking_enabled: bool, thinking_effort: str, summary: str) -> Optional[Dict[str, Any]]:
    """
    The 'reasoning' object: the effort plus, when thinking is on, the summary detail
    level that makes the model's reasoning visible at all. Returns None for models
    that take no reasoning parameter, which reject it outright.
    """
    effort = effort_for(model_id, thinking_enabled, thinking_effort)
    if effort is None:
        return None

    reasoning: Dict[str, Any] = {"effort": effort}
    if thinking_enabled and summary != "none" and model_id not in SUMMARY_BLOCKED_MODELS:
        reasoning["summary"] = summary
    return reasoning


def supports_sampling(model_id: str) -> bool:
    """
    False for OpenAI models that reject sampling controls: every reasoning model and
    every *-chat-latest snapshot refuses top_p outright and accepts no temperature
    other than the default 1.
    """
    return not (is_openai_model(model_id) and effort_ladder(model_id) is not None)


def resolve_thinking() -> None:
    """
    Reports how the shared thinking settings map onto the selected model's reasoning
    parameter. Unlike the Anthropic backend there is no capability metadata to check
    and none of its parameter constraints apply, so nothing is adjusted here.
    """
    ladder = effort_ladder(cfg.model)
    if ladder is None:
        print(f"Backend '{cfg.backend}' has no thinking passthrough for '{cfg.model}'. Configure thinking through EXTRA_BODY.")
        return
    if not ladder:
        print(f"Thinking passthrough: nothing to send ('{cfg.model}' has no thinking controls).")
        return

    effort = effort_for(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    if not cfg.thinking_enabled:
        if effort in OFF_EFFORTS : print("Thinking passthrough: disabled.")
        else                     : print(f"Thinking passthrough: '{cfg.model}' cannot stop reasoning; sending the weakest setting it has ('{effort}').")
        return

    print(f"Thinking passthrough: enabled with effort '{effort}' (from '{cfg.thinking_effort}').")


def after_model_switch() -> None:
    """
    Post-switch hook for this backend (v1_messages and v1_chat_completions have their
    own). There is nothing to validate against the model here, so this only reports how
    the thinking settings land on it.
    """
    resolve_thinking()


def print_think_status() -> None:
    """
    CLI 'think' status for this endpoint
    (v1_messages.print_think_status and v1_chat_completions.print_think_status are the counterparts).
    """
    ladder = effort_ladder(cfg.model)
    if ladder is None:
        print(f"  No thinking passthrough for '{cfg.model}'. Configure thinking through EXTRA_BODY.")
        return
    if not ladder:
        print(f"  No thinking controls for '{cfg.model}'. Neither setting changes the request.")
        return

    # What the current settings actually send, versus what the model can do at all.
    actual      = effort_for(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    can_disable = any(off in ladder for off in OFF_EFFORTS)

    if   cfg.thinking_enabled : print( "  Thinking enabled    ✅")
    elif can_disable          : print( "  Thinking enabled    ❌")
    else                      : print(f"  Thinking enabled    ❌  (always on for '{cfg.model}', sent as '{actual}')")

    if not cfg.thinking_enabled : print(f"  Thinking effort     ✅  {cfg.thinking_effort} (not sent while thinking is off)")
    else                        : print(f"  Thinking effort     ✅  {cfg.thinking_effort} (sent as '{actual}')")


def apply_sampling(body: Dict[str, Any]) -> None:
    """
    Adds temperature/top_p in place, unless the model refuses them. OpenAI reasoning
    models take no sampling controls at all: top_p is rejected outright, and
    temperature accepts nothing but its default of 1.
    """
    if supports_sampling(cfg.model):
        if cfg.send_temperature : body["temperature"] = cfg.temperature
        if cfg.send_top_p       : body["top_p"      ] = cfg.top_p
    elif cfg.send_temperature or cfg.send_top_p:
        print(f"WARNING: '{cfg.model}' does not support temperature/top_p. Sending the request without them.")
    # top_k is not part of the schema; it can be forced through EXTRA_BODY.


def build_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a /responses request from a prepared chat request.

    Note the model decides whether to reason: on adaptive models an easy turn can
    come back with no reasoning at all, and then there is no summary to show.
    """
    provider = cfg.providers[cfg.backend]

    body: Dict[str, Any] = {
        "model"             : cfg.model,
        "input"             : build_message_list(prepared),
        "max_output_tokens" : prepared["max_tokens"],
        # /responses retains responses for 30 days unless told otherwise.
        "store"             : provider["store"],
    }

    apply_sampling(body)

    reasoning = reasoning_param(cfg.model, cfg.thinking_enabled, cfg.thinking_effort, provider["reasoning_summary"])
    if reasoning is not None:
        body["reasoning"] = reasoning

    body.update(provider["extra_body"])

    return body


def parse_usage(usage: Any) -> Dict[str, Any]:
    """
    Pulls the token counts the proxy tracks out of a /responses usage payload. Same
    counts as the chat endpoint under different names: input/output rather than
    prompt/completion, and the details objects renamed to match.
    """
    usage = usage if isinstance(usage, dict) else {}

    input_tokens  = max(0, int(usage.get("input_tokens" , 0) or 0))
    output_tokens = max(0, int(usage.get("output_tokens", 0) or 0))

    input_details  = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    input_details  = input_details  if isinstance(input_details , dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}

    cached_tokens = max(0, int(input_details.get("cached_tokens", 0) or 0))
    # gpt-5.6+ reports the tokens written to the prompt cache, which it charges a
    # premium for. Models without a write fee simply never send this field.
    write_tokens  = max(0, int(input_details.get("cache_write_tokens", 0) or 0))
    cached_tokens = min(cached_tokens, input_tokens)
    write_tokens  = min(write_tokens, input_tokens - cached_tokens)

    return {
        "prompt"     : input_tokens,
        "completion" : output_tokens,
        "total"      : max(0, int(usage.get("total_tokens", input_tokens + output_tokens) or 0)),
        "cached"     : cached_tokens,
        # One cache rate and no TTL choice here, so every write is a 5m write
        # (providers.apply_model prices both buckets identically).
        "write_1h"   : 0,
        "write_5m"   : write_tokens,
        "uncached"   : input_tokens - cached_tokens - write_tokens,
        "reasoning"  : reported_reasoning(output_details),
    }


# Generation.
# The response is an 'output' list of items rather than a single message: reasoning items
# carry the summary of the model's thinking, message items carry the reply. A turn can
# produce several of each, and on adaptive models the reasoning items may be absent.
STOP_REASONS = {"completed": "stop", "incomplete": "length", "failed": "error"}

# The response statuses that end a turn. A stream that runs out of body without
# reporting one of these was cut mid-response -- see generate_stream(). A mid-stream
# 'error' event ends it too, but that path raises where it is read.
TERMINAL_STATUSES = frozenset(STOP_REASONS)


def truncated_stream_message(text_chars: int, reasoning_chars: int) -> str:
    """
    Explains a stream that stopped without OpenAI ever finishing the response.

    Seen when the model reasons for a long stretch over a large prompt: /responses
    sends nothing at all while reasoning (summary text only arrives once the reasoning
    item closes), and the silent connection gets closed with no error and no terminal
    event. It is not a time limit -- runs that stream output for minutes are fine,
    while ones that sit in reasoning for under a minute are cut.

    What arrived before the cut has already gone to the client, so say how much.
    """
    got = f"{text_chars} characters of reply and {reasoning_chars} of reasoning"
    return (
        f"The response stream ended before the model finished ({got}). OpenAI closed "
        "the connection without completing the response, and did not say why. This "
        "happens while the model is reasoning silently on a long prompt; retrying, "
        "shortening the prompt, or lowering the thinking effort ('t effort low') "
        "usually gets through."
    )


def output_text(data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pulls (reply text, reasoning text) out of a non-streaming response body.
    Multiple summary parts are joined with a blank line, the same way the streaming path separates them.
    """
    text_parts      : List[str] = []
    reasoning_parts : List[str] = []

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            for part in item.get("summary") or []:
                if isinstance(part, dict) and part.get("text"):
                    reasoning_parts.append(str(part["text"]))
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]))

    return "".join(text_parts), "\n\n".join(reasoning_parts)


def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming /responses request. Same result shape as v1_messages.generate_non_stream.
    """
    provider = cfg.providers[cfg.backend]

    def send() -> Any:
        body = build_body(prepared)
        print_payload(body)
        return httpx.post(
            f"{provider['base_url']}/responses",
            json=body,
            headers=request_headers(provider),
            timeout=request_timeout(),
        )

    response = send()
    if response.status_code != 200:
        error = error_from_response(cfg.backend, response)
        # An unverified org rejects the summary request; note_summary_blocked drops it.
        if not note_summary_blocked(error):
            raise error
        response = send()
        if response.status_code != 200:
            raise error_from_response(cfg.backend, response)

    data   = response.json()
    counts = parse_usage(data.get("usage"))
    print_usage(counts)

    # A failed response still arrives as HTTP 200 with the reason in the body.
    # Without this it would be relayed as an ordinary reply, which is usually an empty one.
    if str(data.get("status") or "") == "failed":
        message = deep_get(data, "error.message") or "the model failed to produce a response."
        raise ProviderError(502, {"error": {"message": str(message)}}, f"{cfg.backend}: {message}")

    reply_text, reasoning_text = output_text(data)
    finish_reason = STOP_REASONS.get(str(data.get("status") or ""), "stop")

    warn_truncated_by_reasoning(finish_reason, reply_text, counts)

    if cfg.auto_trim:
        reply_text = trim_to_end_sentence(reply_text)

    return {
        "id"            : str(data.get("id") or cfg.model),
        "stop_reason"   : finish_reason,
        "text"          : wrap_think(reply_text, reasoning_text),
        "usage"         : usage_to_openai_dict(counts),
        "message_extra" : {},
    }


def generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming /responses request. Unlike the chat stream, which relays deltas
    off one message, this is a typed event stream: reasoning summary text and reply
    text arrive as separate event types, which is exactly the split the proxy needs.
    """
    provider = cfg.providers[cfg.backend]

    response_parts  : List[str] = []
    reasoning_parts : List[str] = []
    finish_reason = "stop"
    message_id    = ""
    usage         = None

    # Two attempts at most: an unverified org rejects the summary request before any event is streamed, and note_summary_blocked drops it so the retry succeeds.
    for attempt in (0, 1):
        body = build_body(prepared)
        body["stream"] = True
        print_payload(body)

        retry_without_summary = False
        stream_completed      = False

        with httpx.Client(timeout=request_timeout()) as client:
            with client.stream(
                "POST",
                f"{provider['base_url']}/responses",
                json=body,
                headers=request_headers(provider),
            ) as response:
                if response.status_code != 200:
                    response.read()
                    error = error_from_response(cfg.backend, response)
                    if attempt == 0 and note_summary_blocked(error):
                        retry_without_summary = True
                    else:
                        raise error

                if retry_without_summary:
                    continue

                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try: event = json.loads(data_str)
                    except Exception: continue
                    if not isinstance(event, dict):
                        continue

                    event_type = str(event.get("type") or "")

                    # A mid-stream error arrives as an event; the HTTP status was already 200.
                    if event_type == "error":
                        message = deep_get(event, "message") or json.dumps(event, default=str)
                        raise ProviderError(500, {"error": {"message": str(message)}}, f"{cfg.backend}: {message}")

                    if event_type.startswith("response.") and isinstance(event.get("response"), dict):
                        finished = event["response"]
                        if finished.get("id")    : message_id = str(finished["id"])
                        if finished.get("usage") : usage      = finished["usage"]
                        if finished.get("status"):
                            status = str(finished["status"])
                            finish_reason = STOP_REASONS.get(status, finish_reason)
                            if status in TERMINAL_STATUSES:
                                stream_completed = True

                    # The model can emit several summary blocks per turn; keep them apart.
                    if event_type == "response.reasoning_summary_part.added" and reasoning_parts:
                        reasoning_parts.append("\n\n")
                        yield ("reasoning", "\n\n")

                    if event_type == "response.reasoning_summary_text.delta":
                        delta = event.get("delta") or ""
                        if delta:
                            reasoning_parts.append(delta)
                            yield ("reasoning", delta)

                    if event_type == "response.output_text.delta":
                        delta = event.get("delta") or ""
                        if delta:
                            response_parts.append(delta)
                            yield ("text", delta)

                # Running out of body without a terminal event means the connection was cut mid-response.
                if not stream_completed:
                    raise ProviderError(502, {"error": {"message": truncated_stream_message(
                        len("".join(response_parts)), len("".join(reasoning_parts)))}},
                        f"{cfg.backend}: the response stream ended before the model finished.")

        break

    counts        = parse_usage(usage)
    snapshot_text = "".join(response_parts)
    print_usage(counts)
    warn_truncated_by_reasoning(finish_reason, snapshot_text, counts)

    yield ("final", {
        "id"                 : message_id or cfg.model,
        "stop_reason"        : finish_reason,
        "usage"              : usage_to_openai_dict(counts),
        "snapshot_text"      : snapshot_text,
        "snapshot_reasoning" : "".join(reasoning_parts),
    })
