"""
The /chat/completions backend: the OpenAI-style endpoint every compatible provider
implements. GLM, Kimi and Aion are served from here.

OpenAI's own models are not. They speak /responses in this proxy (see v1_responses),
which is the only endpoint that can return their reasoning text, so nothing here
carries OpenAI model knowledge beyond the name of the token-limit field. Pointing a
provider on this endpoint at OpenAI's catalogue is unsupported; <NAME>_EXTRA_BODY and
<NAME>_MAX_TOKENS_PARAM are the escape hatches if you do it anyway.
"""

import httpx
import json
import re

from packaging.version import Version
from typing            import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    cfg,
    deep_get,
    trim_to_end_sentence,
)
from providers import (
    OFF_EFFORTS,
    build_message_list,
    error_from_response,
    is_openai_model,
    print_payload,
    print_usage,
    reported_reasoning,
    request_headers,
    request_timeout,
    usage_to_openai_dict,
    warn_truncated_by_reasoning,
    wrap_think,
)


# Provider thinking dialects.
# Each dialect maps the shared proxy thinking settings (on/off + effort) onto one
# provider's request parameters. A dialect returns None for model ids it does not
# recognize; models with no dialect fall back to the provider's EXTRA_BODY.
AION_MODEL_ID_RE = re.compile(r"^(?:aion-labs/)?aion-(\d+(?:\.\d+)?)")

# Aion accepts only low|medium|high for reasoning_effort; fold the five shared
# proxy efforts onto that scale (xhigh and max round down to high).
AION_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

def aion_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the Aion (AionLabs) request dialect.
    Returns None when model_id is not a numbered aion model (aion-rp-* does not
    reason at all; EXTRA_BODY is the escape hatch there).

    Aion's only thinking parameter is reasoning_effort (none|low|medium|high,
    default medium), and aion-2.0 is the sole model that takes it. Every other
    model rejects the parameter outright with HTTP 400 and reasons
    unconditionally, so they get an empty dialect: recognized, nothing to send.
    Effort was dropped after 2.0 rather than added, hence the exact version test
    -- guessing wrong here is a failed request, not a silently ignored field.

    Aion also has reasoning_split (default true on reasoning models), which
    already puts the thoughts in the separate 'reasoning' field this proxy
    reads, so it is left at its default.
    """
    match = AION_MODEL_ID_RE.match(model_id)
    if match is None:
        return None
    if Version(match.group(1)) != Version("2.0"):
        return {}

    effort = AION_EFFORT_MAP.get(thinking_effort, "medium") if thinking_enabled else "none"
    return {"reasoning_effort": effort}


GLM_MODEL_ID_RE = re.compile(r"^glm-(\d+(?:\.\d+)?)")

def glm_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the GLM request dialect. Returns None
    when model_id is not a GLM model (no passthrough; EXTRA_BODY is the escape
    hatch there). GLM models think by default, so 'thinking' is always sent
    explicitly. reasoning_effort exists from glm-5.2 on (assumed to stay for
    later models); older GLM models only get the on/off switch.
    """
    match = GLM_MODEL_ID_RE.match(model_id)
    if match is None:
        return None
    if not thinking_enabled:
        return {"thinking": {"type": "disabled"}}

    params: Dict[str, Any] = {"thinking": {"type": "enabled"}}
    if Version(match.group(1)) >= Version("5.2"):
        params["reasoning_effort"] = thinking_effort
    return params


KIMI_MODEL_ID_RE = re.compile(r"^kimi-k(\d+(?:\.\d+)?)")

# kimi-k3 accepts only low|high|max for reasoning_effort; fold the five shared
# proxy efforts onto that scale (medium rounds down, xhigh rounds up).
KIMI_EFFORT_MAP = {"low": "low", "medium": "low", "high": "high", "xhigh": "max", "max": "max"}

def kimi_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the Kimi (Moonshot) request dialect.
    Returns None when model_id is not a kimi-k* model (kimi-latest/moonshot-v1
    have no thinking dialect; EXTRA_BODY is the escape hatch there).
    Control is model-dependent:
        kimi-k3+        thinking always on; depth via reasoning_effort (low|high|max, default max)
        kimi-k2.7-*     thinking always on; thinking.type "enabled" is mandatory
        kimi-k2.5/k2.6  thinking on by default; only an on/off switch, no effort control
    A disable request on an always-on model sends the closest thing the API
    offers: minimal reasoning_effort on k3+, plain enabled on k2.7.
    """
    match = KIMI_MODEL_ID_RE.match(model_id)
    if match is None:
        return None
    version = Version(match.group(1))

    if version >= Version("3"):
        effort = KIMI_EFFORT_MAP.get(thinking_effort, "max") if thinking_enabled else "low"
        return {"reasoning_effort": effort}
    if version >= Version("2.7") or thinking_enabled:
        return {"thinking": {"type": "enabled"}}
    return {"thinking": {"type": "disabled"}}


THINKING_DIALECTS = (aion_thinking_params, glm_thinking_params, kimi_thinking_params)


def provider_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Provider-dialect thinking passthrough. Returns the params of the first dialect
    that recognizes model_id, or None when no dialect matches (no passthrough;
    EXTRA_BODY is the escape hatch). An empty dict means the model is recognized
    but offers no thinking controls at all.
    """
    for dialect in THINKING_DIALECTS:
        params = dialect(model_id, thinking_enabled, thinking_effort)
        if params is not None:
            return params
    return None


def thinking_can_be_disabled(model_id: str) -> bool:
    """
    Whether the model's dialect can actually stop it from reasoning. Each dialect
    spells "off" in its own way: a disabled thinking block (GLM), or an off effort
    level (Aion). Models that keep reasoning regardless (kimi-k3) get the weakest
    setting the API offers instead of a real off switch.
    """
    off = provider_thinking_params(model_id, False, cfg.thinking_effort) or {}
    return deep_get(off, "thinking.type") == "disabled" or off.get("reasoning_effort") in OFF_EFFORTS


def resolve_thinking() -> None:
    """
    Reports how the shared thinking settings map onto the selected model's dialect.
    Unlike the Anthropic backend there is no capability metadata to check and none
    of its parameter constraints apply, so nothing is adjusted here.
    """
    params = provider_thinking_params(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    if params is None:
        print(f"Backend '{cfg.backend}' has no thinking passthrough for '{cfg.model}'. Configure thinking through EXTRA_BODY.")
        return
    if not params:
        print(f"Thinking passthrough: nothing to send ('{cfg.model}' has no thinking controls).")
        return

    effort = params.get("reasoning_effort")
    if not cfg.thinking_enabled:
        if thinking_can_be_disabled(cfg.model) : print("Thinking passthrough: disabled.")
        else                                   : print(f"Thinking passthrough: '{cfg.model}' cannot stop reasoning; sending the weakest setting it has ('{effort}').")
        return

    if effort is None : print(f"Thinking passthrough: enabled ('{cfg.model}' has no effort control).")
    else              : print(f"Thinking passthrough: enabled with effort '{effort}' (from '{cfg.thinking_effort}').")


def print_think_status() -> None:
    """
    CLI 'think' status for this endpoint
    (claude.print_think_status and v1_responses.print_think_status are the counterparts).
    """
    probe = provider_thinking_params(cfg.model, True, cfg.thinking_effort)
    if probe is None:
        print(f"  No thinking passthrough for '{cfg.model}'. Configure thinking through EXTRA_BODY.")
        return
    if not probe:
        print(f"  No thinking controls for '{cfg.model}'. Neither setting changes the request.")
        return

    # What the current settings actually send, versus what the model can do at all.
    actual = provider_thinking_params(cfg.model, cfg.thinking_enabled, cfg.thinking_effort) or {}
    if   cfg.thinking_enabled                : print( "  Thinking enabled    ✅")
    elif thinking_can_be_disabled(cfg.model) : print( "  Thinking enabled    ❌")
    else                                     : print(f"  Thinking enabled    ❌  (always on for '{cfg.model}', sent as '{actual.get('reasoning_effort')}')")

    if "reasoning_effort" not in probe:
        print(f"  Thinking effort     ❌  {cfg.thinking_effort} (model has no effort control)")
    elif not cfg.thinking_enabled:
        print(f"  Thinking effort     ✅  {cfg.thinking_effort} (not sent while thinking is off)")
    else:
        print(f"  Thinking effort     ✅  {cfg.thinking_effort} (sent as '{actual['reasoning_effort']}')")


def max_tokens_param_name(provider: Dict[str, Any], model_id: str) -> str:
    """
    The request field carrying the output token limit. Providers on this endpoint
    expect max_tokens. The exception is an OpenAI-compatible gateway serving OpenAI's
    own catalogue: gpt-5+ and the o-series reject max_tokens and demand
    max_completion_tokens. That is not a supported configuration, but guessing right
    costs one regex and guessing wrong is an HTTP 400 with nothing to explain it.
    """
    configured = provider.get("max_tokens_param", "auto")
    if configured != "auto":
        return configured
    return "max_completion_tokens" if is_openai_model(model_id) else "max_tokens"


def apply_sampling(body: Dict[str, Any]) -> None:
    """
    Adds temperature/top_p in place. Every provider on this endpoint accepts them;
    the models that refuse sampling outright are OpenAI's reasoning models, which
    are served by v1_responses.
    """
    if cfg.send_temperature : body["temperature"] = cfg.temperature
    if cfg.send_top_p       : body["top_p"      ] = cfg.top_p
    # top_k is not part of the OpenAI schema; providers that accept it can get it via EXTRA_BODY.


def build_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the chat completion request from a prepared chat request.
    The provider's EXTRA_BODY is merged in verbatim last.
    """
    provider = cfg.openai_providers[cfg.backend]
    messages = build_message_list(prepared)

    body: Dict[str, Any] = {
        "model"    : cfg.model,
        "messages" : messages,

        # See max_tokens_param_name(). On the models that want the new name, the
        # budget also covers the invisible reasoning tokens, so a small limit can be
        # spent entirely on thinking (see warn_truncated_by_reasoning).
        max_tokens_param_name(provider, cfg.model): prepared["max_tokens"],
    }

    apply_sampling(body)

    # Aion, GLM and Kimi models get the shared thinking settings in their provider
    # dialect. EXTRA_BODY is merged afterwards, so an explicit override still wins.
    thinking_params = provider_thinking_params(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    if thinking_params is not None:
        body.update(thinking_params)

    body.update(provider["extra_body"])

    return body


def parse_usage(usage: Any) -> Dict[str, Any]:
    """
    Pulls the token counts the proxy tracks out of a /chat/completions usage payload.
    cached_tokens and cache_write_tokens are both subsets of prompt_tokens.
    """
    usage = usage if isinstance(usage, dict) else {}

    prompt_tokens     = max(0, int(usage.get("prompt_tokens"    , 0) or 0))
    completion_tokens = max(0, int(usage.get("completion_tokens", 0) or 0))

    prompt_details     = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    prompt_details     = prompt_details     if isinstance(prompt_details    , dict) else {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}

    cached_tokens = max(0, int(prompt_details.get("cached_tokens", 0) or 0))
    # Providers that charge a premium for cache writes report them; those without a
    # write fee simply never send this field.
    write_tokens = max(0, int(prompt_details.get("cache_write_tokens", 0) or 0))
    # Clamp both so an unexpected payload can never make uncached input go negative.
    cached_tokens = min(cached_tokens, prompt_tokens)
    write_tokens  = min(write_tokens, prompt_tokens - cached_tokens)

    return {
        "prompt"     : prompt_tokens,
        "completion" : completion_tokens,
        "total"      : max(0, int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)),
        "cached"     : cached_tokens,
        "write"      : write_tokens,
        "uncached"   : prompt_tokens - cached_tokens - write_tokens,
        "reasoning"  : reported_reasoning(completion_details),
    }


# Generation
def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming /chat/completions request.
    Same result shape as claude.generate_non_stream.
    """
    provider = cfg.openai_providers[cfg.backend]
    body     = build_body(prepared)

    print_payload(body)

    response = httpx.post(
        f"{provider['base_url']}/chat/completions",
        json=body,
        headers=request_headers(provider),
        timeout=request_timeout(),
    )
    if response.status_code != 200:
        raise error_from_response(cfg.backend, response)

    data = response.json()

    counts = parse_usage(data.get("usage"))
    print_usage(counts)

    choices       = data.get("choices") or [{}]
    message       = choices[0].get("message") or {}
    finish_reason = str(choices[0].get("finish_reason") or "stop")

    output_text    = str(message.get("content") or "")
    # DeepSeek-style APIs (GLM, ...) use reasoning_content; OpenRouter-style ones (Aion, ...) use reasoning.
    reasoning_text = str(message.get("reasoning_content") or message.get("reasoning") or "")

    warn_truncated_by_reasoning(finish_reason, output_text, counts)

    if cfg.auto_trim:
        output_text = trim_to_end_sentence(output_text)

    return {
        "id"            : str(data.get("id") or cfg.model),
        "stop_reason"   : finish_reason,
        "text"          : wrap_think(output_text, reasoning_text),
        "usage"         : usage_to_openai_dict(counts),
        "message_extra" : {},
    }


def generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming /chat/completions request, yielding the same backend-neutral
    events as claude.generate_stream. Provider SSE chunks are relayed nearly verbatim.

    Note: not every provider sends usage in the stream. Providers that support the
    option can enable it via EXTRA_BODY, e.g. {"stream_options": {"include_usage": true}}.
    Without usage the request is tracked as zero cost.
    """
    provider = cfg.openai_providers[cfg.backend]
    body     = build_body(prepared)
    body["stream"] = True

    print_payload(body)

    response_parts  : List[str] = []
    reasoning_parts : List[str] = []
    finish_reason = "stop"
    message_id    = ""
    usage         = None

    with httpx.Client(timeout=request_timeout()) as client:
        with client.stream(
            "POST",
            f"{provider['base_url']}/chat/completions",
            json=body,
            headers=request_headers(provider),
        ) as response:
            if response.status_code != 200:
                response.read()
                raise error_from_response(cfg.backend, response)

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try: chunk = json.loads(data_str)
                except Exception: continue
                if not isinstance(chunk, dict):
                    continue

                if chunk.get("usage") : usage      = chunk["usage"]
                if chunk.get("id")    : message_id = str(chunk["id"])

                choices = chunk.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]

                # Kimi sends the final-chunk usage inside the choice instead of at the
                # top level of the chunk (where stream_options puts it).
                if isinstance(choice.get("usage"), dict):
                    usage = choice["usage"]

                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

                delta = choice.get("delta") or {}

                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    yield ("reasoning", reasoning_delta)

                text_delta = delta.get("content")
                if text_delta:
                    response_parts.append(text_delta)
                    yield ("text", text_delta)

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
