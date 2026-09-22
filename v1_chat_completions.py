"""
The /chat/completions backend: the OpenAI-style endpoint every compatible provider implements.
GLM, Kimi, MiMo and Aion are served from here, and so is the NanoGPT aggregator (see nano_gpt).

OpenAI's own models are better served over /responses (see v1_responses), which returns reasoning.
So nothing here carries OpenAI model knowledge beyond the name of the token-limit field.
An OpenAI-compatible gateway declared here still works -- you just lose the reasoning.
<NAME>_EXTRA_BODY and <NAME>_MAX_TOKENS_PARAM are the escape hatches for whatever else it wants.
"""

import httpx
import json
import re

import nano_gpt

from packaging.version import Version
from typing            import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    THINK_EFFORT_ORDER,
    cfg,
    print_error,
    print_payload,
    print_usage,
    trim_to_end_sentence,
    usage_to_openai_dict,
)
from providers import (
    ProviderError,
    build_message_list,
    effort_params,
    error_from_response,
    is_openai_model,
    reported_reasoning,
    request_headers,
    request_timeout,
    warn_truncated_by_reasoning,
    wrap_think,
)


# Provider thinking dialects.
# Each maps the shared proxy thinking settings (on/off + effort) onto one provider's parameters.
# A dialect returns None for model ids it does not recognize.
# Models with no dialect fall back to the provider's EXTRA_BODY.
AION_MODEL_ID_RE = re.compile(r"^(?:aion-labs/)?aion-(\d+(?:\.\d+)?)")

# Aion accepts only low|medium|high for reasoning_effort.
# Fold the five shared proxy efforts onto that scale (xhigh and max round down to high).
AION_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

def aion_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the Aion (AionLabs) request dialect.
    Returns None when model_id is not a numbered aion model; aion-rp-* does not reason at all.

    Aion's only thinking parameter is reasoning_effort (none|low|medium|high, default medium).
    aion-2.0 is the sole model that takes it; the rest reject it with HTTP 400 and always reason.
    Those get an empty dialect: recognized, nothing to send.
    Effort was dropped after 2.0 rather than added, hence the exact version test.

    reasoning_split defaults to true, putting the thoughts in the separate 'reasoning' field.
    That is the field this proxy reads, so it is left alone.
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
    Maps the shared thinking settings onto the GLM request dialect.
    Returns None when model_id is not a GLM model; EXTRA_BODY is the escape hatch there.
    GLM models think by default, so 'thinking' is always sent explicitly.
    reasoning_effort exists from glm-5.2 on, and is assumed to stay for later models.
    Older GLM models get only the on/off switch.
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

# kimi-k3 accepts only low|high|max for reasoning_effort.
# Fold the five shared proxy efforts onto that scale (medium rounds down, xhigh rounds up).
KIMI_EFFORT_MAP = {"low": "low", "medium": "low", "high": "high", "xhigh": "max", "max": "max"}

def kimi_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the Kimi (Moonshot) request dialect.
    Returns None when model_id is not a kimi-k* model; kimi-latest/moonshot-v1 have no dialect.
    Control is model-dependent:
        kimi-k3+        thinking always on; depth via reasoning_effort (low|high|max, default max)
        kimi-k2.7-*     thinking always on; thinking.type "enabled" is mandatory
        kimi-k2.5/k2.6  thinking on by default; only an on/off switch, no effort control
    A disable request on an always-on model sends the closest thing the API offers.
    That is a minimal reasoning_effort on k3+, plain enabled on k2.7.
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


MIMO_MODEL_ID_RE = re.compile(r"^mimo-v(?!.*(?:asr|tts))")

def mimo_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the MiMo (Xiaomi) request dialect.
    Returns None for anything but a mimo chat model; asr and tts live on their own endpoints.

    The only knob is the thinking block (enabled|disabled), on by default, so it is sent explicitly.
    There is no depth control -- reasoning_effort and thinking_budget are both rejected.

    Thinking also pins temperature to 1.0 and top_p to 0.95, overriding them rather than refusing.
    SEND_TEMPERATURE and SEND_TOP_P are left alone; they just have no effect while it is on.
    """
    if MIMO_MODEL_ID_RE.match(model_id) is None:
        return None
    return {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}


# Atlas Cloud and DeepInfra take reasoning_effort for every model, not the vendors' own controls.
# What each level does varies by model, so these tables are measured, not documented (2026-09-22).
# Each row: model id pattern, the levels that make it think (weakest first), the level that stops it.
# "" means nothing stops it. Unlisted models get nothing; EXTRA_BODY is the escape hatch there.
# Their ids carry a vendor prefix, which keeps them apart from the vendors' own ids and each other:
# Atlas Cloud writes 'zai-org/glm-5.3', DeepInfra 'zai-org/GLM-5.3'.
ATLAS_THINKING = (
    # Always think: 'none' and 'medium' are refused, every other level thinks.
    (re.compile(r"^zai-org/glm-5\.3")       , ("low", "high", "xhigh", "max")            , ""),
    # 'low', 'medium' and 'max' are refused.
    (re.compile(r"^qwen/qwen3\.7-plus")     , ("high", "xhigh")                          , "none"),
    # Always think; minimax-m3 refuses 'none'.
    (re.compile(r"^moonshotai/kimi-k3")     , THINK_EFFORT_ORDER                         , ""),
    (re.compile(r"^minimaxai/minimax-m3")   , THINK_EFFORT_ORDER                         , ""),
    # Always thinks, whatever it is sent.
    (re.compile(r"^xiaomi/mimo-v")          , ()                                         , ""),
    # The rest behave: 'none' stops them, any level makes them think.
    (re.compile(r"^(?:deepseek-ai/deepseek-v4|qwen/qwen3\.8|zai-org/glm-5\.2|moonshotai/kimi-k2\.6|meituan-longcat/)"),
                                              THINK_EFFORT_ORDER                         , "none"),
)

# DeepInfra documents none|low|medium|high; stronger levels fold down to 'high'.
DEEPINFRA_EFFORTS = ("low", "medium", "high")

DEEPINFRA_THINKING = (
    # Always think, whatever they are sent.
    (re.compile(r"^(?:Qwen/Qwen3\.[78]-Max|stepfun-ai/Step-|meta-models/Muse-)"), DEEPINFRA_EFFORTS, ""),
    # The rest behave: 'none' stops them, any level makes them think.
    # DeepSeek and Hy3 do not think unless asked; the others do by default.
    (re.compile(r"^(?:XiaomiMiMo/MiMo-|zai-org/GLM-5|moonshotai/Kimi-K|deepseek-ai/DeepSeek-V(?:3\.2|4)"
                r"|Qwen/Qwen3\.(?:5|8)-|MiniMaxAI/MiniMax-M|tencent/Hy3|thinkingmachines/Inkling)"),
                                                                                  DEEPINFRA_EFFORTS, "none"),
)

def table_thinking_params(table: Tuple[Any, ...], model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """Maps the shared thinking settings onto one of the measured tables above."""
    for pattern, ladder, off in table:
        if pattern.match(model_id):
            return effort_params(ladder, off, thinking_enabled, thinking_effort)
    return None

def atlas_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    return table_thinking_params(ATLAS_THINKING, model_id, thinking_enabled, thinking_effort)

def deepinfra_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    return table_thinking_params(DEEPINFRA_THINKING, model_id, thinking_enabled, thinking_effort)


THINKING_DIALECTS = (aion_thinking_params, glm_thinking_params, kimi_thinking_params, mimo_thinking_params,
                     atlas_thinking_params, deepinfra_thinking_params)


def provider_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Provider-dialect thinking passthrough.
    Returns the params of the first dialect that recognizes model_id, or None when none matches.
    None means no passthrough, and EXTRA_BODY is the escape hatch.
    An empty dict means the model is recognized but offers no thinking controls at all.
    NanoGPT speaks one dialect for every model it serves, so it is asked instead.
    """
    if nano_gpt.is_active():
        return nano_gpt.thinking_params(model_id, thinking_enabled, thinking_effort)
    for dialect in THINKING_DIALECTS:
        params = dialect(model_id, thinking_enabled, thinking_effort)
        if params is not None:
            return params
    return None


def thinking_can_be_disabled(model_id: str) -> bool:
    """
    Whether the model's dialect can actually stop it from reasoning.
    Each spells "off" differently: a disabled thinking block (GLM, MiMo), an off effort (Aion),
    even an ordinary level (glm-5.3 on Atlas Cloud stops at 'low').
    So it can when "off" sends something other than the weakest "on" does.
    Models that keep reasoning regardless (kimi-k3) get the weakest setting the API offers.
    """
    off = provider_thinking_params(model_id, False, cfg.thinking_effort)
    return off != provider_thinking_params(model_id, True, THINK_EFFORT_ORDER[0])


def resolve_thinking() -> None:
    """
    Reports how the shared thinking settings map onto the selected model's dialect.
    Unlike the Anthropic backend there is no capability metadata to check, so nothing is adjusted.
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


def after_model_switch() -> None:
    """
    Post-switch hook for this backend (v1_messages and v1_responses have their own).
    There is nothing to validate against the model here, so this only reports how it lands.
    """
    resolve_thinking()


def print_think_status() -> None:
    """
    CLI 'think' status for this endpoint.
    v1_messages.print_think_status and v1_responses.print_think_status are the counterparts.
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
    The request field carrying the output token limit.
    Providers on this endpoint expect max_tokens.
    The exception is OpenAI's own catalogue: gpt-5+ and the o-series want max_completion_tokens.
    That is not a supported setup, but guessing wrong is an HTTP 400 with nothing to explain it.
    """
    configured = provider.get("max_tokens_param", "auto")
    if configured != "auto":
        return configured
    return "max_completion_tokens" if is_openai_model(model_id) else "max_tokens"


def apply_sampling(body: Dict[str, Any]) -> None:
    """
    Adds temperature/top_p in place.
    Every provider here accepts them; the models that refuse sampling are served by v1_responses.
    """
    if cfg.send_temperature : body["temperature"] = cfg.temperature
    if cfg.send_top_p       : body["top_p"      ] = cfg.top_p
    # top_k is not part of the OpenAI schema; providers that accept it can get it via EXTRA_BODY.


def build_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the chat completion request from a prepared chat request.
    The provider's EXTRA_BODY is merged in verbatim last.
    """
    provider = cfg.providers[cfg.backend]
    messages = build_message_list(prepared)

    body: Dict[str, Any] = {
        "model"    : cfg.model,
        "messages" : messages,

        # See max_tokens_param_name().
        # On the models that want the new name the budget also covers invisible reasoning tokens.
        # A small limit can then be spent entirely on thinking (see warn_truncated_by_reasoning).
        max_tokens_param_name(provider, cfg.model): prepared["max_tokens"],
    }

    apply_sampling(body)

    # Aion, GLM, Kimi and MiMo models get the shared thinking settings in their provider dialect.
    # EXTRA_BODY is merged afterwards, so an explicit override still wins.
    thinking_params = provider_thinking_params(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    if thinking_params is not None:
        body.update(thinking_params)

    if nano_gpt.is_active():
        body.update(nano_gpt.route_params(cfg.model))

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
    # Providers that charge a premium for cache writes report them; the rest never send it.
    write_tokens = max(0, int(prompt_details.get("cache_write_tokens", 0) or 0))
    # Clamp both so an unexpected payload can never make uncached input go negative.
    cached_tokens = min(cached_tokens, prompt_tokens)
    write_tokens  = min(write_tokens, prompt_tokens - cached_tokens)

    # NanoGPT and DeepInfra report what they billed; None when a provider does not.
    raw_cost      = usage.get("cost", usage.get("estimated_cost"))
    reported_cost = None if raw_cost is None else max(0.0, float(raw_cost))

    return {
        "prompt"        : prompt_tokens,
        "completion"    : completion_tokens,
        "total"         : max(0, int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)),
        "cached"        : cached_tokens,
        # One cache rate and no TTL choice here, so every write is a 5m write.
        # providers.apply_model prices both buckets identically.
        "write_1h"      : 0,
        "write_5m"      : write_tokens,
        "uncached"      : prompt_tokens - cached_tokens - write_tokens,
        "reasoning"     : reported_reasoning(completion_details),
        "reported_cost" : reported_cost,
    }


def distrust_zero_reasoning(counts: Dict[str, Any], reasoning_text: str) -> None:
    """
    Some hosts report 0 reasoning tokens while returning reasoning (Kimi-K3, MiMo on some).
    A zero beside reasoning text is not a measurement, so it is taken as unreported.
    The usage report then shows plain output tokens rather than a split that is wrong.
    """
    if counts["reasoning"] == 0 and reasoning_text.strip():
        counts["reasoning"] = None


# Generation
def failed(error: ProviderError) -> ProviderError:
    """Notes a failed request against a pinned NanoGPT provider, then hands the error back."""
    if nano_gpt.is_active():
        nano_gpt.note_error(error)
    return error


def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming /chat/completions request.
    Same result shape as v1_messages.generate_non_stream.
    """
    provider = cfg.providers[cfg.backend]
    body     = build_body(prepared)

    print_payload(body)

    response = httpx.post(
        f"{provider['base_url']}/chat/completions",
        json=body,
        headers=request_headers(provider),
        timeout=request_timeout(),
    )
    if response.status_code != 200:
        raise failed(error_from_response(cfg.backend, response))
    if nano_gpt.is_active():
        nano_gpt.note_alive()

    data = response.json()

    choices       = data.get("choices") or [{}]
    message       = choices[0].get("message") or {}
    finish_reason = str(choices[0].get("finish_reason") or "stop")

    output_text    = str(message.get("content") or "")
    # DeepSeek-style APIs (GLM) use reasoning_content; OpenRouter-style ones (Aion) use reasoning.
    reasoning_text = str(message.get("reasoning_content") or message.get("reasoning") or "")

    counts = parse_usage(data.get("usage"))
    distrust_zero_reasoning(counts, reasoning_text)
    print_usage(counts)

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
    Runs one streaming /chat/completions request, yielding the same events as v1_messages does.
    Provider SSE chunks are relayed nearly verbatim.

    Note: not every provider sends usage in the stream.
    EXTRA_BODY can enable it where supported: {"stream_options": {"include_usage": true}}.
    Without usage the request is tracked as zero cost.
    """
    provider = cfg.providers[cfg.backend]
    body     = build_body(prepared)
    body["stream"] = True
    # Without it NanoGPT streams no usage, and the turn would be tracked as free.
    if nano_gpt.is_active():
        body["stream_options"] = {"include_usage": True}

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
                raise failed(error_from_response(cfg.backend, response))
            if nano_gpt.is_active():
                nano_gpt.note_alive()

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

                # A stream that fails midway ends with an error frame, not an HTTP status.
                # NanoGPT sends one; without this the reply would just stop, as if finished.
                # The text so far is kept and the error appended to it, so the reply shows both.
                # The finish reason stays as it was; the frame's own 'error' is not an OpenAI value.
                error_obj = chunk.get("error")
                if isinstance(error_obj, dict):
                    message = str(error_obj.get("message") or "no reason given")
                    status  = int(error_obj.get("status") or 500)
                    print_error(failed(ProviderError(status, {"error": error_obj}, f"{cfg.backend}: {message}")))

                    note = f"\n\n[Stream failed: {message}]"
                    response_parts.append(note)
                    yield ("text", note)
                    break

                if chunk.get("usage") : usage      = chunk["usage"]
                if chunk.get("id")    : message_id = str(chunk["id"])

                choices = chunk.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]

                # Kimi sends the final-chunk usage inside the choice, not at the chunk top level.
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
    distrust_zero_reasoning(counts, "".join(reasoning_parts))
    print_usage(counts)
    warn_truncated_by_reasoning(finish_reason, snapshot_text, counts)

    yield ("final", {
        "id"                 : message_id or cfg.model,
        "stop_reason"        : finish_reason,
        "usage"              : usage_to_openai_dict(counts),
        "snapshot_text"      : snapshot_text,
        "snapshot_reasoning" : "".join(reasoning_parts),
    })
