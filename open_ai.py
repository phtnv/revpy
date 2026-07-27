import httpx
import json
import re
import threading

from concurrent.futures import ThreadPoolExecutor
from packaging.version  import Version
from typing             import Any, Dict, Iterator, List, Optional, Tuple

from common import (
    THINK_EFFORT_ORDER,
    append_prefill_instruction_to_last_user_message,
    cfg,
    deep_get,
    resolve_api_key,
    track_usage,
    trim_to_end_sentence,
)


def fold_effort(effort: str, ladder: Tuple[str, ...]) -> str:
    """
    Folds a shared proxy effort onto a provider's own ladder by walking THINK_EFFORT_ORDER
    downwards from the requested level and taking the first level the provider supports.
    So 'max' becomes 'xhigh' on a ladder that stops at xhigh, and 'high' on one that stops
    at high. `ladder` must be ordered weakest first and hold only thinking levels.
    """
    try: start = THINK_EFFORT_ORDER.index(effort)
    except ValueError: start = THINK_EFFORT_ORDER.index("medium")

    for name in reversed(THINK_EFFORT_ORDER[: start + 1]):
        if name in ladder:
            return name
    # The request is weaker than anything the model offers; give it the weakest level.
    return ladder[0]


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


OPENAI_MODEL_RE       = re.compile(r"^(?:gpt-|o\d+(?:-|$)|chat-latest$)")
OPENAI_CHAT_LATEST_RE = re.compile(r"(?:^|-)chat-latest$")
OPENAI_GPT_RE         = re.compile(r"^gpt-(\d+(?:\.\d+)?)")
OPENAI_O_SERIES_RE    = re.compile(r"^o(\d+)(?:-|$)")

# Reasoning effort levels each model accepts, weakest first. Established by sending each
# level to each model: the endpoints' own "supported values" error lists the union across
# models, not what the addressed model takes, so it cannot be trusted on its own.
# The two endpoints disagree in both directions -- /responses adds 'max', but only on
# gpt-5.6+, and drops 'xhigh' from the o-series -- so the ladder is picked per endpoint.
OPENAI_EFFORTS_GPT52      = ("none", "low", "medium", "high", "xhigh")         # gpt-5.2..5.5, both
OPENAI_EFFORTS_GPT56_RESP = ("none", "low", "medium", "high", "xhigh", "max")  # gpt-5.6+, responses
OPENAI_EFFORTS_GPT51      = ("none", "low", "medium", "high")                  # gpt-5.1, both
OPENAI_EFFORTS_GPT50      = ("minimal", "low", "medium", "high")               # gpt-5, both
OPENAI_EFFORTS_O          = ("low", "medium", "high", "xhigh")                 # o-series, chat
OPENAI_EFFORTS_O_RESP     = ("low", "medium", "high")                          # o-series, responses

# Effort levels that mean "do not reason", weakest first (Aion spells its off switch
# 'none' too). Not thinking depths, so they are excluded when folding an effort and
# are used only to answer a disable request.
OFF_EFFORTS = ("none", "minimal")


def is_openai_model(model_id: str) -> bool:
    """
    True for OpenAI's own model ids (gpt-*, o-series, chat-latest). Used for the
    request-shape rules that hold across the whole OpenAI catalogue rather than
    just its reasoning models.
    """
    return OPENAI_MODEL_RE.match(model_id) is not None


def provider_api_style(backend: Optional[str] = None) -> str:
    """
    Which endpoint the active (or named) provider speaks: 'chat' or 'responses'.
    Falls back to 'chat' for the Anthropic backend and unknown provider names, so
    callers can ask about any model without guarding.
    """
    provider = cfg.openai_providers.get(backend or cfg.backend) or {}
    return provider.get("api", "chat")


def openai_effort_ladder(model_id: str, api_style: Optional[str] = None) -> Optional[Tuple[str, ...]]:
    """
    The reasoning effort ladder for an OpenAI model on one endpoint, weakest first.
    Returns None when reasoning effort is not a parameter of the model at all
    (gpt-4 and older reject it as an unknown argument, and so does every non-OpenAI
    model), and an empty tuple for the *-chat-latest snapshots, which accept only
    the default and never reason.
    """
    if api_style is None:
        api_style = provider_api_style()
    responses = api_style == "responses"

    if OPENAI_CHAT_LATEST_RE.search(model_id):
        return ()
    if OPENAI_O_SERIES_RE.match(model_id) is not None:
        return OPENAI_EFFORTS_O_RESP if responses else OPENAI_EFFORTS_O

    match = OPENAI_GPT_RE.match(model_id)
    if match is None:
        return None

    version = Version(match.group(1))
    if version < Version("5")   : return None
    if version < Version("5.1") : return OPENAI_EFFORTS_GPT50
    if version < Version("5.2") : return OPENAI_EFFORTS_GPT51
    if version < Version("5.6") : return OPENAI_EFFORTS_GPT52
    return OPENAI_EFFORTS_GPT56_RESP if responses else OPENAI_EFFORTS_GPT52


def openai_effort_for(model_id: str, thinking_enabled: bool, thinking_effort: str,
                      api_style: Optional[str] = None) -> Optional[str]:
    """
    The effort level to send for one model on one endpoint, or None when the model
    takes no effort parameter at all.

    How far thinking can be turned down is model-dependent: gpt-5.1+ take 'none' and
    stop reasoning outright, gpt-5 bottoms out at 'minimal', and the o-series has no
    off switch at all, so a disable request sends the weakest level it does offer.
    """
    ladder = openai_effort_ladder(model_id, api_style)
    if not ladder:
        return None
    if not thinking_enabled:
        return next((off for off in OFF_EFFORTS if off in ladder), ladder[0])
    return fold_effort(thinking_effort, tuple(e for e in ladder if e not in OFF_EFFORTS))


def openai_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto the OpenAI /chat/completions dialect, and
    doubles as the source for the CLI thinking status (the effort is resolved against
    whichever endpoint the provider is set to). Returns None for models with no
    reasoning effort parameter (gpt-4 and older); EXTRA_BODY is the escape hatch there.

    Note that /chat/completions never returns the reasoning text, only its token count.
    Seeing the thoughts requires <NAME>_API=responses; see build_responses_body().
    """
    ladder = openai_effort_ladder(model_id)
    if ladder is None : return None
    if not ladder     : return {}

    return {"reasoning_effort": openai_effort_for(model_id, thinking_enabled, thinking_effort)}


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


def responses_reasoning_param(model_id: str, thinking_enabled: bool, thinking_effort: str,
                              summary: str) -> Optional[Dict[str, Any]]:
    """
    The /responses 'reasoning' object: the effort plus, when thinking is on, the
    summary detail level that makes the model's reasoning visible at all. Returns
    None for models that take no reasoning parameter, which reject it outright.
    """
    effort = openai_effort_for(model_id, thinking_enabled, thinking_effort, "responses")
    if effort is None:
        return None

    reasoning: Dict[str, Any] = {"effort": effort}
    if thinking_enabled and summary != "none" and model_id not in SUMMARY_BLOCKED_MODELS:
        reasoning["summary"] = summary
    return reasoning


def openai_supports_sampling(model_id: str) -> bool:
    """
    False for OpenAI models that reject sampling controls: every reasoning model and
    every *-chat-latest snapshot refuses top_p outright and accepts no temperature
    other than the default 1. Non-OpenAI models are unaffected.
    """
    return not (is_openai_model(model_id) and openai_effort_ladder(model_id) is not None)


def max_tokens_param_name(provider: Dict[str, Any], model_id: str) -> str:
    """
    The request field carrying the output token limit. OpenAI retired max_tokens:
    gpt-5+ and the o-series reject it and demand max_completion_tokens, which the
    older OpenAI chat models accept too. Other providers still expect max_tokens.
    """
    configured = provider.get("max_tokens_param", "auto")
    if configured != "auto":
        return configured
    return "max_completion_tokens" if is_openai_model(model_id) else "max_tokens"


THINKING_DIALECTS = (aion_thinking_params, glm_thinking_params, kimi_thinking_params, openai_thinking_params)


def provider_thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Provider-dialect thinking passthrough for OpenAI-style backends. Returns the
    params of the first dialect that recognizes model_id, or None when no dialect
    matches (no passthrough; EXTRA_BODY is the escape hatch). An empty dict means
    the model is recognized but offers no thinking controls at all.
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
    level (Aion, OpenAI). Models that keep reasoning regardless (kimi-k3, the
    o-series) get the weakest setting the API offers instead of a real off switch.
    """
    off = provider_thinking_params(model_id, False, cfg.thinking_effort) or {}
    return deep_get(off, "thinking.type") == "disabled" or off.get("reasoning_effort") in OFF_EFFORTS


def resolve_thinking() -> None:
    """
    OpenAI-style counterpart of claude.resolve_thinking(): there is no Anthropic
    capability metadata to check and none of the Anthropic parameter constraints
    apply, so this only reports how the shared thinking settings map onto the
    selected provider model's dialect (if any).
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
    CLI 'think' status for OpenAI-style backends
    (claude.print_think_status() is the Anthropic counterpart).
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


# Aggregated model list across every configured OpenAI-style provider.
# Each entry: {"id", "provider"} plus whatever the provider's /models returned.
OPENAI_MODELS : List[Dict[str, Any]] = []
MODEL_LOCK                           = threading.Lock()


class OpenAIBackendError(Exception):
    """
    Provider HTTP error. Carries status_code and a response body dict in the
    same attribute shape the Anthropic SDK errors use, so server.build_error_body
    and claude.anthropic_error_body handle it without special cases.
    """
    def __init__(self, status_code: int, body: Dict[str, Any], message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body        = body


def error_from_response(provider_name: str, response: Any) -> OpenAIBackendError:
    status_code = int(getattr(response, "status_code", 500) or 500)

    body: Dict[str, Any] = {}
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        pass

    if not body:
        text = ""
        try: text = str(response.text or "")[:2000]
        except Exception: pass
        body = {"error": {"message": text or f"HTTP {status_code}"}}

    error_obj = body.get("error")
    if isinstance(error_obj, dict) and error_obj.get("message"):
        message = str(error_obj["message"])
    else:
        message = json.dumps(body, ensure_ascii=False, default=str)

    return OpenAIBackendError(status_code, body, f"{provider_name}: {message}")


def fetch_provider_models(name: str, provider: Dict[str, Any], timeout_s: float) -> List[Dict[str, Any]]:
    """
    Fetches one provider's /models list. Runs in a worker thread during refresh;
    failures raise and are reported by the caller.
    """
    headers = {}
    if provider["api_key"]:
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    response = httpx.get(f"{provider['base_url']}/models", headers=headers, timeout=timeout_s)
    if response.status_code != 200:
        raise error_from_response(name, response)

    data = response.json()
    # OpenAI-style APIs return {"data": [...]}, but not everyone follows the
    # spec: Aion returns {"models": [...]}, and some providers a bare list.
    if   isinstance(data, dict)  : entries = data.get("data") or data.get("models")
    elif isinstance(data, list)  : entries = data
    else                         : entries = None

    # Some providers serve their whole catalogue here, chat models and all
    # (OpenAI's /models also lists tts, image, embedding and realtime models).
    # <NAME>_MODELS_REGEX keeps the CLI list down to the ones worth selecting.
    models_regex = provider["models_regex"]

    got = []
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("id"):
            model_id = str(entry["id"])
            if models_regex is not None and not models_regex.search(model_id):
                continue
            got.append({**entry, "id": model_id, "provider": name})
    return got


def refresh_openai_models(timeout_s: float) -> None:
    """
    Fetches the model list of every configured provider and stores them for CLI use.

    Providers with a <NAME>_MODELS override skip the /models request entirely.
    A failing provider is skipped with a warning; it does not block the others.
    The requests run in parallel, but results are collected in OPENAI_PROVIDERS
    declaration order, so the aggregated list (and the CLI numbering) does not
    depend on response order.
    """
    global OPENAI_MODELS

    models: List[Dict[str, Any]] = []

    if cfg.openai_providers:
        with ThreadPoolExecutor(max_workers=len(cfg.openai_providers)) as pool:
            fetches = [
                (name, provider, None if provider["models"] else pool.submit(fetch_provider_models, name, provider, timeout_s))
                for name, provider in cfg.openai_providers.items()
            ]

            for name, provider, future in fetches:
                if future is None:
                    models.extend({"id": model_id, "provider": name} for model_id in provider["models"])
                    print(f"Using {len(provider['models'])} configured model(s) for provider '{name}'.")
                    continue
                try:
                    got = future.result()
                    models.extend(got)
                    print(f"Retrieved {len(got)} model(s) from provider '{name}'.")
                except Exception as exc:
                    print(f"WARNING: Could not retrieve a model list from provider '{name}'. {exc}")

    with MODEL_LOCK:
        OPENAI_MODELS = models


def print_model_list(number_offset: int = 0) -> None:
    """
    Prints the aggregated provider model list, numbered after the Anthropic list.
    """
    with MODEL_LOCK:
        models = list(OPENAI_MODELS)
    if not models:
        if cfg.openai_providers:
            print("No OpenAI-style provider models available.")
        return

    number_width = len(str(number_offset + len(models)))

    for index, entry in enumerate(models, start=number_offset + 1):
        selected    = (cfg.backend == entry["provider"]) and (cfg.model == entry["id"])
        number      = str(index).rjust(number_width)
        number_cell = f"[{number}]" if selected else f" {number} "

        print(f"{number_cell}  {entry['id']:<42}  {entry['provider']}")


def select_model_by_number(index: int) -> None:
    with MODEL_LOCK:
        if not OPENAI_MODELS:
            print("No OpenAI-style provider models available.")
            return
        if index < 1 or index > len(OPENAI_MODELS):
            print(f"Model number out of range [1:{len(OPENAI_MODELS)}].")
            return
        entry = OPENAI_MODELS[index - 1]
    apply_openai_model(entry)


def print_model_info(index: int) -> None:
    with MODEL_LOCK:
        if not OPENAI_MODELS:
            print("No OpenAI-style provider models available.")
            return
        if index < 1 or index > len(OPENAI_MODELS):
            print(f"Model number out of range. Use 1 through {len(OPENAI_MODELS)}.")
            return
        entry = dict(OPENAI_MODELS[index - 1])

    print(json.dumps(entry, indent=2, ensure_ascii=False, default=str))


def apply_openai_model(entry: Dict[str, Any]) -> None:
    provider = cfg.openai_providers[entry["provider"]]

    print(f"=== Switching to {entry['provider']}/{entry['id']} ===")
    cfg.backend    = entry["provider"]
    cfg.model      = entry["id"]
    cfg.info       = dict(entry)
    cfg.model_info = dict(entry)
    cfg.version    = Version("0.0")

    # Per-model cost family when one matches, provider-level costs otherwise.
    cost_source = provider
    cost_family = entry["provider"]
    for family in provider["cost_families"]:
        if family["regex"].search(entry["id"]):
            cost_source = family
            cost_family = f"{entry['provider']}:{family['name']}"
            break

    print(f"Using cost family '{cost_family}'.")
    cfg.model_cost_family     = cost_family
    cfg.input_token_cost_usd  = cost_source["input_cost"]
    cfg.output_token_cost_usd = cost_source["output_cost"]
    cfg.cache_read_cost_usd   = cost_source["cache_read_cost"]
    # OpenAI-style providers cache automatically, with a single rate and no TTL choice,
    # so both write buckets get the same price. It defaults to the input cost, which
    # nets stray write tokens to zero for the providers that do not charge for writes.
    cfg.cache_write_5m_cost_usd = cost_source["cache_write_cost"]
    cfg.cache_write_1h_cost_usd = cost_source["cache_write_cost"]
    resolve_thinking()
    print(f"=== Switching to {entry['provider']}/{entry['id']} complete ===")


def apply_model_by_id(model_id: str) -> bool:
    """
    Applies a provider model matching either "model-id" or "provider/model-id".
    Returns False quietly when nothing matches (the caller falls back to Anthropic).
    """
    with MODEL_LOCK:
        models = list(OPENAI_MODELS)

    for entry in models:
        if model_id in (entry["id"], f"{entry['provider']}/{entry['id']}"):
            apply_openai_model(entry)
            return True
    return False


def request_headers(provider: Dict[str, Any]) -> Dict[str, str]:
    key = resolve_api_key(provider["api_key"], provider["api_key_name"])
    return {"Authorization": f"Bearer {key}"}


def request_timeout() -> httpx.Timeout:
    return httpx.Timeout(cfg.openai_request_timeout_seconds, connect=10.0)


def print_payload(body: Dict[str, Any]) -> None:
    if not cfg.debug_log:
        return
    print()
    print(f"=== {cfg.backend} payload start ===")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print(f"=== {cfg.backend} payload end ===")


def build_message_list(prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turns a prepared chat request into the role/content message list both OpenAI
    endpoints take (/chat/completions calls it 'messages', /responses 'input').

    Since the frontend already speaks OpenAI format this is a near-passthrough:
    system segments become one leading system message, and the moved lorebook suffix
    becomes a trailing system message (OpenAI-style APIs allow system anywhere).
    """
    messages: List[Dict[str, Any]] = []

    system_parts = [segment.strip() for segment in prepared["system_segments"] if segment.strip()]
    if prepared["system_summary_text"].strip():
        system_parts.append(prepared["system_summary_text"].strip())
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for msg in prepared["messages"]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    if prepared["lorebook_at_end_text"]:
        messages.append({"role": "system", "content": prepared["lorebook_at_end_text"].strip()})

    if cfg.assistant_prefill.strip() and cfg.assistant_prefill_mode != "none":
        if cfg.assistant_prefill_mode == "instruction":
            append_prefill_instruction_to_last_user_message(messages, cfg.assistant_prefill)
        elif cfg.assistant_prefill_mode == "assistant":
            # Trailing-assistant behavior varies wildly between OpenAI-style providers
            # (continue vs. new turn vs. error), so only instruction mode is supported.
            print("WARNING: assistant prefill mode is not supported for OpenAI-style backends. Use 'prefill instruction'.")

    return messages


def apply_sampling(body: Dict[str, Any]) -> None:
    """
    Adds temperature/top_p in place, unless the model refuses them. OpenAI reasoning
    models take no sampling controls at all: top_p is rejected outright, and
    temperature accepts nothing but its default of 1.
    """
    if openai_supports_sampling(cfg.model):
        if cfg.send_temperature : body["temperature"] = cfg.temperature
        if cfg.send_top_p       : body["top_p"      ] = cfg.top_p
    elif cfg.send_temperature or cfg.send_top_p:
        print(f"WARNING: '{cfg.model}' does not support temperature/top_p. Sending the request without them.")
    # top_k is not part of either OpenAI schema; providers that accept it can get it via EXTRA_BODY.


def build_responses_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds an OpenAI /responses request. This endpoint exists here for one reason:
    it is the only one that returns the model's reasoning. Ask for a summary and the
    response carries reasoning items whose text the proxy wraps in a <think> block,
    which /chat/completions cannot do at all.

    Note the model decides whether to reason: on adaptive models an easy turn can
    come back with no reasoning at all, and then there is no summary to show.
    """
    provider = cfg.openai_providers[cfg.backend]

    body: Dict[str, Any] = {
        "model"             : cfg.model,
        "input"             : build_message_list(prepared),
        "max_output_tokens" : prepared["max_tokens"],
        # /responses retains responses for 30 days unless told otherwise.
        "store"             : provider["store"],
    }

    apply_sampling(body)

    reasoning = responses_reasoning_param(cfg.model, cfg.thinking_enabled, cfg.thinking_effort,
                                          provider["reasoning_summary"])
    if reasoning is not None:
        body["reasoning"] = reasoning

    body.update(provider["extra_body"])

    return body


def build_openai_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the OpenAI-style chat completion request from a prepared chat request.
    The provider's EXTRA_BODY is merged in verbatim last.
    """
    provider = cfg.openai_providers[cfg.backend]
    messages = build_message_list(prepared)

    body: Dict[str, Any] = {
        "model"    : cfg.model,
        "messages" : messages,

        # OpenAI renamed this field; see max_tokens_param_name(). On the models that
        # want the new name, the budget also covers the invisible reasoning tokens,
        # so a small limit can be spent entirely on thinking (see generate_non_stream).
        max_tokens_param_name(provider, cfg.model): prepared["max_tokens"],
    }

    apply_sampling(body)

    # GPT, Aion, GLM and Kimi models get the shared thinking settings in their provider
    # dialect. EXTRA_BODY is merged afterwards, so an explicit override still wins.
    thinking_params = provider_thinking_params(cfg.model, cfg.thinking_enabled, cfg.thinking_effort)
    if thinking_params is not None:
        body.update(thinking_params)

    body.update(provider["extra_body"])

    return body


def reported_reasoning(details: Dict[str, Any]) -> Optional[int]:
    """
    The reasoning token count, or None when the provider does not report one.
    OpenAI, GLM and Kimi all send completion_tokens_details.reasoning_tokens, OpenAI
    even when it is 0. Aion omits the details object entirely, and None keeps the
    proxy from reporting its thinking as zero. Where the count does appear it is a
    subset of the output tokens, which is how the usage report splits them.
    """
    raw = details.get("reasoning_tokens")
    return None if raw is None else max(0, int(raw or 0))


def parse_usage(usage: Any) -> Dict[str, Any]:
    """
    Pulls the token counts the proxy tracks out of an OpenAI-style usage payload.
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
    # gpt-5.6+ reports the tokens written to the prompt cache, which it charges a
    # premium for. Providers without a write fee simply never send this field.
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


def parse_responses_usage(usage: Any) -> Dict[str, Any]:
    """
    The /responses counterpart of parse_usage(). Same counts, different field names:
    input/output rather than prompt/completion, and the details objects renamed to match.
    """
    usage = usage if isinstance(usage, dict) else {}

    input_tokens  = max(0, int(usage.get("input_tokens" , 0) or 0))
    output_tokens = max(0, int(usage.get("output_tokens", 0) or 0))

    input_details  = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    input_details  = input_details  if isinstance(input_details , dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}

    cached_tokens = max(0, int(input_details.get("cached_tokens", 0) or 0))
    write_tokens  = max(0, int(input_details.get("cache_write_tokens", 0) or 0))
    cached_tokens = min(cached_tokens, input_tokens)
    write_tokens  = min(write_tokens, input_tokens - cached_tokens)

    return {
        "prompt"     : input_tokens,
        "completion" : output_tokens,
        "total"      : max(0, int(usage.get("total_tokens", input_tokens + output_tokens) or 0)),
        "cached"     : cached_tokens,
        "write"      : write_tokens,
        "uncached"   : input_tokens - cached_tokens - write_tokens,
        "reasoning"  : reported_reasoning(output_details),
    }


def usage_to_cost_tokens(counts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps parsed usage counts to the normalized dict common.track_usage() expects.
    OpenAI-style caching has one rate and no TTL choice, so writes all land in the
    5m bucket (apply_openai_model prices both buckets identically).
    """
    return {
        "uncached_input" : counts["uncached"],
        "cache_read"     : counts["cached"],
        "cache_write_1h" : 0,
        "cache_write_5m" : counts["write"],
        "output"         : counts["completion"],
        # Splits the output line into thinking and visible text; None when unreported.
        "reasoning"      : counts["reasoning"],
    }


def print_usage(counts: Dict[str, Any]) -> None:
    track_usage(usage_to_cost_tokens(counts))


def usage_to_openai_dict(counts: Dict[str, Any]) -> Dict[str, int]:
    """
    Normalizes parsed usage counts to the same shape the Claude backend emits, so
    clients see one consistent usage format regardless of backend.
    """
    return {
        "prompt_tokens"               : counts["prompt"],
        "completion_tokens"           : counts["completion"],
        "total_tokens"                : counts["total"],
        "input_tokens_uncached"       : counts["uncached"],
        "cache_creation_input_tokens" : counts["write"],
        "cache_read_input_tokens"     : counts["cached"],
    }


def warn_truncated_by_reasoning(finish_reason: str, output_text: str, counts: Dict[str, Any]) -> None:
    """
    On OpenAI reasoning models the output limit also covers the invisible reasoning
    tokens, so a budget that is small next to the effort can be spent entirely on
    thinking, ending the request with no text at all. Janitor supplies its own
    max_tokens, so this is easy to hit with nothing on screen to explain it.
    """
    if finish_reason != "length" or output_text.strip() or not counts["reasoning"]:
        return

    print(f"WARNING: '{cfg.model}' spent its entire output budget ({counts['completion']} tokens) on reasoning and returned no text.")
    print("         Raise max_tokens in the client (or MAX_TOKENS in .env), or lower the thinking effort.")


def wrap_think(output_text: str, reasoning_text: str) -> str:
    """
    Prepends the model's reasoning as a <think> block, which is how Janitor and
    similar clients render it.
    """
    if not reasoning_text.strip():
        return output_text
    return f"<think>\n{reasoning_text.strip()}\n</think>\n\n" + output_text


# Generation
def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming completion against whichever endpoint the provider speaks.
    Same result shape as claude.generate_non_stream.
    """
    if provider_api_style() == "responses":
        return responses_generate_non_stream(prepared)
    return chat_generate_non_stream(prepared)


def generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming completion against whichever endpoint the provider speaks,
    yielding the same backend-neutral events as claude.generate_stream.
    """
    if provider_api_style() == "responses":
        return responses_generate_stream(prepared)
    return chat_generate_stream(prepared)


def chat_generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming /chat/completions request.
    """
    provider = cfg.openai_providers[cfg.backend]
    body     = build_openai_body(prepared)

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


def chat_generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming /chat/completions request, yielding the same backend-neutral
    events as claude.generate_stream. Provider SSE chunks are relayed nearly verbatim.

    Note: not every provider sends usage in the stream. OpenAI needs to be asked for
    it, which is done below; others that support the same option can enable it via
    EXTRA_BODY, e.g. {"stream_options": {"include_usage": true}}. Without usage the
    request is tracked as zero cost.
    """
    provider = cfg.openai_providers[cfg.backend]
    body     = build_openai_body(prepared)
    body["stream"] = True

    # OpenAI omits usage from streamed responses unless asked. EXTRA_BODY is merged
    # before this point, so an explicit setting there still wins.
    if is_openai_model(cfg.model):
        body.setdefault("stream_options", {"include_usage": True})

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
                # top level of the chunk (where OpenAI's stream_options puts it).
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


# /responses generation.
# The response is an 'output' list of items rather than a single message: reasoning items
# carry the summary of the model's thinking, message items carry the reply. A turn can
# produce several of each, and on adaptive models the reasoning items may be absent.
RESPONSES_STOP_REASONS = {"completed": "stop", "incomplete": "length", "failed": "error"}


def responses_output_text(data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pulls (reply text, reasoning text) out of a non-streaming /responses body.
    Multiple summary parts are joined with a blank line, the same way the streaming
    path separates them.
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


def responses_generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming /responses request.
    """
    provider = cfg.openai_providers[cfg.backend]

    def send() -> Any:
        body = build_responses_body(prepared)
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
    counts = parse_responses_usage(data.get("usage"))
    print_usage(counts)

    output_text, reasoning_text = responses_output_text(data)
    finish_reason = RESPONSES_STOP_REASONS.get(str(data.get("status") or ""), "stop")

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


def responses_generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming /responses request. Unlike the chat stream, which relays deltas
    off one message, this is a typed event stream: reasoning summary text and reply
    text arrive as separate event types, which is exactly the split the proxy needs.
    """
    provider = cfg.openai_providers[cfg.backend]

    response_parts  : List[str] = []
    reasoning_parts : List[str] = []
    finish_reason = "stop"
    message_id    = ""
    usage         = None

    # Two attempts at most: an unverified org rejects the summary request before any
    # event is streamed, and note_summary_blocked drops it so the retry succeeds.
    for attempt in (0, 1):
        body = build_responses_body(prepared)
        body["stream"] = True
        print_payload(body)

        retry_without_summary = False

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
                        raise OpenAIBackendError(500, {"error": {"message": str(message)}}, f"{cfg.backend}: {message}")

                    if event_type.startswith("response.") and isinstance(event.get("response"), dict):
                        finished = event["response"]
                        if finished.get("id")    : message_id = str(finished["id"])
                        if finished.get("usage") : usage      = finished["usage"]
                        if finished.get("status"):
                            finish_reason = RESPONSES_STOP_REASONS.get(str(finished["status"]), finish_reason)

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

        break

    counts        = parse_responses_usage(usage)
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
