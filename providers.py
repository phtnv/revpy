"""
Shared plumbing for the OpenAI-style backends.

Everything here is common to both OpenAI-style wire protocols -- /chat/completions
and /responses -- and belongs to neither: the provider registry and its model list,
the HTTP transport, the request message list, and the usage normalization the cost
tracker consumes. The wire modules import from here; this module imports from none
of them, so a provider can never depend on the endpoint that happens to be selected.

Anthropic does not pass through here at all; it speaks its own protocol with its own
SDK and keeps its own model list.
"""

import httpx
import json
import threading

from concurrent.futures import ThreadPoolExecutor
from packaging.version  import Version
from typing             import Any, Dict, List, Optional, Tuple

from common import (
    THINK_EFFORT_ORDER,
    append_prefill_instruction_to_last_user_message,
    cfg,
    resolve_api_key,
    track_usage,
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


# Aggregated model list across every configured OpenAI-style provider.
# Each entry: {"id", "provider"} plus whatever the provider's /models returned.
MODELS : List[Dict[str, Any]] = []
MODEL_LOCK                    = threading.Lock()


class ProviderError(Exception):
    """
    Provider HTTP error. Carries status_code and a response body dict in the
    same attribute shape the Anthropic SDK errors use, so server.build_error_body
    and common.error_body handle it without special cases.
    """
    def __init__(self, status_code: int, body: Dict[str, Any], message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body        = body


def error_from_response(provider_name: str, response: Any) -> ProviderError:
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

    return ProviderError(status_code, body, f"{provider_name}: {message}")


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


def refresh_models(timeout_s: float) -> None:
    """
    Fetches the model list of every configured provider and stores them for CLI use.

    Providers with a <NAME>_MODELS override skip the /models request entirely.
    A failing provider is skipped with a warning; it does not block the others.
    The requests run in parallel, but results are collected in OPENAI_PROVIDERS
    declaration order, so the aggregated list (and the CLI numbering) does not
    depend on response order.
    """
    global MODELS

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
        MODELS = models


def print_model_list(number_offset: int = 0) -> None:
    """
    Prints the aggregated provider model list, numbered after the Anthropic list.
    """
    with MODEL_LOCK:
        models = list(MODELS)
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


def select_model_by_number(index: int) -> bool:
    """
    Selects a provider model by its number in the aggregated list. Returns False
    when nothing was selected, so the caller knows not to re-resolve thinking.
    """
    with MODEL_LOCK:
        if not MODELS:
            print("No OpenAI-style provider models available.")
            return False
        if index < 1 or index > len(MODELS):
            print(f"Model number out of range [1:{len(MODELS)}].")
            return False
        entry = MODELS[index - 1]
    apply_model(entry)
    return True


def print_model_info(index: int) -> None:
    with MODEL_LOCK:
        if not MODELS:
            print("No OpenAI-style provider models available.")
            return
        if index < 1 or index > len(MODELS):
            print(f"Model number out of range. Use 1 through {len(MODELS)}.")
            return
        entry = dict(MODELS[index - 1])

    print(json.dumps(entry, indent=2, ensure_ascii=False, default=str))


def apply_model(entry: Dict[str, Any]) -> None:
    """
    Points cfg at a provider model and its costs.

    Thinking is deliberately not resolved here: which module owns that depends on
    the wire protocol the provider speaks, and this module must not import them.
    The caller resolves it once the switch is done (see server.finish_model_switch).
    """
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
    print(f"=== Switching to {entry['provider']}/{entry['id']} complete ===")


def apply_model_by_id(model_id: str) -> bool:
    """
    Applies a provider model matching either "model-id" or "provider/model-id".
    Returns False quietly when nothing matches (the caller falls back to Anthropic).
    """
    with MODEL_LOCK:
        models = list(MODELS)

    for entry in models:
        if model_id in (entry["id"], f"{entry['provider']}/{entry['id']}"):
            apply_model(entry)
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


def usage_to_cost_tokens(counts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps parsed usage counts to the normalized dict common.track_usage() expects.
    OpenAI-style caching has one rate and no TTL choice, so writes all land in the
    5m bucket (apply_model prices both buckets identically).
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
