"""
The provider registry, shared by all three backends.

Everything here belongs to no single wire protocol.
It holds the registry and model list, selection and pricing, transport, and the shared message list.
The wire modules import from here and this one imports from none of them.
So a provider can never depend on whichever endpoint happens to be selected.

The OpenAI-style modules use everything here.
The Anthropic one keeps its own transport and message formatting, speaking through the SDK.
It is still registered, listed, selected and priced here like any other provider.
"""

import httpx
import json
import re
import threading

from concurrent.futures import ThreadPoolExecutor
from packaging.version  import Version
from typing             import Any, Dict, List, Optional, Tuple

from common import (
    THINK_EFFORT_ORDER,
    append_prefill_instruction_to_last_user_message,
    cfg,
    extract_claude_version,
    resolve_api_key,
)


def fold_effort(effort: str, ladder: Tuple[str, ...]) -> str:
    """
    Folds a shared proxy effort onto a provider's own ladder.
    Walks THINK_EFFORT_ORDER down from the request, taking the first level the provider has.
    So 'max' becomes 'xhigh' on a ladder that stops at xhigh, and 'high' on one that stops at high.
    `ladder` must be ordered weakest first and hold only thinking levels.
    """
    try: start = THINK_EFFORT_ORDER.index(effort)
    except ValueError: start = THINK_EFFORT_ORDER.index("medium")

    for name in reversed(THINK_EFFORT_ORDER[: start + 1]):
        if name in ladder:
            return name
    # The request is weaker than anything the model offers; give it the weakest level.
    return ladder[0]


def effort_params(ladder: Tuple[str, ...], off: str, thinking_enabled: bool, thinking_effort: str) -> Dict[str, Any]:
    """
    Thinking as a single reasoning_effort, for providers that take nothing else.
    'ladder' holds the levels that make the model think, weakest first.
    'off' is the level that stops it, or "" when nothing does.
    On/off models have a one-level ladder, so any effort enables them.
    """
    if thinking_enabled and ladder:
        return {"reasoning_effort": fold_effort(thinking_effort, ladder)}
    if not thinking_enabled and off:
        return {"reasoning_effort": off}
    # Asked to stop, but the model cannot; send the weakest level it has.
    if ladder:
        return {"reasoning_effort": ladder[0]}
    return {}


# Effort levels that mean "do not reason", weakest first.
# Not thinking depths, so folding skips them; they only answer a disable request.
# OpenAI spells its floor 'minimal' on gpt-5; Aion and later OpenAI models use 'none'.
OFF_EFFORTS = ("none", "minimal")

OPENAI_MODEL_RE = re.compile(r"^(?:gpt-|o\d+(?:-|$)|chat-latest$)")


def is_openai_model(model_id: str) -> bool:
    """
    True for OpenAI's own model ids (gpt-*, o-series, chat-latest).
    Used for request-shape rules holding across the catalogue, not just reasoning models.
    """
    return OPENAI_MODEL_RE.match(model_id) is not None


def api_style(backend: str = "") -> str:
    """
    Which wire protocol the active (or named) provider speaks: 'messages', 'chat' or 'responses'.
    This picks the backend module (see server.active_backend).
    It is the list the provider was declared in; nothing about it is guessed.

    Raises for a name that is not configured.
    There is no safe default: answering 'chat' would route the request to the wrong protocol.
    """
    name     = backend or cfg.backend
    provider = cfg.providers.get(name)
    if provider is None:
        raise KeyError(f"No provider named '{name}' is configured.")
    return provider["api"]


# Aggregated model list across every configured OpenAI-style provider.
# Each entry: {"id", "provider"} plus whatever the provider's /models returned.
MODELS : List[Dict[str, Any]] = []
MODEL_LOCK                    = threading.Lock()


class ProviderError(Exception):
    """
    Provider HTTP error.
    Carries status_code and a body dict in the attribute shape the Anthropic SDK errors use.
    So server.build_error_body and common.error_body handle it without special cases.
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
    Fetches one provider's /models list.
    Runs in a worker thread during refresh; failures raise and are reported by the caller.

    Anthropic's /models answers the same {"data": [...]} shape, so only the auth header differs.
    """
    headers = auth_headers(provider, provider["api_key"]) if provider["api_key"] else {}

    response = httpx.get(f"{provider['base_url']}/models", headers=headers, timeout=timeout_s)
    if response.status_code != 200:
        raise error_from_response(name, response)

    data = response.json()
    # OpenAI-style APIs return {"data": [...]}, but not everyone follows the spec.
    # Aion returns {"models": [...]}, and some providers a bare list.
    if   isinstance(data, dict)  : entries = data.get("data") or data.get("models")
    elif isinstance(data, list)  : entries = data
    else                         : entries = None

    # Some providers serve their whole catalogue here, chat models and all.
    # OpenAI's /models also lists tts, image, embedding and realtime models.
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
    Aggregators are skipped too; their catalogues have their own CLI command (see nano_gpt).
    A failing provider is skipped with a warning; it does not block the others.
    The requests run in parallel but results are collected in declaration order.
    So the list, and the CLI numbering, does not depend on response order.
    """
    global MODELS

    models: List[Dict[str, Any]] = []
    listed = {name: provider for name, provider in cfg.providers.items() if not provider["aggregator"]}

    if listed:
        with ThreadPoolExecutor(max_workers=len(listed)) as pool:
            fetches = [
                (name, provider, None if provider["models"] else pool.submit(fetch_provider_models, name, provider, timeout_s))
                for name, provider in listed.items()
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


def print_model_list(terms: List[str]) -> None:
    """
    Prints the aggregated model list of every configured provider.
    With terms, only the models matching any of them: a provider name takes that provider's
    models, anything else is looked for in the model id and display name.
    Numbers stay those of the whole list, so a filtered one can still be selected from.
    """
    with MODEL_LOCK:
        models = list(MODELS)

    # An aggregator model is never in this list, so say what is selected instead.
    active = cfg.providers.get(cfg.backend)
    if active is not None and active["aggregator"]:
        print(f"Selected: {cfg.backend}/{cfg.model}, from an aggregator catalogue. See 'nano'.")

    if not models:
        print_no_models_available()
        return

    names  = [term.lower() for term in terms if term.lower() in cfg.providers]
    needle = [term.lower() for term in terms if term.lower() not in cfg.providers]

    number_width = len(str(len(models)))
    shown        = 0

    for index, entry in enumerate(models, start=1):
        # Anthropic's model list carries a display name; the OpenAI-style ones do not.
        display_name = str(entry.get("display_name") or "")
        haystack     = f"{entry['id']} {display_name}".lower()
        if terms and entry["provider"] not in names and not any(term in haystack for term in needle):
            continue
        shown += 1

        selected    = (cfg.backend == entry["provider"]) and (cfg.model == entry["id"])
        number      = str(index).rjust(number_width)
        number_cell = f"[{number}]" if selected else f" {number} "
        suffix      = f"  {display_name}" if display_name else ""

        print(f"{number_cell}  {entry['id']:<42}  {entry['provider']:<10}{suffix}".rstrip())

    if not shown:
        print(f"No model matches '{'|'.join(terms)}'.")


def print_no_models_available() -> None:
    if cfg.providers : print("No provider returned a model list. Use 'model refresh' to try again.")
    else             : print("No providers are configured.")


def select_model_by_number(index: int) -> bool:
    """
    Selects a model by its number in the aggregated list.
    Returns False when nothing was selected, so the caller knows not to run the post-switch hook.
    """
    with MODEL_LOCK:
        if not MODELS:
            print_no_models_available()
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
            print_no_models_available()
            return
        if index < 1 or index > len(MODELS):
            print(f"Model number out of range. Use 1 through {len(MODELS)}.")
            return
        entry = dict(MODELS[index - 1])

    print(json.dumps(entry, indent=2, ensure_ascii=False, default=str))


def model_record_prices(entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Prices from the model record, in OpenRouter's shape: 'pricing' holding USD per token, as strings.
    Atlas Cloud publishes these, and its billing matches them exactly.
    Returns {} when the record has none, so the provider-level prices apply.
    A configured cost family still wins over these; see apply_model().
    """
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return {}
    try:
        input_  = float(pricing["prompt"])*1_000_000
        output  = float(pricing["completion"])*1_000_000
        # Without a cache price, cached tokens bill as input; without a write price, so do writes.
        read    = float(pricing.get("input_cache_read") or pricing["prompt"])*1_000_000
        write   = float(pricing.get("input_cache_write") or pricing["prompt"])*1_000_000
    except (KeyError, TypeError, ValueError):
        return {}

    return {
        "input_cost"          : input_,
        "output_cost"         : output,
        "cache_read_cost"     : read,
        "cache_write_5m_cost" : write,
        "cache_write_1h_cost" : write,
    }


def apply_model(entry: Dict[str, Any]) -> None:
    """
    Points cfg at a model, its provider and its costs.
    This binds the active backend: the provider decides the protocol, and so the serving module.

    Nothing model-specific is resolved here.
    Thinking and prefill depend on the protocol, and this module must not import the backends.
    The caller runs the backend's own hook once the switch is done (see server.finish_model_switch).
    """
    provider = cfg.providers[entry["provider"]]

    print(f"=== Switching to {entry['provider']}/{entry['id']} ===")
    cfg.backend    = entry["provider"]
    cfg.model      = entry["id"]
    cfg.info       = dict(entry)
    cfg.model_info = dict(entry)

    # Only the Anthropic backend reads the version, for its prefill and system-message rules.
    # Extracting it for everyone is simpler than asking who is asking.
    version = extract_claude_version(entry.get("display_name") or "")
    if version == Version("0.0"):
        version = extract_claude_version(entry["id"])
    cfg.version = version

    # Per-model cost family when one matches, provider-level costs otherwise.
    cost_source = provider
    cost_family = entry["provider"]
    for family in provider["cost_families"]:
        if family["regex"].search(entry["id"]):
            cost_source = family
            cost_family = f"{entry['provider']}:{family['name']}"
            break

    # No family, but the provider may price its models itself.
    if cost_source is provider:
        record_prices = model_record_prices(entry)
        if record_prices:
            cost_source = record_prices
            cost_family = f"{entry['provider']}:model"

    print(f"Using cost family '{cost_family}'.")
    cfg.model_cost_family       = cost_family
    cfg.input_token_cost_usd    = cost_source["input_cost"]
    cfg.output_token_cost_usd   = cost_source["output_cost"]
    cfg.cache_read_cost_usd     = cost_source["cache_read_cost"]
    cfg.cache_write_5m_cost_usd = cost_source["cache_write_5m_cost"]
    cfg.cache_write_1h_cost_usd = cost_source["cache_write_1h_cost"]

    if not cost_source["input_cost"] and not cost_source["output_cost"]:
        print(f"WARNING: no prices are configured for '{cost_family}'. This model will be reported as free.")

    print(f"=== Switching to {entry['provider']}/{entry['id']} complete ===")


def apply_model_by_id(model_id: str) -> bool:
    """
    Applies the model matching either "model-id" or "provider/model-id" in the fetched list.
    Returns False quietly when nothing matches, so the caller can report it.

    The prefixed form names its own provider, so it applies even when it is not in the list.
    That happens whenever a provider's /models request failed.
    It is why MODEL=provider/model-id is the form worth configuring.
    """
    with MODEL_LOCK:
        models = list(MODELS)

    for entry in models:
        if model_id in (entry["id"], f"{entry['provider']}/{entry['id']}"):
            apply_model(entry)
            return True

    name, separator, bare_id = model_id.partition("/")
    if separator and bare_id and name in cfg.providers:
        print(f"Model '{model_id}' is not in the model list; taking it as configured.")
        apply_model({"id": bare_id, "provider": name})
        return True

    return False


# Anthropic authenticates with its own header, not a bearer token, and wants the API version too.
# The SDK sends both itself; this is for the requests that do not go through it (the model list).
ANTHROPIC_VERSION = "2023-06-01"


def auth_headers(provider: Dict[str, Any], key: str) -> Dict[str, str]:
    if provider["api"] == "messages":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {key}"}


def request_headers(provider: Dict[str, Any]) -> Dict[str, str]:
    return auth_headers(provider, resolve_api_key(provider["api_key"], provider["api_key_name"]))


def request_timeout() -> httpx.Timeout:
    return httpx.Timeout(cfg.request_timeout_seconds, connect=10.0)


def build_message_list(prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turns a prepared chat request into the role/content list both OpenAI endpoints take.
    /chat/completions calls it 'messages', /responses calls it 'input'.

    The frontend already speaks OpenAI format, so this is a near-passthrough.
    System segments become one leading message; a moved lorebook suffix becomes a trailing one.
    OpenAI-style APIs allow system anywhere.
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
            # Trailing-assistant behavior varies wildly here: continue, new turn or error.
            # Only instruction mode is supported.
            print("WARNING: assistant prefill mode is not supported for OpenAI-style backends. Use 'prefill instruction'.")

    return messages


def reported_reasoning(details: Dict[str, Any]) -> Optional[int]:
    """
    The reasoning token count, or None when the provider does not report one.
    OpenAI, GLM and Kimi send completion_tokens_details.reasoning_tokens, OpenAI even when 0.
    Aion omits the details object, and None keeps its thinking from being reported as zero.
    Where it does appear it is a subset of the output tokens, which is how the report splits them.
    """
    raw = details.get("reasoning_tokens")
    return None if raw is None else max(0, int(raw or 0))


def warn_truncated_by_reasoning(finish_reason: str, output_text: str, counts: Dict[str, Any]) -> None:
    """
    On OpenAI reasoning models the output limit also covers the invisible reasoning tokens.
    A budget small next to the effort can be spent entirely on thinking, ending with no text at all.
    Janitor supplies its own max_tokens, so this is easy to hit with nothing to explain it.
    """
    if finish_reason != "length" or output_text.strip() or not counts["reasoning"]:
        return

    print(f"WARNING: '{cfg.model}' spent its entire output budget ({counts['completion']} tokens) on reasoning and returned no text.")
    print("         Raise max_tokens in the client (or MAX_TOKENS in .env), or lower the thinking effort.")


def wrap_think(output_text: str, reasoning_text: str) -> str:
    """
    Prepends the reasoning as a <think> block, which is how Janitor and similar clients render it.
    """
    if not reasoning_text.strip():
        return output_text
    return f"<think>\n{reasoning_text.strip()}\n</think>\n\n" + output_text
