import httpx
import json
import threading

from packaging.version import Version
from typing            import Any, Dict, Iterator, List, Tuple

from common import (
    append_prefill_instruction_to_last_user_message,
    cfg,
    resolve_api_key,
    track_usage,
    trim_to_end_sentence,
)


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


def refresh_openai_models(timeout_s: float) -> None:
    """
    Fetches the model list of every configured provider and stores them for CLI use.

    Providers with a <NAME>_MODELS override skip the /models request entirely.
    A failing provider is skipped with a warning; it does not block the others.
    """
    global OPENAI_MODELS

    models: List[Dict[str, Any]] = []

    for name, provider in cfg.openai_providers.items():
        if provider["models"]:
            models.extend({"id": model_id, "provider": name} for model_id in provider["models"])
            print(f"Using {len(provider['models'])} configured model(s) for provider '{name}'.")
            continue

        try:
            headers = {}
            if provider["api_key"]:
                headers["Authorization"] = f"Bearer {provider['api_key']}"

            response = httpx.get(f"{provider['base_url']}/models", headers=headers, timeout=timeout_s)
            if response.status_code != 200:
                raise error_from_response(name, response)

            data    = response.json()
            entries = data.get("data") if isinstance(data, dict) else None
            got     = []
            for entry in entries or []:
                if isinstance(entry, dict) and entry.get("id"):
                    got.append({**entry, "id": str(entry["id"]), "provider": name})

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

    cfg.model_cost_family     = entry["provider"]
    cfg.input_token_cost_usd  = provider["input_cost"]
    cfg.output_token_cost_usd = provider["output_cost"]
    cfg.cache_read_cost_usd   = provider["cache_read_cost"]
    # OpenAI-style providers have no explicit cache writes. Price writes as plain
    # input so any stray write tokens net to zero in the cache-cost accounting.
    cfg.cache_write_5m_cost_usd = provider["input_cost"]
    cfg.cache_write_1h_cost_usd = provider["input_cost"]
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


def build_openai_body(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the OpenAI-style chat completion request from a prepared chat request.

    Since the frontend already speaks OpenAI format this is a near-passthrough:
    system segments become one leading system message, the moved lorebook suffix
    becomes a trailing system message (OpenAI-style APIs allow system anywhere),
    and the provider's EXTRA_BODY is merged in verbatim.
    """
    provider = cfg.openai_providers[cfg.backend]

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

    body: Dict[str, Any] = {
        "model"      : cfg.model,
        "max_tokens" : prepared["max_tokens"],
        "messages"   : messages,
    }

    if cfg.send_temperature : body["temperature"] = cfg.temperature
    if cfg.send_top_p       : body["top_p"      ] = cfg.top_p
    # top_k is not part of the OpenAI chat schema; providers that accept it can get it via EXTRA_BODY.

    body.update(provider["extra_body"])

    return body


def usage_to_cost_tokens(usage: Any) -> Dict[str, int]:
    """
    Maps an OpenAI-style usage payload to the normalized token-count dict that
    common.track_usage() expects. Cached prompt tokens count as cache reads;
    there is no explicit cache-write concept.
    """
    usage = usage if isinstance(usage, dict) else {}

    prompt_tokens     = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)

    details       = usage.get("prompt_tokens_details")
    cached_tokens = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    cached_tokens = min(cached_tokens, prompt_tokens)

    return {
        "uncached_input" : prompt_tokens - cached_tokens,
        "cache_read"     : cached_tokens,
        "cache_write_1h" : 0,
        "cache_write_5m" : 0,
        "output"         : completion_tokens,
    }


def print_usage(usage: Any) -> None:
    track_usage(usage_to_cost_tokens(usage))


def usage_to_openai_dict(usage: Any) -> Dict[str, int]:
    """
    Normalizes the provider's usage payload to the same shape the Claude backend
    emits, so clients see one consistent usage format regardless of backend.
    """
    usage = usage if isinstance(usage, dict) else {}

    prompt_tokens     = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens      = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    details       = usage.get("prompt_tokens_details")
    cached_tokens = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    cached_tokens = min(cached_tokens, prompt_tokens)

    return {
        "prompt_tokens"               : prompt_tokens,
        "completion_tokens"           : completion_tokens,
        "total_tokens"                : total_tokens,
        "input_tokens_uncached"       : prompt_tokens - cached_tokens,
        "cache_creation_input_tokens" : 0,
        "cache_read_input_tokens"     : cached_tokens,
    }


# Generation
def generate_non_stream(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs one non-streaming completion. Same result shape as claude.generate_non_stream.
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

    usage = data.get("usage")
    print_usage(usage)

    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}

    output_text    = str(message.get("content") or "")
    reasoning_text = str(message.get("reasoning_content") or "")

    if cfg.auto_trim:
        output_text = trim_to_end_sentence(output_text)

    # Keep ordinary <think> output for Janitor/client compatibility.
    if reasoning_text.strip():
        output_text = f"<think>\n{reasoning_text.strip()}\n</think>\n\n" + output_text

    return {
        "id"            : str(data.get("id") or cfg.model),
        "stop_reason"   : str(choices[0].get("finish_reason") or "stop"),
        "text"          : output_text,
        "usage"         : usage_to_openai_dict(usage),
        "message_extra" : {},
    }


def generate_stream(prepared: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Runs one streaming completion, yielding the same backend-neutral events as
    claude.generate_stream. Provider SSE chunks are relayed nearly verbatim.

    Note: not every provider sends usage in the stream. Those that support it can
    enable it via EXTRA_BODY, e.g. {"stream_options": {"include_usage": true}};
    without usage the request is tracked as zero cost.
    """
    provider = cfg.openai_providers[cfg.backend]
    body     = build_openai_body(prepared)
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

                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

                delta = choice.get("delta") or {}

                reasoning_delta = delta.get("reasoning_content")
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    yield ("reasoning", reasoning_delta)

                text_delta = delta.get("content")
                if text_delta:
                    response_parts.append(text_delta)
                    yield ("text", text_delta)

    print_usage(usage)

    yield ("final", {
        "id"                 : message_id or cfg.model,
        "stop_reason"        : finish_reason,
        "usage"              : usage_to_openai_dict(usage),
        "snapshot_text"      : "".join(response_parts),
        "snapshot_reasoning" : "".join(reasoning_parts),
    })
