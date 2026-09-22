"""
NanoGPT, a model aggregator served over /chat/completions.

Its catalogue holds hundreds of models, so it stays out of the shared model list.
The 'nano' CLI command searches it, selects a model, then selects the provider serving it.
Requests still go through v1_chat_completions; this module supplies what NanoGPT does its own way.
That is the thinking control, the provider routing and the prices.

A /chat/completions provider is declared NanoGPT with <NAME>_AGGREGATOR=nano_gpt.
The catalogue is saved to <NAME>_CATALOGUE_PATH and read back at startup.
It is refetched once older than <NAME>_CATALOGUE_REFRESH_HOURS, or on 'nano refresh'.
Selections are runtime-only, like the rest of the CLI.
"""

import httpx
import json
import re
import threading
import time

from typing       import Any, Dict, List, Optional
from urllib.parse import quote

from common import (
    THINK_EFFORT_ORDER,
    cfg,
    extract_claude_version,
)
from providers import (
    OFF_EFFORTS,
    auth_headers,
    error_from_response,
    fold_effort,
)


AGGREGATOR = "nano_gpt"

# The chat models of the catalogue, sorted by id.
# A ':thinking' twin is merged into its base where the base can think on its own.
CATALOGUE : List[Dict[str, Any]] = []
# When the catalogue was fetched from NanoGPT, in epoch seconds; read back with the saved file.
CATALOGUE_FETCHED_AT = 0.0
# The provider listing of each model id, fetched when the model is selected.
LISTINGS  : Dict[str, Dict[str, Any]] = {}
# The provider pinned for each model id; "" lets NanoGPT route.
# Kept per model, so returning to a model returns to its provider.
ROUTES    : Dict[str, str] = {}
LOCK = threading.Lock()

THINKING_SUFFIX = ":thinking"
# Providers named per row of 'nano list'; the rest are elided.
LIST_PROVIDERS_SHOWN = 8


def provider_name() -> str:
    """The configured NanoGPT provider, or "" when there is none. The first one wins."""
    for name, provider in cfg.providers.items():
        if provider["aggregator"] == AGGREGATOR:
            return name
    return ""


def is_active() -> bool:
    provider = cfg.providers.get(cfg.backend)
    return provider is not None and provider["aggregator"] == AGGREGATOR


def request_headers_for(provider: Dict[str, Any]) -> Dict[str, str]:
    """The listings are public; the key is sent anyway, in case pricing is per account."""
    return auth_headers(provider, provider["api_key"]) if provider["api_key"] else {}


# Catalogue
def merge_thinking_twins(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drops each 'X:thinking' whose base X accepts every effort level the twin does.
    That base thinks through reasoning_effort, so the twin is the same model with another default.
    Some bases list no effort levels (claude-sonnet-5); their twin is the only way to think.
    A twin listing none itself shows nothing the base can do, so it is kept too.
    """
    efforts = {model["id"]: set(model.get("reasoning_efforts") or []) for model in models}

    kept = []
    for model in models:
        model_id = model["id"]
        if model_id.endswith(THINKING_SUFFIX):
            base = model_id[: -len(THINKING_SUFFIX)]
            if base in efforts and efforts[model_id] and efforts[model_id] <= efforts[base]:
                continue
        kept.append(model)
    return kept


def set_catalogue(entries: List[Any], fetched_at: float) -> int:
    """
    Keeps the chat models of a raw /models list, sorted and with their thinking twins merged.
    Anything that does not answer in text (the 'decisions' models) is left out.
    Selected providers are kept; their listings are refetched on the next selection.
    """
    global CATALOGUE, CATALOGUE_FETCHED_AT

    models = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get("id")
        and (entry.get("architecture") or {}).get("output_modalities", ["text"]) == ["text"]
    ]
    models.sort(key=lambda entry: str(entry["id"]).lower())
    models = merge_thinking_twins(models)

    with LOCK:
        CATALOGUE            = models
        CATALOGUE_FETCHED_AT = fetched_at
        LISTINGS.clear()
    return len(models)


def refresh_catalogue(timeout_s: float) -> bool:
    """
    Fetches the detailed model list: prices, effort levels and provider names per model.
    It is saved as fetched, unfiltered, so a later filter change needs no refetch.
    """
    name = provider_name()
    if not name:
        print_not_configured()
        return False
    provider = cfg.providers[name]

    try:
        response = httpx.get(
            f"{provider['base_url']}/models",
            params={"detailed": "true"},
            headers=request_headers_for(provider),
            timeout=timeout_s,
        )
        if response.status_code != 200:
            raise error_from_response(name, response)
        entries = response.json().get("data") or []
    except Exception as exc:
        print(f"WARNING: Could not retrieve the NanoGPT catalogue. {exc}")
        return False

    fetched_at = time.time()
    count      = set_catalogue(entries, fetched_at)
    print(f"Retrieved {count} model(s) from NanoGPT.")

    path = provider["catalogue_path"]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": fetched_at, "data": entries}, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        print(f"WARNING: Could not save the NanoGPT catalogue to '{path}'. {exc}")
    return True


def load_catalogue(provider: Dict[str, Any]) -> bool:
    """Reads the saved catalogue. A missing file is no warning; it is the first run."""
    path = provider["catalogue_path"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        entries    = saved["data"]
        fetched_at = float(saved["fetched_at"])
    except FileNotFoundError:
        return False
    except Exception as exc:
        print(f"WARNING: Could not read the NanoGPT catalogue from '{path}'. {exc}")
        return False

    count = set_catalogue(entries, fetched_at)
    print(f"Read {count} NanoGPT model(s) from '{path}', {catalogue_age()}.")
    return True


def catalogue_age() -> str:
    with LOCK:
        hours = (time.time() - CATALOGUE_FETCHED_AT)/3600.0
    return f"fetched {hours:.1f}h ago"


def is_stale(provider: Dict[str, Any]) -> bool:
    """Older than <NAME>_CATALOGUE_REFRESH_HOURS. 0 never goes stale; 'nano refresh' still works."""
    hours = provider["catalogue_hours"]
    with LOCK:
        fetched_at = CATALOGUE_FETCHED_AT
    return hours > 0 and time.time() - fetched_at > hours*3600.0


def ensure_catalogue() -> bool:
    """
    Makes sure a catalogue is loaded: the saved one, refetched once it is older than its period.
    A stale catalogue that cannot be refetched is still used, with a warning.
    """
    name = provider_name()
    if not name:
        print_not_configured()
        return False
    provider = cfg.providers[name]

    with LOCK:
        loaded = bool(CATALOGUE)
    if not loaded:
        loaded = load_catalogue(provider)

    if loaded and not is_stale(provider):
        return True
    if refresh_catalogue(cfg.model_list_timeout_seconds):
        return True
    if loaded:
        print(f"Using the saved NanoGPT catalogue, {catalogue_age()}.")
    return loaded


def print_not_configured() -> None:
    print("No NanoGPT provider is configured. Set <NAME>_AGGREGATOR=nano_gpt on a /chat/completions provider.")


def find_model(model_id: str) -> Dict[str, Any]:
    with LOCK:
        for entry in CATALOGUE:
            if entry["id"] == model_id:
                return entry
    return {}


def squash(text: str) -> str:
    """Lowercase letters and digits only, so 'mimo pro' finds 'mimo-v2.5-pro' and 'v25' 'V2.5'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def print_catalogue(terms: List[str]) -> None:
    """
    Lists the catalogue, or only the models matching every term.
    Numbers are positions in the whole catalogue, so a search does not renumber anything.
    """
    if not ensure_catalogue():
        return
    with LOCK:
        models = list(CATALOGUE)

    keys   = [key for key in map(squash, terms) if key]
    width  = len(str(len(models)))
    active = cfg.model if is_active() else ""
    shown  = 0

    for index, entry in enumerate(models, start=1):
        haystack = squash(f"{entry['id']} {entry.get('name') or ''}")
        if not all(key in haystack for key in keys):
            continue
        shown += 1

        number      = str(index).rjust(width)
        number_cell = f"[{number}]" if entry["id"] == active else f" {number} "

        pricing = entry.get("pricing") or {}
        price   = f"${float(pricing.get('prompt') or 0):.3f}/${float(pricing.get('completion') or 0):.3f}"
        efforts = ",".join(entry.get("reasoning_efforts") or []) or "-"

        names = ["Auto"] + list(entry.get("providers") or [])
        if len(names) > LIST_PROVIDERS_SHOWN:
            names = names[:LIST_PROVIDERS_SHOWN] + ["..."]

        print(f"{number_cell}  {entry['id']:<48}  {price:<16}  think: {efforts:<24}  Providers: {', '.join(names)}")

    if not shown:
        print(f"No NanoGPT model matches '{' '.join(terms)}'.")


# Providers
def listing_url(provider: Dict[str, Any], model_id: str) -> str:
    """
    The provider listing lives outside /v1: <root>/api/models/<id>/providers.
    The id is a single path segment, so its own slash is encoded too.
    """
    root = re.sub(r"/v1$", "", provider["base_url"])
    return f"{root}/models/{quote(model_id, safe='')}/providers"


def fetch_listing(model_id: str) -> Dict[str, Any]:
    """
    Fetches and keeps the provider listing of one model.
    Returns {} when it is unavailable; the model can then only be routed by NanoGPT.
    """
    name     = provider_name()
    provider = cfg.providers[name]
    try:
        response = httpx.get(
            listing_url(provider, model_id),
            headers=request_headers_for(provider),
            timeout=cfg.model_list_timeout_seconds,
        )
        if response.status_code != 200:
            raise error_from_response(name, response)
        listing = response.json()
    except Exception as exc:
        print(f"WARNING: Could not retrieve the providers of '{model_id}'. {exc}")
        listing = {}

    if not isinstance(listing, dict):
        listing = {}
    with LOCK:
        LISTINGS[model_id] = listing
    return listing


def listing_rows(model_id: str) -> List[Dict[str, Any]]:
    """The selectable providers of a model, in NanoGPT's order. Numbered from 1; 0 is Auto."""
    with LOCK:
        listing = LISTINGS.get(model_id) or {}
    if not listing.get("supportsProviderSelection"):
        return []
    return [row for row in listing.get("providers") or [] if isinstance(row, dict) and row.get("provider")]


# Prices
def catalogue_price(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    The catalogue pricing, in the per-1k-token shape of the provider listing.
    The catalogue gives prompt and completion per million tokens, but the cache prices per 1k.
    """
    pricing = entry.get("pricing") or {}
    price   = {
        "inputPer1kTokens"  : float(pricing.get("prompt") or 0.0) / 1000.0,
        "outputPer1kTokens" : float(pricing.get("completion") or 0.0) / 1000.0,
    }
    for key in ("cacheReadInputPer1kTokens", "cacheWriteInputPer1kTokens"):
        if key in pricing:
            price[key] = pricing[key]
    return price


def route_price(model_id: str, route: str) -> Dict[str, Any]:
    """
    What a request is billed at, per 1k tokens.
    A pinned provider bills its own price, which includes NanoGPT's selection markup.
    Automatic routing bills the model's default price.
    The catalogue price stands in while the listing is unavailable.
    """
    with LOCK:
        listing = LISTINGS.get(model_id) or {}
    if route:
        for row in listing_rows(model_id):
            if row["provider"] == route:
                return row.get("pricing") or {}
    if listing.get("defaultPrice"):
        return listing["defaultPrice"]
    return catalogue_price(find_model(model_id))


def per_million(price: Dict[str, Any], key: str) -> float:
    return float(price.get(key) or 0.0)*1000.0


def apply_prices(model_id: str, route: str) -> None:
    """
    Points cfg at the prices of a model and route.
    These only check the cost NanoGPT reports with each response (see common.track_usage).
    A missing cache read price bills cached tokens as input; a zero write price bills writes so.
    """
    price  = route_price(model_id, route)
    input_ = per_million(price, "inputPer1kTokens")
    write  = per_million(price, "cacheWriteInputPer1kTokens") or input_

    cfg.model_cost_family       = f"{cfg.backend}:{route or 'auto'}"
    cfg.input_token_cost_usd    = input_
    cfg.output_token_cost_usd   = per_million(price, "outputPer1kTokens")
    cfg.cache_read_cost_usd     = per_million(price, "cacheReadInputPer1kTokens") if "cacheReadInputPer1kTokens" in price else input_
    cfg.cache_write_5m_cost_usd = write
    cfg.cache_write_1h_cost_usd = write

    if not input_ and not cfg.output_token_cost_usd:
        print(f"WARNING: NanoGPT lists no prices for '{model_id}'. Only the reported cost will be tracked.")


def price_cells(price: Dict[str, Any]) -> str:
    """Input, output and cache read per million tokens, as the columns of the provider table."""
    cells = []
    for key in ("inputPer1kTokens", "outputPer1kTokens", "cacheReadInputPer1kTokens"):
        cell = f"${per_million(price, key):.4f}" if key in price else "-"
        cells.append(f"{cell:>9}")
    return "  ".join(cells)


# Selection
def apply_model(entry: Dict[str, Any]) -> None:
    """
    Points cfg at a NanoGPT model, binding the NanoGPT provider as the active backend.
    The provider last pinned for this model is pinned again, if it still serves it.
    """
    name     = provider_name()
    model_id = entry["id"]

    print(f"=== Switching to {name}/{model_id} ===")
    fetch_listing(model_id)

    route = ROUTES.get(model_id, "")
    if route and route not in [row["provider"] for row in listing_rows(model_id)]:
        print(f"Provider '{route}' no longer serves '{model_id}'. NanoGPT will route it.")
        route = ""
    ROUTES[model_id] = route

    cfg.backend    = name
    cfg.model      = model_id
    cfg.info       = dict(entry)
    cfg.model_info = dict(entry)
    # Only the Anthropic backend reads it; kept in step with providers.apply_model.
    cfg.version    = extract_claude_version(model_id)

    apply_prices(model_id, route)
    print(f"=== Switching to {name}/{model_id} complete ===")
    print_providers()


def select_model_by_number(index: int) -> bool:
    """Returns False when nothing was selected, so the caller skips the post-switch hook."""
    if not ensure_catalogue():
        return False
    with LOCK:
        if index < 1 or index > len(CATALOGUE):
            print(f"Model number out of range [1:{len(CATALOGUE)}].")
            return False
        entry = CATALOGUE[index - 1]
    apply_model(entry)
    return True


def apply_model_by_id(model_id: str) -> bool:
    """
    Applies "<nanogpt provider>/<model id>", which is how MODEL names a NanoGPT model.
    Returns False for anything else, including a model missing from the catalogue.
    The shared registry then takes it as configured, like any model missing from a list.
    """
    name, separator, bare_id = model_id.partition("/")
    if not separator or not name or name != provider_name():
        return False
    if not ensure_catalogue():
        return False

    entry = find_model(bare_id)
    if not entry:
        print(f"Model '{bare_id}' is not in the NanoGPT catalogue.")
        return False
    apply_model(entry)
    return True


def print_providers() -> None:
    if not is_active():
        print_nothing_selected()
        return

    rows  = listing_rows(cfg.model)
    route = ROUTES.get(cfg.model, "")

    print(f"Provider: {route or 'Auto'}. Prices per million tokens.")
    print(f"       {'provider':<14}  {'input':>9}  {'output':>9}  {'cache rd':>9}  {'quant':<8}  {'t/s':>5}  {'ttft':>6}  {'cache':<5}  {'privacy':<12}  region")

    number_width = len(str(len(rows)))
    number       = "0".rjust(number_width)
    number_cell  = f"[{number}]" if not route else f" {number} "
    print(f"  {number_cell}  {'Auto':<14}  {price_cells(route_price(cfg.model, ''))}")

    for index, row in enumerate(rows, start=1):
        number      = str(index).rjust(number_width)
        number_cell = f"[{number}]" if row["provider"] == route else f" {number} "
        quant       = str(row.get("quantization") or "-")
        tps         = f"{float(row['tps']):.0f}" if row.get("tps") else "-"
        ttft        = f"{float(row['ttftMs'])/1000.0:.1f}s" if row.get("ttftMs") else "-"
        caching     = "yes" if row.get("supportsPromptCaching") else "no"
        privacy     = str((row.get("privacy") or {}).get("classification") or "-")
        region      = str((row.get("region") or {}).get("code") or "-")
        available   = "" if row.get("available", True) else "  unavailable"

        prices = price_cells(row.get("pricing") or {})
        print(f"  {number_cell}  {row['provider']:<14}  {prices}  {quant:<8}  {tps:>5}  {ttft:>6}  {caching:<5}  {privacy:<12}  {region}{available}")

    if not rows:
        print("  NanoGPT offers no provider selection for this model; it is always routed automatically.")


def select_provider(index: int) -> None:
    """
    Pins a provider for the selected model, or lets NanoGPT route it (0).
    A pinned provider is strict: when it is down the request fails rather than go elsewhere.
    """
    if not is_active():
        print_nothing_selected()
        return

    rows = listing_rows(cfg.model)
    if index < 0 or index > len(rows):
        print(f"Provider number out of range [0:{len(rows)}].")
        return

    route = rows[index - 1]["provider"] if index else ""
    if route and not rows[index - 1].get("available", True):
        print(f"WARNING: NanoGPT lists '{route}' as unavailable. Selecting it anyway.")

    ROUTES[cfg.model] = route
    apply_prices(cfg.model, route)
    print(f"Selected provider {route or 'Auto'} for {cfg.model}.")


def refresh() -> bool:
    """
    Fetches the catalogue again, and the listing of the selected model.
    Returns True when the selected model was re-applied, so the caller runs the post-switch hook.
    """
    if not refresh_catalogue(cfg.model_list_timeout_seconds) or not is_active():
        return False
    entry = find_model(cfg.model)
    if not entry:
        print(f"'{cfg.model}' is no longer in the NanoGPT catalogue. It stays selected, at its last prices.")
        return False
    apply_model(entry)
    return True


def print_nothing_selected() -> None:
    print("No NanoGPT model is selected. Use 'nano list [terms]', then 'nano <number>'.")


def print_status() -> None:
    if not is_active():
        print_nothing_selected()
        return
    route = ROUTES.get(cfg.model, "")
    print(f"  Model      {cfg.backend}/{cfg.model}")
    print(f"  Provider   {route or 'Auto'}")
    print(f"  Prices     ${cfg.input_token_cost_usd:.4f} input, ${cfg.output_token_cost_usd:.4f} output, ${cfg.cache_read_cost_usd:.4f} cache read per million tokens")
    print(f"  Thinking   {','.join(cfg.info.get('reasoning_efforts') or []) or 'no effort levels listed'}")
    print(f"  Catalogue  {catalogue_age()}, '{cfg.providers[cfg.backend]['catalogue_path']}'")


def print_cache_status() -> None:
    route = ROUTES.get(cfg.model, "")
    if cfg.cache_en : print("  Cache enabled   ✅")
    else            : print("  Cache enabled   ❌")
    if route:
        print(f"  Provider '{route}' is pinned. It caches on its own if it supports caching, whatever this says.")
    elif cfg.cache_en:
        print("  Automatic routing picks a provider that caches, and keeps to it.")


# Request
def thinking_params(model_id: str, thinking_enabled: bool, thinking_effort: str) -> Optional[Dict[str, Any]]:
    """
    Maps the shared thinking settings onto reasoning_effort, NanoGPT's one thinking control.
    Provider-native controls (a 'thinking' block) are ignored by most of its providers.
    The ladder is the model's own list of effort levels.
    On/off models list one level besides 'none', so any effort enables it.
    Returns None for a model missing from the catalogue, {} for one listing no levels.

    A provider may still ignore the effort (atlascloud always thinks on mimo-v2.5-pro).
    """
    entry = find_model(model_id)
    if not entry:
        return None

    efforts = entry.get("reasoning_efforts") or []
    ladder  = tuple(effort for effort in THINK_EFFORT_ORDER if effort in efforts)
    off     = next((effort for effort in OFF_EFFORTS if effort in efforts), "")

    if thinking_enabled and ladder:
        return {"reasoning_effort": fold_effort(thinking_effort, ladder)}
    if not thinking_enabled and off:
        return {"reasoning_effort": off}
    # Asked to stop, but the model cannot; send the weakest level it has.
    if ladder:
        return {"reasoning_effort": ladder[0]}
    return {}


def route_params(model_id: str) -> Dict[str, Any]:
    """
    Provider routing for one request.
    A pinned provider is a hard pin, never a preference NanoGPT may route around.
    'caching: true' cannot be combined with a pinned provider, so it applies to Auto only.
    It then routes to a provider that caches, and keeps to it while it can.
    """
    route = ROUTES.get(model_id, "")
    if route:
        return {"provider": {"only": [route], "allow_fallbacks": False}}
    if cfg.cache_en:
        return {"caching": True}
    return {}
