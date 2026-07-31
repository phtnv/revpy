import json
import json5
import os
import re
import threading

from flask             import abort, request
from packaging.version import Version
from typing            import Any, Dict, List, Optional

ENABLE_VALUES  = {"1", "y", "true" , "yes", "enable" , "on" }
DISABLE_VALUES = {"0", "n", "false", "no" , "disable", "off"}
THINK_EFFORTS  = {"low", "medium", "high", "xhigh", "max"}
INF_VALUES     = {"inf", "all", "infinite", "infinity", "*", "∞"}
UINT64_MAX     = 2**64 - 1

# The same thinking efforts, weakest first. Provider dialects fold this ladder onto
# their own supported subset by walking it downwards from the requested level.
THINK_EFFORT_ORDER  = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PARAMS   = {"auto", "max_tokens", "max_completion_tokens"}
REASONING_SUMMARIES = {"none", "auto", "concise", "detailed"}

# Image generation enums, as accepted by /images/generations on the gpt-image family.
# 'transparent' is deliberately absent from the backgrounds: gpt-image-2 does not support
# it, and offering a value the model will reject is worse than not offering it at all.
IMAGE_QUALITIES   = {"auto", "low", "medium", "high"}
IMAGE_FORMATS     = {"png", "jpeg", "webp"}
IMAGE_BACKGROUNDS = {"opaque", "automatic"}

# Size constraints rather than a size allowlist. gpt-image-2 accepts any resolution
# meeting all of these, so an enum of the four popular sizes would reject valid requests
# and would go stale the moment the model's range changes.
IMAGE_SIZE_MAX_EDGE     = 3840
IMAGE_SIZE_EDGE_MULTIPLE = 16
IMAGE_SIZE_MAX_ASPECT   = 3.0
IMAGE_SIZE_MIN_PIXELS   = 655_360
IMAGE_SIZE_MAX_PIXELS   = 8_294_400

# Which wire protocol a provider speaks, and the variable declaring the providers that
# speak it. A provider is served by the module for the list it was declared in; nothing
# is derived from its endpoint, so pointing a provider at any host that implements the
# protocol is a matter of putting its name in the right list.
#
# Declaration order across the three lists is the order of this dict, which is also the
# order of the CLI model list.
API_STYLE_VARS = {
    "messages"  : "V1_MESSAGES_PROVIDERS",
    "chat"      : "V1_CHAT_COMPLETIONS_PROVIDERS",
    "responses" : "V1_RESPONSES_PROVIDERS",
}

def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try: return float(raw)
    except Exception:
        print(f"WARNING: {name} must be a number. Defaulting to {default}.")
        return default
def getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ENABLE_VALUES  : return True
    if value in DISABLE_VALUES : return False
    print(f"WARNING: {name} must be boolean. Defaulting to {default}.")
    return default
def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try: return int(raw)
    except Exception:
        print(f"WARNING: {name} must be an integer. Defaulting to {default}.")
        return default
def getenv_preserve_thinking_blocks(name: str, default: str = "0") -> int:
    raw   = os.getenv(name, default)
    value = str(raw).strip().lower()
    if value in INF_VALUES:
        return UINT64_MAX
    try: return max(0, int(value))
    except Exception:
        print(f"WARNING: {name} must be 0, a positive integer, or inf. Defaulting to {default}.")
        try: return max(0, int(default))
        except Exception:
            return 0
def getenv_choice(name: str, default: str, allowed: set) -> str:
    """One value out of a fixed set, lowercased. An unknown value warns and takes the default."""
    value = os.getenv(name, default).strip().lower()
    if value in allowed:
        return value
    print(f"WARNING: {name} must be one of {sorted(allowed)}. Defaulting to {default}.")
    return default


def getenv_cache_ttl(name: str, default: str) -> str:
    if default not in {"5m", "1h"}:
        default = "1h"
    raw = os.getenv(name, default)
    value = str(raw).strip().lower()
    if value in {"5m", "1h"}:
        return value
    print(f"WARNING: {name} must be '5m' or '1h'. Defaulting to {default}.")
    return default

# Unique sentinel for deep_get. A fresh object() per lookup would never compare
# `is`-equal, making every missing path return truthy garbage instead of default.
_MISSING = object()

def deep_get(obj: Any, path: str, default: Any = None) -> Any:
    """
    Safe lookup for nested dict/list JSON.
    Example:
        deep_get(model_info, "thinking.supported", False)
        deep_get(model_info, "thinking.types.adaptive.supported", False)
    """
    cur = obj

    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, _MISSING)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else _MISSING
        else:
            cur = _MISSING

        if cur is _MISSING:
            return default

    return cur

def cost_value(costs: Dict[str, Any], key: str, default: float) -> float:
    """
    One price out of a <NAME>_MODEL_<FAMILY>_COST object, in USD per 1 million tokens.
    A missing key takes the default; an explicit 0 means free and is kept as such.
    """
    raw = costs.get(key)
    if raw is None:
        return default
    try: return float(raw)
    except (TypeError, ValueError):
        print(f"WARNING: cost '{key}' must be a number. Defaulting to {default}.")
        return default


def resolve_costs(costs: Dict[str, Any]) -> Dict[str, float]:
    """
    The five prices every provider and cost family resolves to, from whichever subset
    was configured. Both cache write buckets default to cache_write, which defaults to
    the input price -- that is what a provider charging no write fee looks like, and it
    nets those tokens to zero in the cache report. Anthropic-style providers, the only
    ones that let you pick a TTL, set the two buckets apart.
    """
    input_cost = cost_value(costs, "input", 0.0)
    write_cost = cost_value(costs, "cache_write", input_cost)

    return {
        "input_cost"          : input_cost,
        "output_cost"         : cost_value(costs, "output"        , 0.0),
        "cache_read_cost"     : cost_value(costs, "cache_read"    , 0.0),
        "cache_write_5m_cost" : cost_value(costs, "cache_write_5m", write_cost),
        "cache_write_1h_cost" : cost_value(costs, "cache_write_1h", write_cost),
    }


def resolve_image_costs(costs: Dict[str, Any]) -> Dict[str, float]:
    """
    The four prices an image model is billed at, in USD per 1 million tokens. These are
    separate from resolve_costs() because image billing splits its input by modality
    (prompt text vs. reference images) rather than by cache state, and because forcing
    image output into the text output_cost field would corrupt the text session totals.
    Cached input defaults to the uncached image input rate, which is what a provider
    charging no cache discount looks like.
    """
    image_input = cost_value(costs, "image_input", 0.0)

    return {
        "text_input_cost"   : cost_value(costs, "text_input"   , 0.0),
        "image_input_cost"  : image_input,
        "cached_input_cost" : cost_value(costs, "cached_input" , image_input),
        "image_output_cost" : cost_value(costs, "image_output" , 0.0),
    }


def extract_claude_version(value: Any) -> Version:
    """
    Extracts a Claude major.minor model version from either display names like
    "Claude Opus 4.8" or ids like "claude-opus-4-8-YYYYMMDD".
    """
    text = str(value or "")

    dot_match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", text)
    if dot_match is not None:
        try: return Version(dot_match.group(1))
        except Exception: pass

    hyphen_match = re.search(r"(?<!\d)(\d+)[_-](\d+)(?!\d)", text)
    if hyphen_match is not None:
        try: return Version(f"{hyphen_match.group(1)}.{hyphen_match.group(2)}")
        except Exception: pass

    return Version("0.0")


PREFILL_MODES = {"none", "assistant", "instruction"}
class RuntimeConfig:

    def parse_cost_families(self, prefix: str) -> List[Dict[str, Any]]:
        """
        Per-model cost families: <PREFIX>_MODEL_<FAMILY>_REGEX is matched (re.search)
        against the selected model id; the first declared match wins. The costs live in
        <PREFIX>_MODEL_<FAMILY>_COST as a json5 object, USD per 1 million tokens:
            {input: 0.80, output: 1.60, cache_read: 0.20, cache_write: 1.00}
        Anthropic-style providers can price the two cache TTLs apart with cache_write_5m
        and cache_write_1h. See resolve_costs() for what a missing key falls back to.
        Models matching no family use the provider-level costs.
        """
        families: List[Dict[str, Any]] = []
        family_var_re = re.compile(rf"^{re.escape(prefix)}_MODEL_([A-Za-z0-9]+)_REGEX$")

        for env_name in os.environ:
            match = family_var_re.match(env_name)
            if match is None:
                continue
            family   = match.group(1)
            cost_var = f"{prefix}_MODEL_{family}_COST"

            try:
                pattern = re.compile(os.environ[env_name].strip())
            except re.error as exc:
                print(f"WARNING: {env_name} is not a valid regex ({exc}). Skipping cost family.")
                continue

            try:
                costs = json5.loads(os.getenv(cost_var, "").strip() or "null")
                if not isinstance(costs, dict):
                    raise ValueError("must be a json5 object like {input: 0.8, output: 1.6, cache_read: 0.2}")
                families.append({"name": family.lower(), "regex": pattern, **resolve_costs(costs)})
            except Exception as exc:
                print(f"WARNING: {cost_var}: {exc}. Skipping cost family.")

        return families


    def parse_image_cost_families(self, prefix: str) -> List[Dict[str, Any]]:
        """
        Per-model image prices: <PREFIX>_IMAGE_MODEL_<FAMILY>_REGEX is matched (re.search)
        against the selected image model id; the first declared match wins. The costs live
        in <PREFIX>_IMAGE_MODEL_<FAMILY>_COST as a json5 object, USD per 1 million tokens:
            {text_input: 5.00, image_input: 8.00, image_output: 30.00}
        See resolve_image_costs() for what a missing key falls back to.

        This is deliberately a separate namespace from parse_cost_families(): the same
        provider serves text and image models, and <PREFIX>_MODEL_* must keep pricing only
        the text ones.
        """
        families: List[Dict[str, Any]] = []
        family_var_re = re.compile(rf"^{re.escape(prefix)}_IMAGE_MODEL_([A-Za-z0-9]+)_REGEX$")

        for env_name in os.environ:
            match = family_var_re.match(env_name)
            if match is None:
                continue
            family   = match.group(1)
            cost_var = f"{prefix}_IMAGE_MODEL_{family}_COST"

            try:
                pattern = re.compile(os.environ[env_name].strip())
            except re.error as exc:
                print(f"WARNING: {env_name} is not a valid regex ({exc}). Skipping image cost family.")
                continue

            try:
                costs = json5.loads(os.getenv(cost_var, "").strip() or "null")
                if not isinstance(costs, dict):
                    raise ValueError("must be a json5 object like {text_input: 5.0, image_input: 8.0, image_output: 30.0}")
                families.append({"name": family.lower(), "regex": pattern, **resolve_image_costs(costs)})
            except Exception as exc:
                print(f"WARNING: {cost_var}: {exc}. Skipping image cost family.")

        return families


    def parse_provider(self, name: str, api: str) -> Optional[Dict[str, Any]]:
        """
        One provider entry, from the <NAME>_* variables. Returns None when the provider
        cannot be used at all, which is a missing base URL and nothing else.

        Required:
            <NAME>_BASE_URL            the /v1 root, without a trailing slash
            <NAME>_API_KEY             provider key
        Optional:
            <NAME>_MODELS              comma-separated model ids; skips the /models request
            <NAME>_MODELS_REGEX        keeps only matching ids from the fetched /models list
                                       (providers like OpenAI also serve tts/image/embedding models)
            <NAME>_MAX_TOKENS_PARAM    'auto' (default), 'max_tokens' or 'max_completion_tokens'
            <NAME>_REASONING_SUMMARY   none|auto|concise|detailed; /responses only
            <NAME>_STORE               let the provider retain the response (default false);
                                       /responses only
            <NAME>_BACKGROUND          run the response as a background job, so a dropped
                                       connection can be resumed instead of losing the
                                       turn (default false); /responses only
            <NAME>_EXTRA_BODY          json5 object merged verbatim into every request
                                       (the escape hatch for provider thinking/caching dialects)
            <NAME>_INPUT_TOKEN_COST_USD, <NAME>_OUTPUT_TOKEN_COST_USD,
            <NAME>_CACHE_READ_COST_USD, <NAME>_CACHE_WRITE_COST_USD,
            <NAME>_CACHE_WRITE_5M_COST_USD, <NAME>_CACHE_WRITE_1H_COST_USD
            <NAME>_MODEL_<FAMILY>_REGEX / _COST   see parse_cost_families()

        The wire protocol is not among these. It is the list the name was declared in.
        """
        prefix = re.sub(r"[^A-Z0-9]", "_", name.upper())

        base_url = os.getenv(f"{prefix}_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            print(f"WARNING: provider '{name}' is declared in {API_STYLE_VARS[api]} but {prefix}_BASE_URL is missing. Skipping.")
            return None

        extra_body = {}
        extra_raw  = os.getenv(f"{prefix}_EXTRA_BODY", "").strip()
        if extra_raw:
            try:
                parsed = json5.loads(extra_raw)
                if isinstance(parsed, dict) : extra_body = parsed
                else                        : print(f"WARNING: {prefix}_EXTRA_BODY must be a JSON object. Ignoring.")
            except Exception as exc:
                print(f"WARNING: {prefix}_EXTRA_BODY is not valid json5 ({exc}). Ignoring.")

        models_regex = None
        models_regex_raw = os.getenv(f"{prefix}_MODELS_REGEX", "").strip()
        if models_regex_raw:
            try: models_regex = re.compile(models_regex_raw)
            except re.error as exc:
                print(f"WARNING: {prefix}_MODELS_REGEX is not a valid regex ({exc}). Ignoring.")

        max_tokens_param = os.getenv(f"{prefix}_MAX_TOKENS_PARAM", "auto").strip().lower()
        if max_tokens_param not in MAX_TOKENS_PARAMS:
            print(f"WARNING: {prefix}_MAX_TOKENS_PARAM must be in {MAX_TOKENS_PARAMS}. Defaulting to 'auto'.")
            max_tokens_param = "auto"

        reasoning_summary = os.getenv(f"{prefix}_REASONING_SUMMARY", "auto").strip().lower()
        if reasoning_summary not in REASONING_SUMMARIES:
            print(f"WARNING: {prefix}_REASONING_SUMMARY must be in {REASONING_SUMMARIES}. Defaulting to 'auto'.")
            reasoning_summary = "auto"

        input_cost = getenv_float(f"{prefix}_INPUT_TOKEN_COST_USD", 0.0)
        write_cost = getenv_float(f"{prefix}_CACHE_WRITE_COST_USD", input_cost)

        return {
            "api"                 : api,
            "base_url"            : base_url,
            "api_key"             : os.getenv(f"{prefix}_API_KEY", "").strip(),
            "api_key_name"        : f"{prefix}_API_KEY",
            "models"              : [m.strip() for m in os.getenv(f"{prefix}_MODELS", "").split(",") if m.strip()],
            "models_regex"        : models_regex,
            "max_tokens_param"    : max_tokens_param,
            "reasoning_summary"   : reasoning_summary,
            # The /responses endpoint retains responses for 30 days by default. Chat
            # content is nobody else's business, so the proxy opts out unless asked.
            "store"               : getenv_bool(f"{prefix}_STORE", False),
            # Background responses survive the connection that started them, which is what
            # makes a cut stream resumable rather than a lost turn. OpenAI keeps them for
            # about 10 minutes so they can be polled, whatever 'store' says -- see
            # v1_responses.generate_stream and the README.
            "background"          : getenv_bool(f"{prefix}_BACKGROUND", False),
            "extra_body"          : extra_body,
            "cost_families"       : self.parse_cost_families(prefix),
            "input_cost"          : input_cost,
            "output_cost"         : getenv_float(f"{prefix}_OUTPUT_TOKEN_COST_USD"   , 0.0),
            "cache_read_cost"     : getenv_float(f"{prefix}_CACHE_READ_COST_USD"     , 0.0),
            "cache_write_5m_cost" : getenv_float(f"{prefix}_CACHE_WRITE_5M_COST_USD" , write_cost),
            "cache_write_1h_cost" : getenv_float(f"{prefix}_CACHE_WRITE_1H_COST_USD" , write_cost),
        }


    def reload_from_env(self) -> None:
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = getenv_int("PORT", 5001)

        # The model to start on, as a bare id or "provider/model-id". The prefixed form
        # names its own provider and so resolves without a model list, which is what to
        # use when a provider's /models request is unavailable (see server startup).
        self.model = os.getenv("MODEL", "").strip()
        self.version = extract_claude_version(self.model)
        self.model_info = {}
        # Full model record from the provider model list. Empty until providers.apply_model
        # runs, so capability checks (v1_messages.resolve_thinking) fail closed instead of
        # crashing.
        self.info = {}

        # Active backend: the name of a configured provider. Empty until a model is
        # selected, which is what binds a backend (see providers.apply_model).
        self.backend = ""

        # Every configured provider, keyed by name, in declaration order.
        # See parse_provider() for what a name is configured with.
        self.providers: Dict[str, Dict[str, Any]] = {}
        for api, list_var in API_STYLE_VARS.items():
            for name in [p.strip().lower() for p in os.getenv(list_var, "").split(",") if p.strip()]:
                if name in self.providers:
                    print(f"WARNING: provider '{name}' is declared more than once. Keeping the '{self.providers[name]['api']}' one.")
                    continue
                provider = self.parse_provider(name, api)
                if provider is not None:
                    self.providers[name] = provider

        self.request_timeout_seconds = getenv_float("REQUEST_TIMEOUT_SECONDS", 600.0)

        # Background /responses recovery (see v1_responses). How long a whole turn may
        # take, which is a different question from how long one HTTP call may take:
        # a recovered turn is many calls, and the model reasons silently between them.
        # This is the only thing that ends a turn whose job is still running.
        self.responses_turn_timeout_seconds = max(0.0, getenv_float("RESPONSES_TURN_TIMEOUT_SECONDS", 1800.0))
        self.responses_poll_seconds         = max(0.1, getenv_float("RESPONSES_POLL_SECONDS", 2.0))

        self.proxy_key             = os.getenv("PROXY_KEY", "").strip()
        self.require_proxy_key     = getenv_bool("REQUIRE_PROXY_KEY", True)
        self.allow_key_passthrough = getenv_bool("ALLOW_KEY_PASSTHROUGH", False)

        self.debug_log = getenv_bool("DEBUG_LOG", True)
        self.auto_trim = getenv_bool("AUTO_TRIM", True)
        self.summary_blocks_enabled = getenv_bool("SUMMARY_BLOCKS_ENABLED", True)

        # Leave empty by default. The original notebook used a strong assistant prefill.
        # For safety and reliability, keep this blank unless you have a benign reason to use it.
        self.assistant_prefill = os.getenv("ASSISTANT_PREFILL", "")

        # assistant   : sends assistant_prefill as an assistant message/prefill.
        # instruction : appends an OOC instruction containing assistant_prefill to the last user message.
        # PREFILL_MODE is accepted as a shorter backwards-compatible alias.
        self.assistant_prefill_mode = os.getenv("ASSISTANT_PREFILL_MODE", os.getenv("PREFILL_MODE", "assistant")).strip().lower()

        if self.assistant_prefill_mode not in PREFILL_MODES:
            print(f"WARNING: ASSISTANT_PREFILL_MODE must be in {PREFILL_MODES}. Defaulting to 'none'.")
            self.assistant_prefill_mode = "none"

        # Generation defaults
        self.max_tokens       = getenv_int("MAX_TOKENS", 8192)
        self.send_temperature = getenv_bool("SEND_TEMPERATURE", False)
        self.temperature      = getenv_float("TEMPERATURE", 0.9)
        self.send_top_p       = getenv_bool("SEND_TOP_P", False)
        self.top_p            = getenv_float("TOP_P", 0.95)
        self.send_top_k       = getenv_bool("SEND_TOP_K", False)
        self.top_k            = getenv_int("TOP_K", 75)

        # Thinking
        self.thinking_enabled = getenv_bool("THINKING_ENABLED", False)
        self.use_adaptive     = False
        self.thinking_budget  = getenv_int("THINKING_BUDGET", 2048)
        self.thinking_effort  = os.getenv("THINKING_EFFORT", "medium").lower()

        # Round-trip Anthropic signed thinking blocks through clients that only preserve message.content.
        # 0 disables preservation, N preserves the last N assistant messages, and inf/all preserves every assistant message.
        self.preserve_thinking_blocks = getenv_preserve_thinking_blocks("PRESERVE_THINKING_BLOCKS", "0")

        # Cost tracking. Values are USD per 1 million tokens, and are configured per
        # provider rather than here; these are the inert values a request would be billed
        # at before a model has been selected (see providers.apply_model).
        self.model_cost_family    = ""
        self.input_token_cost_usd = 0.0
        self.output_token_cost_usd   = 0.0
        self.cache_write_5m_cost_usd = 0.0
        self.cache_write_1h_cost_usd = 0.0
        self.cache_read_cost_usd     = 0.0

        # Prompt caching.
        # Anthropic supports automatic top-level caching and explicit block-level caching.
        # This script uses explicit block-level caching because assistant prefill can otherwise
        # become the final cacheable block. Up to four explicit markers are used:
        #   1-2  system / lorebook blocks (when split_lorebook=true and lorebook_at_end=false)
        #    3   manual first-N-message breakpoint
        #    4   automatic end-relative breakpoint before optional prefill
        self.cache_en             = getenv_bool("CACHE_EN", False)
        self.cache_system         = getenv_bool("CACHE_SYSTEM", True)
        self.cache_system_ttl     = getenv_cache_ttl("CACHE_SYSTEM_TTL", "1h")
        self.split_lorebook       = getenv_bool("SPLIT_LOREBOOK", True)
        self.lorebook_at_end      = getenv_bool("LOREBOOK_AT_END", False)
        self.lorebook_xml_at_end  = getenv_bool("LOREBOOK_XML_AT_END", False)
        self.cache_manual_ttl     = getenv_cache_ttl("CACHE_MANUAL_TTL", "1h")
        self.cache_manual_msg     = max(0, getenv_int("CACHE_MANUAL_MSG", 0))
        self.cache_auto_ttl       = getenv_cache_ttl("CACHE_AUTO_TTL", "1h")
        self.cache_auto_msg       = max(0, getenv_int("CACHE_AUTO_MSG", 0))
        self.cache_anthropic_auto = getenv_bool("CACHE_ANTHROPIC_AUTO", False)
        self.cache_anthropic_ttl  = getenv_cache_ttl("CACHE_ANTHROPIC_TTL", "1h")

        # Image generation. A secondary service: none of this touches the active text
        # provider or model, and the image model is never bound through providers.apply_model.
        self.image_enabled      = getenv_bool("IMAGE_GENERATION_ENABLED", False)
        # Whether <IMAGE_REQUEST_TAG> blocks in user messages are honored. Turning this off
        # leaves the direct /v1/images/generations endpoint working.
        self.image_chat_enabled = getenv_bool("IMAGE_CHAT_ENABLED", True)
        self.image_provider     = os.getenv("IMAGE_PROVIDER", "").strip().lower()
        self.image_model        = os.getenv("IMAGE_MODEL", "").strip()
        self.image_output_dir   = os.getenv("IMAGE_OUTPUT_DIR", "generated_images").strip() or "generated_images"

        # Defaults applied to every request, direct or chat-triggered, that does not
        # override them. 'auto' lets the provider decide.
        self.image_default_size       = os.getenv("IMAGE_DEFAULT_SIZE", "1024x1024").strip().lower()
        self.image_default_quality    = getenv_choice("IMAGE_DEFAULT_QUALITY"   , "medium", IMAGE_QUALITIES)
        self.image_default_format     = getenv_choice("IMAGE_DEFAULT_FORMAT"    , "png"   , IMAGE_FORMATS)
        self.image_default_background = getenv_choice("IMAGE_DEFAULT_BACKGROUND", "opaque", IMAGE_BACKGROUNDS)
        self.image_default_n          = max(1, getenv_int("IMAGE_DEFAULT_N", 1))
        self.image_default_batch      = getenv_bool("IMAGE_DEFAULT_BATCH", False)

        # Validation limits, checked before the provider is contacted.
        self.image_max_n            = max(1, getenv_int("IMAGE_MAX_N", 4))
        self.image_max_prompt_chars = max(1, getenv_int("IMAGE_MAX_PROMPT_CHARS", 20000))
        if self.image_default_n > self.image_max_n:
            print(f"WARNING: IMAGE_DEFAULT_N ({self.image_default_n}) exceeds IMAGE_MAX_N ({self.image_max_n}). Clamping.")
            self.image_default_n = self.image_max_n

        # The tag a user message carries an image request in. Assistant output is never
        # scanned, so this is the only trigger there is.
        self.image_request_tag = os.getenv("IMAGE_REQUEST_TAG", "image_generation").strip() or "image_generation"

        # The image model list is fetched separately from the conversational one, because
        # <NAME>_MODELS_REGEX exists precisely to keep image models out of that list.
        self.image_models_regex = os.getenv("IMAGE_MODELS_REGEX", "image").strip()

        self.image_cost_reporting   = getenv_bool("IMAGE_COST_REPORTING"  , True)
        self.image_manifest_enabled = getenv_bool("IMAGE_MANIFEST_ENABLED", True)
        # Prompts are chat content. Writing them into a sidecar that outlives the session is
        # a separate decision from printing them to a debug console, so it gets its own switch.
        self.image_manifest_prompts = getenv_bool("IMAGE_MANIFEST_PROMPTS", True)

        self.image_batch_window     = os.getenv("IMAGE_BATCH_COMPLETION_WINDOW", "24h").strip() or "24h"
        # What a batch is billed at relative to immediate generation. The provider reports
        # the same token counts either way, so nothing in the usage payload reveals the
        # discount; it has to be configured. Defaults to 1.0 rather than to any provider's
        # published rate -- over-reporting a budget is the safe direction to be wrong in.
        self.image_batch_multiplier = max(0.0, getenv_float("IMAGE_BATCH_COST_MULTIPLIER", 1.0))

        # Background batch retrieval. A batch completes on the provider's schedule rather
        # than the proxy's, so without this its images sit finished but never collected
        # until somebody runs the CLI. The floor keeps a misconfigured interval from
        # turning the poller into a request loop.
        self.image_batch_auto_poll    = getenv_bool("IMAGE_BATCH_AUTO_POLL", True)
        self.image_batch_poll_seconds = max(10.0, getenv_float("IMAGE_BATCH_POLL_SECONDS", 300.0))

        # Image editing. Reference images are read off this machine and uploaded to the
        # provider, which makes every path here a read primitive -- hence the allowlist.
        # Gateway failures (Cloudflare 520s in front of the provider) are common enough to
        # make a single-attempt image call unreliable. Only failures that produced no image
        # are retried, so this never pays for the same picture twice.
        self.image_retry_attempts         = max(1, getenv_int("IMAGE_RETRY_ATTEMPTS", 3))
        self.image_retry_backoff_seconds  = max(0.0, getenv_float("IMAGE_RETRY_BACKOFF_SECONDS", 5.0))

        self.image_edit_enabled      = getenv_bool("IMAGE_EDIT_ENABLED", True)
        self.image_edit_max_images   = max(1, getenv_int("IMAGE_EDIT_MAX_IMAGES", 4))
        self.image_edit_max_bytes    = max(1, getenv_int("IMAGE_EDIT_MAX_BYTES", 20*1024*1024))
        # Edits usually want the source geometry kept, which is what 'auto' asks for.
        self.image_edit_default_size = os.getenv("IMAGE_EDIT_DEFAULT_SIZE", "auto").strip().lower() or "auto"

        # Whether a path written in a chat block may be read at all. Off by default: this
        # proxy is built to sit behind a public tunnel, and a prompt-supplied path is an
        # arbitrary-file-read-and-exfiltrate primitive for anyone who reaches it. The CLI
        # slots need none of this, since paths there come from whoever runs the console.
        self.image_edit_allow_prompt_paths = getenv_bool("IMAGE_EDIT_ALLOW_PROMPT_PATHS", False)
        self.image_edit_roots = [
            os.path.realpath(os.path.expanduser(root.strip()))
            for root in os.getenv("IMAGE_EDIT_ROOTS", "").split(",")
            if root.strip()
        ]

        # Fallback per-image price estimates, for providers that return no usage object.
        # {"model-regex": {quality: {size: usd}}}, with "*" accepted for either key.
        self.image_price_table: Dict[str, Any] = {}
        price_table_raw = os.getenv("IMAGE_PRICE_TABLE", "").strip()
        if price_table_raw:
            try:
                parsed = json5.loads(price_table_raw)
                if isinstance(parsed, dict) : self.image_price_table = parsed
                else                        : print("WARNING: IMAGE_PRICE_TABLE must be a json5 object. Ignoring.")
            except Exception as exc:
                print(f"WARNING: IMAGE_PRICE_TABLE is not valid json5 ({exc}). Ignoring.")

        # Resolved prices for the selected image model. Populated by
        # v1_images.apply_image_model(); inert until then.
        self.image_cost_family     = ""
        self.image_text_input_cost   = 0.0
        self.image_image_input_cost  = 0.0
        self.image_cached_input_cost = 0.0
        self.image_output_cost       = 0.0

        self.error_log_path = os.getenv("ERROR_LOG_PATH", "revpy_error.log")
        self.model_list_timeout_seconds = getenv_float("MODEL_LIST_TIMEOUT_SECONDS", 10.0)


    def set_prefill_mode(self, mode: str) -> None:
        if not mode in PREFILL_MODES : print(f"WARNING: ASSISTANT_PREFILL_MODE must be in {PREFILL_MODES}. Defaulting to 'none'."); return
        if mode == "assistant" :
            if self.version >= Version("4.6") : print("Mythos class models (>= 4.6) do not support assistant prefill."); return
            if self.thinking_enabled          : print("While thinking is enabled, prefill mode cannot be assistant."); return
        self.assistant_prefill_mode = mode

    def set_prefill(self, prefill: str) -> None:
        self.assistant_prefill = prefill

    def set_think_effort(self, effort: str) -> bool:
        if not effort in THINK_EFFORTS:
            print(f"Allowed thinking efforts: {THINK_EFFORTS}.")
            return False
        self.thinking_effort = effort
        return True

    def set_think_budget(self, budget: int) -> bool:
        if budget < 0 or budget > self.max_tokens:
            print(f"Thinking budget {budget} must be in range (0:max_tokens] - (0:{self.max_tokens}].")
            return False
        self.thinking_budget = budget
        return True

    def print_lorebook_status(self) -> None:
        if cfg.split_lorebook      : print("  Lorebook split   ✅")
        else                       : print("  Lorebook split   ❌")
        if cfg.lorebook_at_end     : print("  Lorebook at end  ✅")
        else                       : print("  Lorebook at end  ❌")
        if cfg.lorebook_xml_at_end : print("  XML at end       ✅")
        else                       : print("  XML at end       ❌")

    def print_cache_status(self) -> None:
        if not self.split_lorebook  : system_str = "Lorebook not split"
        else:
            if self.lorebook_at_end : system_str = "Lorebook split and moved to end"
            else                    : system_str = "Lorebook split"
        if self.lorebook_xml_at_end:
            system_str = f"{system_str}; XML lorebook moved to end"

        if self.cache_en              : print( "  Cache enabled   ✅")
        else                          : print( "  Cache enabled   ❌")
        if self.cache_system          : print(f"  System cache    ✅  {self.cache_system_ttl} | {system_str}")
        else                          : print(f"  System cache    ❌  {self.cache_system_ttl} | {system_str}")
        if self.cache_manual_msg <= 0 : print(f"  Manual cache    ❌  {self.cache_manual_ttl} | 1 is the first (intro) message")
        else                          : print(f"  Manual cache   {self.cache_manual_msg:3d}  {self.cache_manual_ttl} | 1 is the first (intro) message")
        if self.cache_manual_msg <= 0 : print(f"  Auto   cache    ❌  {self.cache_auto_ttl} | 1 is the last user message")
        else                          : print(f"  Auto   cache   {self.cache_auto_msg:3d}  {self.cache_auto_ttl} | 1 is the last user message")
        if self.cache_anthropic_auto  : print(f"  Anthropic auto  ✅  {self.cache_anthropic_ttl}")
        else                          : print(f"  Anthropic auto  ❌  {self.cache_anthropic_ttl}")

    def print_image_status(self) -> None:
        if self.image_enabled      : print("  Image generation ✅")
        else                       : print("  Image generation ❌  (IMAGE_GENERATION_ENABLED)")
        if self.image_chat_enabled : print(f"  Chat trigger     ✅  <{self.image_request_tag}> in a user message")
        else                       : print(f"  Chat trigger     ❌  direct endpoint only")
        print(f"  Provider/model      {self.image_provider or '(unset)'}/{self.image_model or '(unset)'}")
        print(f"  Cost family         {self.image_cost_family or '(unresolved)'}")
        print(f"  Output dir          {self.image_output_dir}")
        print(f"  Defaults            size={self.image_default_size} quality={self.image_default_quality} format={self.image_default_format} background={self.image_default_background} n={self.image_default_n} batch={self.image_default_batch}")
        print(f"  Limits              max_n={self.image_max_n} max_prompt_chars={self.image_max_prompt_chars}")
        if self.image_batch_auto_poll : print(f"  Batch auto-poll  ✅  every {self.image_batch_poll_seconds:.0f}s, window {self.image_batch_window}, billed x{self.image_batch_multiplier:g}")
        else                          : print(f"  Batch auto-poll  ❌  retrieve with 'image batch get <number>'")

    def check_cache_block_num(self) -> None:
        cache_blocks_active : int = 0
        if self.cache_system:
            if self.split_lorebook and not self.lorebook_at_end : cache_blocks_active += 2
            else                                                : cache_blocks_active += 1
        if self.cache_auto_msg       : cache_blocks_active += 1
        if self.cache_manual_msg     : cache_blocks_active += 1
        if self.cache_anthropic_auto : cache_blocks_active += 1
        if (cache_blocks_active > 4):
            print("Not more than four cache blocks can be active at a time. Disabling auto cache block.")
            self.cache_auto_msg = 0

    def set_lorebook_split(self, en: bool) -> None:
        cfg.split_lorebook = en;
        self.check_cache_block_num()

    def set_cache_msg_num(self, type: str, msg_num: int) -> bool:
        if type in {"m", "man", "manual"}:
            self.cache_manual_msg = msg_num
            if msg_num == 0 : print("Manual cache marker disabled.")
            else            : print(f"Manual cache marker targets {cfg.cache_manual_msg} message(s) from start.")
        elif type in {"a", "auto"}:
            self.cache_auto_msg = msg_num
            if msg_num == 0 : print("Auto cache marker disabled.")
            else            : print(f"Auto cache marker targets {cfg.cache_auto_msg} message(s) from end.")
        elif type in {"s", "sys", "system"}:
            self.cache_system = msg_num > 0
            if msg_num <= 0 : print("System message caching disabled.")
            else            : print("System message caching enabled.")
        elif type in ("ant", "anthropic"):
            self.cache_anthropic_auto = msg_num > 0
            if msg_num <= 0 : print("Anthropic auto caching disabled.")
            else            : print("Anthropic auto caching enabled.")
        else:
            print(f"Unknown cache type '{type}'")
            return False
        self.check_cache_block_num();
        return True

    def set_cache_dur(self, type: str, dur: str) -> bool:
        if not dur in {"5m", "1h"}:
            print(f"Invalid duration type '{dur}'.")
            return False
        if type in {"m", "man", "manual"}:
            self.cache_manual_ttl= dur
            print(f"Manual cache marker duration is now {dur}.")
        elif type in {"a", "auto"}:
            self.cache_auto_ttl = dur
            print(f"Auto cache marker duration is now {dur}.")
        elif type in {"s", "sys", "system"}:
            self.cache_system_ttl = dur
            print(f"System cache marker duration is now {dur}.")
        elif type in ("ant", "anthropic"):
            self.cache_anthropic_ttl = dur
            print(f"Anthropic cache marker duration is now {dur}.")
        else:
            print(f"Unknown cache type '{type}'.")
            return False
        return True

    def set_think_blocks_to_preserve(self, block_num: int) -> bool:
        if block_num < 0:
            print("Number of blocks to preserve must be a natural numbers.")
            return False
        self.preserve_thinking_blocks = block_num
        return True

    def print_status(self) -> None:
        preserve_str = "inf" if self.preserve_thinking_blocks == UINT64_MAX else str(self.preserve_thinking_blocks)

        print()
        print("=== Runtime config start ===")
        print(f"host                   = {self.host} (restart required to change)")
        print(f"port                   = {self.port} (restart required to change)")
        print(f"backend                = {self.backend}")
        print(f"model                  = {self.model}")
        print(f"model_cost_family      = {self.model_cost_family}")
        print(f"  input_token_cost     = {self.input_token_cost_usd}")
        print(f"  output_token_cost    = {self.output_token_cost_usd}")
        print(f"  cache_write_5m_cost  = {self.cache_write_5m_cost_usd}")
        print(f"  cache_write_1h_cost  = {self.cache_write_1h_cost_usd}")
        print(f"  cache_read_cost      = {self.cache_read_cost_usd}")
        print(f"require_proxy_key      = {self.require_proxy_key}")
        print(f"allow_key_passthrough  = {self.allow_key_passthrough}")
        print(f"debug_log              = {self.debug_log}")
        print(f"auto_trim              = {self.auto_trim}")
        print(f"summary_blocks_enabled = {self.summary_blocks_enabled}")
        print(f"assistant_prefill      = {'set' if self.assistant_prefill.strip() else 'empty'}")
        print(f"assistant_prefill_mode = {self.assistant_prefill_mode}")
        print(f"temperature            = {self.temperature}")
        print(f"top_p                  = {self.top_p}")
        print(f"top_k                  = {self.top_k}")
        print(f"max_tokens             = {self.max_tokens}")
        print(f"cache_en               = {self.cache_en}")
        print(f"cache_system           = {self.cache_system}")
        print(f"cache_system_ttl       = {self.cache_system_ttl}")
        print(f"split_lorebook         = {self.split_lorebook}")
        print(f"lorebook_at_end        = {self.lorebook_at_end}")
        print(f"lorebook_xml_at_end    = {self.lorebook_xml_at_end}")
        print(f"cache_manual_ttl       = {self.cache_manual_ttl}")
        print(f"cache_manual_msg       = {self.cache_manual_msg}")
        print(f"cache_auto_ttl         = {self.cache_auto_ttl}")
        print(f"cache_auto_msg         = {self.cache_auto_msg}")
        print(f"cache_anthropic_auto   = {self.cache_anthropic_auto}")
        print(f"cache_anthropic_ttl    = {self.cache_anthropic_ttl}")
        print(f"thinking               = {self.thinking_enabled}")
        print(f"adaptive_thinking      = {self.use_adaptive}")
        print(f"thinking_budget        = {self.thinking_budget}")
        print(f"thinking_effort        = {self.thinking_effort}")
        print(f"preserve_thinking      = {preserve_str}")
        print(f"error_log_path         = {self.error_log_path}")
        print(f"model_list_timeout_sec = {self.model_list_timeout_seconds}")
        print(f"image_enabled          = {self.image_enabled}")
        print(f"image_provider         = {self.image_provider}")
        print(f"image_model            = {self.image_model}")
        print(f"image_cost_family      = {self.image_cost_family}")
        print(f"  text_input_cost      = {self.image_text_input_cost}")
        print(f"  image_input_cost     = {self.image_image_input_cost}")
        print(f"  image_output_cost    = {self.image_output_cost}")
        print(f"image_output_dir       = {self.image_output_dir}")
        print("=== Runtime config end ===")
        print()


# The single runtime configuration instance shared by every module.
# It is created empty here and populated by cfg.reload_from_env() at startup.
# Always mutate it in place; never rebind the name.
cfg = RuntimeConfig()


def average_cost_per_token_usd(total_cost_usd: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd/total_tokens


# Session cost tracking variables
SESSION_TTL_SPENT_USD       = 0.0
SESSION_TTL_INPUT_COST_USD  = 0.0
SESSION_TTL_OUTPUT_COST_USD = 0.0
SESSION_TTL_INPUT_TOK       = 0
SESSION_TTL_OUTPUT_TOK      = 0
SESSION_CACHE_NET_COST_USD  = 0.0
SESSION_COST_LOCK           = threading.Lock()


def session_cost_totals_locked() -> Dict[str, Any]:
    return {
        "total_spent_usd"        : SESSION_TTL_SPENT_USD,
        "input_cost_usd"         : SESSION_TTL_INPUT_COST_USD,
        "output_cost_usd"        : SESSION_TTL_OUTPUT_COST_USD,
        "input_tokens"           : SESSION_TTL_INPUT_TOK,
        "output_tokens"          : SESSION_TTL_OUTPUT_TOK,
        "cache_net_cost_usd"     : SESSION_CACHE_NET_COST_USD,
        "average_input_cost_usd" : average_cost_per_token_usd(SESSION_TTL_INPUT_COST_USD, SESSION_TTL_INPUT_TOK),
    }


def add_session_cost(request_total_cost: float, total_input_cost: float, output_cost: float, input_tokens: int, output_tokens: int, cache_net_cost: float) -> Dict[str, Any]:
    global SESSION_TTL_SPENT_USD, SESSION_TTL_INPUT_COST_USD, SESSION_TTL_OUTPUT_COST_USD, SESSION_TTL_INPUT_TOK, SESSION_TTL_OUTPUT_TOK, SESSION_CACHE_NET_COST_USD

    with SESSION_COST_LOCK:
        SESSION_TTL_SPENT_USD       += request_total_cost
        SESSION_TTL_INPUT_COST_USD  += total_input_cost
        SESSION_TTL_OUTPUT_COST_USD += output_cost
        SESSION_TTL_INPUT_TOK       += input_tokens
        SESSION_TTL_OUTPUT_TOK      += output_tokens
        SESSION_CACHE_NET_COST_USD  += cache_net_cost
        return session_cost_totals_locked()


def session_cost_snapshot() -> Dict[str, Any]:
    with SESSION_COST_LOCK:
        return session_cost_totals_locked()


# Image session cost tracking. Deliberately separate from the text totals above: an
# image is not an output token, and folding its cost into SESSION_TTL_OUTPUT_COST_USD
# would make the average-cost-per-token report meaningless. Immediate and Batch API
# spending are tracked apart because they are billed at different rates.
SESSION_IMAGE_IMMEDIATE_COST_USD = 0.0
SESSION_IMAGE_BATCH_COST_USD     = 0.0
SESSION_IMAGE_COUNT              = 0
SESSION_IMAGE_BATCH_COUNT        = 0
SESSION_IMAGE_TEXT_INPUT_TOK     = 0
SESSION_IMAGE_IMAGE_INPUT_TOK    = 0
SESSION_IMAGE_OUTPUT_TOK         = 0
SESSION_IMAGE_ESTIMATED          = False
IMAGE_COST_LOCK                  = threading.Lock()


def image_cost_totals_locked() -> Dict[str, Any]:
    return {
        "immediate_cost_usd" : SESSION_IMAGE_IMMEDIATE_COST_USD,
        "batch_cost_usd"     : SESSION_IMAGE_BATCH_COST_USD,
        "total_cost_usd"     : SESSION_IMAGE_IMMEDIATE_COST_USD + SESSION_IMAGE_BATCH_COST_USD,
        "images"             : SESSION_IMAGE_COUNT,
        "batch_images"       : SESSION_IMAGE_BATCH_COUNT,
        "text_input_tokens"  : SESSION_IMAGE_TEXT_INPUT_TOK,
        "image_input_tokens" : SESSION_IMAGE_IMAGE_INPUT_TOK,
        "image_output_tokens": SESSION_IMAGE_OUTPUT_TOK,
        # True once any total includes a price the provider did not actually report.
        "contains_estimates" : SESSION_IMAGE_ESTIMATED,
    }


def add_image_cost(cost_usd: float, text_input_tok: int, image_input_tok: int, output_tok: int, images: int, batch: bool, estimated: bool) -> Dict[str, Any]:
    global SESSION_IMAGE_IMMEDIATE_COST_USD, SESSION_IMAGE_BATCH_COST_USD, SESSION_IMAGE_COUNT
    global SESSION_IMAGE_BATCH_COUNT, SESSION_IMAGE_TEXT_INPUT_TOK, SESSION_IMAGE_IMAGE_INPUT_TOK
    global SESSION_IMAGE_OUTPUT_TOK, SESSION_IMAGE_ESTIMATED

    with IMAGE_COST_LOCK:
        if batch:
            SESSION_IMAGE_BATCH_COST_USD += cost_usd
            SESSION_IMAGE_BATCH_COUNT    += images
        else:
            SESSION_IMAGE_IMMEDIATE_COST_USD += cost_usd
            SESSION_IMAGE_COUNT              += images
        SESSION_IMAGE_TEXT_INPUT_TOK  += text_input_tok
        SESSION_IMAGE_IMAGE_INPUT_TOK += image_input_tok
        SESSION_IMAGE_OUTPUT_TOK      += output_tok
        SESSION_IMAGE_ESTIMATED        = SESSION_IMAGE_ESTIMATED or estimated
        return image_cost_totals_locked()


def image_cost_snapshot() -> Dict[str, Any]:
    with IMAGE_COST_LOCK:
        return image_cost_totals_locked()


def get_bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def resolve_api_key(configured_key: str, key_name: str) -> str:
    """
    Per-request auth shared by every backend.

    Recommended public-tunnel mode:
        .env contains the provider API key and PROXY_KEY. JanitorAI uses PROXY_KEY as the proxy key.

    Optional compatibility mode:
        ALLOW_KEY_PASSTHROUGH=true lets the incoming Bearer token act as the provider key.
    """
    provided_key = get_bearer_token()

    if configured_key:
        if cfg.require_proxy_key:
            if not cfg.proxy_key             : abort(500, description=("Server is configured with REQUIRE_PROXY_KEY=true, but PROXY_KEY is missing from .env."))
            if provided_key != cfg.proxy_key : abort(401, description="Invalid proxy key.")
        return configured_key
    if cfg.allow_key_passthrough:
        if not provided_key : abort(401, description="Missing Authorization bearer token.")
        return provided_key
    abort(500, description=(f"{key_name} is not configured. Either set {key_name} and PROXY_KEY in .env, or set ALLOW_KEY_PASSTHROUGH=true."))
    raise RuntimeError("unreachable")


# Backend error rendering. Every backend raises errors carrying the same two things:
# a status code and a JSON body with an 'error' object -- the Anthropic SDK does it
# natively, and providers.ProviderError is built to match. So these are shared rather
# than Anthropic-only, despite having started out that way.
def error_body(exc: Exception) -> Optional[Dict[str, Any]]:
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


def error_message(body: Optional[Dict[str, Any]], fallback: str) -> str:
    if isinstance(body, dict):
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if message:
                return str(message)

        return json.dumps(body, ensure_ascii=False, default=str)

    return fallback


def print_error(exc: Exception) -> None:
    """
    Prints an error to the console in red.

    An upstream refusal arrives with a JSON body, which is printed above the message.
    Anything else is a fault in this proxy rather than an answer from a provider, and
    says so plainly: the two mean very different things to whoever is reading the
    terminal. The full traceback for those goes to the error log, not here.
    """
    ANSI_RED   : str = "\033[31m"
    ANSI_RESET : str = "\033[0m"

    body     = error_body(exc)
    fallback = str(exc) or exc.__class__.__name__
    # The Anthropic SDK does not always populate a body, so anything it raises counts
    # as an API error regardless; every other backend carries one (providers.ProviderError).
    from_api = body is not None or exc.__class__.__module__.split(".", 1)[0] == "anthropic"

    if not from_api:
        print(f"{ANSI_RED}Proxy error (not an API response): {exc.__class__.__name__}: {fallback}{ANSI_RESET}")
        print(f"{ANSI_RED}This is a bug in the proxy. See the error log for the traceback.{ANSI_RESET}")
        return

    if body is None:
        body = {
            "type"  : "error",
            "error" : {
                "type"    : exc.__class__.__name__,
                "message" : fallback,
            },
        }

    message = error_message(body, fallback)
    print(json.dumps(body, indent=2, ensure_ascii=False, default=str))
    print(f"{ANSI_RED}{message}{ANSI_RESET}")


def content_to_plain_text(content: Any) -> str:
    """
    The proxy primarily expects text-only OpenAI-style messages.

    If a client sends a list of text parts, this joins text parts.
    Non-text parts are serialized. This is intentionally conservative;
    it does not implement OpenAI-image-to-Anthropic-image conversion.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" : parts.append(str(item.get("text", "")))
                else                          : parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def append_text_to_content(content: Any, text: str) -> Any:
    """
    Appends text while preserving Anthropic list-form content blocks.
    """
    if text is None:
        text = ""

    if isinstance(content,  str) : return content + "\n" + text
    if isinstance(content, list) : return content + [{"type": "text", "text": "\n" + text}]

    return str(content) + "\n" + text


def trim_to_end_sentence(input_str: str, include_newline: bool = False) -> str:
    punctuation = set([".", "!", "?", "*", '"', ")", "}", "`", "]", "$", "。", "！", "？", "”", "）", "】", "’", "」"])

    last = -1
    for i in range(len(input_str) - 1, -1, -1):
        char = input_str[i]

        if char in punctuation:
            if i > 0 and input_str[i - 1] in [" ", "\n"] : last = i - 1
            else                                         : last = i
            break

        if include_newline and char == "\n":
            last = i
            break

    if last == -1:
        return input_str.rstrip()

    return input_str[: last + 1].rstrip()


def make_prefill_instruction(prefix_text: str) -> str:
    """
    Creates the instruction-mode version of ASSISTANT_PREFILL.

    This avoids assistant prefill by telling the model, inside the
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


# Normalized token counts. Every backend's parse_usage() returns this shape, whatever
# its provider called the fields, and everything downstream reads only this:
#   prompt      all input tokens, including the cached and newly written ones
#   completion  output tokens, reasoning included
#   total       prompt + completion
#   uncached    input tokens billed at the full input rate
#   cached      input tokens served from cache (a cache read)
#   write_1h    input tokens written to a 1h cache; Anthropic only, 0 elsewhere
#   write_5m    input tokens written to cache at the 5m rate, and the single rate
#               charged by providers that offer no TTL choice
#   reasoning   reasoning tokens, or None when the provider does not report a count
#               (see track_usage: a zero there would be a claim, not a measurement)
def print_payload(body: Dict[str, Any]) -> None:
    if not cfg.debug_log:
        return
    print()
    print(f"=== {cfg.backend} payload start ===")
    print(json.dumps(body, indent=2, ensure_ascii=False, default=str))
    print(f"=== {cfg.backend} payload end ===")


def usage_to_cost_tokens(counts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps normalized counts to the dict track_usage() bills from.
    """
    return {
        "uncached_input" : counts["uncached"],
        "cache_read"     : counts["cached"],
        "cache_write_1h" : counts["write_1h"],
        "cache_write_5m" : counts["write_5m"],
        "output"         : counts["completion"],
        # Splits the output line into thinking and visible text; None when unreported.
        "reasoning"      : counts["reasoning"],
    }


def print_usage(counts: Dict[str, Any]) -> None:
    track_usage(usage_to_cost_tokens(counts))


def usage_to_openai_dict(counts: Dict[str, Any]) -> Dict[str, int]:
    """
    Maps normalized counts to the usage object returned to the client, so it sees one
    consistent shape regardless of which backend served the request.
    """
    return {
        "prompt_tokens"               : counts["prompt"],
        "completion_tokens"           : counts["completion"],
        "total_tokens"                : counts["total"],
        "input_tokens_uncached"       : counts["uncached"],
        "cache_creation_input_tokens" : counts["write_1h"] + counts["write_5m"],
        "cache_read_input_tokens"     : counts["cached"],
    }


def tok_usd(tokens: int, usd_per_million_tokens: float) -> float:
    return (tokens*usd_per_million_tokens)/1_000_000.0


def fmt_usd(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.6f}"


def track_image_usage(counts: Dict[str, Any], images: int, batch: bool, model: str) -> float:
    """
    Costs one image request and folds it into the image session totals. Returns the
    request cost so the caller can report it back to the client.

    'counts' carries text_input, image_input, output and an 'estimated' flag. The
    providers seen so far return an exact usage object, so estimation is the fallback
    path for those that do not (see v1_images.estimate_image_cost).
    """
    text_input_tok  = max(0, int(counts.get("text_input" , 0) or 0))
    image_input_tok = max(0, int(counts.get("image_input", 0) or 0))
    output_tok      = max(0, int(counts.get("output"     , 0) or 0))
    estimated       = bool(counts.get("estimated", False))

    # A batch reports the same token counts as an immediate request but is invoiced at a
    # different rate, so the discount is applied here rather than hidden in the prices.
    multiplier = cfg.image_batch_multiplier if batch else 1.0

    if estimated:
        request_cost = float(counts.get("estimated_cost_usd", 0.0) or 0.0)*multiplier
        text_cost = image_cost = output_cost = 0.0
    else:
        text_cost    = tok_usd(text_input_tok , cfg.image_text_input_cost) *multiplier
        image_cost   = tok_usd(image_input_tok, cfg.image_image_input_cost)*multiplier
        output_cost  = tok_usd(output_tok     , cfg.image_output_cost)     *multiplier
        request_cost = text_cost + image_cost + output_cost

    session = add_image_cost(
        cost_usd        = request_cost,
        text_input_tok  = text_input_tok,
        image_input_tok = image_input_tok,
        output_tok      = output_tok,
        images          = images,
        batch           = batch,
        estimated       = estimated,
    )

    if not cfg.debug_log or not cfg.image_cost_reporting:
        return request_cost

    mode = f"batch x{multiplier:g}" if batch else "immediate"
    print()
    print(f"=== image usage start ({cfg.image_provider}/{model}, {mode}) ===")
    print("Request:")
    print(f"    Images             = {images}")
    if estimated:
        print(f"    Tokens             = not reported by the provider; cost estimated from IMAGE_PRICE_TABLE")
        print(f"    Estimated cost     = {fmt_usd(request_cost)}")
    else:
        print( "    Input tokens       =       text +      image")
        print( "    {:18d} = {:10d} + {:10d}".format(text_input_tok + image_input_tok, text_input_tok, image_input_tok))
        print( "    {:>18s} = {:>10s} + {:>10s}".format(fmt_usd(text_cost + image_cost), fmt_usd(text_cost), fmt_usd(image_cost)))
        print(f"    Output tokens      = {output_tok:d} ({fmt_usd(output_cost)})")
        print(f"    Total cost         = {fmt_usd(request_cost)}")
    print("Session:")
    print("    Images             = {:d} immediate + {:d} batch".format(session["images"], session["batch_images"]))
    print("    Immediate cost     = {}".format(fmt_usd(session["immediate_cost_usd"])))
    print("    Batch cost         = {}".format(fmt_usd(session["batch_cost_usd"])))
    print("    Image total        = {}{}".format(fmt_usd(session["total_cost_usd"]), " (contains estimates)" if session["contains_estimates"] else ""))
    print("    Text total         = {}".format(fmt_usd(session_cost_snapshot()["total_spent_usd"])))
    print("    Combined session   = {}".format(fmt_usd(session["total_cost_usd"] + session_cost_snapshot()["total_spent_usd"])))
    print(f"=== image usage end ===")
    print("> ", end="", flush=True)

    return request_cost


def track_usage(tokens: Dict[str, Any]) -> None:
    """
    Model-agnostic cost accounting over a normalized token-count dict:
        uncached_input, cache_read, cache_write_1h, cache_write_5m, output
    Backends are responsible for mapping their provider's usage payload to this
    shape (for providers without cache writes, the write counts are simply 0).

    The optional 'reasoning' key splits the output tokens into thinking and visible
    text, both billed at the output rate. Leave it out (or None) when the provider
    does not report the count -- Anthropic and Aion reason without ever saying how
    much, and printing a zero there would be a lie rather than a measurement.
    """
    def cache_lbl(net_cost_usd: float) -> str:
        if net_cost_usd < 0: return f"{fmt_usd(abs(net_cost_usd))} saved"
        if net_cost_usd > 0: return f"{fmt_usd(net_cost_usd)} lost"
        return "$0.000000 break-even"

    input_tok            = tokens["uncached_input"]
    cache_read           = tokens["cache_read"]
    ephemeral_1h         = tokens["cache_write_1h"]
    ephemeral_5m         = tokens["cache_write_5m"]
    output_tok           = tokens["output"]
    cache_creation_input = ephemeral_1h + ephemeral_5m
    ttl_tokens           = input_tok + cache_read + cache_creation_input

    # None when the backend cannot report it; see the docstring.
    reasoning_tok = tokens.get("reasoning")
    if reasoning_tok is not None:
        reasoning_tok = min(max(0, int(reasoning_tok)), output_tok)

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

    print(f"=== {cfg.backend} usage start ===")
    print("Request:")
    print("    Input tokens       =   uncached + cache read + cache write (        1h +         5m)")
    print("    {:18d} = {:10d} + {:10d} + {:11d} ({:10d} + {:10d})".format(ttl_tokens, input_tok, cache_read, cache_creation_input, ephemeral_1h, ephemeral_5m))
    print("    {:>18s} = {:>10s} + {:>10s} + {:>11s} ({:>10s} + {:>10s})".format(fmt_usd(total_input_cost), fmt_usd(input_cost), fmt_usd(cache_read_cost), fmt_usd(cache_write_cost), fmt_usd(cache_write_1h_cost), fmt_usd(cache_write_5m_cost)))
    if reasoning_tok is None:
        print("    Output tokens      = {:d} ({})".format(output_tok, fmt_usd(output_cost)))
    else:
        visible_tok    = output_tok - reasoning_tok
        reasoning_cost = tok_usd(reasoning_tok, cfg.output_token_cost_usd)
        visible_cost   = tok_usd(visible_tok  , cfg.output_token_cost_usd)
        print("    Output tokens      =  reasoning +    visible")
        print("    {:18d} = {:10d} + {:10d}".format(output_tok, reasoning_tok, visible_tok))
        print("    {:>18s} = {:>10s} + {:>10s}".format(fmt_usd(output_cost), fmt_usd(reasoning_cost), fmt_usd(visible_cost)))
    print("    Cache cost         = {} ({})".format(fmt_usd(request_cache_net_cost), cache_lbl(request_cache_net_cost)))
    print("    Total cost         = {}".format(fmt_usd(request_total_cost)))
    print("Session:")
    print("    Input tokens       = {:d} ({})".format(session["input_tokens"], fmt_usd(session["input_cost_usd"])))
    print("    Output tokens      = {:d} ({})".format(session["output_tokens"], fmt_usd(session["output_cost_usd"])))
    print("    Cache cost         = {} ({})".format(fmt_usd(session["cache_net_cost_usd"]), cache_lbl(session["cache_net_cost_usd"])))
    print("    Average input cost = {} / MTok.".format(fmt_usd(session["average_input_cost_usd"]*1_000_000)))
    print("    Total cost         = {} ({} input / {} output)".format(fmt_usd(session["total_spent_usd"]), fmt_usd(session["input_cost_usd"]), fmt_usd(session["output_cost_usd"])))
    print(f"=== {cfg.backend} usage end ===")
    print("> ", end="", flush=True)
