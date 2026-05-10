import anthropic
import base64
import json
import os
import re
import time
import traceback
import threading

from dotenv            import load_dotenv
from flask             import Flask, Response, abort, jsonify, request, stream_with_context
from flask_cors        import CORS
from packaging.version import Version
from typing            import Any, Dict, List, Optional, Tuple
from waitress          import serve

ENABLE_VALUES  = {"true" , "1", "yes", "y", "enable" , "on" }
DISABLE_VALUES = {"false", "0", "no" , "n", "disable", "off"}
THINK_EFFORTS  = {"low", "medium", "high", "xhigh", "max"}

def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
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
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: {name} must be an integer. Defaulting to {default}.")
        return default

def getenv_preserve_thinking_blocks(name: str, default: str = "0") -> Optional[int]:
    """
    Parses PRESERVE_THINKING_BLOCKS.

    Returns:
        0    : disabled
        N    : preserve the last N assistant messages with hidden signed thinking blocks
        None : preserve all assistant messages with hidden signed thinking blocks (inf/all)
    """
    raw = os.getenv(name, default)
    value = str(raw).strip().lower()

    if value in {"inf", "infinite", "infinity", "all", "*"}:
        return None

    try:
        return max(0, int(value))
    except ValueError:
        print(f"WARNING: {name} must be 0, a positive integer, or inf. Defaulting to {default}.")
        try:
            return max(0, int(default))
        except ValueError:
            return 0

def getenv_cache_ttl(name: str, default: str = "5m") -> str:
    normalized_default = str(default).strip().lower()
    if normalized_default not in {"5m", "1h"}:
        normalized_default = "5m"

    raw = os.getenv(name, normalized_default)
    value = str(raw).strip().lower()
    if value in {"5m", "1h"}:
        return value

    print(f"WARNING: {name} must be '5m' or '1h'. Defaulting to {normalized_default}.")
    return normalized_default

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

PREFILL_MODES = {"none", "assistant", "instruction"}
class RuntimeConfig:

    def reload_from_env(self) -> None:
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = getenv_int("PORT", 5001)

        self.model = os.getenv("MODEL", "claude-sonnet-4-5-20250929")

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.proxy_key         = os.getenv("PROXY_KEY", "").strip()

        # Safer default for a public Cloudflare Tunnel:
        # JanitorAI sends proxy_key, while your real Anthropic key stays local in .env.
        self.require_proxy_key = getenv_bool("REQUIRE_PROXY_KEY", True)

        # Compatibility fallback: if true and anthropic_api_key is not set,
        # the proxy will use the Bearer token from the incoming request as the Anthropic key.
        # For a public tunnel, I recommend leaving this false.
        self.allow_key_passthrough = getenv_bool("ALLOW_KEY_PASSTHROUGH", False)

        self.debug_log = getenv_bool("DEBUG_LOG", True)
        self.auto_trim = getenv_bool("AUTO_TRIM", True)

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
        self.temperature_override = getenv_float("TEMPERATURE_OVERRIDE", -1.0)
        self.default_temperature  = getenv_float("DEFAULT_TEMPERATURE", 0.9)
        self.send_temperature     = True
        self.send_top_p           = getenv_bool("SEND_TOP_P", False)
        self.top_p                = getenv_float("TOP_P", 0.9)
        self.send_top_k           = True
        self.top_k                = getenv_int("TOP_K", 75)
        self.default_max_tokens   = getenv_int("DEFAULT_MAX_TOKENS", 8192)

        # Thinking
        self.thinking_enabled = getenv_bool("THINKING_ENABLED", False)
        self.use_adaptive     = False
        self.thinking_budget  = getenv_int("THINKING_BUDGET", 2048)
        self.thinking_effort  = os.getenv("THINKING_EFFORT", "medium").lower()

        # Round-trip Anthropic signed thinking blocks through clients that only preserve message.content.
        # 0 disables preservation, N preserves the last N assistant messages, and inf/all preserves every assistant message.
        self.preserve_thinking_blocks = getenv_preserve_thinking_blocks("PRESERVE_THINKING_BLOCKS", "0")

        # Cost tracking.
        # Values are USD per 1 million tokens. Defaults are Anthropic's Claude Sonnet 4.5 API prices.
        self.input_token_cost_usd    = getenv_float("INPUT_TOKEN_COST_USD"   ,  3.00)
        self.output_token_cost_usd   = getenv_float("OUTPUT_TOKEN_COST_USD"  , 15.00)
        self.cache_write_5m_cost_usd = getenv_float("CACHE_WRITE_5M_COST_USD",  3.75)
        self.cache_write_1h_cost_usd = getenv_float("CACHE_WRITE_1H_COST_USD",  6.00)
        self.cache_read_cost_usd     = getenv_float("CACHE_READ_COST_USD"    ,  0.30)

        # Prompt caching.
        # Anthropic supports automatic top-level caching and explicit block-level caching.
        # This script uses explicit block-level caching because assistant prefill can otherwise
        # become the final cacheable block. Up to four explicit markers are used:
        #   1-2  system / lorebook blocks (when split_lorebook=true)
        #    3   manual first-N-message breakpoint
        #    4   automatic end-relative breakpoint before optional prefill
        self.use_cache        = getenv_bool("USE_CACHE", getenv_bool("PROMPT_CACHE", True))
        self.cache_system     = getenv_bool("CACHE_SYSTEM", getenv_bool("CACHE_SYSTEM_PROMPT", True))
        self.cache_system_ttl = getenv_cache_ttl("CACHE_SYSTEM_TTL")
        self.split_lorebook   = getenv_bool("SPLIT_LOREBOOK", True)
        self.cache_manual_ttl = getenv_cache_ttl("CACHE_MANUAL_TTL")
        self.cache_manual_msg = max(0, getenv_int("CACHE_MANUAL_MSG", getenv_int("CACHE_FIRST_MESSAGES", 0)))
        self.cache_auto_ttl   = getenv_cache_ttl("CACHE_AUTO_TTL")
        self.cache_auto_msg   = max(0, getenv_int("CACHE_AUTO_MSG", 0))

        self.error_log_path = os.getenv("ERROR_LOG_PATH", "claude_error_log.txt")
        self.model_list_timeout_seconds = getenv_float("MODEL_LIST_TIMEOUT_SECONDS", 10.0)


    def set_prefill_mode(self, mode: str) -> None:
        if not mode in PREFILL_MODES : print(f"WARNING: ASSISTANT_PREFILL_MODE must be in {PREFILL_MODES}. Defaulting to 'none'."); return
        if mode == "assistant" :
            if self.thinking_enabled          : print("While thinking is enabled, prefill mode cannot be assistant."); return
            if self.version >= Version("4.6") : print("Mythos class models do not support assistant prefill."); return
        self.assistant_prefill_mode = mode

    def set_prefill(self, prefill: str) -> None:
        self.assistant_prefill = prefill

    def enable_thinking(self, en: bool):
        self.thinking_enabled = en
        self.resolve_thinking()

    def set_think_effort(self, effort: str) -> None:
        if not effort in THINK_EFFORTS : print(f"Allowed thinking efforts: {THINK_EFFORTS}."); return
        self.thinking_effort = effort

    def resolve_thinking(self) -> None:
        if not self.thinking_enabled:
            return
        name = deep_get(self.info, "id")
        print(f"Name is {name}")

        if not deep_get(self.info, "capabilities.thinking.supported"):
            print(f"Model {name} does not support thinking. Disabling.")
            self.thinking_enabled = False;
            return
        if deep_get(self.info, "capabilities.thinking.types.adaptive.supported"):
            print(f"Models supports adaptive thinking. Using with effort '{self.thinking_effort}'.")
            self.use_adaptive = True
        elif deep_get(self.info, "capabilities.thinking.types.enabled.supported"):
            print(f"Using thinking with a budget of {self.thinking_budget} tokens")
            self.use_adaptive = False
        else:
            print("Neither adaptive nor budget thinking are supported. Disabling.")
            self.thinking_enabled = False
            return

        if (self.assistant_prefill_mode == "assistant") and (self.assistant_prefill != ""):
            print("When thinking is enabled, only instruction mode prefill is supported. Switching.")
            self.assistant_prefill_mode = "instruction"
        if self.send_temperature:
            print("Temperature is not compatible with thinking. Disabling.")
            self.send_temperature = False
        if self.send_top_k:
            print("top_k is not compatible with thinking. Disabling.")
            self.send_top_k = False
        if self.send_top_p:
            if   self.top_p < 0.95 : self.top_p = 0.95
            elif self.top_p > 1.00 : self.top_p = 1.00
            if (self.top_p < 0.95) or (self.top_p > 1.00) :
                print("Thinking supports top_p in the range [0.95:1]. Clamping.")


    def apply_model(self, i_info: Dict[str, Any]) -> None:
        self.info  = i_info
        self.model = deep_get(self.info, "id")
        print(f"=== Switching to {self.model} ===")
        self.model_info  = self.info
        display_name_str = deep_get(self.info, "display_name")
        version_str      = re.search(r'\d+(?:\.\d+)+', display_name_str)
        if version_str is not None : self.version = Version(version_str.group())
        else                       : self.version = Version("0.0")
        # Validate the configured prefill mode against the selected model.
        # Do not call set_prefill() here: that would overwrite ASSISTANT_PREFILL
        # with the mode string (for example "none", "assistant", or "instruction").
        self.set_prefill_mode(self.assistant_prefill_mode)
        self.resolve_thinking()
        print(f"=== Switching to {self.model} complete ===")

    def find_cfg(self, models: List[Dict[str, Any]]) -> str:
        for model in models:
            model_id = deep_get(model, "id")
            if model_id != self.model:
                continue
            self.apply_model(model)
            return model_id
        print(f"Requested model {self.model} not found in model list from Anthropic.")
        print("Unless you know what you're doing, it is recommended to do 'model list' followed by 'models select <number>'.")
        print("Otherwise payload correctness cannot be guaranteed.")
        return ""


    def print_status(self) -> None:
        print()
        print("=== Runtime config start ===")
        print(f"host                   = {self.host} (restart required to change)")
        print(f"port                   = {self.port} (restart required to change)")
        print(f"model                  = {self.model}")
        print(f"require_proxy_key      = {self.require_proxy_key}")
        print(f"allow_key_passthrough  = {self.allow_key_passthrough}")
        print(f"debug_log              = {self.debug_log}")
        print(f"auto_trim              = {self.auto_trim}")
        print(f"assistant_prefill      = {'set' if self.assistant_prefill.strip() else 'empty'}")
        print(f"assistant_prefill_mode = {self.assistant_prefill_mode}")
        print(f"temperature_override   = {self.temperature_override}")
        print(f"default_temperature    = {self.default_temperature}")
        print(f"top_p                  = {self.top_p}")
        print(f"top_k                  = {self.top_k}")
        print(f"default_max_tokens     = {self.default_max_tokens}")
        print(f"use_cache              = {self.use_cache}")
        print(f"cache_system           = {self.cache_system}")
        print(f"cache_system_ttl       = {self.cache_system_ttl}")
        print(f"split_lorebook         = {self.split_lorebook}")
        print(f"cache_manual_ttl       = {self.cache_manual_ttl}")
        print(f"cache_manual_msg       = {self.cache_manual_msg}")
        print(f"cache_auto_ttl         = {self.cache_auto_ttl}")
        print(f"cache_auto_msg         = {self.cache_auto_msg}")
        print(f"thinking               = {self.thinking_enabled}")
        print(f"adaptive_thinking      = {self.use_adaptive}")
        print(f"thinking_budget        = {self.thinking_budget}")
        print(f"thinking_effort        = {self.thinking_effort}")
        preserve_str = "inf" if self.preserve_thinking_blocks is None else str(self.preserve_thinking_blocks)
        print(f"preserve_thinking      = {preserve_str}")
        print(f"error_log_path         = {self.error_log_path}")
        print(f"model_list_timeout_sec = {self.model_list_timeout_seconds}")
        print("=== Runtime config end ===")
        print()

cfg      : RuntimeConfig
cfg_init : RuntimeConfig

# Session cost tracking variables
SESSION_TTL_SPENT_USD       = 0.0
SESSION_TTL_INPUT_COST_USD  = 0.0
SESSION_TTL_OUTPUT_COST_USD = 0.0
SESSION_TTL_INPUT_TOK       = 0
SESSION_TTL_OUTPUT_TOK      = 0
SESSION_CACHE_NET_COST_USD  = 0.0
SESSION_COST_LOCK           = threading.Lock()

ANTHROPIC_TIMEOUT_ERROR                      = getattr(anthropic, "APITimeoutError", TimeoutError)
ANTHROPIC_MODELS      : List[Dict[str, Any]] = []
MODEL_LIST_LAST_ERROR : Optional[str]        = None
MODEL_LIST_LAST_TIMEOUT                      = False
MODEL_LOCK                                   = threading.Lock()

# Flask app
app = Flask(__name__)
CORS(app)

# Runtime CLI config
def reload_runtime_env() -> None:
    """
    Reloads runtime configuration from .env.

    cfg.host and cfg.port are intentionally not reloaded because Waitress is already bound to them.
    """
    load_dotenv(override=True)

    bound_host = cfg.host
    bound_port = cfg.port

    cfg.reload_from_env()

    cfg.host = bound_host
    cfg.port = bound_port

    refresh_anthropic_models(cfg.anthropic_api_key, cfg.model_list_timeout_seconds)

    print("Reloaded runtime configuration from .env.")
    print("HOST and PORT were not changed; restart the process to change bind address.")


def set_cache_manual_msg(value: int) -> None:
    if value < 0 : raise ValueError("CACHE_MANUAL_MSG must be >= 0.")
    cfg.cache_manual_msg = value
    print(f"Manual cache marker now targets {cfg.cache_manual_msg} message(s) from the start.")


def set_cache_auto_msg(value: int) -> None:
    if value < 0 : raise ValueError("CACHE_AUTO_MSG must be >= 0.")
    cfg.cache_auto_msg = value
    print(f"Auto cache marker now targets {cfg.cache_auto_msg} message(s) from the end.")


def anthropic_object_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Converts Anthropic SDK model objects into JSON-printable dictionaries.
    """
    if isinstance(obj, dict)      : return dict(obj)
    if hasattr(obj, "model_dump") : return obj.model_dump(mode="json")
    if hasattr(obj, "dict")       : return obj.dict()

    if hasattr(obj, "__dict__"):
        return {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }

    return {"value": str(obj)}


def model_id_from_info(model_info: Dict[str, Any]) -> str:
    return str(model_info.get("id") or model_info.get("model") or "").strip()


def anthropic_exception_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)

    if isinstance(body, dict):
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if message:
                return str(message)

        return json.dumps(body, ensure_ascii=False, default=str)

    return str(exc) or exc.__class__.__name__


def refresh_anthropic_models(key: str, timeout_s: float) -> bool:
    """
    Fetches the available Anthropic models and stores them for CLI use.
    """
    global ANTHROPIC_MODELS, MODEL_LIST_LAST_ERROR, MODEL_LIST_LAST_TIMEOUT

    try:
        if not key: raise RuntimeError("ANTHROPIC_API_KEY is not configured; model list cannot be retrieved at startup.")

        client = anthropic.Anthropic(api_key=key)
        page   = client.models.list(limit=100, timeout=timeout_s)

        raw_models = getattr(page, "data", None)
        if raw_models is None:
            raw_models = list(page)

        models = [anthropic_object_to_dict(model) for model in raw_models]

        with MODEL_LOCK:
            ANTHROPIC_MODELS        = models
            MODEL_LIST_LAST_ERROR   = None
            MODEL_LIST_LAST_TIMEOUT = False

        if models : print(f"Retrieved {len(models)} Anthropic model(s).")
        else      : print(f"Anthropic returned an empty model list.")

        return True

    except (ANTHROPIC_TIMEOUT_ERROR, TimeoutError) as exc:
        with MODEL_LOCK:
            ANTHROPIC_MODELS        = []
            MODEL_LIST_LAST_ERROR   = None
            MODEL_LIST_LAST_TIMEOUT = True
        print("WARNING: Could not retrieve a model list from Anthropic. Timeout.")
        return False

    except Exception as exc:
        with MODEL_LOCK:
            ANTHROPIC_MODELS        = []
            MODEL_LIST_LAST_ERROR   = anthropic_exception_message(exc)
            MODEL_LIST_LAST_TIMEOUT = False
        print(f"WARNING: Could not retrieve a model list from Anthropic. {anthropic_exception_message(exc)}.")
        return False

def print_no_model_list_available() -> None:
    print("No Anthropic model list is available.")

    with MODEL_LOCK:
        last_timeout = MODEL_LIST_LAST_TIMEOUT
        last_error   = MODEL_LIST_LAST_ERROR

    if   last_timeout : print("Last retrieval failed: timeout")
    elif last_error   : print(f"Last retrieval failed: error: {last_error}")


def print_model_list() -> None:
    with MODEL_LOCK:
        models            = list(ANTHROPIC_MODELS)
        selected_model_id = cfg.model
    if not models:
        print_no_model_list_available()
        return

    number_width = len(str(len(models)))

    for index, model_info in enumerate(models, start=1):
        model_id     = model_id_from_info(model_info)
        display_name = str(model_info.get("display_name") or model_info.get("name") or "")
        created_at   = str(model_info.get("created_at") or "")

        number      = str(index).rjust(number_width)
        number_cell = f"[{number}]" if model_id == selected_model_id else f" {number} "

        extra_parts = []
        if display_name : extra_parts.append(display_name)
        if created_at   : extra_parts.append(created_at)

        extra  = "  ".join(extra_parts)
        suffix = f"  {extra}" if extra else ""

        print(f"{number_cell}  {model_id:<42}{suffix}")


def select_model_by_number(index: int) -> None:
    with MODEL_LOCK:
        if not ANTHROPIC_MODELS:
            print_no_model_list_available()
            return
        if index < 1 or index > len(ANTHROPIC_MODELS):
            print(f"Model number out of range. Use 1 through {len(ANTHROPIC_MODELS)}.")
            return
        model_info = ANTHROPIC_MODELS[index - 1]
        model_id   = model_id_from_info(model_info)
        if not model_id:
            print(f"Model {index} does not have an id and cannot be selected.")
            return
        cfg.reload_from_env()
        cfg.apply_model(model_info)


def print_model_info(index: int) -> None:
    with MODEL_LOCK:
        if not ANTHROPIC_MODELS:
            print_no_model_list_available()
            return
        if index < 1 or index > len(ANTHROPIC_MODELS):
            print(f"Model number out of range. Use 1 through {len(ANTHROPIC_MODELS)}.")
            return
        model_info = dict(ANTHROPIC_MODELS[index - 1])

    print(json.dumps(model_info, indent=2, ensure_ascii=False, default=str))


def admin_cli_loop() -> None:
    """
    Tiny local CLI for changing runtime-only settings.
    """

    print("Runtime CLI ready. Type 'help' for commands.")
    print()

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return
        except KeyboardInterrupt:
            print()
            return

        if not line:
            continue

        parts   = line.split()
        command = parts[0].lower()

        try:
            if command == "cmm":
                if len(parts) != 2 : print("Usage: cache_manual_msg <number>"); continue
                set_cache_manual_msg(int(parts[1]))

            if command == "cam":
                if len(parts) != 2 : print("Usage: cache_manual_msg <number>"); continue
                set_cache_auto_msg(int(parts[1]))

            if command == "cache":
                if len(parts) < 2 : print("Usage : cache <sub_cmd>"); continue
                arg1 = parts[1].lower()
                if arg1 in DISABLE_VALUES : cfg.use_cache = False; continue
                if arg1 in ENABLE_VALUES  : cfg.use_cache = True; continue
                if len(parts) != 3 : print("Usage : cache <sub_cmd> <number>"); continue
                arg2 = parts[2].lower()
                if arg1 in {"manual", "man", "m"} :
                    if (arg2 == "1h") : cfg.cache_manual_ttl = "1h"; continue
                    if (arg2 == "5m") : cfg.cache_manual_ttl = "5m"; continue
                    if (arg2 in DISABLE_VALUES) : set_cache_manual_msg(0); continue
                    set_cache_manual_msg(int(arg2)); continue
                if arg1 in {"auto", "a"} :
                    if (arg2 == "1h") : cfg.cache_auto_ttl = "1h"; continue
                    if (arg2 == "5m") : cfg.cache_auto_ttl = "5m"; continue
                    if (arg2 in DISABLE_VALUES) : set_cache_auto_msg(0); continue
                    set_cache_auto_msg(int(arg2));  continue
                if arg1 in {"system", "sys", "s"} :
                    if (arg2 in DISABLE_VALUES) : cfg.cache_system = False; continue
                    if (arg2 in ENABLE_VALUES ) : cfg.cache_system = True; continue
                    if (arg2 == "1h") : cfg.cache_system_ttl = "1h"; continue
                    if (arg2 == "5m") : cfg.cache_system_ttl = "5m"; continue
                print(f"Unknown arg1 {arg1}.")
                continue

            if command == "prefill":
                if len(parts) < 2:
                    print("Usage:")
                    print("  prefill <none|assistant|instruction>")
                    print("  prefill set <text>")
                    continue

                arg1 = parts[1].lower()
                if arg1 in PREFILL_MODES:
                    cfg.set_prefill_mode(arg1)
                    continue

                if arg1 != "set":
                    print("Invalid argument. Use: prefill <none|assistant|instruction> or prefill set <text>")
                    continue

                split_line = line.split(maxsplit=2)
                if len(split_line) < 3:
                    print("Usage: prefill set <text>")
                    continue

                cfg.set_prefill(split_line[2])
                continue

            if command in {"think", "thinking"}:
                if len(parts) < 2:
                    print("Usage: think <on|off|effort> [effort]")
                    continue

                arg1 = parts[1].lower()
                if arg1 in DISABLE_VALUES:
                    cfg.enable_thinking(False)
                    continue
                if arg1 in ENABLE_VALUES:
                    cfg.enable_thinking(True)
                    continue

                if arg1 == "effort":
                    if len(parts) != 3:
                        print("Usage: think effort <low|medium|high|xhigh|max>")
                        continue
                    cfg.set_think_effort(parts[2].lower())
                    continue

                print("Usage: think <on|off|effort> [effort]")
                continue

            if command in {"reload_env", "reload", "env"}:
                if len(parts) != 1:
                    print("Usage: reload_env")
                    continue
                reload_runtime_env()
                cfg.print_status()
                continue

            if command in {"model", "models"}:
                if len(parts) == 2 and parts[1].lower() == "list":
                    print_model_list()
                    continue
                if len(parts) == 3 and parts[1].lower() == "select":
                    value = int(parts[2])
                    select_model_by_number(value)
                    continue
                if len(parts) == 3 and parts[1].lower() == "info":
                    value = int(parts[2])
                    print_model_info(value)
                    continue
                print("Usage:")
                print("  model list")
                print("  model select <number>")
                print("  model info   <number>")
                continue

            if command in {"show", "status"}:
                cfg.print_status()
                continue

            if command == "help":
                print()
                print("Commands:")
                print("  cache_manual_msg <number>      Set manual first-N cache marker")
                print("    cmm <number>")
                print("    cache_first_messages / cfm   Backwards-compatible aliases")
                print("  cache_auto_msg <number>        Set automatic end-relative cache marker")
                print("    cam <number>")
                print("  model list                     List available Anthropic models")
                print("  model select <number>          Select model by list number")
                print("  model info   <number>          Show all stored info for a model")
                print("  reload_env                     Reload runtime settings from .env")
                print("    reload")
                print("    env")
                print("  show                           Show runtime settings")
                print("  help                           Show this help")
                print("  quit                           Stop the process")
                print("    exit")
                print()
                continue

            if command in {"quit", "exit"}:
                print("Stopping proxy.")
                os._exit(0)

            print(f"Unknown command: {command}")
            print("Type 'help' for commands.")

        except ValueError as exc : print(f"Invalid value: {exc}")
        except Exception  as exc : print(f"CLI error: {exc}")


# Utility helpers

def write_error_log(body: Any) -> None:
    try:
        with open(cfg.error_log_path, "a", encoding="utf-8") as f:
            f.write(str(body) + "\n\n")
    except Exception:
        print("Failed to write error log:")
        traceback.print_exc()


def get_bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth.strip()


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Recommended public-tunnel mode:
        .env contains ANTHROPIC_API_KEY and PROXY_KEY. JanitorAI uses PROXY_KEY as the reverse proxy key.

    Optional compatibility mode:
        ALLOW_KEY_PASSTHROUGH=true lets incoming Bearer token act as Anthropic key.
    """
    provided_key = get_bearer_token()

    if cfg.anthropic_api_key:
        if cfg.require_proxy_key:
            if not cfg.proxy_key             : abort(500, description=("Server is configured with REQUIRE_PROXY_KEY=true, but PROXY_KEY is missing from .env."))
            if provided_key != cfg.proxy_key : abort(401, description="Invalid proxy key.")
        return anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    if cfg.allow_key_passthrough:
        if not provided_key : abort(401, description="Missing Authorization bearer token.")
        return anthropic.Anthropic(api_key=provided_key)
    abort(500, description=("ANTHROPIC_API_KEY is not configured. Either set ANTHROPIC_API_KEY and PROXY_KEY in .env, or set ALLOW_KEY_PASSTHROUGH=true."))


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

    Anthropic prompt caching is enabled by adding cache_control either at the
    request level or on content blocks. This script uses explicit block-level
    caching to avoid caching the assistant prefill as the final block.
    """
    if not cfg.use_cache:
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


def append_text_to_content(content: Any, text: str) -> Any:
    """
    Appends text while preserving Anthropic list-form content blocks.
    """
    if text is None:
        text = ""

    if isinstance(content,  str) : return content + "\n" + text
    if isinstance(content, list) : return content + [{"type": "text", "text": "\n" + text}]

    return str(content) + "\n" + text


def make_prefill_instruction(prefix_text: str) -> str:
    """
    Creates the instruction-mode version of ASSISTANT_PREFILL.

    This avoids Anthropic assistant prefill by telling Claude, inside the
    last user message, to continue as though the prefix was already present.
    """
    # return (
    #     "\n<OOC>\n"
    #     "The assistant will begin its reply with the following prefix:\n"
    #     f"<prefix>{prefix_text}</prefix>\n"
    #     "Continue immediately after that prefix. Do not display the prefix in your answer.\n"
    #     "</OOC>"
    # )

    return (
        "\n<OOC>\n"
        f"{prefix_text}"
        "</OOC>"
    )

    # return (
        # "\n<OOC>\n"
        # "The assistant has began its response with:\n"
        # "<prefix>\n"
        # f"{prefix_text}\n"
        # "</prefix>\n"
        # "You are generating only the continuation after that already-rendered text.\n"
        # "</OOC>"
    # )


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


# =============================================================================
# Anthropic thinking block round-tripping
# =============================================================================
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
    return (cfg.preserve_thinking_blocks is None) or (cfg.preserve_thinking_blocks > 0)


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
        "version": 1,
        "kind": "anthropic_thinking_blocks",
        "blocks": blocks,
    }

    # Keep the envelope ASCII-safe and line-wrapped for text-only clients.
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    wrapped = [encoded[i:i + 120] for i in range(0, len(encoded), 120)]
    body = "\n".join(f"~~~{line}" for line in wrapped)

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


def openai_stream_chunk(model_used: str, delta: Dict[str, Any], finish_reason: Optional[str] = None, usage: Optional[Dict[str, int]] = None, message_id: str = "claude") -> str:
    chunk: Dict[str, Any] = {
        "id": message_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": f"anthropic/{model_used}",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "delta": delta,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


PERSONA_END_RE = re.compile(r"</[^<>]*\bPersona>", re.IGNORECASE)

def split_system_prompt_into_text_blocks(system_prompt: str) -> List[Dict[str, str]]:
    """
    For more efficient caching, we split the character definition into the core definition (which never changes),
    and the lorebook / user script additions (which can change in one chat). That way we never have to re-cache
    the core definition.

    Split priority:
        1. After </example_dialogs>.
        1. After </Scenario>.
        2. After the last </* Persona> marker.
        3. Otherwise keep the whole system prompt as one block.

    Anything after it becomes the second block, usually lorebook / long term memory.
    Should the split be performed incorrectly (due to user scripts doing something 'interesting'),
    the definition will still be sent correctly. Just the caching efficiency will degrade.
    """
    if not system_prompt or not system_prompt.strip():
        return []

    text = system_prompt.strip()

    split_marker = "</example_dialogs>"
    split_idx    = text.find(split_marker)
    if (split_idx == -1):
        split_marker = "</Scenario>"
        split_idx    = text.find(split_marker)
    if (split_idx == -1):
        persona_matches = list(PERSONA_END_RE.finditer(text))
        if persona_matches : split_at = persona_matches[-1].end()
        else               : split_at = -1
    else:
        split_at = split_idx + len(split_marker)

    if split_at == -1 or split_at >= len(text):
        return [{"type": "text", "text": text}]

    before = text[:split_at].rstrip()
    after  = text[split_at:].strip()

    blocks = [{"type": "text", "text": before}]

    if after:
        # Keep a clean visual/semantic separator between Scenario/Persona and suffix.
        blocks.append({"type": "text", "text": "\n\n" + after})

    return blocks


def format_system_for_claude(system_prompt: Optional[str]) -> Optional[Any]:
    """
    Optionally cache the system prompt separately.

    When SPLIT_LOREBOOK=true, the system prompt is split into core definition and lorebook/suffix blocks.
    Each non-empty system block receives its own system cache marker when CACHE_SYSTEM=true.
    """
    if system_prompt is None:
        return None

    if cfg.split_lorebook:
        blocks = split_system_prompt_into_text_blocks(system_prompt)
    else:
        stripped = system_prompt.strip()
        blocks = [{"type": "text", "text": stripped}] if stripped else []

    if not blocks:
        return None

    formatted_system: List[Dict[str, Any]] = []

    for block in blocks:
        new_block: Dict[str, Any] = dict(block)
        if (cfg.use_cache and cfg.cache_system and new_block.get("type") == "text" and new_block.get("text", "").strip()):
            new_block["cache_control"] = make_cache_control(cfg.cache_system_ttl)
        formatted_system.append(new_block)

    return formatted_system


def format_to_claude_messages(mlist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts OpenAI-style chat messages to Anthropic Messages format.

    Consecutive same-role messages are merged because Anthropic expects
    alternating user/assistant turns.

    Manual caching marks the configured first-N-message prefix. Automatic caching marks
    an end-relative conversation point before ASSISTANT_PREFILL is applied, so the prefill
    is never accidentally included in that breakpoint.
    """

    manual_cache_target_index = 0
    if cfg.use_cache and cfg.cache_manual_msg > 0 and mlist:
        manual_cache_target_index = min(cfg.cache_manual_msg, len(mlist))

    auto_cache_target_index = 0
    if cfg.use_cache and cfg.cache_auto_msg > 0 and mlist:
        auto_cache_target_index = max(1, len(mlist) - cfg.cache_auto_msg + 1)

    formatted: List[Dict[str, Any]] = [{"role": "user", "content": "<OOC>\nBegin the scenario.\n</OOC>"}]
    old_role  = "user"

    for idx, msg in enumerate(mlist, start=1):
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

        if claude_role == old_role:
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

        # Manual marker: cache the first N incoming chat messages, preserving the old behavior.
        if idx == manual_cache_target_index:
            formatted[-1]["content"] = add_cache_control_to_content(formatted[-1]["content"], cfg.cache_manual_ttl)

        # Auto marker: count backwards from the final incoming chat message, before optional prefill is added.
        if idx == auto_cache_target_index:
            formatted[-1]["content"] = add_cache_control_to_content(formatted[-1]["content"], cfg.cache_auto_ttl)

    # Optional Claude prefill.
    # assistant mode preserves the original assistant-message/prefill behavior.
    # instruction mode avoids assistant prefill and appends an OOC instruction to the last user message instead.
    if cfg.assistant_prefill.strip() and cfg.assistant_prefill_mode != "none":
        if cfg.assistant_prefill_mode == "instruction":
            append_prefill_instruction_to_last_user_message(formatted, cfg.assistant_prefill)
        elif cfg.assistant_prefill_mode == "assistant":
            if formatted[-1]["role"] == "user" : formatted.append({"role" : "assistant", "content" : cfg.assistant_prefill})
            else                               : formatted[-1]["content"] = append_text_to_content(formatted[-1]["content"], cfg.assistant_prefill)

    return formatted


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


def extract_text_from_anthropic_message(message: Any) -> str:
    """
    Collects text blocks from an Anthropic response.
    """
    chunks = []
    for block in getattr(message, "content", []) or []:
        if   getattr(block, "type", None) == "text"                  : chunks.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text" : chunks.append(block.get("text", ""))

    return "".join(chunks)

def average_cost_per_token_usd(total_cost_usd: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd/total_tokens


def fallback_cache_write_ttl() -> str:
    """
    Older SDK usage payloads may not split cache creation by 5m/1h.
    If any active marker is configured for 1h, assume 1h for unknown write tokens to avoid under-counting cost.
    """
    if not cfg.use_cache:
        return "5m"
    active_ttl: List[str] = []
    if cfg.cache_system         : active_ttl.append(cfg.cache_system_ttl)
    if cfg.cache_manual_msg > 0 : active_ttl.append(cfg.cache_manual_ttl)
    if cfg.cache_auto_msg   > 0 : active_ttl.append(cfg.cache_auto_ttl)
    return "1h" if "1h" in active_ttl else "5m"


def print_usage(usage: Any) -> None:
    global SESSION_TTL_SPENT_USD, SESSION_CACHE_NET_COST_USD, SESSION_TTL_INPUT_COST_USD, SESSION_TTL_OUTPUT_COST_USD, SESSION_TTL_INPUT_TOK, SESSION_TTL_OUTPUT_TOK

    def tok_usd(tokens: int, usd_per_million_tokens: float) -> float:
        return (tokens*usd_per_million_tokens)/1_000_000.0
    def fmt_usd(amount: float) -> str:
        sign = "-" if amount < 0 else ""
        return f"{sign}${abs(amount):,.6f}"
    def cache_lbl(net_cost_usd: float) -> str:
        if net_cost_usd < 0: return f"{fmt_usd(abs(net_cost_usd))} saved"
        if net_cost_usd > 0: return f"{fmt_usd(net_cost_usd)} lost"
        return "$0.000000 break-even"

    cache_creation       = getattr(usage, "cache_creation", {}) or {}
    ephemeral_1h         = int(getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
    ephemeral_5m         = int(getattr(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
    input_tok            = int(getattr(usage, "input_tokens", 0) or 0)
    output_tok           = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read           = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_input = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    ttl_tokens = input_tok + cache_read + cache_creation_input;

    # Older SDK responses may expose only cache_creation_input_tokens without the 5m/1h split.
    # Mixed-TTL requests cannot be reconstructed from that legacy shape, so use a conservative fallback.
    known_cache_write = ephemeral_1h + ephemeral_5m
    if cache_creation_input > known_cache_write:
        unknown_cache_write = cache_creation_input - known_cache_write
        if fallback_cache_write_ttl() == "1h" : ephemeral_1h += unknown_cache_write
        else                                  : ephemeral_5m += unknown_cache_write

    cache_creation_input = ephemeral_1h + ephemeral_5m
    ttl_tokens           = input_tok + cache_read + cache_creation_input

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

    with SESSION_COST_LOCK:
        SESSION_TTL_SPENT_USD       += request_total_cost
        SESSION_TTL_INPUT_COST_USD  += total_input_cost
        SESSION_TTL_OUTPUT_COST_USD += output_cost
        SESSION_TTL_INPUT_TOK       += ttl_tokens
        SESSION_TTL_OUTPUT_TOK      += output_tok
        SESSION_CACHE_NET_COST_USD  += request_cache_net_cost
        session_total_spent          = SESSION_TTL_SPENT_USD
        session_total_input_cost     = SESSION_TTL_INPUT_COST_USD
        session_total_output_cost    = SESSION_TTL_OUTPUT_COST_USD
        session_total_input_tok      = SESSION_TTL_INPUT_TOK
        session_total_output_tok     = SESSION_TTL_OUTPUT_TOK
        session_cache_net_cost       = SESSION_CACHE_NET_COST_USD
        session_average_input_cost   = average_cost_per_token_usd(session_total_input_cost, session_total_input_tok)

    if not cfg.debug_log:
        return

    print("=== Claude usage start ===")
    print("Request:")
    print("    Input tokens       =   uncached + cache read + cache write (        1h +         5m)")
    print("    {:18d} = {:10d} + {:10d} + {:11d} ({:10d} + {:10d})".format(ttl_tokens, input_tok, cache_read, cache_creation_input, ephemeral_1h, ephemeral_5m))
    print("    {:>18s} = {:>10s} + {:>10s} + {:>11s} ({:>10s} + {:>10s})".format(fmt_usd(total_input_cost), fmt_usd(input_cost), fmt_usd(cache_read_cost), fmt_usd(cache_write_cost), fmt_usd(cache_write_1h_cost), fmt_usd(cache_write_5m_cost)))
    print("    Output tokens      = {:d} ({})".format(output_tok, fmt_usd(output_cost)))
    print("    Cache cost         = {} ({})".format(fmt_usd(request_cache_net_cost), cache_lbl(request_cache_net_cost)))
    print("    Total cost         = {}".format(fmt_usd(request_total_cost)))
    print("Session:")
    print("    Input tokens       = {:d} ({})".format(session_total_input_tok, fmt_usd(session_total_input_cost)))
    print("    Output tokens      = {:d} ({})".format(session_total_output_tok, fmt_usd(session_total_output_cost)))
    print("    Cache cost         = {} ({})".format(fmt_usd(session_cache_net_cost), cache_lbl(session_cache_net_cost)))
    print("    Average input cost = {} / MTok.".format(fmt_usd(session_average_input_cost*1_000_000)))
    print("    Total cost         = {} ({} input / {} output)".format(fmt_usd(session_total_spent), fmt_usd(session_total_input_cost), fmt_usd(session_total_output_cost)))
    print("=== Claude usage end ===")
    print("> ", end="", flush=True)


def usage_to_openai_dict(usage: Any) -> Dict[str, int]:
    """
    Anthropic separates cache usage into:
      input_tokens
      cache_creation_input_tokens
      cache_read_input_tokens
      output_tokens

    OpenAI-compatible clients usually expect:
      prompt_tokens
      completion_tokens
      total_tokens

    This preserves both.
    """
    input_tokens   = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens  = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read     = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    prompt_tokens = input_tokens + cache_read + cache_creation

    return {
        "prompt_tokens"               : prompt_tokens,
        "completion_tokens"           : output_tokens,
        "total_tokens"                : prompt_tokens + output_tokens,
        "input_tokens_uncached"       : input_tokens,
        "cache_creation_input_tokens" : cache_creation,
        "cache_read_input_tokens"     : cache_read,
    }


def get_temperature(payload: Dict[str, Any]) -> float:
    if cfg.temperature_override != -1:
        return cfg.temperature_override
    return float(payload.get("temperature", cfg.default_temperature))


def split_system_and_messages(raw_messages: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Validates and normalizes OpenAI-style chat messages.

    Accepts untrusted request payload data.
    Returns:
        - system_prompt: joined system messages, or None
        - chat_messages: list of normalized message dicts

    Invalid message lists abort with 400 and do not return.
    """
    if not isinstance(raw_messages, list):
        abort(400, description="Request body must include a messages list.")
        raise RuntimeError("unreachable")

    system_parts  : List[str]            = []
    chat_messages : List[Dict[str, Any]] = []

    for idx, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            abort(400, description=f"Message at index {idx} must be an object.")
            raise RuntimeError("unreachable")

        raw_role = msg.get("role", "user")
        role     = raw_role if isinstance(raw_role, str) else "user"
        content  = content_to_plain_text(msg.get("content", ""))

        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
            continue

        if role not in ("user", "assistant"):
            role = "user"

        # Strip every preservation envelope, but keep assistant envelopes only when preservation is enabled.
        content, thinking_blocks = extract_hidden_thinking_envelopes(content)

        msg_obj: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and thinking_preservation_enabled() and thinking_blocks:
            msg_obj["anthropic_thinking_blocks"] = thinking_blocks
        chat_messages.append(msg_obj)

    if thinking_preservation_enabled():
        # Mark only the last N assistant messages for signed-block rehydration.
        remaining = cfg.preserve_thinking_blocks
        for i in range(len(chat_messages) - 1, -1, -1):
            msg = chat_messages[i]
            if msg.get("role") != "assistant":
                continue
            if not msg.get("anthropic_thinking_blocks"):
                continue
            if remaining is None:
                msg["send_anthropic_thinking_blocks"] = True
                continue
            if remaining <= 0:
                break
            msg["send_anthropic_thinking_blocks"] = True
            remaining -= 1

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, chat_messages


def build_claude_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:

    system_prompt, chat_messages = split_system_and_messages(payload.get("messages"))

    if chat_messages and chat_messages[0].get("role") == "user" and chat_messages[0].get("content", "").strip() == ".":
        chat_messages = chat_messages[1:]

    formatted_messages = format_to_claude_messages(chat_messages)

    kwargs: Dict[str, Any] = {
        "model"      : cfg.model,
        "max_tokens" : int(payload.get("max_tokens", cfg.default_max_tokens)),
    }
    if cfg.send_temperature : kwargs["temperature"] = get_temperature(payload)
    if cfg.send_top_k       : kwargs["top_k"] = cfg.top_k
    if cfg.send_top_p       : kwargs["top_p"] = cfg.top_p
    if cfg.thinking_enabled:
        if   cfg.use_adaptive:
            kwargs["thinking"]      = { "type": "adaptive", "display": "summarized" };
            kwargs["output_config"] = { "effort": cfg.thinking_effort }
        else:
            kwargs["thinking"] = { "type": "enabled", "budget_tokens": cfg.thinking_budget }

    formatted_system = format_system_for_claude(system_prompt)
    if formatted_system is not None:
        kwargs["system"] = formatted_system
    kwargs["messages"] = formatted_messages

    return kwargs


def anthropic_blocks_to_dicts(message: Any) -> List[Dict[str, Any]]:
    blocks = []
    for block in getattr(message, "content", []) or []:
        if hasattr(block, "model_dump") : blocks.append(block.model_dump(mode="json"))
        elif isinstance(block, dict)    : blocks.append(block)
        else                            : blocks.append({"type": getattr(block, "type", "unknown"), "value": str(block)})
    return blocks


def make_openai_non_stream_response(message: Any, model: str) -> Dict[str, Any]:
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
        "id": getattr(message, "id", "claude"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"anthropic/{model}",
        "choices": [
            {
                "index": 0,
                "finish_reason": getattr(message, "stop_reason", "stop"),
                "message": {
                    "role": "assistant",
                    "content": output_text,
                    "anthropic_content": anthropic_content,
                    "anthropic_thinking_preserved": bool(thinking_preservation_enabled() and thinking_blocks),
                },
            }
        ],
        "usage": usage_to_openai_dict(getattr(message, "usage", None)),
    }


def make_error_response(exc: Exception, payload: Optional[Dict[str, Any]] = None) -> Response:
    status_code = 500
    message = str(exc)
    error_type = exc.__class__.__name__

    # Flask abort errors
    if hasattr(exc, "code"):
        status_code = getattr(exc, "code", 500)
        message = getattr(exc, "description", message)

    # Anthropic SDK errors often expose status_code/body.
    if hasattr(exc, "status_code"):
        status_code = getattr(exc, "status_code", status_code)

    if hasattr(exc, "body"):
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_obj  = body.get("error", {})
            message    = error_obj.get("message", message)
            error_type = error_obj.get("type", error_type)

    error_body = { "error": { "message": message, "type": error_type, "code": status_code } }
    log_body   = { "error": error_body, "request": payload, "traceback": traceback.format_exc() }
    write_error_log(log_body)

    return Response(json.dumps(error_body, ensure_ascii=False), status=status_code, content_type="application/json")


# =============================================================================
# Generation
# =============================================================================
def print_payload(kwargs: Dict[str, Any]) -> None:
    if not cfg.debug_log:
        return
    print()
    print("=== Claude payload start ===")
    print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    print("=== Claude payload end ===")


def generate_non_stream(payload: Dict[str, Any], route_model: str) -> Dict[str, Any]:
    client = get_anthropic_client()
    kwargs = build_claude_kwargs(payload)

    print_payload(kwargs)

    message = client.messages.create(**kwargs)

    usage = getattr(message, "usage", None)
    print_usage(usage)

    model_used = kwargs.get("model", route_model)
    return make_openai_non_stream_response(message, model_used)


def generate_stream(payload: Dict[str, Any], route_model: str):
    client = get_anthropic_client()
    kwargs = build_claude_kwargs(payload)

    print_payload(kwargs)

    model_used = kwargs.get("model", route_model)

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    # Stream reasoning_content only; Janitor already renders it as <think> text.
                    yield openai_stream_chunk(model_used, {
                        "role": "assistant",
                        "reasoning_content": event.delta.thinking,
                    })

                elif event.delta.type == "text_delta":
                    yield openai_stream_chunk(model_used, {
                        "role": "assistant",
                        "content": event.delta.text,
                    })
            time.sleep(0.01)

        final_message = stream.get_final_message()

        # The visible <think> text has already gone through reasoning_content above.
        thinking_blocks = extract_preservable_thinking_blocks(anthropic_blocks_to_dicts(final_message))
        if thinking_preservation_enabled() and thinking_blocks:
            # Send only the hidden signed-block envelope for next-turn rehydration.
            yield openai_stream_chunk(model_used, {
                "role": "assistant",
                "content": make_hidden_thinking_envelope(thinking_blocks),
            })

        usage = getattr(final_message, "usage", None)
        print_usage(usage)

        yield openai_stream_chunk(
            model_used,
            {},
            finish_reason=getattr(final_message, "stop_reason", "stop"),
            usage=usage_to_openai_dict(getattr(final_message, "usage", None)),
            message_id=getattr(final_message, "id", "claude"),
        )

    yield "data: [DONE]\n\n"

def handle_chat_completion():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return Response(json.dumps({"error": {"message": "Invalid JSON body."}}), status=400, content_type="application/json")

    try:
        stream = bool(payload.get("stream", False))

        if stream:
            return Response(
                stream_with_context(generate_stream(payload, cfg.model)),
                content_type="text/event-stream",
                headers={
                    "Cache-Control"     : "no-cache",
                    "X-Accel-Buffering" : "no",
                },
            )

        response = generate_non_stream(payload, cfg.model)
        return jsonify(response)

    except Exception as exc:
        return make_error_response(exc, payload)


@app.route("/", methods=["GET"])
def running():
    base_url = request.base_url.rstrip("/")

    with SESSION_COST_LOCK:
        session_total_spent         = SESSION_TTL_SPENT_USD
        session_total_input_cost    = SESSION_TTL_INPUT_COST_USD
        session_total_output_cost   = SESSION_TTL_OUTPUT_COST_USD
        session_total_input_tokens  = SESSION_TTL_INPUT_TOK
        session_total_output_tokens = SESSION_TTL_OUTPUT_TOK
        session_cache_net_cost      = SESSION_CACHE_NET_COST_USD
        session_average_input_cost  = average_cost_per_token_usd(session_total_input_cost, session_total_input_tokens)

    return jsonify(
        {
            "status"        : "ok",
            "model"         : cfg.model,
            "prompt_cache"  : cfg.use_cache,
            "cache"         : {
                "use_cache"        : cfg.use_cache,
                "cache_system"     : cfg.cache_system,
                "cache_system_ttl" : cfg.cache_system_ttl,
                "split_lorebook"   : cfg.split_lorebook,
                "cache_manual_ttl" : cfg.cache_manual_ttl,
                "cache_manual_msg" : cfg.cache_manual_msg,
                "cache_auto_ttl"   : cfg.cache_auto_ttl,
                "cache_auto_msg"   : cfg.cache_auto_msg,
            },
            "cost_tracking" : {
                "input_token_cost_usd"                             : cfg.input_token_cost_usd,
                "output_token_cost_usd"                            : cfg.output_token_cost_usd,
                "cache_write_5m_cost_usd"                          : cfg.cache_write_5m_cost_usd,
                "cache_write_1h_cost_usd"                          : cfg.cache_write_1h_cost_usd,
                "cache_read_cost_usd"                              : cfg.cache_read_cost_usd,
                "session_total_spent_usd"                          : session_total_spent,
                "session_total_input_token_cost_usd"               : session_total_input_cost,
                "session_total_output_token_cost_usd"              : session_total_output_cost,
                "session_total_input_tokens"                       : session_total_input_tokens,
                "session_total_output_tokens"                      : session_total_output_tokens,
                "session_average_input_token_cost_usd_per_million" : session_average_input_cost*1_000_000,
                "session_cache_net_cost_usd"                       : session_cache_net_cost,
            },
            "thinking" : {
                "thinking_enabled"          : cfg.thinking_enabled,
                "adaptive_thinking"         : cfg.use_adaptive,
                "thinking_budget"           : cfg.thinking_budget,
                "thinking_effort"           : cfg.thinking_effort,
                "preserve_thinking_blocks"  : "inf" if cfg.preserve_thinking_blocks is None else str(cfg.preserve_thinking_blocks),
            }
        }
    )


@app.route("/"                   , methods=["POST"])
@app.route("/chat/completions"   , methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def short_baseurl() : return handle_chat_completion()
def baseurl()       : return handle_chat_completion()
def v1_baseurl()    : return handle_chat_completion()


if __name__ == "__main__":
    load_dotenv()
    cfg_init = RuntimeConfig()
    cfg_init.reload_from_env()
    refresh_anthropic_models(cfg_init.anthropic_api_key, cfg_init.model_list_timeout_seconds)

    cfg = RuntimeConfig()
    cfg.reload_from_env()
    cfg.find_cfg(ANTHROPIC_MODELS)

    print("Starting Claude reverse proxy")
    print(f"Local URL: http://{cfg.host}:{cfg.port}")
    print(f"Chat completions: http://{cfg.host}:{cfg.port}/chat/completions")
    print("Cloudflare Tunnel service URL should point to this local address:")
    print(f"  http://{cfg.host}:{cfg.port}")
    print()

    if cfg.require_proxy_key and not cfg.proxy_key:
        print("WARNING: REQUIRE_PROXY_KEY=true but PROXY_KEY is missing.")
        print("Set PROXY_KEY in .env before exposing this through Cloudflare Tunnel.")
        print()

    if not cfg.anthropic_api_key and not cfg.allow_key_passthrough:
        print("WARNING: ANTHROPIC_API_KEY is missing and ALLOW_KEY_PASSTHROUGH=false.")
        print("Requests will fail until you configure one of these modes.")
        print()

    thread = threading.Thread(target=admin_cli_loop, daemon=True)
    thread.start()
    serve(app, host=cfg.host, port=cfg.port)
