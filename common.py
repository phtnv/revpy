import json
import os
import re
import threading

from packaging.version import Version
from typing            import Any, Dict, List

ENABLE_VALUES  = {"1", "y", "true" , "yes", "enable" , "on" }
DISABLE_VALUES = {"0", "n", "false", "no" , "disable", "off"}
THINK_EFFORTS  = {"low", "medium", "high", "xhigh", "max"}
INF_VALUES     = {"inf", "all", "infinite", "infinity", "*", "∞"}
UINT64_MAX     = 2**64 - 1

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
def getenv_cache_ttl(name: str, default: str) -> str:
    if default not in {"5m", "1h"}:
        default = "1h"
    raw = os.getenv(name, default)
    value = str(raw).strip().lower()
    if value in {"5m", "1h"}:
        return value
    print(f"WARNING: {name} must be '5m' or '1h'. Defaulting to {default}.")
    return default

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
            cur = cur.get(part, object())
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else object()
        else:
            cur = object()

        if cur is object():
            return default

    return cur

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

    def reload_from_env(self) -> None:
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = getenv_int("PORT", 5001)

        self.model = os.getenv("MODEL", "claude-sonnet-4-6")
        self.version = extract_claude_version(self.model)
        self.model_info = {}

        self.anthropic_api_key     = os.getenv("ANTHROPIC_API_KEY", "").strip()
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
        self.send_top_k       = getenv_bool("SEND_TOP_P", False)
        self.top_k            = getenv_int("TOP_K", 75)

        # Thinking
        self.thinking_enabled = getenv_bool("THINKING_ENABLED", False)
        self.use_adaptive     = False
        self.thinking_budget  = getenv_int("THINKING_BUDGET", 2048)
        self.thinking_effort  = os.getenv("THINKING_EFFORT", "medium").lower()

        # Round-trip Anthropic signed thinking blocks through clients that only preserve message.content.
        # 0 disables preservation, N preserves the last N assistant messages, and inf/all preserves every assistant message.
        self.preserve_thinking_blocks = getenv_preserve_thinking_blocks("PRESERVE_THINKING_BLOCKS", "0")

        # Cost tracking. Values are USD per 1 million tokens.
        self.cost_table: Dict[str, Dict[str, float]] = {
            "fable": {
                "input"          : getenv_float("FABLE_INPUT_TOKEN_COST_USD"   , 10.00),
                "output"         : getenv_float("FABLE_OUTPUT_TOKEN_COST_USD"  , 50.00),
                "cache_write_5m" : getenv_float("FABLE_CACHE_WRITE_5M_COST_USD", 12.50),
                "cache_write_1h" : getenv_float("FABLE_CACHE_WRITE_1H_COST_USD", 20.00),
                "cache_read"     : getenv_float("FABLE_CACHE_READ_COST_USD"    ,  1.00),
            },
            "opus": {
                "input"          : getenv_float("OPUS_INPUT_TOKEN_COST_USD"   ,  5.00),
                "output"         : getenv_float("OPUS_OUTPUT_TOKEN_COST_USD"  , 25.00),
                "cache_write_5m" : getenv_float("OPUS_CACHE_WRITE_5M_COST_USD",  6.25),
                "cache_write_1h" : getenv_float("OPUS_CACHE_WRITE_1H_COST_USD", 10.00),
                "cache_read"     : getenv_float("OPUS_CACHE_READ_COST_USD"    ,  0.50),
            },
            "sonnet": {
                "input"          : getenv_float("SONNET_INPUT_TOKEN_COST_USD"   ,  3.00),
                "output"         : getenv_float("SONNET_OUTPUT_TOKEN_COST_USD"  , 15.00),
                "cache_write_5m" : getenv_float("SONNET_CACHE_WRITE_5M_COST_USD",  3.75),
                "cache_write_1h" : getenv_float("SONNET_CACHE_WRITE_1H_COST_USD",  6.00),
                "cache_read"     : getenv_float("SONNET_CACHE_READ_COST_USD"    ,  0.30),
            },
            "haiku": {
                "input"          : getenv_float("HAIKU_INPUT_TOKEN_COST_USD"   ,  1.00),
                "output"         : getenv_float("HAIKU_OUTPUT_TOKEN_COST_USD"  ,  5.00),
                "cache_write_5m" : getenv_float("HAIKU_CACHE_WRITE_5M_COST_USD",  1.25),
                "cache_write_1h" : getenv_float("HAIKU_CACHE_WRITE_1H_COST_USD",  2.00),
                "cache_read"     : getenv_float("HAIKU_CACHE_READ_COST_USD"    ,  0.10),
            }
        }
        self.sync_active_costs()

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

        self.error_log_path = os.getenv("ERROR_LOG_PATH", "claude_error_log.txt")
        self.model_list_timeout_seconds = getenv_float("MODEL_LIST_TIMEOUT_SECONDS", 10.0)


    def sync_active_costs(self) -> None:
        model_l = str(self.model or "").lower()
        if   "haiku"  in model_l : self.model_cost_family = "haiku"
        elif "sonnet" in model_l : self.model_cost_family = "sonnet"
        elif "opus"   in model_l : self.model_cost_family = "opus"
        elif "fable"  in model_l : self.model_cost_family = "fable"
        else:
            print(f"Unknown model family {model_l}. Using fable's (most expensive) pricing estimates.")
            self.model_cost_family = "fable"

        costs = self.cost_table[self.model_cost_family]
        self.input_token_cost_usd    = costs["input"]
        self.output_token_cost_usd   = costs["output"]
        self.cache_write_5m_cost_usd = costs["cache_write_5m"]
        self.cache_write_1h_cost_usd = costs["cache_write_1h"]
        self.cache_read_cost_usd     = costs["cache_read"]


    def set_prefill_mode(self, mode: str) -> None:
        if not mode in PREFILL_MODES : print(f"WARNING: ASSISTANT_PREFILL_MODE must be in {PREFILL_MODES}. Defaulting to 'none'."); return
        if mode == "assistant" :
            if self.version >= Version("4.6") : print("Mythos class models (>= 4.6) do not support assistant prefill."); return
            if self.thinking_enabled          : print("While thinking is enabled, prefill mode cannot be assistant."); return
        self.assistant_prefill_mode = mode

    def set_prefill(self, prefill: str) -> None:
        self.assistant_prefill = prefill

    def enable_thinking(self, en: bool):
        self.thinking_enabled = en
        self.resolve_thinking()

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

    def print_think_status(self) -> None:
        if self.preserve_thinking_blocks == UINT64_MAX : preserve_str = "inf"
        else                                           : preserve_str = str(self.preserve_thinking_blocks)

        if self.thinking_enabled              : print( "  Thinking enabled    ✅")
        else                                  : print( "  Thinking enabled    ❌")
        if self.use_adaptive                  :
                                                print(f"  Thinking effort     ✅  {self.thinking_effort}")
                                                print(f"  Thinking budget     ❌  {self.thinking_budget}")
        else                                  :
                                                print(f"  Thinking effort     ❌  {self.thinking_effort}")
                                                print(f"  Thinking budget     ✅  {self.thinking_budget}")
        if self.preserve_thinking_blocks <= 0 : print( "  Thinking preserved  ❌")
        else                                  : print(f"  Thinking preserved  {preserve_str}")

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
            if (self.top_p < 0.95) or (self.top_p > 1.00) :
                print("Thinking supports top_p in the range [0.95:1]. Clamping.")
            if   self.top_p < 0.95 : self.top_p = 0.95
            elif self.top_p > 1.00 : self.top_p = 1.00


    def apply_model(self, i_info: Dict[str, Any]) -> None:
        self.info  = i_info
        self.model = deep_get(self.info, "id")
        print(f"=== Switching to {self.model} ===")
        self.model_info  = self.info
        display_name_str = deep_get(self.info, "display_name")
        self.version     = extract_claude_version(display_name_str)
        if self.version == Version("0.0"):
            self.version = extract_claude_version(self.model)
        self.sync_active_costs()
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
        preserve_str = "inf" if self.preserve_thinking_blocks == UINT64_MAX else str(self.preserve_thinking_blocks)

        print()
        print("=== Runtime config start ===")
        print(f"host                   = {self.host} (restart required to change)")
        print(f"port                   = {self.port} (restart required to change)")
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
        print("=== Runtime config end ===")
        print()


# The single runtime configuration instance shared by every module.
# It is created empty here and populated by cfg.reload_from_env() at startup.
# Always mutate it in place; never rebind the name, or modules will desync.
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
