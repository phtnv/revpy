import builtins
import contextlib
import importlib
import io
import json
import sys
import tempfile

from pathlib           import Path
from types             import SimpleNamespace
from typing            import Any
from packaging.version import Version

import json5


ROOT     = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GREEN    = "\033[32m"
RED      = "\033[31m"
RESET    = "\033[0m"

MODEL      = "claude-test-model"
CREATED    = 1234567890
MAX_TOKENS = 256
USAGE      = {
    "input_tokens"                : 10,
    "output_tokens"               : 3,
    "cache_creation_input_tokens" : 0,
    "cache_read_input_tokens"     : 0,
}

sys.path.insert(0, str(ROOT))
common      : Any = importlib.import_module("common")
v1_messages : Any = importlib.import_module("v1_messages")
chat_api    : Any = importlib.import_module("v1_chat_completions")
resp_api    : Any = importlib.import_module("v1_responses")
server      : Any = importlib.import_module("server")

# The two OpenAI-style request builders, keyed by the wire protocol they speak.
BUILDERS = {
    "chat"      : lambda prepared: chat_api.build_body(prepared),
    "responses" : lambda prepared: resp_api.build_body(prepared),
}

# Where each endpoint carries the message list build_message_list() produces.
BUILDER_MESSAGE_KEY = {"chat": "messages", "responses": "input"}


class FakeMessages:
    def __init__(self, text: str) -> None:
        self.text  = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id          = "msg_test",
            stop_reason = "end_turn",
            content     = [SimpleNamespace(type="text", text=self.text)],
            usage       = SimpleNamespace(**USAGE, cache_creation=SimpleNamespace()),
        )


class FakeAnthropic:
    def __init__(self, text: str) -> None:
        self.messages = FakeMessages(text)


def make_config() -> Any:
    # common.cfg is a shared singleton; configure it in place instead of rebinding it.
    cfg = common.cfg
    cfg.reload_from_env()
    cfg.model                    = MODEL
    cfg.version                  = Version("4.0")
    cfg.max_tokens               = MAX_TOKENS
    cfg.debug_log                = False
    cfg.auto_trim                = False
    cfg.summary_blocks_enabled   = True
    cfg.cache_en                 = False
    cfg.split_lorebook           = False
    cfg.lorebook_at_end          = False
    cfg.lorebook_xml_at_end      = False
    cfg.assistant_prefill        = ""
    cfg.assistant_prefill_mode   = "none"
    cfg.send_temperature         = False
    cfg.send_top_k               = False
    cfg.send_top_p               = False
    cfg.thinking_enabled         = False
    cfg.use_adaptive             = False
    cfg.preserve_thinking_blocks = 0
    cfg.error_log_path           = str(ROOT / "test_error_log.txt")
    return cfg


def make_provider(**overrides: Any) -> dict[str, Any]:
    """
    A synthetic OPENAI_PROVIDERS entry, in the shape RuntimeConfig builds them.
    Defaults are the ones a provider gets with only BASE_URL and API_KEY set, so a
    case only has to name what it actually exercises.
    """
    provider = {
        "base_url"          : "https://provider.test/v1",
        "api_key"           : "test-key",
        "api_key_name"      : "TEST_API_KEY",
        "models"            : [],
        "models_regex"      : None,
        "max_tokens_param"  : "auto",
        "api"               : "chat",
        "reasoning_summary" : "auto",
        "store"             : False,
        "extra_body"        : {},
        "cost_families"     : [],
        "input_cost"        : 0.0,
        "output_cost"       : 0.0,
        "cache_read_cost"   : 0.0,
        "cache_write_cost"  : 0.0,
    }
    provider.update(overrides)
    return provider


def reference_from_fixture(name) -> dict[str, Any]:
    fixture : dict[str, Any] = json5.loads((FIXTURES / name).read_text(encoding="utf-8"))
    messages = []
    if fixture.get("system"):
        messages.append({"role": "system", "content": fixture["system"]})
    messages.extend(fixture["messages"])

    expected = {
        "status_code"          : 200,
        "anthropic_model"      : MODEL,
        "anthropic_max_tokens" : MAX_TOKENS,
        "anthropic_messages"   : fixture.get("expected_anthropic_messages", fixture["messages"]),
        "openai_model"         : f"anthropic/{MODEL}",
        "openai_created"       : CREATED,
        "openai_content"       : fixture.get("expected_openai_assistant", fixture["anthropic_response"]),
    }
    if "expected_usage" in fixture:
        expected["usage"] = fixture["expected_usage"]

    return {
        "request"  : {"model": "ignored-by-proxy", "max_tokens": MAX_TOKENS, "messages": messages},
        "reply"    : fixture["anthropic_response"],
        "expected" : expected,
    }


def received_from(response: Any, fake: FakeAnthropic) -> dict[str, Any]:
    body     = response.get_json(silent=True) or {}
    call     = fake.messages.calls[0] if fake.messages.calls else {}
    received = {"status_code": response.status_code}

    if call:
        received.update({
            "anthropic_model"      : call.get("model"),
            "anthropic_max_tokens" : call.get("max_tokens"),
            "anthropic_messages"   : call.get("messages"),
        })

    if body:
        received.update({
            "openai_model"   : body.get("model"),
            "openai_created" : body.get("created"),
            "openai_content" : body.get("choices", [{}])[0].get("message", {}).get("content"),
            "usage"          : body.get("usage"),
        })

    return received


def check_equal(expected: dict[str, Any], received: dict[str, Any]) -> bool:
    if expected == received:
        return True

    all_ok : bool = True
    for key in expected:
        if expected.get(key) == received.get(key):
            continue
        print(f"key={key} exp={expected.get(key)!r}, rec={received.get(key)!r}")
        all_ok = False

    return all_ok

def run_cli_commands(commands: list[str]) -> None:
    """
    Feeds commands to the admin CLI loop, then EOF. CLI output is swallowed.
    """
    pending = iter(commands)

    def fake_input(prompt: str = "") -> str:
        try                  : return next(pending)
        except StopIteration : raise EOFError

    original_input = builtins.input
    builtins.input = fake_input
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            server.admin_cli_loop()
    finally:
        builtins.input = original_input


tests_ttl : int = 0

def test_basic_non_streaming_roundtrip(name: str) -> bool:
    global tests_ttl
    tests_ttl += 1
    print(f"Testing non-streaming roundtrip with '{name}'... ", end="")

    ref_msg = reference_from_fixture(name)
    rx_msg  = FakeAnthropic(ref_msg["reply"])

    v1_messages.get_anthropic_client = lambda: rx_msg
    v1_messages.print_usage          = lambda counts: None
    server.time.time                 = lambda: CREATED
    response = server.app.test_client().post("/v1/chat/completions", json=ref_msg["request"])

    rx_msg = received_from(response, rx_msg)
    passed = check_equal(ref_msg["expected"], rx_msg)
    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_chat_dump_formats(name: str) -> bool:
    global tests_ttl
    tests_ttl += 1
    print(f"Testing chat dump (json and natural) with '{name}'... ", end="")

    fixture = json5.loads((FIXTURES / name).read_text(encoding="utf-8"))
    ref_msg = reference_from_fixture(name)
    fake    = FakeAnthropic(ref_msg["reply"])

    v1_messages.get_anthropic_client = lambda: fake
    v1_messages.print_usage          = lambda counts: None
    server.time.time                 = lambda: CREATED
    server.app.test_client().post("/v1/chat/completions", json=ref_msg["request"])

    expected_snapshot = fixture["expected_snapshot"]
    expected_markdown = (FIXTURES / fixture["expected_markdown_file"]).read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "chat_snapshot.json"
        md_path   = Path(tmp_dir) / "chat_snapshot.md"

        run_cli_commands([f"dump json {json_path}", f"dump natural {md_path}"])

        if not json_path.is_file() or not md_path.is_file():
            print(f"{RED}FAIL{RESET} (dump command wrote no file)")
            return False

        received_snapshot = json.loads(json_path.read_text(encoding="utf-8"))
        received_markdown = md_path.read_text(encoding="utf-8")

    passed = check_equal(expected_snapshot, received_snapshot)
    if expected_markdown != received_markdown:
        print(f"markdown exp={expected_markdown!r}, rec={received_markdown!r}")
        passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


WIRE_PROTOCOL_CASES = [
    # (base_url, expected protocol)
    ("https://api.openai.com/v1"                  , "responses"),
    ("https://API.OpenAI.com/v1"                  , "responses"),
    ("https://my-resource.openai.azure.com/openai/v1", "responses"),
    ("https://api.z.ai/api/paas/v4"               , "chat"),
    ("https://api.moonshot.ai/v1"                 , "chat"),
    ("https://api.aionlabs.ai/v1"                 , "chat"),
    ("https://openrouter.ai/api/v1"               , "chat"),
    # Must key on the host, not a substring: these are not OpenAI.
    ("https://api.openai.com.evil.test/v1"        , "chat"),
    ("https://not-openai.example.com/v1"          , "chat"),
]


def test_wire_protocol_derivation() -> bool:
    """
    The wire protocol is derived from the provider endpoint rather than configured,
    so the derivation is the only thing deciding which backend module serves a
    provider. Anything but an OpenAI host must land on /chat/completions.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing wire protocol derivation from base_url... ", end="")

    passed = True
    for base_url, expected in WIRE_PROTOCOL_CASES:
        received = "responses" if common.openai_native_endpoint(base_url) else "chat"
        if received != expected:
            print(f"\n  {base_url}: exp={expected} rec={received}", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def anthropic_usage(inp=0, out=0, read=0, creation=0, e1h=None, e5m=None) -> Any:
    """An Anthropic SDK usage object, in the attribute shape parse_usage() reads."""
    cache_creation = SimpleNamespace()
    if e1h is not None: cache_creation.ephemeral_1h_input_tokens = e1h
    if e5m is not None: cache_creation.ephemeral_5m_input_tokens = e5m
    return SimpleNamespace(
        input_tokens                = inp,
        output_tokens               = out,
        cache_read_input_tokens     = read,
        cache_creation_input_tokens = creation,
        cache_creation              = cache_creation,
    )


# Every backend's parse_usage() feeds the same two consumers, so both are pinned here:
# what the cost tracker bills from, and what the client is told. These are the numbers
# the per-request cost report is computed from, so they are asserted exactly.
#   (label, backend, raw usage, expected cost, expected client[, cfg overrides])
USAGE_CASES: list[tuple] = [
    ("messages: cache read", "messages",
     anthropic_usage(inp=10, out=50, read=90),
     {"uncached_input": 10, "cache_read": 90, "cache_write_1h": 0, "cache_write_5m": 0, "output": 50, "reasoning": None},
     {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
      "input_tokens_uncached": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 90}),

    # Anthropic is the only backend that splits cache writes by TTL.
    ("messages: 1h/5m write split", "messages",
     anthropic_usage(inp=10, out=50, creation=100, e1h=60, e5m=40),
     {"uncached_input": 10, "cache_read": 0, "cache_write_1h": 60, "cache_write_5m": 40, "output": 50, "reasoning": None},
     {"prompt_tokens": 110, "completion_tokens": 50, "total_tokens": 160,
      "input_tokens_uncached": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0}),

    # A legacy payload reports only a total cache write, with no 5m/1h split to read.
    # fallback_cache_write_ttl() then guesses from the configured markers, so the same
    # payload is billed differently depending on them. Both directions are pinned.
    ("messages: legacy write, 1h marker active", "messages",
     anthropic_usage(inp=10, out=50, creation=100),
     {"uncached_input": 10, "cache_read": 0, "cache_write_1h": 100, "cache_write_5m": 0, "output": 50, "reasoning": None},
     {"prompt_tokens": 110, "completion_tokens": 50, "total_tokens": 160,
      "input_tokens_uncached": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0},
     {"cache_en": True, "cache_system": True, "cache_system_ttl": "1h",
      "cache_manual_msg": 0, "cache_auto_msg": 0}),

    # Caching off: nothing was written on purpose, so the cheaper bucket is assumed.
    ("messages: legacy write, caching off", "messages",
     anthropic_usage(inp=10, out=50, creation=100),
     {"uncached_input": 10, "cache_read": 0, "cache_write_1h": 0, "cache_write_5m": 100, "output": 50, "reasoning": None},
     {"prompt_tokens": 110, "completion_tokens": 50, "total_tokens": 160,
      "input_tokens_uncached": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 0},
     {"cache_en": False}),

    ("chat: cached + write", "chat",
     {"prompt_tokens": 100, "completion_tokens": 50,
      "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30}},
     {"uncached_input": 20, "cache_read": 50, "cache_write_1h": 0, "cache_write_5m": 30, "output": 50, "reasoning": None},
     {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
      "input_tokens_uncached": 20, "cache_creation_input_tokens": 30, "cache_read_input_tokens": 50}),

    # A payload claiming more cached/written tokens than there were input tokens must
    # not drive uncached input negative.
    ("chat: overclaimed cache clamps", "chat",
     {"prompt_tokens": 10, "completion_tokens": 5,
      "prompt_tokens_details": {"cached_tokens": 999, "cache_write_tokens": 999}},
     {"uncached_input": 0, "cache_read": 10, "cache_write_1h": 0, "cache_write_5m": 0, "output": 5, "reasoning": None},
     {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
      "input_tokens_uncached": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 10}),

    # A reported zero is a measurement and must survive as 0, not become None.
    ("chat: reported zero reasoning", "chat",
     {"prompt_tokens": 40, "completion_tokens": 90,
      "completion_tokens_details": {"reasoning_tokens": 0}},
     {"uncached_input": 40, "cache_read": 0, "cache_write_1h": 0, "cache_write_5m": 0, "output": 90, "reasoning": 0},
     {"prompt_tokens": 40, "completion_tokens": 90, "total_tokens": 130,
      "input_tokens_uncached": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),

    ("responses: cached + write", "responses",
     {"input_tokens": 100, "output_tokens": 50,
      "input_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 30}},
     {"uncached_input": 20, "cache_read": 50, "cache_write_1h": 0, "cache_write_5m": 30, "output": 50, "reasoning": None},
     {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
      "input_tokens_uncached": 20, "cache_creation_input_tokens": 30, "cache_read_input_tokens": 50}),

    ("responses: reasoning tokens", "responses",
     {"input_tokens": 40, "output_tokens": 90,
      "output_tokens_details": {"reasoning_tokens": 70}},
     {"uncached_input": 40, "cache_read": 0, "cache_write_1h": 0, "cache_write_5m": 0, "output": 90, "reasoning": 70},
     {"prompt_tokens": 40, "completion_tokens": 90, "total_tokens": 130,
      "input_tokens_uncached": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}),
]

USAGE_BACKENDS = {"messages": v1_messages, "chat": chat_api, "responses": resp_api}


def test_usage_normalization(case: tuple) -> bool:
    """
    One backend's parse_usage() plus the two shared consumers in common.py.
    All three backends normalize to one counts shape; this pins what comes out of it.
    """
    global tests_ttl
    tests_ttl += 1
    label, backend, raw, expected_cost, expected_client = case[:5]
    overrides = case[5] if len(case) > 5 else {}
    print(f"Testing usage normalization for {label}... ", end="")

    cfg = make_config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    counts = USAGE_BACKENDS[backend].parse_usage(raw)

    passed  = check_equal(expected_cost  , common.usage_to_cost_tokens(counts))
    passed &= check_equal(expected_client, common.usage_to_openai_dict(counts))

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def load_body_cases(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixture = json5.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return fixture["inputs"], fixture["cases"]


def test_provider_request_body(inputs: dict[str, Any], case: dict[str, Any]) -> bool:
    """
    Builds one OpenAI-style request body and compares it to the golden dict.

    Each case re-runs make_config() so it is isolated from the last one: cfg is a
    shared singleton and these cases leave a provider backend selected on it.
    """
    global tests_ttl
    tests_ttl += 1
    print(f"Testing request body for {case['name']}... ", end="")

    entry = inputs[case["input"]]

    # Config warnings from .env, and the sampling-refused warning, are not under test.
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = make_config()
        cfg.backend          = case["backend"]
        cfg.model            = case["model"]
        cfg.openai_providers = {case["backend"]: make_provider(**case.get("provider", {}))}
        cfg.thinking_enabled = False
        cfg.thinking_effort  = "medium"
        for key, value in case.get("cfg", {}).items():
            setattr(cfg, key, value)

        received = BUILDERS[case["builder"]](entry["prepared"])

    expected = {**case["expected"], BUILDER_MESSAGE_KEY[case["builder"]]: entry["messages"]}

    passed = check_equal(expected, received)
    # check_equal only walks the keys it was given, so a field the builder should not
    # have sent at all would slip past it. For a golden body that is a failure too.
    for key in received:
        if key not in expected:
            print(f"unexpected key={key} rec={received[key]!r}")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


if __name__ == "__main__":
    make_config()

    tests_passed : int = 0
    tests_passed += test_basic_non_streaming_roundtrip("basic_no_ooc.json5")
    tests_passed += test_basic_non_streaming_roundtrip("basic_with_ooc.json5")
    tests_passed += test_chat_dump_formats("dump_chat.json5")

    tests_passed += test_wire_protocol_derivation()

    for usage_case in USAGE_CASES:
        tests_passed += test_usage_normalization(usage_case)

    body_inputs, body_cases = load_body_cases("provider_bodies.json5")
    for body_case in body_cases:
        tests_passed += test_provider_request_body(body_inputs, body_case)

    tests_failed : int = tests_ttl - tests_passed

    if tests_failed == 0 : print(f"{GREEN}All {tests_ttl} tests passed.{RESET}")
    else                 : print(f"{RED}{tests_failed} out of {tests_ttl} tests failed.{RESET}")
