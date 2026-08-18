import base64
import builtins
import contextlib
import importlib
import io
import json
import os
import struct
import sys
import tempfile
import threading
import zlib

from pathlib           import Path
from types             import SimpleNamespace
from typing            import Any
from packaging.version import Version

import httpx
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
providers   : Any = importlib.import_module("providers")
v1_messages : Any = importlib.import_module("v1_messages")
chat_api    : Any = importlib.import_module("v1_chat_completions")
resp_api    : Any = importlib.import_module("v1_responses")
v1_images   : Any = importlib.import_module("v1_images")
server      : Any = importlib.import_module("server")

image_orchestrator : Any = importlib.import_module("image_orchestrator")

BUILDERS = {
    "messages"  : lambda prepared: v1_messages.build_body(prepared),
    "chat"      : lambda prepared: chat_api.build_body(prepared),
    "responses" : lambda prepared: resp_api.build_body(prepared),
}

# OpenAI-style bodies share one message list under different field names.
BUILDER_MESSAGE_KEY = {"chat": "messages", "responses": "input"}

PROVIDER = "claude"


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
    cfg.error_log_path           = os.path.join(tempfile.gettempdir(), "revpy_test_error_log.txt")
    cfg.providers = {PROVIDER: make_provider(api="messages", api_key_name="CLAUDE_API_KEY")}
    cfg.backend   = PROVIDER
    return cfg


def make_provider(**overrides: Any) -> dict[str, Any]:
    """A RuntimeConfig-style provider entry with only the exercised fields overridden."""
    provider = {
        "api"                 : "chat",
        "base_url"            : "https://provider.test/v1",
        "api_key"             : "test-key",
        "api_key_name"        : "TEST_API_KEY",
        "models"              : [],
        "models_regex"        : None,
        "max_tokens_param"    : "auto",
        "reasoning_summary"   : "auto",
        "store"               : False,
        "background"          : False,
        "extra_body"          : {},
        "cost_families"       : [],
        "input_cost"          : 0.0,
        "output_cost"         : 0.0,
        "cache_read_cost"     : 0.0,
        "cache_write_5m_cost" : 0.0,
        "cache_write_1h_cost" : 0.0,
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
        "openai_model"         : f"{PROVIDER}/{MODEL}",
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


def check_case_equal(label: str, expected: dict[str, Any], received: dict[str, Any]) -> bool:
    if expected == received:
        return True
    print(f"[{label}] ", end="")
    return check_equal(expected, received)


def run_cli_commands(commands: list[str]) -> None:
    """Feed commands to the admin CLI loop, then EOF."""
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


# The environment one config-parsing pass reads. Exercises all three lists, a name
# declared twice, a name with no base URL, and a cost family with the two cache TTLs
# priced apart.
PROVIDER_ENV = {
    "V1_MESSAGES_PROVIDERS"        : "claude",
    "V1_CHAT_COMPLETIONS_PROVIDERS": "glm,twice,nourl",
    "V1_RESPONSES_PROVIDERS"       : "gpt,twice",

    "CLAUDE_BASE_URL"              : "https://api.anthropic.com/v1/",
    "CLAUDE_INPUT_TOKEN_COST_USD"  : "3.00",
    "CLAUDE_OUTPUT_TOKEN_COST_USD" : "15.00",
    "CLAUDE_MODEL_OPUS_REGEX"      : "opus",
    "CLAUDE_MODEL_OPUS_COST"       : "{input: 5.00, output: 25.00, cache_read: 0.50, cache_write_5m: 6.25, cache_write_1h: 10.00}",
    "GLM_BASE_URL"                 : "https://api.z.ai/api/paas/v4",
    "GPT_BASE_URL"                 : "https://api.openai.com/v1",
    "TWICE_BASE_URL"               : "https://twice.test/v1",
}


def parse_provider_env() -> Any:
    """
    Reloads the config over PROVIDER_ENV, then restores the test config. Config parsing
    reads os.environ directly, so this is the only way to exercise it.
    """
    for key, value in PROVIDER_ENV.items():
        os.environ[key] = value
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            common.cfg.reload_from_env()
            return dict(common.cfg.providers)
    finally:
        for key in PROVIDER_ENV:
            os.environ.pop(key, None)
        make_config()


def test_provider_config_parsing() -> bool:
    """
    The wire protocol is the list a provider was declared in, and nothing else decides
    which backend module serves it. This pins that mapping, the declaration order the
    CLI numbering follows, and the two ways a declaration is rejected.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing provider config parsing... ", end="")

    providers_parsed = parse_provider_env()

    # Order is messages, then chat, then responses -- not the order names appear in .env.
    expected_order = ["claude", "glm", "twice", "gpt"]
    expected_api   = {"claude": "messages", "glm": "chat", "twice": "chat", "gpt": "responses"}

    passed = check_equal(
        {"order": expected_order, "api": expected_api},
        {"order": list(providers_parsed), "api": {n: p["api"] for n, p in providers_parsed.items()}},
    )

    # 'nourl' has no base URL and is dropped; 'twice' keeps its first declaration (chat).
    if "nourl" in providers_parsed:
        print("provider without a base URL was not skipped ", end="")
        passed = False

    # A trailing slash must not survive into the request URL.
    if providers_parsed["claude"]["base_url"] != "https://api.anthropic.com/v1":
        print(f"base_url exp='https://api.anthropic.com/v1' rec={providers_parsed['claude']['base_url']!r} ", end="")
        passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


# First matching cost family wins; otherwise provider prices are used.
COST_CASES = [
    ("cost family wins, TTLs priced apart", "claude-opus-4-8",
     {"family": "claude:opus", "input": 5.00, "output": 25.00, "read": 0.50, "write_5m": 6.25, "write_1h": 10.00}),

    ("no family matches, provider prices", "claude-sonnet-4-6",
     {"family": "claude", "input": 3.00, "output": 15.00, "read": 0.00, "write_5m": 3.00, "write_1h": 3.00}),
]


def test_cost_resolution() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing cost resolution... ", end="")

    providers_parsed = parse_provider_env()
    passed = True

    for label, model_id, expected in COST_CASES:
        cfg = make_config()
        cfg.providers = providers_parsed
        with contextlib.redirect_stdout(io.StringIO()):
            providers.apply_model({"id": model_id, "provider": "claude"})

        received = {
            "family"   : cfg.model_cost_family,
            "input"    : cfg.input_token_cost_usd,
            "output"   : cfg.output_token_cost_usd,
            "read"     : cfg.cache_read_cost_usd,
            "write_5m" : cfg.cache_write_5m_cost_usd,
            "write_1h" : cfg.cache_write_1h_cost_usd,
        }
        passed &= check_case_equal(label, expected, received)

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


# One normalized usage shape feeds both billing and the OpenAI-compatible response.
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


def test_usage_normalization() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing usage normalization across backends... ", end="")

    passed = True
    for label, backend, raw, expected_cost, expected_client, *rest in USAGE_CASES:
        cfg = make_config()
        for key, value in (rest[0] if rest else {}).items():
            setattr(cfg, key, value)
        counts = USAGE_BACKENDS[backend].parse_usage(raw)

        passed &= check_case_equal(f"{label} cost", expected_cost, common.usage_to_cost_tokens(counts))
        passed &= check_case_equal(f"{label} client", expected_client, common.usage_to_openai_dict(counts))

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


class FakeStream:
    """A /responses SSE body, replayed from a list of already-encoded 'data:' lines."""
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines      = lines

    def iter_lines(self): return iter(self._lines)
    def __enter__(self): return self
    def __exit__(self, *exc): return False


class FakeStreamClient:
    """
    Serves one canned body per connection, in the order the case lists them: a
    background turn that is cut reconnects, and each connection gets the next body.
    """
    def __init__(self, bodies: list[list[str]], calls: list[tuple[str, str]]) -> None:
        self._bodies = list(bodies)
        self._calls  = calls

    def __enter__(self): return self
    def __exit__(self, *exc): return False

    def stream(self, method: str, url: str, **kwargs: Any):
        self._calls.append((method, url))
        body = self._bodies.pop(0) if self._bodies else []
        # A body of TIMEOUT is a connection that goes quiet rather than closing.
        if body == TIMEOUT:
            raise httpx.ReadTimeout("timed out")
        return FakeStream(body)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self._payload    = payload
        self.text        = json.dumps(payload)

    def json(self): return self._payload


def fake_httpx(bodies: list[list[str]], objects: list[dict[str, Any]], calls: list[tuple[str, str]],
               posts: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    """
    Stands in for the httpx module in v1_responses: streamed bodies for the connections,
    canned response objects for the retrievals, and one list recording every call so a
    case can assert what the recovery path actually did. 'posts' answers the
    non-streaming request; without it a POST is a cancellation.
    """
    def get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(("GET", url))
        return FakeResponse(objects.pop(0) if objects else {"id": "resp_1", "status": "in_progress"})

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(("POST", url))
        if posts:
            return FakeResponse(posts.pop(0))
        return FakeResponse({"id": "resp_1", "status": "cancelled"})

    return SimpleNamespace(Client=lambda **kw: FakeStreamClient(bodies, calls), get=get, post=post,
                           TimeoutException=httpx.TimeoutException)


# A connection that never delivers anything, as opposed to one that closes early.
TIMEOUT = "TIMEOUT"


def sse(*events: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(e)}" for e in events]


def seq(number: int, event: dict[str, Any]) -> dict[str, Any]:
    """The same event, numbered. A resumed stream is asked to start after a number."""
    return {**event, "sequence_number": number}


CREATED_EVENT   = {"type": "response.created",  "response": {"id": "resp_1", "status": "in_progress"}}
TEXT_EVENT      = {"type": "response.output_text.delta", "delta": "Hello."}
REASONING_EVENT = {"type": "response.reasoning_summary_text.delta", "delta": "Thinking..."}
COMPLETED_EVENT = {"type": "response.completed", "response": {
    "id": "resp_1", "status": "completed",
    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}

# The response object as a retrieval returns it, for the recovery paths.
def response_object(status: str, text: str = "", reasoning: str = "") -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if reasoning : output.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    if text      : output.append({"type": "message"  , "content": [{"type": "output_text" , "text": text     }]})
    return {"id": "resp_1", "status": status, "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}


# A /responses stream is only finished when the response reports a terminal status.
# Running out of body without one means the connection was cut mid-response; relaying
# that as a clean stop hands the client an empty message and no reason for it, which
# is indistinguishable from the model choosing to say nothing.
#
# In background mode a cut is recoverable instead: the response outlives the connection,
# so the proxy reconnects to it while it is still running, and reads it off the response
# object once it has finished (a finished response does not replay its events).
STREAM_CASES = [
    {
        "name"   : "completed stream is relayed",
        "bodies" : [sse(CREATED_EVENT, TEXT_EVENT, COMPLETED_EVENT)],
        "text"   : "Hello.",
    },
    {
        "name"   : "truncated after reasoning raises",
        "bodies" : [sse(CREATED_EVENT, REASONING_EVENT)],
        "error"  : True,
    },
    {
        "name"   : "truncated after text raises",
        "bodies" : [sse(CREATED_EVENT, TEXT_EVENT)],
        "error"  : True,
    },
    {
        "name"   : "truncated before any content raises",
        "bodies" : [sse(CREATED_EVENT)],
        "error"  : True,
    },
    {
        # The job is still running, so reconnecting picks the rest of it up live.
        "name"       : "background resumes a running response",
        "background" : True,
        "bodies"     : [
            sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT)),
            sse(seq(2, {"type": "response.output_text.delta", "delta": " More."}), seq(3, COMPLETED_EVENT)),
        ],
        "objects"    : [response_object("in_progress")],
        "text"       : "Hello. More.",
        "resumed"    : "starting_after=1",
    },
    {
        # It finished while the proxy was away: no events left to stream, only the object.
        "name"       : "background recovers a finished response from the object",
        "background" : True,
        "bodies"     : [sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT))],
        "objects"    : [response_object("completed", text="Hello. The rest.", reasoning="Thinking...")],
        "text"       : "Hello. The rest.",
        # Nothing of the reasoning was streamed before the cut, so all of it is recovered.
        "reasoning"  : "Thinking...",
    },
    {
        # The recovered reply should continue what the client already has. When it does
        # not, the two cannot be stitched together, and guessing at a join would corrupt
        # the message -- so the reply is left as it arrived.
        "name"       : "background will not graft on a reply that does not continue",
        "background" : True,
        "bodies"     : [sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT))],
        "objects"    : [response_object("completed", text="Something else entirely.")],
        "text"       : "Hello.",
    },
    {
        # A replayed number must not become duplicated text.
        "name"       : "background drops events it has already seen",
        "background" : True,
        "bodies"     : [
            sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT)),
            sse(seq(1, TEXT_EVENT), seq(2, COMPLETED_EVENT)),
        ],
        "objects"    : [response_object("in_progress")],
        "text"       : "Hello.",
    },
    {
        # A job that is still running is not given up on for bringing nothing: that is
        # what silent reasoning looks like from here. Only the turn timeout ends it, and
        # then the job is cancelled rather than left running.
        "name"         : "background keeps resuming a job that is still running",
        "background"   : True,
        "turn_timeout" : 0.0,
        "bodies"       : [sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT))] + [[] for _ in range(12)],
        "objects"      : [response_object("in_progress") for _ in range(12)],
        "error"        : True,
        "cancelled"    : True,
        # Nothing is retried once the budget is already spent, so exactly one cut is seen.
        "resumes"      : 0,
    },
    {
        # The same job, with room to work: reconnecting continues until it finishes,
        # however many cuts that takes and however little some of them carry.
        "name"         : "background outlasts several empty reconnects",
        "background"   : True,
        "bodies"       : [
            sse(seq(0, CREATED_EVENT), seq(1, REASONING_EVENT)),
            [], [], [],
            sse(seq(2, TEXT_EVENT), seq(3, COMPLETED_EVENT)),
        ],
        "objects"      : [response_object("in_progress") for _ in range(4)],
        "text"         : "Hello.",
        "reasoning"    : "Thinking...",
        "resumes"      : 4,
    },
    {
        # A connection that goes quiet is the same cut in another shape.
        "name"       : "background recovers a connection that times out",
        "background" : True,
        "bodies"     : [sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT)), TIMEOUT,
                        sse(seq(2, COMPLETED_EVENT))],
        "objects"    : [response_object("in_progress"), response_object("in_progress")],
        "text"       : "Hello.",
    },
    {
        # Without a background job behind it, a quiet connection is a real failure.
        "name"    : "a connection that times out without background raises",
        "bodies"  : [TIMEOUT],
        "error"   : True,
    },
    {
        # The client hung up. The response would otherwise keep running, and billing.
        "name"       : "background cancels when the client walks away",
        "background" : True,
        "bodies"     : [sse(seq(0, CREATED_EVENT), seq(1, TEXT_EVENT), seq(2, COMPLETED_EVENT))],
        "abandon"    : True,
        "cancelled"  : True,
    },
]


def test_responses_stream_termination(case: dict[str, Any]) -> bool:
    global tests_ttl
    tests_ttl += 1
    print(f"Testing /responses stream: {case['name']}... ", end="")

    cfg = make_config()
    cfg.backend   = "gpt"
    cfg.model     = "gpt-5.6-sol"
    cfg.providers = {"gpt": make_provider(api="responses", background=case.get("background", False))}
    cfg.responses_poll_seconds          = 0.0
    # Short on purpose: a recovery loop that fails to terminate should fail the case
    # quickly rather than sit there until a real timeout expires.
    cfg.responses_turn_timeout_seconds  = case.get("turn_timeout", 5.0)

    prepared = {"messages": [{"role": "user", "content": "hi"}], "system_segments": [],
                "system_summary_text": "", "lorebook_at_end_text": "", "max_tokens": 64}

    calls  : list[tuple[str, str]] = []
    chunks : dict[str, list[str]]  = {"text": [], "reasoning": []}

    original_httpx, original_headers = resp_api.httpx, resp_api.request_headers
    resp_api.httpx = fake_httpx(case["bodies"], list(case.get("objects", [])), calls)
    resp_api.request_headers = lambda provider: {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            raised, final = None, None
            stream = resp_api.generate_stream(prepared)
            try:
                for kind, data in stream:
                    if kind == "final" : final = data
                    else               : chunks[kind].append(data)
                    # Walking away mid-turn is what a disconnected client looks like from
                    # here: the generator is closed while it is still producing.
                    if case.get("abandon") and kind == "text":
                        stream.close()
                        break
            except Exception as exc:
                raised = exc
    finally:
        resp_api.httpx, resp_api.request_headers = original_httpx, original_headers

    passed = True

    if case.get("error"):
        if raised is None or final is not None:
            print(f"expected an error, got final={final!r} raised={raised!r} ", end="")
            passed = False
    elif case.get("abandon"):
        if raised is not None:
            print(f"expected no error on abandon, got {raised!r} ", end="")
            passed = False
    else:
        if raised is not None or final is None or final["stop_reason"] != "stop":
            print(f"expected a clean final, got final={final!r} raised={raised!r} ", end="")
            passed = False

    for kind in ("text", "reasoning"):
        if kind not in case:
            continue
        received = "".join(chunks[kind])
        if received != case[kind]:
            print(f"expected {kind}={case[kind]!r}, got {received!r} ", end="")
            passed = False

    if case.get("resumed") and not any(method == "GET" and case["resumed"] in url for method, url in calls):
        print(f"expected a resume carrying {case['resumed']!r}, got {calls!r} ", end="")
        passed = False

    if "resumes" in case:
        resumes = sum(1 for method, url in calls if method == "GET" and "starting_after=" in url)
        if resumes != case["resumes"]:
            print(f"expected {case['resumes']} resume attempts, got {resumes} ", end="")
            passed = False

    cancelled = any(method == "POST" and url.endswith("/cancel") for method, url in calls)
    if cancelled != bool(case.get("cancelled")):
        print(f"expected cancelled={bool(case.get('cancelled'))}, got {cancelled} from {calls!r} ", end="")
        passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


# A background request is answered with 'queued' and no reply at all, so the
# non-streaming path has to wait for the job and collect it. Relaying that first body
# would hand the client an empty message -- the failure this endpoint is prone to hiding.
NON_STREAM_CASES = [
    {
        "name"       : "background waits for the job and collects the reply",
        "background" : True,
        "posts"      : [{"id": "resp_1", "status": "queued"}],
        "objects"    : [response_object("in_progress"),
                        response_object("completed", text="Hello.", reasoning="Thinking...")],
        "text"       : "Hello.",
        "reasoning"  : "Thinking...",
    },
    {
        "name"       : "background relays a failure instead of an empty reply",
        "background" : True,
        "posts"      : [{"id": "resp_1", "status": "queued"}],
        "objects"    : [{"id": "resp_1", "status": "failed", "error": {"message": "the model gave up"}}],
        "error"      : True,
    },
    {
        # Without background the POST already carries the whole response, and nothing
        # about this path changes: no retrieval at all.
        "name"       : "a finished response is used as it arrives",
        "posts"      : [response_object("completed", text="Hello.")],
        "text"       : "Hello.",
        "retrievals" : 0,
    },
]


def test_responses_non_stream(case: dict[str, Any]) -> bool:
    global tests_ttl
    tests_ttl += 1
    print(f"Testing /responses non-streaming: {case['name']}... ", end="")

    cfg = make_config()
    cfg.backend   = "gpt"
    cfg.model     = "gpt-5.6-sol"
    cfg.providers = {"gpt": make_provider(api="responses", background=case.get("background", False))}
    cfg.responses_poll_seconds         = 0.0
    cfg.responses_turn_timeout_seconds = 600.0

    prepared = {"messages": [{"role": "user", "content": "hi"}], "system_segments": [],
                "system_summary_text": "", "lorebook_at_end_text": "", "max_tokens": 64}

    calls : list[tuple[str, str]] = []

    original_httpx, original_headers = resp_api.httpx, resp_api.request_headers
    resp_api.httpx = fake_httpx([], list(case.get("objects", [])), calls, list(case.get("posts", [])))
    resp_api.request_headers = lambda provider: {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            raised, result = None, None
            try: result = resp_api.generate_non_stream(prepared)
            except Exception as exc: raised = exc
    finally:
        resp_api.httpx, resp_api.request_headers = original_httpx, original_headers

    passed = True

    if case.get("error"):
        if raised is None or result is not None:
            print(f"expected an error, got result={result!r} raised={raised!r} ", end="")
            passed = False
    elif raised is not None or result is None:
        print(f"expected a reply, got result={result!r} raised={raised!r} ", end="")
        passed = False
    else:
        for key in ("text", "reasoning"):
            if key in case and case[key] not in result["text"]:
                print(f"expected {key}={case[key]!r} in {result['text']!r} ", end="")
                passed = False

    if "retrievals" in case:
        retrievals = sum(1 for method, _ in calls if method == "GET")
        if retrievals != case["retrievals"]:
            print(f"expected {case['retrievals']} retrievals, got {retrievals} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def load_body_cases(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fixture = json5.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return fixture["inputs"], fixture["cases"]


def test_provider_request_bodies() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing provider request bodies... ", end="")

    inputs, cases = load_body_cases("provider_bodies.json5")
    passed = True

    for case in cases:
        entry = inputs[case["input"]]

        with contextlib.redirect_stdout(io.StringIO()):
            cfg = make_config()
            cfg.backend          = case["backend"]
            cfg.model            = case["model"]
            cfg.providers        = {case["backend"]: make_provider(api=case["builder"], **case.get("provider", {}))}
            cfg.thinking_enabled = False
            cfg.thinking_effort  = "medium"
            for key, value in case.get("cfg", {}).items():
                setattr(cfg, key, value)

            received = BUILDERS[case["builder"]](entry["prepared"])

        if case["builder"] == "messages":
            expected = {**entry["anthropic"], **case["expected"]}
        else:
            expected = {**case["expected"], BUILDER_MESSAGE_KEY[case["builder"]]: entry["messages"]}

        passed &= check_case_equal(case["name"], expected, received)
        for key in received:
            if key not in expected:
                print(f"[{case['name']}] unexpected key={key} rec={received[key]!r}")
                passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


# Image generation. None of these contact a provider.
IMAGE_EXTRACTION_CASES = [
    {
        "name"       : "a block in the last user message triggers and is stripped",
        "messages"   : [{"role": "user", "content": "<image_generation>prompt: \"a fox\"</image_generation>"}],
        "prompts"    : ["a fox"],
        "needs_text" : False,
        "cleaned"    : [""],
    },
    {
        "name"       : "history is stripped but never re-triggered",
        "messages"   : [
            {"role": "user"     , "content": "before <image_generation>prompt: \"an old fox\"</image_generation> after"},
            {"role": "assistant", "content": "done"},
            {"role": "user"     , "content": "just talking"},
        ],
        "prompts"    : [],
        "needs_text" : True,
        "cleaned"    : ["before  after", "done", "just talking"],
    },
    {
        "name"       : "a mixed turn keeps its text and still generates",
        "messages"   : [{"role": "user", "content": "describe the room\n<image_generation>prompt: \"a lit chamber\"</image_generation>"}],
        "prompts"    : ["a lit chamber"],
        "needs_text" : True,
        "cleaned"    : ["describe the room"],
    },
    {
        "name"       : "a body with no field syntax is taken as a bare prompt",
        "messages"   : [{"role": "user", "content": "<image_generation>a gray tabby cat hugging an otter</image_generation>"}],
        "prompts"    : ["a gray tabby cat hugging an otter"],
        "needs_text" : False,
        "cleaned"    : [""],
    },
    {
        "name"       : "a body that already carries braces parses as-is",
        "messages"   : [{"role": "user", "content": "<image_generation>{prompt: \"a braced fox\"}</image_generation>"}],
        "prompts"    : ["a braced fox"],
        "needs_text" : False,
        "cleaned"    : [""],
    },
    {
        "name"       : "an unsupported field is reported, not silently defaulted",
        "messages"   : [{"role": "user", "content": "<image_generation>prompt: \"x\", qualtiy: \"high\"</image_generation>"}],
        "prompts"    : [],
        "needs_text" : False,
        "errors"     : 1,
        "cleaned"    : [""],
    },
    {
        "name"       : "assistant output is never scanned",
        "messages"   : [
            {"role": "user"     , "content": "hello"},
            {"role": "assistant", "content": "<image_generation>prompt: \"sneaky\"</image_generation>"},
        ],
        "prompts"    : [],
        "needs_text" : True,
        "cleaned"    : ["hello", "<image_generation>prompt: \"sneaky\"</image_generation>"],
    },
]


def make_image_config() -> None:
    """Turns image generation on with known defaults, without touching the chat model."""
    cfg = common.cfg
    cfg.image_enabled            = True
    cfg.image_chat_enabled       = True
    cfg.image_provider           = "testimg"
    cfg.image_model              = "gpt-image-2"
    cfg.image_request_tag        = "image_generation"
    cfg.image_default_size       = "1024x1024"
    cfg.image_default_quality    = "medium"
    cfg.image_default_format     = "png"
    cfg.image_default_background = "opaque"
    cfg.image_default_n          = 1
    cfg.image_default_batch      = False
    cfg.image_max_n              = 4
    cfg.image_max_prompt_chars   = 20000
    cfg.image_manifest_enabled   = False
    cfg.image_cost_reporting     = False
    cfg.image_edit_enabled       = True
    cfg.image_edit_max_images    = 4
    cfg.image_edit_max_bytes     = 20*1024*1024
    cfg.image_edit_default_size  = "auto"
    # Off by default in the tests too: the cases that need it turn it on deliberately.
    cfg.image_edit_allow_prompt_paths = False
    cfg.image_edit_roots              = []


def test_image_extraction() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing image block extraction... ", end="")

    passed = True
    for case in IMAGE_EXTRACTION_CASES:
        make_image_config()
        extraction = image_orchestrator.extract({"messages": case["messages"]})

        received = {
            "prompts"    : [req.prompt for req in extraction.requests],
            "needs_text" : extraction.needs_text,
            "errors"     : len(extraction.errors),
            "cleaned"    : [msg["content"] for msg in extraction.messages],
        }
        expected = {
            "prompts"    : case["prompts"],
            "needs_text" : case["needs_text"],
            "errors"     : case.get("errors", 0),
            "cleaned"    : case["cleaned"],
        }
        passed &= check_case_equal(case["name"], expected, received)

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


IMAGE_VALIDATION_CASES = [
    ({"prompt": "ok"}                                   , None),
    ({}                                                 , "prompt is required"),
    ({"prompt": "   "}                                  , "prompt is required"),
    ({"prompt": "x", "size": "1024x1536"}               , None),
    ({"prompt": "x", "size": "auto"}                    , None),
    ({"prompt": "x", "size": "1000x1000"}               , "multiples of 16"),
    ({"prompt": "x", "size": "4096x1024"}               , "at most 3840px"),
    ({"prompt": "x", "size": "2048x512"}                , "aspect ratio"),
    ({"prompt": "x", "size": "512x512"}                 , "pixels"),
    ({"prompt": "x", "size": "wide"}                    , "WIDTHxHEIGHT"),
    ({"prompt": "x", "quality": "ultra"}                , "quality must be one of"),
    # gpt-image-2 has no transparent background; offering the value would only produce
    # a provider-side rejection later.
    ({"prompt": "x", "background": "transparent"}       , "background must be one of"),
    ({"prompt": "x", "output_format": "gif"}            , "output_format must be one of"),
    ({"prompt": "x", "n": 0}                            , "n must be between"),
    ({"prompt": "x", "n": 5}                            , "n must be between"),
    ({"prompt": "x", "n": "two"}                        , "n must be an integer"),
    ({"prompt": "x", "model": "dall-e-3", "user": "u"}  , None),
    ({"prompt": "x", "stream": True}                    , "streaming image generation is not supported"),
    # Everything the caller must never choose. 'model' is accepted above only because
    # OpenAI clients always send it; it is not an override.
    ({"prompt": "x", "qualtiy": "low"}                  , "unsupported field"),
    ({"prompt": "x", "provider": "gpt"}                 , "unsupported field"),
    ({"prompt": "x", "base_url": "http://evil.test"}    , "unsupported field"),
    ({"prompt": "x", "api_key": "sk-test"}              , "unsupported field"),
    ({"prompt": "x", "output_dir": "/etc"}              , "unsupported field"),
    # A filename is a name, never a path.
    ({"prompt": "x", "filename": "portrait"}            , None),
    ({"prompt": "x", "filename": "../../etc/passwd"}    , "no path separators"),
    ({"prompt": "x", "filename": "/tmp/evil"}           , "no path separators"),
    ({"prompt": "x", "filename": "a/b"}                 , "no path separators"),
    ({"prompt": "x", "filename": "..\\..\\win"}         , "no path separators"),
    ({"prompt": "x", "filename": ".hidden"}             , "no path separators"),
    ({"prompt": "x", "filename": ".."}                  , "no path separators"),
    # An output directory gets copied and read on the other platform, so a name is judged by
    # what both will accept rather than by what the machine writing it happens to allow.
    ({"prompt": "x", "filename": "nul"}                 , "reserved device name"),
    ({"prompt": "x", "filename": "CON"}                 , "reserved device name"),
    ({"prompt": "x", "filename": "com4.png"}            , "reserved device name"),
    # 'console' only starts like one, and 'nul_1' is what indexing a reserved name would
    # produce -- neither is reserved, and rejecting them would be superstition.
    ({"prompt": "x", "filename": "console"}             , None),
    ({"prompt": "x", "filename": "nul_1"}               , None),
    # Windows strips a trailing dot silently, so the file that appears is not the one asked
    # for. A single one is already the extension by the time it is checked, so this is 'a..'.
    ({"prompt": "x", "filename": "trailing.."}          , "must not end in a dot"),
    ({"prompt": "x", "filename": "trailing."}           , None),
]


def test_image_request_validation() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing image request validation... ", end="")

    passed = True
    for overrides, expected_error in IMAGE_VALIDATION_CASES:
        make_image_config()

        received_error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                v1_images.build_request(overrides)
        except Exception as exc:
            received_error = str(exc)

        label = str(overrides)[:58]
        if expected_error is None:
            if received_error is not None:
                print(f"[{label}] exp=accepted, rec={received_error!r} ", end="")
                passed = False
        elif received_error is None or expected_error not in received_error:
            print(f"[{label}] exp contains {expected_error!r}, rec={received_error!r} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_storage_confinement() -> bool:
    """
    Storage is the last line able to catch a path escaping the output directory, and it
    is what guarantees an existing image is never overwritten.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image storage: collisions index, paths stay confined... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        request = v1_images.build_request({"prompt": "x", "filename": "shot"})

        saved = [v1_images.save_image_bytes(request, b"\x89PNG-one", "img_aaaa"),
                 v1_images.save_image_bytes(request, b"\x89PNG-two", "img_bbbb")]

        names = sorted(os.path.basename(image.path) for image in saved)
        if names != ["shot.png", "shot_1.png"]:
            print(f"exp=['shot.png', 'shot_1.png'], rec={names} ", end="")
            passed = False

        # The first file must still hold its own bytes; an index that overwrote it
        # would leave both names pointing at the second image.
        with open(os.path.join(tmp, "shot.png"), "rb") as handle:
            if handle.read() != b"\x89PNG-one":
                print("exp=first image intact, rec=overwritten ", end="")
                passed = False

        # A name differing only in case is taken. Windows would refuse it outright and Linux
        # would allow it, and a directory meant to be copied between them cannot behave two
        # ways -- so it indexes here on both.
        cased = v1_images.save_image_bytes(v1_images.build_request({"prompt": "x", "filename": "SHOT"}),
                                           b"\x89PNG-three", "img_cccc")
        if os.path.basename(cased.path) != "SHOT_2.png":
            print(f"exp=SHOT_2.png, rec={os.path.basename(cased.path)} ", end="")
            passed = False

        # No .part file may survive a completed write.
        leftovers = [name for name in os.listdir(tmp) if name.endswith(".part")]
        if leftovers:
            print(f"exp=no temp files, rec={leftovers} ", end="")
            passed = False

        # A stem that somehow bypassed sanitisation still cannot escape.
        try:
            v1_images.allocate_path(tmp, "../escaped", ".png")
            print("exp=escape rejected, rec=allowed ", end="")
            passed = False
        except v1_images.ImageRequestError:
            pass

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


IMAGE_USAGE_CASES = [
    (
        "exact counts, split by modality",
        {"input_tokens": 15, "input_tokens_details": {"text_tokens": 15, "image_tokens": 0},
         "output_tokens": 196, "total_tokens": 211},
        {"text_input": 15, "image_input": 0, "output": 196, "reported": True},
        # 15 text @ $5/MTok + 196 image output @ $30/MTok
        0.005955,
    ),
    (
        "an undivided input counts as text, which is what text-to-image sends",
        {"input_tokens": 20, "output_tokens": 4160},
        {"text_input": 20, "image_input": 0, "output": 4160, "reported": True},
        0.1249,
    ),
    (
        "reference images are billed at the image input rate",
        {"input_tokens": 530, "input_tokens_details": {"text_tokens": 30, "image_tokens": 500},
         "output_tokens": 1056},
        {"text_input": 30, "image_input": 500, "output": 1056, "reported": True},
        # 30 @ $5 + 500 @ $8 + 1056 @ $30, per MTok
        0.035830,
    ),
    (
        "no usage object at all",
        None,
        {"text_input": 0, "image_input": 0, "output": 0, "reported": False},
        0.0,
    ),
]


def test_image_usage_accounting() -> bool:
    global tests_ttl
    tests_ttl += 1
    print("Testing image usage accounting... ", end="")

    passed = True
    for name, payload, expected_counts, expected_cost in IMAGE_USAGE_CASES:
        make_image_config()
        common.cfg.image_text_input_cost  = 5.00
        common.cfg.image_image_input_cost = 8.00
        common.cfg.image_output_cost      = 30.00

        counts = v1_images.parse_usage(payload)
        passed &= check_case_equal(f"{name} counts", expected_counts, counts)

        cost = common.track_image_usage(counts, images=1, batch=False, model="gpt-image-2")
        if round(cost, 6) != round(expected_cost, 6):
            print(f"[{name} cost] exp={expected_cost!r}, rec={round(cost, 6)!r}")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_batch_accounting() -> bool:
    """
    Batch spending is billed at its own rate and kept in its own bucket, so the session
    report can tell immediate from batch spending rather than blending them.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image batch accounting: own rate, own bucket... ", end="")

    make_image_config()
    cfg = common.cfg
    cfg.image_text_input_cost  = 5.00
    cfg.image_image_input_cost = 8.00
    cfg.image_output_cost      = 30.00
    cfg.image_batch_multiplier = 0.5

    counts = {"text_input": 15, "image_input": 0, "output": 196, "reported": True}

    before    = common.image_cost_snapshot()
    immediate = common.track_image_usage(dict(counts), images=1, batch=False, model="gpt-image-2")
    batched   = common.track_image_usage(dict(counts), images=1, batch=True , model="gpt-image-2")
    after     = common.image_cost_snapshot()

    passed = True
    if round(batched, 6) != round(immediate*0.5, 6):
        print(f"exp=half of {immediate!r}, rec={batched!r} ", end="")
        passed = False

    expected = {
        "immediate" : round(before["immediate_cost_usd"] + immediate, 6),
        "batch"     : round(before["batch_cost_usd"] + batched, 6),
        "images"    : before["images"] + 1,
        "batch_imgs": before["batch_images"] + 1,
    }
    received = {
        "immediate" : round(after["immediate_cost_usd"], 6),
        "batch"     : round(after["batch_cost_usd"], 6),
        "images"    : after["images"],
        "batch_imgs": after["batch_images"],
    }
    if not check_equal(expected, received):
        passed = False

    cfg.image_batch_multiplier = 1.0
    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


# Image editing
def make_png(width: int, height: int, alpha: bool = False) -> bytes:
    """
    A minimal valid PNG. Built by hand rather than with an imaging library, because the
    proxy deliberately has no such dependency -- it reads headers, so the tests write them.
    Colour type 6 is RGBA, 2 is RGB; that byte is what the mask check reads.
    """
    channels    = 4 if alpha else 2 + 1  # RGBA = 4 bytes, RGB = 3
    colour_type = 6 if alpha else 2
    pixel       = bytes([200, 100, 50, 255][:channels])
    raw         = b"".join(b"\x00" + pixel*width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1))
            + chunk(b"IEND", b""))


JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00"*64
WEBP_BYTES = b"RIFF" + struct.pack("<I", 64) + b"WEBP" + b"\x00"*56


def write_file(directory: str, name: str, payload: bytes) -> str:
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def test_image_reference_validation() -> bool:
    """
    A reference image is identified by its content, never its name, and anything the
    proxy cannot use is rejected outright rather than dropped from the list.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image references: content decides, bad input is rejected... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        good_png = write_file(tmp, "good.png" , make_png(64, 32))
        jpeg     = write_file(tmp, "photo.jpg", JPEG_BYTES)
        webp     = write_file(tmp, "art.webp" , WEBP_BYTES)
        # The case that matters: a text file wearing an image extension.
        liar     = write_file(tmp, "liar.png" , b"#!/bin/sh\nrm -rf /\n")
        empty    = write_file(tmp, "empty.png", b"")

        for path, expect_format in ((good_png, "png"), (jpeg, "jpeg"), (webp, "webp")):
            try:
                reference = v1_images.load_reference(path, from_prompt=False)
                if reference.format != expect_format:
                    print(f"exp={expect_format}, rec={reference.format} ", end="")
                    passed = False
            except Exception as exc:
                print(f"exp={expect_format} accepted, rec={exc} ", end="")
                passed = False

        # Geometry and alpha are read from the PNG header.
        png_ref = v1_images.load_reference(good_png, from_prompt=False)
        if (png_ref.width, png_ref.height, png_ref.has_alpha) != (64, 32, False):
            print(f"exp=(64, 32, False), rec={(png_ref.width, png_ref.height, png_ref.has_alpha)} ", end="")
            passed = False
        rgba_ref = v1_images.load_reference(write_file(tmp, "a.png", make_png(8, 8, alpha=True)), from_prompt=False)
        if not rgba_ref.has_alpha:
            print("exp=alpha detected, rec=not detected ", end="")
            passed = False

        rejections = [
            (liar                              , "not a PNG, JPEG or WebP"),
            (empty                             , "is empty"),
            (os.path.join(tmp, "nope.png")     , "does not exist"),
            (tmp                               , "not a regular file"),
        ]
        for path, expected in rejections:
            try:
                v1_images.load_reference(path, from_prompt=False)
                print(f"exp={expected!r} rejected, rec=accepted ", end="")
                passed = False
            except v1_images.ImageRequestError as exc:
                if expected not in str(exc):
                    print(f"exp contains {expected!r}, rec={exc} ", end="")
                    passed = False

        # The byte cap is enforced before anything is uploaded.
        common.cfg.image_edit_max_bytes = 100
        try:
            v1_images.load_reference(good_png, from_prompt=False)
            print("exp=size cap enforced, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError as exc:
            if "the limit is" not in str(exc):
                print(f"exp=size-limit message, rec={exc} ", end="")
                passed = False
        common.cfg.image_edit_max_bytes = 20*1024*1024

        # And the count cap.
        common.cfg.image_edit_max_images = 2
        try:
            v1_images.load_references([good_png, jpeg, webp], from_prompt=False)
            print("exp=count cap enforced, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError:
            pass
        common.cfg.image_edit_max_images = 4

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_edit_path_security() -> bool:
    """
    The load-bearing test. A path written in a chat block is a file-read primitive aimed
    at whatever the proxy can reach, so it must be refused unless explicitly enabled and
    inside an allowed root -- and a symlink must not be able to walk out of one.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image edit paths: prompts cannot read outside the allowlist... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        allowed = os.path.join(tmp, "allowed"); os.makedirs(allowed)
        secret  = os.path.join(tmp, "secret") ; os.makedirs(secret)
        inside  = write_file(allowed, "ok.png"    , make_png(16, 16))
        outside = write_file(secret , "secret.png", make_png(16, 16))

        # A symlink planted inside the allowed root, aimed out of it.
        escape = os.path.join(allowed, "escape.png")
        try:
            os.symlink(outside, escape)
            symlinks = True
        except (OSError, NotImplementedError):
            symlinks = False

        common.cfg.image_edit_roots = [os.path.realpath(allowed)]

        # Disabled: even a path inside a root is refused.
        common.cfg.image_edit_allow_prompt_paths = False
        try:
            v1_images.load_reference(inside, from_prompt=True)
            print("exp=prompt paths refused when disabled, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError as exc:
            if "disabled" not in str(exc):
                print(f"exp=disabled message, rec={exc} ", end="")
                passed = False

        # Enabled: inside the root works, outside does not.
        common.cfg.image_edit_allow_prompt_paths = True
        try:
            v1_images.load_reference(inside, from_prompt=True)
        except Exception as exc:
            print(f"exp=allowed path accepted, rec={exc} ", end="")
            passed = False

        for path, label in ((outside, "outside root"), ("/etc/hostname", "absolute system path")):
            try:
                v1_images.load_reference(path, from_prompt=True)
                print(f"exp={label} refused, rec=accepted ", end="")
                passed = False
            except v1_images.ImageRequestError as exc:
                if "IMAGE_EDIT_ROOTS" not in str(exc):
                    print(f"exp=root message for {label}, rec={exc} ", end="")
                    passed = False

        if symlinks:
            try:
                v1_images.load_reference(escape, from_prompt=True)
                print("exp=symlink escape refused, rec=accepted ", end="")
                passed = False
            except v1_images.ImageRequestError:
                pass

        # An empty root list means nothing is readable, whatever the switch says.
        common.cfg.image_edit_roots = []
        try:
            v1_images.load_reference(inside, from_prompt=True)
            print("exp=empty roots refuse everything, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError:
            pass

        # The console is trusted with paths regardless of any of the above.
        common.cfg.image_edit_allow_prompt_paths = False
        try:
            v1_images.load_reference(outside, from_prompt=False)
        except Exception as exc:
            print(f"exp=console path accepted, rec={exc} ", end="")
            passed = False

    common.cfg.image_edit_roots = []
    common.cfg.image_edit_allow_prompt_paths = False
    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_edit_detection() -> bool:
    """The truth table deciding whether a request edits, and against what."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image edit detection: images, edit:true and slots... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        one = write_file(tmp, "one.png", make_png(32, 32))
        two = write_file(tmp, "two.png", make_png(32, 32))

        def check(name: str, fields: dict, expect_images: int, expect_error: str = "") -> None:
            nonlocal passed
            try:
                request = v1_images.build_request({"prompt": "x", **fields}, source="cli")
            except Exception as exc:
                if not expect_error or expect_error not in str(exc):
                    print(f"[{name}] exp={expect_error or 'accepted'}, rec={exc} ", end="")
                    passed = False
                return
            if expect_error:
                print(f"[{name}] exp={expect_error!r} rejected, rec=accepted ", end="")
                passed = False
            elif len(request.images) != expect_images or request.is_edit != bool(expect_images):
                print(f"[{name}] exp={expect_images} refs, rec={len(request.images)} ", end="")
                passed = False

        # No slots filled yet.
        check("plain generation"      , {}                               , 0)
        check("explicit images"       , {"images": [one, two]}           , 2)
        check("edit true, no slots"   , {"edit": True}                   , 0, "no image slots are filled")
        check("edit false + images"   , {"edit": False, "images": [one]} , 0, "edit: false was given together")
        check("mask with edit false"  , {"edit": False, "mask": one}     , 0, "only applies to an edit")

        v1_images.set_slot(1, one)
        v1_images.set_slot(2, two)

        check("edit true uses slots"  , {"edit": True}                   , 2)
        check("slot numbers"          , {"images": [1]}                  , 1)
        check("edit true + images"    , {"edit": True, "images": [two]}  , 1)
        check("still plain generation", {}                               , 0)

        # An empty images list must not silently become an edit.
        check("empty images list"     , {"images": []}                   , 0)

        # Slot numbers survive a round trip through json5-style strings.
        check("numeric string slot"   , {"images": ["2"]}                , 1)
        check("missing slot"          , {"images": [7]}                  , 0, "slot 7 is empty")

        # A slot whose file disappeared is reported, not silently skipped.
        os.unlink(two)
        check("slot file vanished"    , {"edit": True}                   , 0, "does not exist")

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_edit_batch_rules() -> bool:
    """file_ids and local images are different worlds; mixing them is always an error."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image edit batching: file_ids vs local images... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        one  = write_file(tmp, "one.png", make_png(32, 32))
        flat = write_file(tmp, "flat.png", make_png(32, 32))
        alpha_mask = write_file(tmp, "alpha.png", make_png(32, 32, alpha=True))

        cases = [
            ({"file_ids": ["file-abc"], "batch": True}                   , ""),
            ({"file_ids": ["https://example.test/a.png"], "batch": True} , ""),
            ({"file_ids": ["file-abc"]}                                  , "Batch API edits only"),
            ({"images": [one], "batch": True}                            , "cannot be batched"),
            ({"images": [one], "file_ids": ["file-abc"]}                 , "mutually exclusive"),
            ({"file_ids": ["/local/path.png"], "batch": True}            , "must be a provider file id"),
            # A mask rides along in the JSON body, so a batch takes one -- held to the same
            # alpha rule as anywhere else, since that is what marks the editable region.
            ({"file_ids": ["file-abc"], "mask": alpha_mask, "batch": True}, ""),
            ({"file_ids": ["file-abc"], "mask": flat, "batch": True}     , "no alpha channel"),
        ]
        for fields, expected in cases:
            try:
                request = v1_images.build_request({"prompt": "x", **fields}, source="cli")
            except Exception as exc:
                if not expected or expected not in str(exc):
                    print(f"exp={expected or 'accepted'}, rec={exc} ", end="")
                    passed = False
                continue
            if expected:
                print(f"exp={expected!r} rejected, rec=accepted ", end="")
                passed = False
            elif not request.is_edit:
                print("exp=edit request, rec=generation ", end="")
                passed = False

        # A batched edit carries its references in the body, since multipart is unavailable.
        # The shape below was verified against the provider: 'images', always an array,
        # always objects. A bare id string inside the array is rejected with "expected an
        # object", a bare string instead of the array with "expected an array of objects",
        # and the 'input_reference' field the batch guide documents (which belongs to
        # image-guided generation) with "Missing required parameter: 'images'".
        request = v1_images.build_request({"prompt": "x", "file_ids": ["file-abc"], "batch": True}, source="cli")
        body    = v1_images.build_body(request)
        if body.get("images") != [{"file_id": "file-abc"}]:
            print(f"exp=[{{'file_id': 'file-abc'}}], rec={body.get('images')} ", end="")
            passed = False
        if "input_reference" in body:
            print("exp=no input_reference field, rec=sent ", end="")
            passed = False

        # The mask takes the same reference shape as an entry of 'images', carrying its
        # bytes inline because a batch has no upload to make. Verified against the
        # provider: a batch line carrying one passes validation and runs.
        request = v1_images.build_request(
            {"prompt": "x", "file_ids": ["file-abc"], "mask": alpha_mask, "batch": True}, source="cli")
        body    = v1_images.build_body(request)
        mask_ref = body.get("mask")
        if not isinstance(mask_ref, dict) or set(mask_ref) != {"image_url"}:
            print(f"exp={{'image_url': ...}}, rec={mask_ref} ", end="")
            passed = False
        elif not mask_ref["image_url"].startswith("data:image/png;base64,"):
            print(f"exp=png data url, rec={mask_ref['image_url'][:32]!r} ", end="")
            passed = False
        elif base64.b64decode(mask_ref["image_url"].split(",", 1)[1]) != Path(alpha_mask).read_bytes():
            print("exp=the mask's own bytes, rec=something else ", end="")
            passed = False

        # A mask is only ever sent with an edit, so a plain generation must not grow one.
        if "mask" in v1_images.build_body(v1_images.build_request({"prompt": "x"}, source="cli")):
            print("exp=no mask on a generation, rec=sent ", end="")
            passed = False

        request = v1_images.build_request({"prompt": "x", "file_ids": ["file-a", "https://e.test/b.png"], "batch": True}, source="cli")
        body    = v1_images.build_body(request)
        if body.get("images") != [{"file_id": "file-a"}, {"image_url": "https://e.test/b.png"}]:
            print(f"exp=file_id + image_url objects, rec={body.get('images')} ", end="")
            passed = False

        # A batch cannot mix the two endpoints.
        edit_req = v1_images.build_request({"prompt": "x", "file_ids": ["file-a"], "batch": True}, source="cli")
        gen_req  = v1_images.build_request({"prompt": "x", "batch": True}, source="cli")
        try:
            v1_images.submit_image_batch([edit_req, gen_req])
            print("exp=mixed batch rejected, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError as exc:
            if "cannot mix" not in str(exc):
                print(f"exp=mix message, rec={exc} ", end="")
                passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_edit_form() -> bool:
    """The multipart form: repeated image[] fields, scalars as strings, handles closed."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image edit form: multipart assembly and file handles... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        one  = write_file(tmp, "one.png" , make_png(32, 32))
        two  = write_file(tmp, "two.png" , make_png(32, 32))
        mask = write_file(tmp, "mask.png", make_png(32, 32, alpha=True))

        request = v1_images.build_request(
            {"prompt": "make it blue", "images": [one, two], "mask": mask, "quality": "high", "size": "1024x1024"},
            source="cli",
        )
        data, files = v1_images.build_edit_form(request)

        expected_data = {
            "model": "gpt-image-2", "prompt": "make it blue", "n": "1",
            "output_format": "png", "background": "opaque",
            "size": "1024x1024", "quality": "high",
        }
        if not check_equal(expected_data, data):
            passed = False
        # Every multipart value must be a string; a raw int makes httpx reject the form.
        if any(not isinstance(value, str) for value in data.values()):
            print("exp=all form values are strings, rec=non-string present ", end="")
            passed = False
        # gpt-image-2 rejects input_fidelity outright, so it must never be sent.
        if "input_fidelity" in data:
            print("exp=no input_fidelity, rec=sent ", end="")
            passed = False

        field_names = [name for name, _ in files]
        if field_names != ["image[]", "image[]", "mask"]:
            print(f"exp=['image[]', 'image[]', 'mask'], rec={field_names} ", end="")
            passed = False

        v1_images.close_form_files(files)
        if any(not spec[1].closed for _, spec in files):
            print("exp=handles closed, rec=left open ", end="")
            passed = False

        # 'auto' is omitted rather than sent as a literal.
        auto_req = v1_images.build_request({"prompt": "x", "images": [one], "size": "auto", "quality": "auto"}, source="cli")
        auto_data, auto_files = v1_images.build_edit_form(auto_req)
        v1_images.close_form_files(auto_files)
        if "size" in auto_data or "quality" in auto_data:
            print(f"exp=auto omitted, rec={auto_data} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_mask_validation() -> bool:
    """A mask has to be a PNG with alpha, matching the first reference's geometry."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image mask: PNG, alpha, matching geometry... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        base      = write_file(tmp, "base.png"  , make_png(64, 64))
        good_mask = write_file(tmp, "good.png"  , make_png(64, 64, alpha=True))
        no_alpha  = write_file(tmp, "flat.png"  , make_png(64, 64))
        wrong_dim = write_file(tmp, "small.png" , make_png(32, 32, alpha=True))
        jpeg_mask = write_file(tmp, "mask.jpg"  , JPEG_BYTES)

        cases = [
            (good_mask, ""),
            (no_alpha , "no alpha channel"),
            (wrong_dim, "they must match"),
            (jpeg_mask, "must be a PNG"),
        ]
        for path, expected in cases:
            try:
                v1_images.build_request({"prompt": "x", "images": [base], "mask": path}, source="cli")
            except Exception as exc:
                if not expected or expected not in str(exc):
                    print(f"exp={expected or 'accepted'}, rec={exc} ", end="")
                    passed = False
                continue
            if expected:
                print(f"exp={expected!r} rejected, rec=accepted ", end="")
                passed = False

        # A mask set from the console applies to a slot-driven edit without being named.
        v1_images.set_slot(1, base)
        with v1_images.SLOT_LOCK:
            slots = v1_images.read_slots()
            slots[v1_images.MASK_KEY] = good_mask
            v1_images.write_slots(slots)
        request = v1_images.build_request({"prompt": "x", "edit": True}, source="cli")
        if request.mask is None:
            print("exp=stored mask applied, rec=ignored ", end="")
            passed = False
        # The mask key must not be mistaken for a reference slot.
        if len(request.images) != 1:
            print(f"exp=1 reference (mask is not a slot), rec={len(request.images)} ", end="")
            passed = False

        # ...but a caller that never asked for it must be able to say so, or a mask left
        # set from the console would silently shape every request that followed.
        for refusal in (False, "none", "off", "false"):
            refused = v1_images.build_request({"prompt": "x", "edit": True, "mask": refusal}, source="cli")
            if refused.mask is not None:
                print(f"exp=mask:{refusal!r} refuses the stored mask, rec=applied ", end="")
                passed = False

        # Leaving the key out is not the same as refusing: it still inherits.
        if v1_images.build_request({"prompt": "x", "edit": True}, source="cli").mask is None:
            print("exp=an omitted key still inherits, rec=refused ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_manifest_patch() -> bool:
    """The one writable path in: testimony can be corrected, measurement cannot."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image manifest patch: corrections in, measurement out... ", end="")

    make_image_config()
    passed = True

    def fail(what: str) -> None:
        nonlocal passed
        print(f"{what} ", end="")
        passed = False

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir       = tmp
        common.cfg.image_manifest_enabled = True
        base   = write_file(tmp, "base.png", make_png(32, 32))
        mask   = write_file(tmp, "alpha.png", make_png(32, 32, alpha=True))
        region = {"size": {"width": 32, "height": 32}, "rects": [{"x": 1, "y": 2, "w": 3, "h": 4, "mode": "add"}]}

        request = v1_images.build_request(
            {"prompt": "a red coat", "filename": "redcoat", "images": [base], "mask": mask,
             "mask_region": region, "size": "1024x1024"}, source="cli")
        saved = [v1_images.save_image_bytes(request, b"\x89PNG-one", "img_aaaa")]
        v1_images.append_manifest(request, saved, 0.01, {"input_tokens": 7}, True)

        def record() -> dict:
            with open(os.path.join(tmp, v1_images.MANIFEST_FILE), "r", encoding="utf-8") as handle:
                return json.load(handle)[0]

        # Everything a client can know better than the proxy, in one pass.
        moved = {"size": {"width": 32, "height": 32}, "rects": [{"x": 0, "y": 0, "w": 8, "h": 8, "mode": "subtract"}]}
        result = v1_images.patch_manifest([{
            "image_id"          : "img_aaaa",
            "prompt"            : "a red coat, wet cobbles",
            "request_parameters": {"size": "1024x1536", "quality": "high"},
            "mask"              : moved,
            "provider"          : "openai",
            "estimated_cost_usd": 0.25,
            "cost_is_estimate"  : False,
            "job"               : {"id": "job-1", "comment": "warmer"},
            "job_group"         : {"id": "g-1", "name": "coats"},
        }])
        if result["changed"] != 1 or result["missing"]:
            fail(f"exp=one record changed, rec={result}")

        after = record()
        if after.get("prompt") != "a red coat, wet cobbles":
            fail(f"exp=the corrected prompt, rec={after.get('prompt')!r}")
        # Merged, not replaced: a parameter the correction did not mention survives it.
        if after.get("request_parameters", {}).get("size") != "1024x1536":
            fail(f"exp=the corrected size, rec={after.get('request_parameters')}")
        if after.get("request_parameters", {}).get("output_format") != "png":
            fail(f"exp=output_format left alone, rec={after.get('request_parameters')}")
        # The region is corrected; the picture the proxy rasterised is kept, since it cannot
        # render it again from rectangles it was handed afterwards.
        if after.get("mask", {}).get("rects") != moved["rects"]:
            fail(f"exp=the corrected rectangles, rec={after.get('mask')}")
        if after.get("mask", {}).get("file") != "alpha.png":
            fail(f"exp=the rasterised mask kept, rec={after.get('mask')}")
        if after.get("cost_is_estimate") is not False or after.get("estimated_cost_usd") != 0.25:
            fail(f"exp=the corrected cost, rec={after.get('estimated_cost_usd')}")
        if after.get("job", {}).get("comment") != "warmer":
            fail(f"exp=the job stamp, rec={after.get('job')}")
        # Measurement is untouched by a patch that never mentioned it.
        if after.get("file") != "redcoat.png" or after.get("usage", {}).get("input_tokens") != 7:
            fail(f"exp=file and usage left alone, rec={after.get('file')} {after.get('usage')}")

        # Nothing to do is not a write: the manifest must not be rewritten on every poll.
        if v1_images.patch_manifest([{"image_id": "img_aaaa", "prompt": "a red coat, wet cobbles"}])["changed"] != 0:
            fail("exp=an identical patch changes nothing, rec=changed")

        # Null clears; absent leaves alone.
        v1_images.patch_manifest([{"image_id": "img_aaaa", "job": None, "mask": None}])
        cleared = record()
        if "job" in cleared or "mask" in cleared:
            fail(f"exp=job and mask cleared, rec={sorted(cleared)}")
        if cleared.get("prompt") != "a red coat, wet cobbles":
            fail("exp=a key left out is left alone, rec=lost")

        # A record nobody has is reported rather than invented.
        if v1_images.patch_manifest([{"file": "nothing.png", "prompt": "x"}])["missing"] != ["nothing.png"]:
            fail("exp=an unknown record reported missing, rec=otherwise")

        # What the file *is* stays this proxy's to say.
        for update, expected in [
            ({"image_id": "img_aaaa", "bytes": 12}       , "unsupported"),
            ({"image_id": "img_aaaa", "usage": {}}       , "unsupported"),
            ({"image_id": "img_aaaa", "path": "e.png"}   , "unsupported"),
            ({"image_id": "img_aaaa", "operation": "wat"}, "must be 'generate' or 'edit'"),
            ({"prompt": "x"}                             , "must name a record"),
        ]:
            try:
                v1_images.patch_manifest([update])
                fail(f"exp={expected!r} rejected, rec=accepted for {sorted(update)}")
            except Exception as exc:
                if expected not in str(exc):
                    fail(f"exp={expected}, rec={exc}")

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_mask_region() -> bool:
    """The rectangles a mask was drawn as: recorded verbatim, validated, never acted on."""
    global tests_ttl
    tests_ttl += 1
    print("Testing image mask region: the rectangles behind the picture... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        base = write_file(tmp, "base.png", make_png(32, 32))
        mask = write_file(tmp, "alpha.png", make_png(32, 32, alpha=True))
        good = {"size": {"width": 32, "height": 32}, "rects": [{"x": 1, "y": 2, "w": 3, "h": 4, "mode": "add"}]}

        def check(name: str, region: Any, expect_error: str = "", with_mask: bool = True) -> None:
            nonlocal passed
            fields: Dict[str, Any] = {"prompt": "x", "images": [base], "mask_region": region}
            if with_mask:
                fields["mask"] = mask
            try:
                request = v1_images.build_request(fields, source="cli")
            except Exception as exc:
                if not expect_error or expect_error not in str(exc):
                    print(f"[{name}] exp={expect_error or 'accepted'}, rec={exc} ", end="")
                    passed = False
                return
            if expect_error:
                print(f"[{name}] exp={expect_error!r} rejected, rec=accepted ", end="")
                passed = False
            elif request.mask_region != good:
                print(f"[{name}] exp={good}, rec={request.mask_region} ", end="")
                passed = False

        check("rectangles"       , good)
        # Structure has to travel as a string through multipart, as the job stamps do.
        check("as a JSON string" , json.dumps(good))
        # Numbers as some clients spell them. Recorded as numbers either way, since a manifest
        # read back has to give the same answer whichever door the request came in by.
        check("numeric strings"  , {"size": {"width": "32", "height": "32"},
                                    "rects": [{"x": "1", "y": "2", "w": "3", "h": "4", "mode": "add"}]})

        check("not an object"    , "[]"                                        , "must be an object")
        check("no size"          , {"rects": good["rects"]}                    , "size must be an object")
        check("rects not a list" , {"size": good["size"], "rects": {}}         , "rects must be a list")
        check("unknown rect key" , {"size": good["size"], "rects": [{"x": 1, "colour": "red"}]}, "unsupported")
        check("bad mode"         , {"size": good["size"], "rects": [{"x": 1, "y": 1, "w": 1, "h": 1, "mode": "erase"}]},
              "must be 'add' or 'subtract'")
        check("unreadable number", {"size": good["size"], "rects": [{"x": "over there", "y": 1, "w": 1, "h": 1}]},
              "must be a number")
        check("too many rects"   , {"size": good["size"], "rects": [{"x": 0, "y": 0, "w": 1, "h": 1}] * 513},
              "the limit is 512")

        # A region describes a mask, so one arriving alone means the caller sent the two halves
        # down different paths -- worth reporting rather than recording.
        check("region, no mask"  , good, "without a mask", with_mask=False)

        # A region with no rectangles selects nothing, so there is nothing to record.
        empty = v1_images.build_request(
            {"prompt": "x", "images": [base], "mask": mask, "size": "auto",
             "mask_region": {"size": {"width": 32, "height": 32}, "rects": []}}, source="cli")
        if empty.mask_region != {}:
            print(f"exp=an empty region records nothing, rec={empty.mask_region} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_mask_persistence() -> bool:
    """
    A mask that only ever existed as bytes is written out beside the images it shaped, so
    the manifest names a file rather than a bare 'upload:' label with nothing behind it.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image mask persistence: bytes become a file the record can name... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir      = tmp
        common.cfg.image_manifest_enabled = True
        base       = write_file(tmp, "base.png", make_png(64, 64))
        mask_bytes = make_png(64, 64, alpha=True)
        data_url   = "data:image/png;base64," + base64.b64encode(mask_bytes).decode("ascii")

        region  = {"size": {"width": 64, "height": 64}, "rects": [{"x": 4, "y": 8, "w": 16, "h": 32, "mode": "add"}]}
        request = v1_images.build_request(
            {"prompt": "x", "filename": "redcoat", "images": [base], "mask": data_url,
             "mask_region": region, "n": 2}, source="cli")
        saved = [v1_images.save_image_bytes(request, b"\x89PNG-one", "img_aaaa"),
                 v1_images.save_image_bytes(request, b"\x89PNG-two", "img_bbbb")]
        v1_images.append_manifest(request, saved, 0.01, {}, True)

        with open(os.path.join(tmp, v1_images.MANIFEST_FILE), "r", encoding="utf-8") as handle:
            records = json.load(handle)

        masks = [record.get("mask") for record in records]
        # One mask per request, whatever n was: it belongs to the request, not to any one
        # image, so both records say the same thing. The region is the durable half -- it can be
        # read back, corrected and asked for again -- and the file is the picture it rasterised to.
        expected = [{**region, "file": "masks/redcoat.png"}] * 2
        if masks != expected:
            print(f"exp={expected}, rec={masks} ", end="")
            passed = False

        written = os.path.join(tmp, "masks", "redcoat.png")
        if not os.path.exists(written):
            print(f"exp={written} written, rec=missing ", end="")
            passed = False
        elif Path(written).read_bytes() != mask_bytes:
            print("exp=the mask's own bytes on disk, rec=something else ", end="")
            passed = False

        # A mask that already has a path is left where it is rather than copied.
        on_disk = write_file(tmp, "hand.png", make_png(64, 64, alpha=True))
        request = v1_images.build_request(
            {"prompt": "x", "filename": "second", "images": [base], "mask": on_disk}, source="cli")
        saved   = [v1_images.save_image_bytes(request, b"\x89PNG-three", "img_cccc")]
        v1_images.append_manifest(request, saved, 0.01, {}, True)

        with open(os.path.join(tmp, v1_images.MANIFEST_FILE), "r", encoding="utf-8") as handle:
            records = json.load(handle)
        # No region given, so the record says only where the picture is -- which is all a caller
        # that never drew rectangles can honestly claim.
        if records[-1].get("mask") != {"file": "hand.png"}:
            print(f"exp=hand.png named where it lies, rec={records[-1].get('mask')} ", end="")
            passed = False
        if os.path.exists(os.path.join(tmp, "masks", "second.png")):
            print("exp=an on-disk mask is not copied, rec=copied ", end="")
            passed = False

        # A generation has no mask, and must not grow a masks/ folder or a record field.
        request = v1_images.build_request({"prompt": "x", "filename": "plain"}, source="cli")
        v1_images.append_manifest(request, [v1_images.save_image_bytes(request, b"\x89PNG-four", "img_dddd")],
                                  0.01, {}, True)
        with open(os.path.join(tmp, v1_images.MANIFEST_FILE), "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if "mask" in records[-1]:
            print(f"exp=no mask field on a generation, rec={records[-1]['mask']} ", end="")
            passed = False

        leftovers = [name for name in os.listdir(os.path.join(tmp, "masks")) if name.endswith(".part")]
        if leftovers:
            print(f"exp=no temp files, rec={leftovers} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_data_url_references() -> bool:
    """
    A data URL is an upload by another route: bytes the caller already held, reaching no
    filesystem. It is what lets a JSON request carry a mask it has no file for.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image data URLs: inline bytes, no filesystem... ", end="")

    make_image_config()
    passed = True

    def data_url(payload: bytes, mediatype: str = "image/png") -> str:
        return f"data:{mediatype};base64," + base64.b64encode(payload).decode("ascii")

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        base = write_file(tmp, "base.png", make_png(64, 64))

        # A path names the reference, a data URL carries the mask: the two encodings mix,
        # which is the point -- large references stay on disk, a drawn mask rides along.
        request = v1_images.build_request(
            {"prompt": "x", "images": [base], "mask": data_url(make_png(64, 64, alpha=True))}, source="cli")
        if request.mask is None or request.mask.path or request.mask.data is None:
            print(f"exp=mask held as bytes, rec={request.mask and request.mask.path!r} ", end="")
            passed = False
        elif (request.mask.width, request.mask.height) != (64, 64):
            print(f"exp=64x64, rec={request.mask.width}x{request.mask.height} ", end="")
            passed = False

        # A reference may come the same way, so a caller holding no files at all still edits.
        inline = v1_images.build_request(
            {"prompt": "x", "images": [data_url(make_png(64, 64))]}, source="cli")
        if len(inline.images) != 1 or inline.images[0].path:
            print("exp=inline reference held as bytes, rec=otherwise ", end="")
            passed = False

        # Content decides, exactly as it does for a path or an upload: a mediatype that
        # disagrees with the bytes is advisory, and a non-image is refused whatever it claims.
        mislabelled = v1_images.build_request(
            {"prompt": "x", "images": [data_url(make_png(64, 64), "image/jpeg")]}, source="cli")
        if mislabelled.images[0].format != "png":
            print(f"exp=png by content, rec={mislabelled.images[0].format} ", end="")
            passed = False

        cases = [
            (data_url(JPEG_BYTES[:2])                      , "not a PNG, JPEG or WebP"),
            ("data:image/png," + "x"*8                     , "only base64 data URLs"),
            ("data:image/png;base64,!!!not-base64!!!"      , "not valid Base64"),
            ("data:"                                       , "must look like"),
        ]
        for spec, expected in cases:
            try:
                v1_images.build_request({"prompt": "x", "images": [base], "mask": spec}, source="cli")
            except Exception as exc:
                if expected not in str(exc):
                    print(f"exp={expected!r}, rec={exc} ", end="")
                    passed = False
                continue
            print(f"exp={expected!r} rejected, rec=accepted ", end="")
            passed = False

        # Sized before it is decoded, so an oversized payload is never materialised.
        common.cfg.image_edit_max_bytes = 1024
        try:
            v1_images.build_request(
                {"prompt": "x", "images": [base], "mask": data_url(make_png(256, 256, alpha=True))}, source="cli")
            print("exp=oversized inline mask rejected, rec=accepted ", end="")
            passed = False
        except Exception as exc:
            if "IMAGE_EDIT_MAX_BYTES" not in str(exc):
                print(f"exp=size cap cited, rec={exc} ", end="")
                passed = False

        # A prompt-supplied data URL touches no filesystem, so the path allowlist -- which
        # exists to stop a prompt reading this disk -- has nothing to say about it.
        common.cfg.image_edit_max_bytes         = 20*1024*1024
        common.cfg.image_edit_allow_prompt_paths = False
        try:
            v1_images.build_request(
                {"prompt": "x", "images": [data_url(make_png(64, 64))]}, source="chat")
        except Exception as exc:
            print(f"exp=inline reference allowed from a prompt, rec={exc} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_uploaded_references() -> bool:
    """
    Uploaded bytes are validated by the same content rules as a path, and touch the
    filesystem not at all -- which is why the path allowlist does not apply to them.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing uploaded references: same checks, no filesystem... ", end="")

    make_image_config()
    # Deliberately hostile settings for a *path*: nothing is readable from a prompt.
    common.cfg.image_edit_allow_prompt_paths = False
    common.cfg.image_edit_roots              = []
    passed = True

    good = v1_images.load_uploaded_reference("shot.png", make_png(64, 32))
    if (good.format, good.width, good.height, good.path, good.slot) != ("png", 64, 32, "", 0):
        print(f"exp=png 64x32 with no path, rec={(good.format, good.width, good.height, good.path)} ", end="")
        passed = False
    if good.origin() != "upload:shot.png":
        print(f"exp=upload:shot.png, rec={good.origin()} ", end="")
        passed = False
    # The bytes must be readable twice, since a retry rebuilds the form.
    if good.open().read() != good.open().read() or not good.open().read():
        print("exp=stream re-readable, rec=exhausted ", end="")
        passed = False

    rejections = [
        ("liar.png" , b"#!/bin/sh\nrm -rf /", "not a PNG, JPEG or WebP"),
        ("empty.png", b""                   , "is empty"),
    ]
    for name, payload, expected in rejections:
        try:
            v1_images.load_uploaded_reference(name, payload)
            print(f"exp={expected!r} rejected, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError as exc:
            if expected not in str(exc):
                print(f"exp contains {expected!r}, rec={exc} ", end="")
                passed = False

    common.cfg.image_edit_max_bytes = 100
    try:
        v1_images.load_uploaded_reference("big.png", make_png(64, 64))
        print("exp=size cap enforced on uploads, rec=accepted ", end="")
        passed = False
    except v1_images.ImageRequestError:
        pass
    common.cfg.image_edit_max_bytes = 20*1024*1024

    # An already-validated upload passes straight through build_request.
    request = v1_images.build_request({"prompt": "x", "images": [good]}, source="direct")
    if not request.is_edit or request.images[0] is not good:
        print("exp=upload passed through, rec=re-resolved ", end="")
        passed = False

    # A mask may be uploaded too, and is held to the same alpha rule.
    alpha = v1_images.load_uploaded_reference("m.png", make_png(64, 32, alpha=True))
    flat  = v1_images.load_uploaded_reference("f.png", make_png(64, 32))
    try:
        v1_images.build_request({"prompt": "x", "images": [good], "mask": alpha}, source="direct")
    except Exception as exc:
        print(f"exp=uploaded mask accepted, rec={exc} ", end="")
        passed = False
    try:
        v1_images.build_request({"prompt": "x", "images": [good], "mask": flat}, source="direct")
        print("exp=alpha-less uploaded mask rejected, rec=accepted ", end="")
        passed = False
    except v1_images.ImageRequestError:
        pass

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_http_routes() -> bool:
    """
    The HTTP surface: generations and edits are split the way the upstream API splits
    them, and the reply carries what an OpenAI client expects.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image routes: split surface and response formats... ", end="")

    make_config()
    make_image_config()
    common.cfg.image_response_format = "b64_json"
    passed = True

    saved = v1_images.SavedImage(image_id="img_dead", path="generated_images/x.png", data=b"\x89PNG-bytes")
    fake  = v1_images.ImageResult(provider="testimg", model="gpt-image-2", created=CREATED,
                                  images=[saved], cost_usd=0.01, usage={"reported": False}, estimated=False)

    original = v1_images.generate_image
    v1_images.generate_image = lambda request: fake
    client = server.app.test_client()
    headers = {"Authorization": "Bearer test-proxy-key"}
    common.cfg.require_proxy_key = False
    try:
        # Editing fields on the generations route are refused, with a pointer.
        r = client.post("/v1/images/generations", json={"prompt": "x", "images": [1]}, headers=headers)
        if r.status_code != 400 or "/v1/images/edits" not in r.get_json()["error"]["message"]:
            print(f"exp=400 pointing at edits, rec={r.status_code} {r.get_json()} ", end="")
            passed = False

        # The edits route needs references.
        r = client.post("/v1/images/edits", json={"prompt": "x"}, headers=headers)
        if r.status_code != 400 or "reference images" not in r.get_json()["error"]["message"]:
            print(f"exp=400 needing references, rec={r.status_code} ", end="")
            passed = False

        # b64_json by default, since every caller here is an external client.
        r = client.post("/v1/images/generations", json={"prompt": "x"}, headers=headers)
        entry = r.get_json()["data"][0]
        if r.status_code != 200 or "b64_json" not in entry or entry.get("path") != "generations/x.png".replace("generations", "generated_images"):
            print(f"exp=b64_json + path, rec={r.status_code} {sorted(entry)} ", end="")
            passed = False
        if entry["b64_json"] != base64.b64encode(b"\x89PNG-bytes").decode():
            print("exp=encoded bytes, rec=wrong payload ", end="")
            passed = False

        # ...and metadata only when asked.
        r = client.post("/v1/images/generations", json={"prompt": "x", "response_format": "path"}, headers=headers)
        if sorted(r.get_json()["data"][0]) != ["id", "path"]:
            print(f"exp=['id', 'path'], rec={sorted(r.get_json()['data'][0])} ", end="")
            passed = False

        with contextlib.redirect_stdout(io.StringIO()):
            r = client.post("/v1/images/generations", json={"prompt": "x", "response_format": "jpeg"}, headers=headers)
        if r.status_code != 400:
            print(f"exp=400 for a bad response_format, rec={r.status_code} ", end="")
            passed = False

        # A multipart upload reaches the edits route without any path being involved.
        r = client.post(
            "/v1/images/edits",
            data={"prompt": "x", "image": (io.BytesIO(make_png(64, 64)), "a.png")},
            content_type="multipart/form-data",
            headers=headers,
        )
        if r.status_code != 200 or "b64_json" not in r.get_json()["data"][0]:
            print(f"exp=200 with b64_json, rec={r.status_code} {r.get_json()} ", end="")
            passed = False

        # A multipart edit with no file at all is a clear error, not a crash.
        with contextlib.redirect_stdout(io.StringIO()):
            r = client.post("/v1/images/edits", data={"prompt": "x"}, content_type="multipart/form-data", headers=headers)
        if r.status_code != 400:
            print(f"exp=400 for a fileless multipart edit, rec={r.status_code} ", end="")
            passed = False
    finally:
        v1_images.generate_image = original

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_batch_numbering() -> bool:
    """
    Batches are referenced by a short number, not the provider's hash: numbers are
    assigned once, survive a restart, are never reused, and resolve back to the id.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image batch numbering: stable, unique, resolvable... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp

        # Reading state must not be what creates the output directory.
        nested = os.path.join(tmp, "nested")
        common.cfg.image_output_dir = nested
        v1_images.read_batch_state()
        if os.path.exists(nested):
            print("exp=reading state creates nothing, rec=directory created ", end="")
            passed = False
        common.cfg.image_output_dir = tmp

        state = {}
        for index, batch_id in enumerate(["batch_aaa", "batch_bbb", "batch_ccc"], start=1):
            number = v1_images.next_batch_number(state)
            if number != index:
                print(f"exp=number {index}, rec={number} ", end="")
                passed = False
            state[batch_id] = {"number": number, "retrieved": False, "model": "gpt-image-2"}
        v1_images.write_batch_state(state)

        # A number resolves to its id, and a raw id is passed through untouched.
        if v1_images.resolve_batch_id("2") != "batch_bbb":
            print(f"exp=batch_bbb, rec={v1_images.resolve_batch_id('2')!r} ", end="")
            passed = False
        if v1_images.resolve_batch_id("batch_ccc") != "batch_ccc":
            print("exp=raw id passed through, rec=rewritten ", end="")
            passed = False
        try:
            v1_images.resolve_batch_id("99")
            print("exp=unknown number rejected, rec=accepted ", end="")
            passed = False
        except v1_images.ImageRequestError:
            pass

        # Settling one batch removes it from the pending set but must not renumber the rest.
        state["batch_bbb"]["retrieved"] = True
        v1_images.write_batch_state(state)
        if v1_images.pending_batches() != [(1, "batch_aaa"), (3, "batch_ccc")]:
            print(f"exp=[1, 3] pending, rec={v1_images.pending_batches()} ", end="")
            passed = False
        # A retired number is never handed out again.
        if v1_images.next_batch_number(v1_images.read_batch_state()) != 4:
            print("exp=next number 4, rec=reused ", end="")
            passed = False

        # Starting the poller twice (boot, then a 'reload' that re-enables it) must not
        # leave two threads racing for the same batches.
        common.cfg.image_batch_auto_poll = True
        threads_before = threading.active_count()
        with contextlib.redirect_stdout(io.StringIO()):
            v1_images.start_batch_poller()
            v1_images.start_batch_poller()
        if threading.active_count() - threads_before != 1:
            print(f"exp=1 poller thread, rec={threading.active_count() - threads_before} ", end="")
            passed = False

        # An older state file without numbers gets them rather than printing "?".
        v1_images.write_batch_state({"batch_old": {"retrieved": False}})
        with contextlib.redirect_stdout(io.StringIO()):
            v1_images.number_legacy_batches()
        if v1_images.read_batch_state()["batch_old"].get("number") != 1:
            print("exp=legacy batch numbered, rec=unnumbered ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_source_files() -> bool:
    """
    Declared lineage outlives the reference it describes.

    A batched edit names its inputs by provider file id, and those ids expire and are
    deleted -- so the manifest records which file on this machine each one stood for.
    Declared by the caller when only the caller can know, derived from the references
    otherwise, and carried through the batch wait either way.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image source files: lineage the ids cannot carry... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        # The manifest's own directory is what every recorded path is relative to, so it has
        # to be the real one from the first call that records anything.
        common.cfg.image_output_dir = tmp
        base_path = write_file(tmp, "base.png", make_png(16, 16))

        # An object names all three parts; a bare string is a path only when it looks like
        # one, because reading "base.png" as a path would invent a location it never had.
        received = v1_images.validate_source_files([
            {"file_id": "file-abc", "file": "base.png", "path": base_path},
            base_path,
            "base.png",
        ])
        expected = [
            {"file_id": "file-abc", "path": base_path, "file": "base.png"},
            {"path": os.path.realpath(base_path), "file": "base.png"},
            {"file": "base.png"},
        ]
        if not check_equal(expected, received):
            passed = False

        # A path that no longer resolves is kept as given: the file may have moved, and a
        # stale link beats no link.
        moved = v1_images.validate_source_files(["/gone/away/old.png"])
        if moved != [{"path": "/gone/away/old.png", "file": "old.png"}]:
            print(f"exp=unresolvable path kept, rec={moved} ", end="")
            passed = False

        for bad in ([{"nope": "x"}], [""], [123], [{}]):
            try:
                v1_images.validate_source_files(bad)
                print(f"exp={bad} rejected, rec=accepted ", end="")
                passed = False
            except v1_images.ImageRequestError:
                pass

        # Listing the same references in both fields pairs them positionally, so the
        # caller does not repeat every id inside the entry it already lines up with.
        request = v1_images.build_request({
            "prompt"       : "restyle these",
            "batch"        : True,
            "file_ids"     : ["file-a", "file-b"],
            "source_files" : ["a.png", "b.png"],
        }, source="direct")
        expected = [{"file": "a.png", "file_id": "file-a"}, {"file": "b.png", "file_id": "file-b"}]
        if not check_equal(expected, request.source_files):
            passed = False

        # An id the caller paired itself is left alone, even when the counts line up.
        paired = v1_images.build_request({
            "prompt"       : "x",
            "batch"        : True,
            "file_ids"     : ["file-a", "file-b"],
            "source_files" : [{"file_id": "file-b", "file": "b.png"}, {"file": "a.png"}],
        }, source="direct").source_files
        if paired[0].get("file_id") != "file-b" or "file_id" in paired[1]:
            print(f"exp=caller's own pairing kept, rec={paired} ", end="")
            passed = False

        # Nothing declared: an immediate edit still gets the field, derived from what it
        # actually carried. A path knows where it came from, an upload only its name.
        derived = v1_images.manifest_source_files(v1_images.ImageRequest(
            prompt="x", size="auto", quality="auto", output_format="png",
            background="opaque", n=1, batch=False,
            images=[
                v1_images.load_reference(base_path, from_prompt=False),
                v1_images.load_uploaded_reference("dropped.png", make_png(8, 8)),
            ],
        ))
        # Recorded relative to the manifest's directory, which is where base.png sits.
        expected = [{"path": "base.png", "file": "base.png"}, {"file": "dropped.png"}]
        if derived != expected:
            print(f"exp={expected}, rec={derived} ", end="")
            passed = False

        # A source outside the output directory climbs out of it rather than going absolute,
        # and says so with forward slashes on either platform.
        outside = os.path.join(tmp, "refs", "held.png")
        os.makedirs(os.path.dirname(outside), exist_ok=True)
        write_file(os.path.dirname(outside), "held.png", make_png(8, 8))
        nested = v1_images.manifest_source_files(v1_images.ImageRequest(
            prompt="x", size="auto", quality="auto", output_format="png",
            background="opaque", n=1, batch=False,
            images=[v1_images.load_reference(outside, from_prompt=False)],
        ))
        if nested != [{"path": "refs/held.png", "file": "held.png"}]:
            print(f"exp=[refs/held.png], rec={nested} ", end="")
            passed = False

        # The manifest is written when a batch is retrieved, hours after the request that
        # declared the lineage is gone -- so the state file has to carry it.
        restored = v1_images.state_to_request(v1_images.request_to_state(request))
        if not check_equal(request.source_files, restored.source_files):
            passed = False

        # The record itself: a basename that joins it to a directory listing, its size, and
        # the lineage. Every path in it is relative to this manifest's directory, so a
        # generated image's 'path' comes out equal to its 'file'.
        common.cfg.image_manifest_enabled = True
        common.cfg.image_manifest_prompts = True

        payload = make_png(16, 16)
        saved   = v1_images.save_image_bytes(request, payload, "img_abcdef0123456789")
        v1_images.append_manifest(request, [saved], 0.04, {"reported": False}, True, batch_id="batch_x")

        with open(os.path.join(tmp, v1_images.MANIFEST_FILE), "r", encoding="utf-8") as handle:
            records = json.load(handle)

        record   = records[0] if records else {}
        received = {
            "file"         : record.get("file"),
            "path"         : record.get("path"),
            "bytes"        : record.get("bytes"),
            "source_files" : record.get("source_files"),
            "size"         : (record.get("request_parameters") or {}).get("size"),
            "joins"        : record.get("file") in os.listdir(tmp),
        }
        expected = {
            "file"         : os.path.basename(saved.path),
            "path"         : os.path.basename(saved.path),
            "bytes"        : len(payload),
            "source_files" : [{"file": "a.png", "file_id": "file-a"},
                              {"file": "b.png", "file_id": "file-b"}],
            # The size that was *asked* for, recorded as given -- including 'auto'.
            "size"         : request.size,
            "joins"        : True,
        }
        if not check_equal(expected, received):
            passed = False

        common.cfg.image_manifest_enabled = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_batch_listing() -> bool:
    """
    The listing a client rebuilds its job list from: recorded state only, newest first.

    It must not ask the provider about anything. A listing is polled, and one that fanned
    out would both cost and, through retrieval, bill.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image batch listing: recorded state, no provider call... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        v1_images.write_batch_state({
            "batch_aaa": {"number": 1, "retrieved": True, "final_status": "completed",
                          "model": "gpt-image-2", "submitted_at": "2026-07-31T10:00:00+0000",
                          "images": ["generated_images/one.png"],
                          "requests": {"img_0000_aa": {"prompt": "a cat", "size": "1024x1024",
                                                       "n": 1, "file_ids": ["file-a"],
                                                       "source_files": [{"file": "base.png", "file_id": "file-a"}]}}},
            "batch_bbb": {"number": 2, "retrieved": False, "model": "gpt-image-2",
                          "submitted_at": "2026-07-31T11:00:00+0000",
                          "requests": {"img_0000_bb": {"prompt": "a dog", "n": 2}}},
            # Retrieved without a final status: settled, but not by any provider word.
            "batch_ccc": {"number": 3, "retrieved": True, "model": "gpt-image-2"},
        })

        listing  = v1_images.list_batches()
        received = {
            "order"    : [entry["number"] for entry in listing],
            "statuses" : {entry["number"]: entry["status"] for entry in listing},
            "lineage"  : listing[-1]["requests"][0]["source_files"],
            "prompt"   : listing[-1]["requests"][0]["prompt"],
            "images"   : listing[-1]["images"],
        }
        expected = {
            "order"    : [3, 2, 1],
            "statuses" : {1: "completed", 2: "pending", 3: "settled"},
            "lineage"  : [{"file": "base.png", "file_id": "file-a"}],
            "prompt"   : "a cat",
            "images"   : ["generated_images/one.png"],
        }
        if not check_equal(expected, received):
            passed = False

        # A directory nobody has submitted from lists nothing rather than failing.
        common.cfg.image_output_dir = os.path.join(tmp, "empty")
        if v1_images.list_batches() != []:
            print("exp=empty listing, rec=entries ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_batch_polling() -> bool:
    """
    The poller settles finished batches, leaves running ones alone, keeps going past a
    batch that errors, and stops chasing one that failed or expired.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image batch polling: settles the finished, survives the broken... ", end="")

    make_image_config()
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        common.cfg.image_output_dir = tmp
        v1_images.write_batch_state({
            "batch_done"    : {"number": 1, "retrieved": False, "model": "gpt-image-2"},
            "batch_running" : {"number": 2, "retrieved": False, "model": "gpt-image-2"},
            "batch_broken"  : {"number": 3, "retrieved": False, "model": "gpt-image-2"},
            "batch_expired" : {"number": 4, "retrieved": False, "model": "gpt-image-2"},
        })

        seen: list[str] = []
        original = v1_images.retrieve_image_batch
        def fake_retrieve(batch_id: str) -> Any:
            seen.append(batch_id)
            if batch_id == "batch_broken":
                raise providers.ProviderError(500, {"error": {"message": "boom"}}, "boom")
            status = {"batch_done": "completed", "batch_running": "in_progress", "batch_expired": "expired"}[batch_id]
            if status in v1_images.BATCH_TERMINAL_STATUSES:
                state = v1_images.read_batch_state()
                state[batch_id].update({"retrieved": True, "final_status": status})
                v1_images.write_batch_state(state)
            return v1_images.ImageBatchResult(
                provider="testimg", model="gpt-image-2", batch_id=batch_id,
                number=int(v1_images.read_batch_state()[batch_id]["number"]), status=status,
            )

        v1_images.retrieve_image_batch = fake_retrieve
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                settled = v1_images.poll_batches_once()
        finally:
            v1_images.retrieve_image_batch = original

        # Every pending batch is checked, in number order, and one raising does not
        # stop the ones after it.
        if seen != ["batch_done", "batch_running", "batch_broken", "batch_expired"]:
            print(f"exp=all four checked in order, rec={seen} ", end="")
            passed = False

        settled_ids = sorted(result.batch_id for result in settled)
        if settled_ids != ["batch_done", "batch_expired"]:
            print(f"exp=['batch_done', 'batch_expired'] settled, rec={settled_ids} ", end="")
            passed = False

        # A batch that expired will never produce images, so it must stop being polled.
        still_pending = [batch_id for _, batch_id in v1_images.pending_batches()]
        if sorted(still_pending) != ["batch_broken", "batch_running"]:
            print(f"exp=broken+running still pending, rec={sorted(still_pending)} ", end="")
            passed = False

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_chat_integration() -> bool:
    """
    The two chat outcomes: an image-only turn never reaches a text backend, and a mixed
    turn keeps the model's prose with the reference appended.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image chat integration: image-only skips the backend, mixed appends... ", end="")

    make_config()
    make_image_config()
    passed = True

    saved = v1_images.SavedImage(image_id="img_dead", path="generated_images/x.png")
    fake_result = v1_images.ImageResult(
        provider="testimg", model="gpt-image-2", created=CREATED,
        images=[saved], cost_usd=0.05, usage={"reported": True}, estimated=False,
    )

    calls: list[str] = []
    original_generate = v1_images.generate_image
    def fake_generate(request: Any) -> Any:
        calls.append(request.prompt)
        return fake_result

    backend_calls: list[Any] = []
    original_backend = server.active_backend
    def fake_backend() -> Any:
        backend_calls.append(True)
        return SimpleNamespace(generate_non_stream=lambda prepared: {
            "id": "msg_test", "stop_reason": "stop", "text": "The room is lit.",
            "usage": {}, "message_extra": {},
        })

    v1_images.generate_image = fake_generate
    server.active_backend               = fake_backend
    try:
        image_only = server.generate_non_stream({"messages": [
            {"role": "user", "content": "<image_generation>prompt: \"a lit chamber\"</image_generation>"},
        ]})
        if backend_calls:
            print("exp=no backend call for an image-only turn, rec=called ", end="")
            passed = False
        content = image_only["choices"][0]["message"]["content"]
        if content != "[Generated image: img_dead — generated_images/x.png]":
            print(f"exp=reference only, rec={content!r} ", end="")
            passed = False

        mixed = server.generate_non_stream({"messages": [
            {"role": "user", "content": "describe it\n<image_generation>prompt: \"a lit chamber\"</image_generation>"},
        ]})
        if len(backend_calls) != 1:
            print(f"exp=1 backend call for a mixed turn, rec={len(backend_calls)} ", end="")
            passed = False
        content = mixed["choices"][0]["message"]["content"]
        if content != "The room is lit.\n\n[Generated image: img_dead — generated_images/x.png]":
            print(f"exp=prose + reference, rec={content!r} ", end="")
            passed = False

        if calls != ["a lit chamber", "a lit chamber"]:
            print(f"exp=2 generations, rec={calls} ", end="")
            passed = False
    finally:
        v1_images.generate_image = original_generate
        server.active_backend               = original_backend

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_failure_is_inline() -> bool:
    """
    A provider failure must cost the turn its image, never its text -- and an HTML
    gateway error page must not be relayed into the reply verbatim.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image failure: the turn survives and the note stays short... ", end="")

    make_config()
    make_image_config()
    common.cfg.error_log_path = os.path.join(tempfile.gettempdir(), "revpy_test_error_log.txt")
    passed = True

    html = "<html><head><title>520</title></head><body><h1>" + ("Web server error. " * 200) + "</h1></body></html>"

    original_generate = v1_images.generate_image
    def failing_generate(request: Any) -> Any:
        raise providers.ProviderError(520, {"error": {"message": html}}, "gateway error")

    original_backend = server.active_backend
    def fake_backend() -> Any:
        return SimpleNamespace(generate_non_stream=lambda prepared: {
            "id": "msg_test", "stop_reason": "stop", "text": "The room is lit.",
            "usage": {}, "message_extra": {},
        })

    v1_images.generate_image = failing_generate
    server.active_backend               = fake_backend
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            response = server.generate_non_stream({"messages": [
                {"role": "user", "content": "describe it\n<image_generation>prompt: \"a lit chamber\"</image_generation>"},
            ]})
        content = response["choices"][0]["message"]["content"]

        if not content.startswith("The room is lit."):
            print(f"exp=prose kept, rec={content[:60]!r} ", end="")
            passed = False
        if "[Image generation failed:" not in content:
            print(f"exp=inline failure note, rec={content[:80]!r} ", end="")
            passed = False
        if "<" in content or ">" in content:
            print("exp=markup stripped, rec=markup relayed ", end="")
            passed = False
        note = content.split("[Image generation failed:", 1)[1]
        if len(note) > server.IMAGE_NOTE_MAX_CHARS + 8:
            print(f"exp=note under {server.IMAGE_NOTE_MAX_CHARS} chars, rec={len(note)} ", end="")
            passed = False
    finally:
        v1_images.generate_image = original_generate
        server.active_backend               = original_backend

    if passed : print(f"{GREEN}PASS{RESET}")
    else      : print(f"{RED}FAIL{RESET}")
    return passed


def test_image_model_selection_is_isolated() -> bool:
    """
    The rule the whole feature rests on: choosing an image model must leave the
    conversation, its backend and its text prices exactly as they were.
    """
    global tests_ttl
    tests_ttl += 1
    print("Testing image model selection: the chat model is untouched... ", end="")

    make_config()
    make_image_config()
    cfg = common.cfg

    os.environ["TESTIMG_IMAGE_MODEL_GPTIMAGE2_REGEX"] = "^gpt-image-2"
    os.environ["TESTIMG_IMAGE_MODEL_GPTIMAGE2_COST"]  = "{text_input: 5.0, image_input: 8.0, image_output: 30.0}"

    before = (cfg.backend, cfg.model, cfg.input_token_cost_usd, cfg.output_token_cost_usd, cfg.model_cost_family)
    try:
        v1_images.apply_image_model("gpt-image-2")
    finally:
        del os.environ["TESTIMG_IMAGE_MODEL_GPTIMAGE2_REGEX"]
        del os.environ["TESTIMG_IMAGE_MODEL_GPTIMAGE2_COST"]

    after  = (cfg.backend, cfg.model, cfg.input_token_cost_usd, cfg.output_token_cost_usd, cfg.model_cost_family)
    passed = True

    if before != after:
        print(f"exp=chat state unchanged, rec={before} -> {after} ", end="")
        passed = False

    expected_image = {"family": "testimg:gptimage2", "text": 5.0, "image_in": 8.0, "out": 30.0}
    received_image = {"family": cfg.image_cost_family, "text": cfg.image_text_input_cost,
                      "image_in": cfg.image_image_input_cost, "out": cfg.image_output_cost}
    if not check_equal(expected_image, received_image):
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

    tests_passed += test_provider_config_parsing()

    tests_passed += test_cost_resolution()

    tests_passed += test_usage_normalization()

    for stream_case in STREAM_CASES:
        tests_passed += test_responses_stream_termination(stream_case)

    for non_stream_case in NON_STREAM_CASES:
        tests_passed += test_responses_non_stream(non_stream_case)

    tests_passed += test_provider_request_bodies()

    tests_passed += test_image_extraction()

    tests_passed += test_image_request_validation()

    tests_passed += test_image_usage_accounting()

    tests_passed += test_image_reference_validation()
    tests_passed += test_image_uploaded_references()
    tests_passed += test_image_http_routes()
    tests_passed += test_image_edit_path_security()
    tests_passed += test_image_edit_detection()
    tests_passed += test_image_edit_batch_rules()
    tests_passed += test_image_edit_form()
    tests_passed += test_image_mask_validation()
    tests_passed += test_image_manifest_patch()
    tests_passed += test_image_mask_region()
    tests_passed += test_image_mask_persistence()
    tests_passed += test_image_data_url_references()

    tests_passed += test_image_source_files()
    tests_passed += test_image_batch_accounting()
    tests_passed += test_image_batch_numbering()
    tests_passed += test_image_batch_listing()
    tests_passed += test_image_batch_polling()
    tests_passed += test_image_storage_confinement()
    tests_passed += test_image_chat_integration()
    tests_passed += test_image_failure_is_inline()
    tests_passed += test_image_model_selection_is_isolated()

    tests_failed : int = tests_ttl - tests_passed

    if tests_failed == 0 : print(f"{GREEN}All {tests_ttl} tests passed.{RESET}")
    else                 : print(f"{RED}{tests_failed} out of {tests_ttl} tests failed.{RESET}")
