import builtins
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import threading

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
image_api   : Any = importlib.import_module("v1_images_generation")
server      : Any = importlib.import_module("server")

v1_images_generation = image_api
image_orchestrator : Any = importlib.import_module("image_orchestrator")

# The three request builders, keyed by the wire protocol they speak.
BUILDERS = {
    "messages"  : lambda prepared: v1_messages.build_body(prepared),
    "chat"      : lambda prepared: chat_api.build_body(prepared),
    "responses" : lambda prepared: resp_api.build_body(prepared),
}

# Where each OpenAI-style endpoint carries the message list build_message_list() produces.
# The Anthropic body is shaped differently enough -- system is a field of its own, and
# content is blocks rather than strings -- that its cases carry their own expected
# 'system' and 'messages' instead of sharing this one.
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
    # A model is what binds a backend, so the roundtrip tests need one selected. The
    # cases that exercise a provider replace both.
    cfg.providers = {PROVIDER: make_provider(api="messages", api_key_name="CLAUDE_API_KEY")}
    cfg.backend   = PROVIDER
    return cfg


def make_provider(**overrides: Any) -> dict[str, Any]:
    """
    A synthetic provider entry, in the shape RuntimeConfig.parse_provider builds them.
    Defaults are the ones a provider gets with only BASE_URL and API_KEY set, so a
    case only has to name what it actually exercises.
    """
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


# What a model selection resolves its five prices to.
#   (label, model id, expected prices)
# Every provider resolves them the same way: the first matching cost family, else the
# provider-level prices. Both cache write buckets fall back to cache_write, which falls
# back to the input price -- so a provider that charges no write fee nets those to zero.
COST_CASES = [
    ("cost family wins, TTLs priced apart", "claude-opus-4-8",
     {"family": "claude:opus", "input": 5.00, "output": 25.00, "read": 0.50, "write_5m": 6.25, "write_1h": 10.00}),

    ("no family matches, provider prices", "claude-sonnet-4-6",
     {"family": "claude", "input": 3.00, "output": 15.00, "read": 0.00, "write_5m": 3.00, "write_1h": 3.00}),
]


def test_cost_resolution(case: tuple) -> bool:
    global tests_ttl
    tests_ttl += 1
    label, model_id, expected = case
    print(f"Testing cost resolution: {label}... ", end="")

    providers_parsed = parse_provider_env()

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

    passed = check_equal(expected, received)
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
        cfg.providers        = {case["backend"]: make_provider(api=case["builder"], **case.get("provider", {}))}
        cfg.thinking_enabled = False
        cfg.thinking_effort  = "medium"
        for key, value in case.get("cfg", {}).items():
            setattr(cfg, key, value)

        received = BUILDERS[case["builder"]](entry["prepared"])

    if case["builder"] == "messages":
        # The input's system/messages unless the case pins its own, which the cases
        # exercising cache markers do -- those change the message shape itself.
        expected = {**entry["anthropic"], **case["expected"]}
    else:
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


# Image generation
#
# None of these contact a provider: extraction, validation and filename confinement all
# happen before transport, and the two chat cases stub generate_image out. The point is
# that a request is rejected (or an image is placed) without anything being spent.
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
        "messages"   : [{"role": "user", "content": "<image_generation>prompt: \"x\", model: \"gpt-4o\"</image_generation>"}],
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


def test_image_extraction(case: dict[str, Any]) -> bool:
    global tests_ttl
    tests_ttl += 1
    print(f"Testing image extraction: {case['name']}... ", end="")

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

    passed = check_equal(expected, received)
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
    # Everything the caller must never choose.
    ({"prompt": "x", "model": "gpt-4o"}                 , "unsupported field"),
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
]


def test_image_validation(case: tuple) -> bool:
    global tests_ttl
    tests_ttl += 1
    overrides, expected_error = case
    print(f"Testing image validation: {str(overrides)[:58]}... ", end="")

    make_image_config()

    received_error = None
    try:
        v1_images_generation.build_request(overrides)
    except Exception as exc:
        received_error = str(exc)

    if expected_error is None:
        passed = received_error is None
        if not passed:
            print(f"exp=accepted, rec={received_error!r} ", end="")
    else:
        passed = received_error is not None and expected_error in received_error
        if not passed:
            print(f"exp contains {expected_error!r}, rec={received_error!r} ", end="")

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
        request = v1_images_generation.build_request({"prompt": "x", "filename": "shot"})

        saved = [v1_images_generation.save_image_bytes(request, b"\x89PNG-one", "img_aaaa"),
                 v1_images_generation.save_image_bytes(request, b"\x89PNG-two", "img_bbbb")]

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

        # No .part file may survive a completed write.
        leftovers = [name for name in os.listdir(tmp) if name.endswith(".part")]
        if leftovers:
            print(f"exp=no temp files, rec={leftovers} ", end="")
            passed = False

        # A stem that somehow bypassed sanitisation still cannot escape.
        try:
            v1_images_generation.allocate_path(tmp, "../escaped", ".png")
            print("exp=escape rejected, rec=allowed ", end="")
            passed = False
        except v1_images_generation.ImageRequestError:
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


def test_image_usage(case: tuple) -> bool:
    global tests_ttl
    tests_ttl += 1
    name, payload, expected_counts, expected_cost = case
    print(f"Testing image usage: {name}... ", end="")

    make_image_config()
    common.cfg.image_text_input_cost  = 5.00
    common.cfg.image_image_input_cost = 8.00
    common.cfg.image_output_cost      = 30.00

    counts = v1_images_generation.parse_usage(payload)
    passed = check_equal(expected_counts, counts)

    cost = common.track_image_usage(counts, images=1, batch=False, model="gpt-image-2")
    if round(cost, 6) != round(expected_cost, 6):
        print(f"key=cost exp={expected_cost!r}, rec={round(cost, 6)!r}")
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
        v1_images_generation.read_batch_state()
        if os.path.exists(nested):
            print("exp=reading state creates nothing, rec=directory created ", end="")
            passed = False
        common.cfg.image_output_dir = tmp

        state = {}
        for index, batch_id in enumerate(["batch_aaa", "batch_bbb", "batch_ccc"], start=1):
            number = v1_images_generation.next_batch_number(state)
            if number != index:
                print(f"exp=number {index}, rec={number} ", end="")
                passed = False
            state[batch_id] = {"number": number, "retrieved": False, "model": "gpt-image-2"}
        v1_images_generation.write_batch_state(state)

        # A number resolves to its id, and a raw id is passed through untouched.
        if v1_images_generation.resolve_batch_id("2") != "batch_bbb":
            print(f"exp=batch_bbb, rec={v1_images_generation.resolve_batch_id('2')!r} ", end="")
            passed = False
        if v1_images_generation.resolve_batch_id("batch_ccc") != "batch_ccc":
            print("exp=raw id passed through, rec=rewritten ", end="")
            passed = False
        try:
            v1_images_generation.resolve_batch_id("99")
            print("exp=unknown number rejected, rec=accepted ", end="")
            passed = False
        except v1_images_generation.ImageRequestError:
            pass

        # Settling one batch removes it from the pending set but must not renumber the rest.
        state["batch_bbb"]["retrieved"] = True
        v1_images_generation.write_batch_state(state)
        if v1_images_generation.pending_batches() != [(1, "batch_aaa"), (3, "batch_ccc")]:
            print(f"exp=[1, 3] pending, rec={v1_images_generation.pending_batches()} ", end="")
            passed = False
        # A retired number is never handed out again.
        if v1_images_generation.next_batch_number(v1_images_generation.read_batch_state()) != 4:
            print("exp=next number 4, rec=reused ", end="")
            passed = False

        # Starting the poller twice (boot, then a 'reload' that re-enables it) must not
        # leave two threads racing for the same batches.
        common.cfg.image_batch_auto_poll = True
        threads_before = threading.active_count()
        with contextlib.redirect_stdout(io.StringIO()):
            v1_images_generation.start_batch_poller()
            v1_images_generation.start_batch_poller()
        if threading.active_count() - threads_before != 1:
            print(f"exp=1 poller thread, rec={threading.active_count() - threads_before} ", end="")
            passed = False

        # An older state file without numbers gets them rather than printing "?".
        v1_images_generation.write_batch_state({"batch_old": {"retrieved": False}})
        with contextlib.redirect_stdout(io.StringIO()):
            v1_images_generation.number_legacy_batches()
        if v1_images_generation.read_batch_state()["batch_old"].get("number") != 1:
            print("exp=legacy batch numbered, rec=unnumbered ", end="")
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
        v1_images_generation.write_batch_state({
            "batch_done"    : {"number": 1, "retrieved": False, "model": "gpt-image-2"},
            "batch_running" : {"number": 2, "retrieved": False, "model": "gpt-image-2"},
            "batch_broken"  : {"number": 3, "retrieved": False, "model": "gpt-image-2"},
            "batch_expired" : {"number": 4, "retrieved": False, "model": "gpt-image-2"},
        })

        seen: list[str] = []
        original = v1_images_generation.retrieve_image_batch
        def fake_retrieve(batch_id: str) -> Any:
            seen.append(batch_id)
            if batch_id == "batch_broken":
                raise providers.ProviderError(500, {"error": {"message": "boom"}}, "boom")
            status = {"batch_done": "completed", "batch_running": "in_progress", "batch_expired": "expired"}[batch_id]
            if status in v1_images_generation.BATCH_TERMINAL_STATUSES:
                state = v1_images_generation.read_batch_state()
                state[batch_id].update({"retrieved": True, "final_status": status})
                v1_images_generation.write_batch_state(state)
            return v1_images_generation.ImageBatchResult(
                provider="testimg", model="gpt-image-2", batch_id=batch_id,
                number=int(v1_images_generation.read_batch_state()[batch_id]["number"]), status=status,
            )

        v1_images_generation.retrieve_image_batch = fake_retrieve
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                settled = v1_images_generation.poll_batches_once()
        finally:
            v1_images_generation.retrieve_image_batch = original

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
        still_pending = [batch_id for _, batch_id in v1_images_generation.pending_batches()]
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

    saved = v1_images_generation.SavedImage(image_id="img_dead", path="generated_images/x.png")
    fake_result = v1_images_generation.ImageResult(
        provider="testimg", model="gpt-image-2", created=CREATED,
        images=[saved], cost_usd=0.05, usage={"reported": True}, estimated=False,
    )

    calls: list[str] = []
    original_generate = v1_images_generation.generate_image
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

    v1_images_generation.generate_image = fake_generate
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
        v1_images_generation.generate_image = original_generate
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
    common.cfg.error_log_path = str(ROOT / "test_error_log.txt")
    passed = True

    html = "<html><head><title>520</title></head><body><h1>" + ("Web server error. " * 200) + "</h1></body></html>"

    original_generate = v1_images_generation.generate_image
    def failing_generate(request: Any) -> Any:
        raise providers.ProviderError(520, {"error": {"message": html}}, "gateway error")

    original_backend = server.active_backend
    def fake_backend() -> Any:
        return SimpleNamespace(generate_non_stream=lambda prepared: {
            "id": "msg_test", "stop_reason": "stop", "text": "The room is lit.",
            "usage": {}, "message_extra": {},
        })

    v1_images_generation.generate_image = failing_generate
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
        v1_images_generation.generate_image = original_generate
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
        v1_images_generation.apply_image_model("gpt-image-2")
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

    for cost_case in COST_CASES:
        tests_passed += test_cost_resolution(cost_case)

    for usage_case in USAGE_CASES:
        tests_passed += test_usage_normalization(usage_case)

    for stream_case in STREAM_CASES:
        tests_passed += test_responses_stream_termination(stream_case)

    for non_stream_case in NON_STREAM_CASES:
        tests_passed += test_responses_non_stream(non_stream_case)

    body_inputs, body_cases = load_body_cases("provider_bodies.json5")
    for body_case in body_cases:
        tests_passed += test_provider_request_body(body_inputs, body_case)

    for extraction_case in IMAGE_EXTRACTION_CASES:
        tests_passed += test_image_extraction(extraction_case)

    for validation_case in IMAGE_VALIDATION_CASES:
        tests_passed += test_image_validation(validation_case)

    for image_usage_case in IMAGE_USAGE_CASES:
        tests_passed += test_image_usage(image_usage_case)

    tests_passed += test_image_batch_accounting()
    tests_passed += test_image_batch_numbering()
    tests_passed += test_image_batch_polling()
    tests_passed += test_image_storage_confinement()
    tests_passed += test_image_chat_integration()
    tests_passed += test_image_failure_is_inline()
    tests_passed += test_image_model_selection_is_isolated()

    tests_failed : int = tests_ttl - tests_passed

    if tests_failed == 0 : print(f"{GREEN}All {tests_ttl} tests passed.{RESET}")
    else                 : print(f"{RED}{tests_failed} out of {tests_ttl} tests failed.{RESET}")
