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
common: Any = importlib.import_module("common")
claude: Any = importlib.import_module("claude")
server: Any = importlib.import_module("server")


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

    claude.get_anthropic_client = lambda: rx_msg
    server.time.time            = lambda: CREATED
    claude.print_usage          = lambda usage: None
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

    claude.get_anthropic_client = lambda: fake
    server.time.time            = lambda: CREATED
    claude.print_usage          = lambda usage: None
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


if __name__ == "__main__":
    make_config()

    tests_passed : int = 0
    tests_passed += test_basic_non_streaming_roundtrip("basic_no_ooc.json5")
    tests_passed += test_basic_non_streaming_roundtrip("basic_with_ooc.json5")
    tests_passed += test_chat_dump_formats("dump_chat.json5")

    tests_failed : int = tests_ttl - tests_passed

    if tests_failed == 0 : print(f"{GREEN}All {tests_ttl} tests passed.{RESET}")
    else                 : print(f"{RED}{tests_failed} out of {tests_ttl} tests failed.{RESET}")
