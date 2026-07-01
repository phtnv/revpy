import importlib.util
import json5
import types

from pathlib           import Path
from typing            import Any, Callable, Protocol, cast
from packaging.version import Version


ROOT     = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GREEN    = "\033[32m"
RED      = "\033[31m"
RESET    = "\033[0m"


class ServerModule(Protocol):
    cfg                  : Any
    app                  : Any
    time                 : "ClockModule"
    get_anthropic_client : Callable[[], Any]


class ClockModule(Protocol):
    time: Callable[[], float]


def load_server_module() -> ServerModule:
    spec = importlib.util.spec_from_file_location("revpy_server", ROOT / "server.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load server module from {ROOT / 'server.py'}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    loader.exec_module(module)
    return cast(ServerModule, module)


def load_json5_fixture(path: Path) -> dict[str, Any]:
    text    = path.read_text(encoding="utf-8")
    fixture = json5.loads(text)

    if not isinstance(fixture, dict):
        raise TypeError(f"Fixture {path} must contain an object.")

    return cast(dict[str, Any], fixture)


def object_from_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return types.SimpleNamespace(**{key: object_from_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [object_from_dict(item) for item in value]
    return value


def fake_anthropic_message(raw: dict[str, Any]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id          = raw["id"],
        stop_reason = raw["stop_reason"],
        content     = raw["content"],
        usage       = object_from_dict(raw["usage"]),
    )


class FakeMessagesClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

    def create(self, **kwargs: Any) -> types.SimpleNamespace:
        self.calls.append(kwargs)
        return fake_anthropic_message(self.response)


class FakeAnthropicClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.messages = FakeMessagesClient(response)


def make_test_config(model: str, max_tokens: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model                    = model,
        version                  = Version("4.0"),
        max_tokens               = max_tokens,
        debug_log                = False,
        auto_trim                = False,
        summary_blocks_enabled   = True,
        cache_en                 = False,
        cache_system             = False,
        cache_system_ttl         = "1h",
        cache_manual_msg         = 0,
        cache_manual_ttl         = "1h",
        cache_auto_msg           = 0,
        cache_auto_ttl           = "5m",
        cache_anthropic_auto     = False,
        cache_anthropic_ttl      = "1h",
        input_token_cost_usd     = 3.0,
        output_token_cost_usd    = 15.0,
        cache_write_5m_cost_usd  = 3.75,
        cache_write_1h_cost_usd  = 6.0,
        cache_read_cost_usd      = 0.3,
        split_lorebook           = False,
        lorebook_at_end          = False,
        lorebook_xml_at_end      = False,
        assistant_prefill        = "",
        assistant_prefill_mode   = "none",
        send_temperature         = False,
        temperature              = 0.9,
        send_top_k               = False,
        top_k                    = 75,
        send_top_p               = False,
        top_p                    = 0.95,
        thinking_enabled         = False,
        use_adaptive             = False,
        thinking_effort          = "medium",
        thinking_budget          = 2048,
        preserve_thinking_blocks = 0,
        error_log_path           = str(ROOT / "test_error_log.txt"),
    )


def check_equal(expected: Any, received: Any, label: str) -> bool:
    if expected == received:
        return True

    print(f"Expected: {expected!r}")
    print(f"Received: {received!r}")
    return False


def test_basic_non_streaming_roundtrip() -> bool:
    print("Testing basic non-streaming roundtrip... ", end="")
    server      = load_server_module()
    fixture     = load_json5_fixture(FIXTURES / "basic.json5")
    fake_client = FakeAnthropicClient(fixture["anthropic_response"])

    server.cfg = make_test_config(
        model      = fixture["config"]["model"],
        max_tokens = fixture["config"]["max_tokens"],
    )

    original_get_anthropic_client = server.get_anthropic_client
    original_time                 = server.time.time
    server.get_anthropic_client   = lambda: fake_client
    server.time.time              = lambda: fixture["expected_openai_response"]["created"]
    try:
        response = server.app.test_client().post(
            "/v1/chat/completions",
            json=fixture["openai_request"],
        )
    finally:
        server.get_anthropic_client = original_get_anthropic_client
        server.time.time = original_time

    expected_anthropic_content = fixture["expected_anthropic_request"]["messages"][0]["content"]
    received_anthropic_content = fake_client.messages.calls[0]["messages"][0]["content"]

    expected_openai_content = fixture["expected_openai_response"]["choices"][0]["message"]["content"]
    received_openai_content = response.get_json()["choices"][0]["message"]["content"]

    passed = True
    passed &= check_equal(200, response.status_code, "status code")
    passed &= check_equal(expected_anthropic_content, received_anthropic_content, "anthropic request content")
    passed &= check_equal(expected_openai_content, received_openai_content, "openai response content")

    if passed : print(f"{GREEN}PASS!{RESET}")
    else      : print(f"{RED}Test failed!{RESET}")
    return passed


if __name__ == "__main__":
    tests: list[Callable[[], bool]] = [
        test_basic_non_streaming_roundtrip,
    ]

    tests_ttl    : int = 0
    tests_passed : int = 0
    tests_failed : int = 0

    for test in tests:
        tests_ttl += 1
        passed = test()
        if passed : tests_passed += 1
        else      : tests_failed += 1

    if tests_failed == 0 : print(f"{GREEN}Finished. All {tests_ttl} tests passed{RESET}.")
    else                 : print(f"Finished. {tests_passed} out of {tests_ttl} passed.")
