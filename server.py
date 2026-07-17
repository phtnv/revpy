import json
import re
import os
import time
import traceback
import threading

from dotenv     import load_dotenv
from flask      import Flask, Response, abort, jsonify, request, stream_with_context
from flask_cors import CORS
from typing     import Any, Dict, List, Optional, Tuple
from waitress   import serve

import claude
import open_ai

from claude import (
    anthropic_error_body,
    extract_hidden_thinking_envelopes,
    print_anthropic_error,
    print_model_info,
    print_model_list,
    refresh_anthropic_models,
    select_model_by_number,
    thinking_preservation_enabled,
)
from common import (
    DISABLE_VALUES,
    ENABLE_VALUES,
    INF_VALUES,
    PREFILL_MODES,
    UINT64_MAX,
    cfg,
    content_to_plain_text,
    session_cost_snapshot,
)

LATEST_CHAT_SNAPSHOT : Dict[str, Any] = {}
LATEST_CHAT_LOCK                      = threading.Lock()


def active_backend():
    """
    The backend module serving requests: claude, or open_ai for any provider
    configured through OPENAI_PROVIDERS. Both expose the same generate functions.
    """
    return claude if cfg.backend == "anthropic" else open_ai


def model_label() -> str:
    return f"{cfg.backend}/{cfg.model}"

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
    open_ai.refresh_openai_models(cfg.model_list_timeout_seconds)

    print("Reloaded runtime configuration from .env.")
    print("HOST and PORT were not changed; restart the process to change bind address.")


def cli_print_model_list() -> None:
    print_model_list()
    open_ai.print_model_list(number_offset=len(claude.ANTHROPIC_MODELS))


def cli_select_model(number: int) -> None:
    anthropic_count = len(claude.ANTHROPIC_MODELS)
    if number <= anthropic_count : select_model_by_number(number)
    else                         : open_ai.select_model_by_number(number - anthropic_count)


def cli_print_model_info(number: int) -> None:
    anthropic_count = len(claude.ANTHROPIC_MODELS)
    if number <= anthropic_count : print_model_info(number)
    else                         : open_ai.print_model_info(number - anthropic_count)


def cli_refresh_models() -> None:
    refresh_anthropic_models(cfg.anthropic_api_key, cfg.model_list_timeout_seconds)
    open_ai.refresh_openai_models(cfg.model_list_timeout_seconds)
    if cfg.backend == "anthropic":
        cfg.find_cfg(claude.ANTHROPIC_MODELS)
    elif not open_ai.apply_model_by_id(f"{cfg.backend}/{cfg.model}"):
        print(f"Model {cfg.backend}/{cfg.model} is no longer in the refreshed provider list.")


CLI_CMD_MODEL_INFO = """\
  model command. Alias: models, m.
    model              List all available models.
    model <uint>       Select a model from list.
    model info         Display information on the currently selected model.
    model info <uint>  Display information on the specified model.
      Alias: i
    model refresh      Request the available model list again.
      Alias: r
"""

CLI_CMD_CACHE_INFO = """\
  cache command. Alias: c.
    cache <bool>          Toggle all caching on/off.
    cache system <bool>   Toggle caching of system messages.
    cache system <5m|1h>  Set cache duration for the system messages.
      Alias: sys, s
    cache manual <uint>   Number of messages from start at which to place the manual cache marker. 0 to disable.
    cache manual <5m|1h>  Cache duration for the manual marker.
      Alias: man, m
    cache auto   <uint>   Message number from end at which to place the auto marker. 0 to disable.
    cache auto   <5m|1h>  Cache duration for the auto marker.
      Alias: a
    cache help            Display this message
      Alias: ?
"""

CLI_CMD_PREFILL_INFO = """\
  prefill command. Alias: p.
    prefill <none|assistant|instruction>  Select prefill mode.
    prefill set <string>                  Set prefill to <string>.
      Alias: s
"""

CLI_CMD_THINK_INFO = """\
  think command. Alias: t, thinking.
    think <bool>                              Turn thinking on/off.
    think effort <low|medium|high|xhigh|max>  Set thinking effort for models with adaptive thinking.
      Alias: e
    think budget <uint>                       Set thinking budget to <uint> for older models.
      Alias: b
    think preserve <uint>                     Set number of thinking blocks to preserve. 0 to disable. inf for all blocks.
      Alias: p
    think help                                Display this message.
      Alias: ?
"""

CLI_CMD_LOREBOOK_INFO = """\
  lorebook command. Alias l, lore.
    lorebook split <bool>  Split lorebook from system messages split on/off.
      Alias: s
    lorebook end <bool>    Move the lorebook to end of chat. As a system message for >=4.8 models. As OOC for others.
      Alias: e
    lorebook xml <bool>    Extract <lorebook>...</lorebook> from system prompt and move it to end of chat.
      Alias: x
    lorebook help          Display this message
      Alias: ?
"""

def admin_cli_loop() -> None:
    print("Runtime CLI ready. Type 'help' for commands.\n")

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
        parts_l = len(parts)
        cmd     = parts[0].lower()

        try:
            if cmd in {"c", "cache"}:
                if cfg.backend != "anthropic":
                    print(f"Cache markers are Anthropic-only. Backend '{cfg.backend}' has no explicit cache control (use EXTRA_BODY if the provider supports one).")
                    continue
                if parts_l < 2:
                    cfg.print_cache_status()
                    continue

                arg1 = parts[1].lower()
                if arg1 in {"?", "help"}  : print(CLI_CMD_CACHE_INFO); continue
                if arg1 in DISABLE_VALUES : cfg.cache_en = False     ; continue
                if arg1 in ENABLE_VALUES  : cfg.cache_en = True      ; continue

                if parts_l < 3:
                    print(CLI_CMD_CACHE_INFO)
                    continue
                arg2 = parts[2].lower()
                if arg2 in {"5m", "1h"}   : cfg.set_cache_dur(arg1, arg2) ; continue
                if arg2 in DISABLE_VALUES : cfg.set_cache_msg_num(arg1, 0); continue
                if arg2 in ENABLE_VALUES  : cfg.set_cache_msg_num(arg1, 1); continue
                try: msg_num = int(arg2)
                except Exception: pass
                else:
                    cfg.set_cache_msg_num(arg1, msg_num)
                    continue
                print(CLI_CMD_CACHE_INFO)
                continue

            if cmd in {"t", "think", "thinking"}:
                if cfg.backend != "anthropic":
                    print(f"Thinking controls are Anthropic-only. For backend '{cfg.backend}', configure thinking through the provider's EXTRA_BODY.")
                    continue
                if parts_l < 2:
                    cfg.print_think_status()
                    continue

                arg1 = parts[1].lower()
                if arg1 in {"?", "help"}  : print(CLI_CMD_THINK_INFO) ; continue
                if arg1 in DISABLE_VALUES : cfg.enable_thinking(False); continue
                if arg1 in ENABLE_VALUES  : cfg.enable_thinking(True) ; continue

                if parts_l < 3:
                    print(CLI_CMD_THINK_INFO)
                    continue
                arg2 = parts[2].lower()
                if arg1 in {"e", "effort"}:
                    cfg.set_think_effort(arg2)
                    continue
                if arg1 in {"b", "budget"}:
                    try: budget = int(arg2)
                    except Exception: pass
                    else:
                        cfg.set_think_budget(budget)
                        continue
                if arg1 in {"p", "preserve"}:
                    if arg2 in INF_VALUES:
                        cfg.set_think_blocks_to_preserve(UINT64_MAX)
                        continue
                    try: preserve_blocks = int(arg2)
                    except Exception: pass
                    else:
                        cfg.set_think_blocks_to_preserve(preserve_blocks)
                        continue
                print(CLI_CMD_THINK_INFO)
                continue

            if cmd in {"l", "lore", "lorebook"}:
                if parts_l < 2:
                    cfg.print_lorebook_status()
                    continue

                arg1 = parts[1].lower()
                if arg1 in {"?", "help"}:
                    print(CLI_CMD_LOREBOOK_INFO)
                    continue

                if parts_l < 3:
                    print(CLI_CMD_LOREBOOK_INFO)
                    continue
                arg2 = parts[2].lower()

                if arg1 in {"s", "split"}:
                    if   arg2 in ENABLE_VALUES  : cfg.set_lorebook_split(True ); continue
                    elif arg2 in DISABLE_VALUES : cfg.set_lorebook_split(False); continue
                    else:
                        print(CLI_CMD_LOREBOOK_INFO)
                        continue
                if arg1 in {"e", "end"}:
                    if   arg2 in ENABLE_VALUES  : cfg.lorebook_at_end = True ; continue
                    elif arg2 in DISABLE_VALUES : cfg.lorebook_at_end = False; continue
                    else:
                        print(CLI_CMD_LOREBOOK_INFO)
                        continue
                if arg1 in {"x", "xml"}:
                    if   arg2 in ENABLE_VALUES  : cfg.lorebook_xml_at_end = True ; continue
                    elif arg2 in DISABLE_VALUES : cfg.lorebook_xml_at_end = False; continue
                    else:
                        print(CLI_CMD_LOREBOOK_INFO)
                        continue
                print(CLI_CMD_LOREBOOK_INFO)
                continue

            if cmd == "prefill":
                if parts_l < 2:
                    print(CLI_CMD_PREFILL_INFO)
                    continue

                arg1 = parts[1].lower()
                if arg1 in PREFILL_MODES:
                    cfg.set_prefill_mode(arg1)
                    continue

                if arg1 != "set":
                    print(CLI_CMD_PREFILL_INFO)
                    continue

                split_line = line.split(maxsplit=2)
                if len(split_line) < 3:
                    print(CLI_CMD_PREFILL_INFO)
                    continue

                cfg.set_prefill(split_line[2])
                continue

            if cmd in {"reload"}:
                reload_runtime_env()
                cfg.print_status()
                continue

            if cmd in {"m", "model", "models"}:
                if parts_l < 2:
                    cli_print_model_list()
                    continue

                arg1 = parts[1].lower()
                if parts_l == 2:
                    try: model_id = int(arg1)
                    except Exception: pass
                    else:
                        cli_select_model(model_id)
                        continue
                    if arg1 in {"i", "info"}:
                        print(json.dumps(cfg.model_info, indent=2, ensure_ascii=False, default=str))
                        continue
                    if arg1 in {"r", "refresh"}:
                        cli_refresh_models()
                        continue
                    print(CLI_CMD_MODEL_INFO)
                    continue

                if parts_l < 3:
                    print(CLI_CMD_MODEL_INFO)
                    continue
                arg2 = parts[2].lower()
                if arg1 in {"i", "info"}:
                    try: model_id = int(arg2)
                    except Exception: pass
                    else:
                        cli_print_model_info(model_id)
                        continue
                print(CLI_CMD_MODEL_INFO)
                continue

            if cmd == "status":
                cfg.print_status()
                continue

            if cmd in {"d", "dump"}:
                fmt  = "json"
                path = ""
                if parts_l > 1:
                    arg1 = parts[1].lower()
                    if arg1 in {"n", "nat", "natural", "md", "markdown"}:
                        fmt = "natural"
                        split_line = line.split(maxsplit=2)
                        if len(split_line) > 2 : path = split_line[2]
                    elif arg1 in {"j", "json"}:
                        split_line = line.split(maxsplit=2)
                        if len(split_line) > 2 : path = split_line[2]
                    else:
                        path = line.split(maxsplit=1)[1]
                if not path:
                    path = "chat_snapshot.md" if fmt == "natural" else "chat_snapshot.json"
                with LATEST_CHAT_LOCK:
                    snapshot = LATEST_CHAT_SNAPSHOT
                if not snapshot:
                    print("No chat snapshot captured yet.")
                    continue
                with open(path, "w", encoding="utf-8") as f:
                    if fmt == "natural":
                        f.write(snapshot_to_markdown(snapshot))
                    else:
                        json.dump(snapshot, f, indent=2, ensure_ascii=False)
                        f.write("\n")
                print(f"Wrote latest chat snapshot to {path} ({fmt}).")
                continue

            if cmd in {"help", "?"}:
                print()
                print("Commands:")
                print(CLI_CMD_CACHE_INFO)
                print(CLI_CMD_MODEL_INFO)
                print(CLI_CMD_PREFILL_INFO)
                print(CLI_CMD_THINK_INFO)
                print("  reload         Reload runtime settings from .env.")
                print("  status         Show runtime settings.")
                print("  dump command. Alias d.")
                print("    dump  json    JSON snapshot (default).")
                print("      Alias: j")
                print("    dump natural  Human-readable markdown.")
                print("      Alias: n, nat, md, markdown")
                print("  help           Display this message.")
                print("    Alias: ?")
                print("  quit           Stop the server.")
                print("    Alias: q, exit")
                print()
                continue

            if cmd in {"q", "quit", "exit"}:
                print("Stopping proxy.")
                os._exit(0)

            print(f"Unknown command: {cmd}")
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


# Summary block replacement
SUMMARY_BLOCK_ANY_RE = re.compile(r"<summary_block_(?:beg|end)\b", re.IGNORECASE)
SUMMARY_BLOCK_BEG_RE = re.compile(r"<summary_block_beg\b(?P<attrs>[^>]*)>", re.IGNORECASE)
SUMMARY_BLOCK_END_RE = re.compile(
    r"<summary_block_end\b(?P<attrs>[^>]*)>(?P<body>.*?)</summary_block_end\s*>",
    re.IGNORECASE | re.DOTALL,
)
SUMMARY_BLOCK_ATTR_RE = re.compile(r"""([A-Za-z_][A-Za-z0-9_:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
SUMMARY_BLOCK_TAG_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SUMMARY_BLOCK_ROLES = {"assistant", "user", "system"}


def warn_summary_block(message: str) -> None:
    print(f"WARNING: Summary block ignored. {message}")


def parse_summary_block_attrs(raw_attrs: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for match in SUMMARY_BLOCK_ATTR_RE.finditer(raw_attrs or ""):
        value = match.group(2) if match.group(2) is not None else match.group(3)
        attrs[match.group(1).lower()] = value
    return attrs


def parse_summary_block_tag(attrs: Dict[str, str], msg_num: int, kind: str) -> Optional[str]:
    tag = attrs.get("tag", "").strip()
    if not tag:
        warn_summary_block(f"Message {msg_num} has a {kind} tag without tag=\"...\".")
        return None
    if not SUMMARY_BLOCK_TAG_VALUE_RE.match(tag):
        warn_summary_block(f"Message {msg_num} has invalid summary tag {tag!r}.")
        return None
    if kind == "begin" and tag.lower() == "all":
        warn_summary_block(f"Message {msg_num} uses reserved summary tag \"all\" as a begin tag.")
        return None
    return tag


def parse_summary_block_role(attrs: Dict[str, str], msg_num: int, tag: str) -> Optional[str]:
    role = attrs.get("role", "assistant").strip().lower()
    if not role:
        role = "assistant"
    if role not in SUMMARY_BLOCK_ROLES:
        warn_summary_block(f"Message {msg_num} has invalid summary role {role!r}.")
        return None
    if role == "system" and tag.lower() != "all":
        warn_summary_block(f"Message {msg_num} uses summary role \"system\" with non-all tag {tag!r}.")
        return None
    return role


def extract_summary_block_control(content: str, msg_num: int) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Extracts at most one summary control tag from a message.

    The containing message is always discarded when a summary tag is present.
    Normal text outside the control tag is intentionally ignored.
    """
    text = content or ""
    if not SUMMARY_BLOCK_ANY_RE.search(text):
        return None, False

    end_matches = list(SUMMARY_BLOCK_END_RE.finditer(text))
    if len(end_matches) > 1:
        warn_summary_block(f"Message {msg_num} contains multiple summary end tags.")
        return None, True
    if len(end_matches) == 1:
        match = end_matches[0]
        outside = text[:match.start()] + text[match.end():]
        if SUMMARY_BLOCK_ANY_RE.search(outside):
            warn_summary_block(f"Message {msg_num} contains multiple or malformed summary control tags.")
            return None, True

        attrs = parse_summary_block_attrs(match.group("attrs"))
        tag   = parse_summary_block_tag(attrs, msg_num, "end")
        if tag is None:
            return None, True
        role = parse_summary_block_role(attrs, msg_num, tag)
        if role is None:
            return None, True

        return {
            "kind" : "end",
            "tag"  : tag,
            "role" : role,
            "text" : match.group("body").strip(),
        }, True

    begin_matches = list(SUMMARY_BLOCK_BEG_RE.finditer(text))
    if len(begin_matches) > 1:
        warn_summary_block(f"Message {msg_num} contains multiple summary begin tags.")
        return None, True
    if len(begin_matches) == 1:
        match = begin_matches[0]
        outside = text[:match.start()] + text[match.end():]
        if SUMMARY_BLOCK_ANY_RE.search(outside):
            warn_summary_block(f"Message {msg_num} contains multiple or malformed summary control tags.")
            return None, True

        attrs = parse_summary_block_attrs(match.group("attrs"))
        tag   = parse_summary_block_tag(attrs, msg_num, "begin")
        if tag is None:
            return None, True

        return {
            "kind" : "begin",
            "tag"  : tag,
        }, True

    warn_summary_block(f"Message {msg_num} contains a malformed summary control tag.")
    return None, True


def build_summary_groups(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not summaries:
        return []

    intervals = sorted(summaries, key=lambda item: (item["start"], item["end"], item["ordinal"]))
    groups: List[Dict[str, Any]] = []

    for summary in intervals:
        if not groups or summary["start"] > groups[-1]["end"] + 1:
            groups.append({
                "start"     : summary["start"],
                "end"       : summary["end"],
                "summaries" : [summary],
            })
            continue

        groups[-1]["end"] = max(groups[-1]["end"], summary["end"])
        groups[-1]["summaries"].append(summary)

    for group in groups:
        group["summaries"].sort(key=lambda item: (item["end"], item["start"], item["ordinal"]))

    return groups


def apply_summary_blocks(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Removes summary control messages and optionally collapses covered ranges.

    Valid summary ranges are computed against the original normalized chat list.
    Overlapping ranges become one removed span, with their summaries inserted in
    closing-message order.
    """
    if not messages:
        return messages, ""

    open_starts : Dict[str, int]       = {}
    summaries   : List[Dict[str, Any]] = []
    control_indices = set()


    for idx, msg in enumerate(messages):
        control, should_discard = extract_summary_block_control(msg.get("content", ""), idx + 1)
        if should_discard:
            control_indices.add(idx)
        if control is None:
            continue

        tag = control["tag"]
        if control["kind"] == "begin":
            if tag in open_starts:
                warn_summary_block(f"Message {idx + 1} starts duplicate active summary tag {tag!r}.")
                continue
            open_starts[tag] = idx
            continue

        if tag.lower() == "all":
            summaries.append({
                "start"   : 0,
                "end"     : idx,
                "role"    : control["role"],
                "text"    : control["text"],
                "tag"     : tag,
                "ordinal" : len(summaries),
            })
            continue

        start_idx = open_starts.pop(tag, None)
        if start_idx is None:
            warn_summary_block(f"Message {idx + 1} closes summary tag {tag!r} without a matching begin tag.")
            continue

        summaries.append({
            "start"   : start_idx,
            "end"     : idx,
            "role"    : control["role"],
            "text"    : control["text"],
            "tag"     : tag,
            "ordinal" : len(summaries),
        })

    for tag, start_idx in sorted(open_starts.items(), key=lambda item: item[1]):
        warn_summary_block(f"Message {start_idx + 1} starts summary tag {tag!r} without a matching end tag.")

    if not cfg.summary_blocks_enabled:
        return [msg for idx, msg in enumerate(messages) if idx not in control_indices], ""

    groups = build_summary_groups(summaries)
    if not groups:
        return [msg for idx, msg in enumerate(messages) if idx not in control_indices], ""

    result: List[Dict[str, Any]] = []
    system_summary_text = ""
    idx = 0

    for group in groups:
        while idx < group["start"]:
            if idx not in control_indices:
                result.append(messages[idx])
            idx += 1

        for summary in group["summaries"]:
            if summary["role"] == "system":
                stripped = summary["text"].strip()
                if stripped:
                    system_summary_text += f"{stripped}\n\n"
                continue
            result.append({
                "role"    : summary["role"],
                "content" : summary["text"],
            })

        idx = group["end"] + 1

    while idx < len(messages):
        if idx not in control_indices:
            result.append(messages[idx])
        idx += 1

    return result, system_summary_text.strip()


def openai_stream_chunk(model_label_str: str, delta: Dict[str, Any], finish_reason: Optional[str] = None, usage: Optional[Dict[str, int]] = None, message_id: str = "claude") -> str:
    chunk: Dict[str, Any] = {
        "id"      : message_id,
        "object"  : "chat.completion.chunk",
        "created" : int(time.time()),
        "model"   : model_label_str,
        "choices" : [{
            "index"         : 0,
            "finish_reason" : finish_reason,
            "delta"         : delta,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


PERSONA_END_RE  = re.compile(r"</[^<>]*\bPersona>", re.IGNORECASE)
LOREBOOK_XML_RE = re.compile(r"<lorebook\b[^>]*>.*?</lorebook>", re.IGNORECASE | re.DOTALL)
def split_system_text(system_prompt: str) -> Tuple[List[str], str]:
    """
    Splits the joined system prompt into segments and an optional moved-to-end suffix.

    This is pure string surgery driven by the lorebook settings; how the segments
    are represented on the wire (Anthropic system blocks with cache markers, plain
    OpenAI system messages, ...) is up to the backend.

    When SPLIT_LOREBOOK=true, the prompt is split into:
        1. stable core definition
        2. dynamic lorebook / user-script suffix

    Split priority:
        1. After </summary>
        2. After </example_dialogs>
        3. After </UserPersona>
        4. After </Scenario>
        5. After the last </* Persona> marker
        6. Otherwise keep the whole system prompt as one segment

    When LOREBOOK_AT_END=true, the suffix is returned as plain text instead of being
    kept as a system segment. That moved suffix deliberately does not receive the old
    system/lorebook cache marker; it is handled later as an ordinary end-of-conversation item.

    When LOREBOOK_XML_AT_END=true, every <lorebook>...</lorebook> block is removed from the
    system prompt and appended after any other moved end-of-chat lorebook text.

    Returns:
        - system_segments: system prompt segments, or an empty list
        - lorebook_at_end_text: moved lorebook/suffix text, or an empty string
    """
    text = system_prompt.strip()

    segments: List[str] = []
    lorebook_at_end_text     = ""
    lorebook_xml_at_end_text = ""

    if text and cfg.lorebook_xml_at_end:
        matches = [match.group(0).strip() for match in LOREBOOK_XML_RE.finditer(text) if match.group(0).strip()]
        if matches:
            text = LOREBOOK_XML_RE.sub("", text).strip()
            lorebook_xml_at_end_text = "\n\n".join(matches)

    if text:
        if cfg.split_lorebook:
            split_at = -1

            for split_marker in ("</summary>", "</example_dialogs>", "</UserPersona>", "</Scenario>"):
                split_idx = text.find(split_marker)
                if split_idx != -1:
                    split_at = split_idx + len(split_marker)
                    break

            if split_at == -1:
                persona_matches = list(PERSONA_END_RE.finditer(text))
                if persona_matches:
                    split_at = persona_matches[-1].end()

            if split_at == -1 or split_at >= len(text):
                segments.append(text)
            else:
                before = text[:split_at].rstrip()
                after  = text[split_at:].strip()

                if before:
                    segments.append(before)

                if after:
                    if cfg.lorebook_at_end:
                        existing_text = lorebook_at_end_text.strip()
                        lorebook_at_end_text = f"{existing_text}\n\n{after}" if existing_text else after
                    else:
                        # Keep a clean visual/semantic separator between Scenario/Persona and suffix.
                        segments.append("\n\n" + after)
        else:
            segments.append(text)

    if lorebook_xml_at_end_text:
        existing_text = lorebook_at_end_text.strip()
        lorebook_at_end_text = f"{existing_text}\n\n{lorebook_xml_at_end_text}" if existing_text else lorebook_xml_at_end_text

    return segments, lorebook_at_end_text


def split_system_and_messages(raw_messages: Any) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Validates and normalizes OpenAI-style chat messages.

    Accepts untrusted request payload data.
    Returns:
        - system_prompt: joined system messages, or an empty string
        - chat_messages: list of normalized message dicts
        - system_summary_text: role="system" all-summary text to append to system, or an empty string

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

    chat_messages, system_summary_text = apply_summary_blocks(chat_messages)

    if thinking_preservation_enabled():
        # Mark only the last N assistant messages for signed-block rehydration.
        remaining = cfg.preserve_thinking_blocks
        for i in range(len(chat_messages) - 1, -1, -1):
            msg = chat_messages[i]
            if msg.get("role") != "assistant":
                continue
            if not msg.get("anthropic_thinking_blocks"):
                continue
            if remaining <= 0:
                break
            msg["send_anthropic_thinking_blocks"] = True
            remaining -= 1

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, chat_messages, system_summary_text


def capture_chat_snapshot(payload: Dict[str, Any], assistant_content: str, assistant_reasoning: str = "") -> None:
    global LATEST_CHAT_SNAPSHOT

    system_parts : List[str]            = []
    messages     : List[Dict[str, Any]] = []
    raw_messages = payload.get("messages", [])

    if isinstance(raw_messages, list):
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            raw_role = msg.get("role", "user")
            role     = raw_role if isinstance(raw_role, str) else "user"
            content  = content_to_plain_text(msg.get("content", ""))

            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
                continue
            if role not in ("user", "assistant"):
                role = "user"

            messages.append({"role": role, "content": content})

    assistant_message : Dict[str, Any] = {"role": "assistant", "content": assistant_content or ""}
    if assistant_reasoning:
        assistant_message["reasoning"] = assistant_reasoning
    messages.append(assistant_message)

    now         = time.time()
    exported_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int((now % 1)*1000):03d}Z"

    with LATEST_CHAT_LOCK:
        LATEST_CHAT_SNAPSHOT = {
            "app"          : "mini-chat",
            "version"      : 8,
            "exportedAt"   : exported_at,
            "systemPrompt" : "\n\n".join(system_parts),
            "messages"     : messages,
        }


NATURAL_DUMP_ESCAPE_RE  = re.compile(r"""\\(\\|n|t|r|"|')""")
NATURAL_DUMP_ESCAPE_MAP = {"\\": "\\", "n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'"}

def naturalize_dump_text(text: str) -> str:
    """
    Replaces literal JSON escape sequences (\\n, \\t, \\', ...) that leaked into
    message text with the natural characters they represent.
    """
    return NATURAL_DUMP_ESCAPE_RE.sub(lambda m: NATURAL_DUMP_ESCAPE_MAP[m.group(1)], text)


def snapshot_to_markdown(snapshot: Dict[str, Any]) -> str:
    """
    Renders a chat snapshot as human-readable markdown.
    """
    lines: List[str] = []

    lines.append(f"# Chat dump {snapshot.get('exportedAt', '')}".rstrip())
    lines.append("")

    lines.append("## System")
    system_text = naturalize_dump_text(str(snapshot.get("systemPrompt", "")))
    lines.append(system_text if system_text.strip() else "(empty)")
    lines.append("")

    lines.append("## Chat")
    lines.append("")

    messages = snapshot.get("messages", [])
    if isinstance(messages, list):
        for index, msg in enumerate(messages, start=1):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            lines.append(f"### Message {index} ({role})")

            reasoning = naturalize_dump_text(str(msg.get("reasoning", "")))
            if reasoning.strip():
                lines.append("\n".join(f"> {ln}" for ln in reasoning.splitlines()))
                lines.append("")

            lines.append(naturalize_dump_text(str(msg.get("content", ""))))
            lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def prepare_chat_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs every model-agnostic transform on the incoming OpenAI-style payload
    and returns the prepared request that backends build their API call from:
        messages, system_segments, system_summary_text, lorebook_at_end_text, max_tokens
    """
    system_prompt, chat_messages, system_summary_text = split_system_and_messages(payload.get("messages"))

    # JanitorAI sends '.' as the first fake user message (because all chats start with a user message).
    # We're replacing it with the <OOC>\nBegin the scenario.\n</OOC> version since it seems more natural.
    if chat_messages and chat_messages[0].get("role") == "user" and chat_messages[0].get("content", "").strip() == ".":
        chat_messages = [{"role": "user", "content": "<OOC>\nBegin the scenario.\n</OOC>"}] + chat_messages[1:]

    system_segments, lorebook_at_end_text = split_system_text(system_prompt)

    return {
        "messages"             : chat_messages,
        "system_segments"      : system_segments,
        "system_summary_text"  : system_summary_text,
        "lorebook_at_end_text" : lorebook_at_end_text,
        "max_tokens"           : int(payload.get("max_tokens", cfg.max_tokens)),
    }


def make_openai_non_stream_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps a backend generate_non_stream() result into an OpenAI chat completion.
    """
    message: Dict[str, Any] = {
        "role"    : "assistant",
        "content" : result["text"],
    }
    message.update(result["message_extra"])

    return {
        "id"      : result["id"],
        "object"  : "chat.completion",
        "created" : int(time.time()),
        "model"   : model_label(),
        "choices" : [
            {
                "index"         : 0,
                "finish_reason" : result["stop_reason"],
                "message"       : message,
            }
        ],
        "usage"   : result["usage"],
    }


def build_error_body(exc: Exception) -> Tuple[int, Dict[str, Any]]:
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

    body = anthropic_error_body(exc)
    if isinstance(body, dict):
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            message    = error_obj.get("message", message)
            error_type = error_obj.get("type", error_type)

    error_body = { "error": { "message": message, "type": error_type, "code": status_code } }
    return status_code, error_body


def make_error_response(exc: Exception, payload: Optional[Dict[str, Any]] = None) -> Response:
    status_code, error_body = build_error_body(exc)
    print_anthropic_error(exc)

    log_body   = { "error": error_body, "request": payload, "traceback": traceback.format_exc() }
    write_error_log(log_body)

    return Response(json.dumps(error_body, ensure_ascii=False), status=status_code, content_type="application/json")


# Generation
def generate_non_stream(payload: Dict[str, Any]) -> Dict[str, Any]:
    prepared = prepare_chat_request(payload)
    result   = active_backend().generate_non_stream(prepared)

    response = make_openai_non_stream_response(result)
    capture_chat_snapshot(payload, response["choices"][0]["message"].get("content", ""))
    return response


def generate_stream(payload: Dict[str, Any]):
    try:
        prepared = prepare_chat_request(payload)
        label    = model_label()

        for kind, data in active_backend().generate_stream(prepared):
            if kind == "reasoning":
                # Stream reasoning_content only; Janitor already renders it as <think> text.
                yield openai_stream_chunk(label, {
                    "role"              : "assistant",
                    "reasoning_content" : data,
                })
            elif kind == "text":
                yield openai_stream_chunk(label, {
                    "role"    : "assistant",
                    "content" : data,
                })
            elif kind == "final":
                capture_chat_snapshot(payload, data["snapshot_text"], data["snapshot_reasoning"])
                yield openai_stream_chunk(
                    label,
                    {},
                    finish_reason=data["stop_reason"],
                    usage=data["usage"],
                    message_id=data["id"],
                )

    except Exception as exc:
        _, error_body = build_error_body(exc)
        print_anthropic_error(exc)
        log_body = { "error": error_body, "request": payload, "traceback": traceback.format_exc() }
        write_error_log(log_body)
        yield "data: " + json.dumps(error_body, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    yield "data: [DONE]\n\n"

def handle_chat_completion():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return Response(json.dumps({"error": {"message": "Invalid JSON body."}}), status=400, content_type="application/json")

    try:
        stream = bool(payload.get("stream", False))

        if stream:
            return Response(
                stream_with_context(generate_stream(payload)),
                content_type="text/event-stream",
                headers={
                    "Cache-Control"     : "no-cache",
                    "X-Accel-Buffering" : "no",
                },
            )

        response = generate_non_stream(payload)
        return jsonify(response)

    except Exception as exc:
        return make_error_response(exc, payload)


@app.route("/", methods=["GET"])
def running():
    session = session_cost_snapshot()

    return jsonify(
        {
            "status"        : "ok",
            "backend"       : cfg.backend,
            "model"         : cfg.model,
            "prompt_cache"  : cfg.cache_en,
            "cache"         : {
                "cache_en"            : cfg.cache_en,
                "cache_system"         : cfg.cache_system,
                "cache_system_ttl"     : cfg.cache_system_ttl,
                "split_lorebook"       : cfg.split_lorebook,
                "lorebook_at_end"      : cfg.lorebook_at_end,
                "lorebook_xml_at_end"  : cfg.lorebook_xml_at_end,
                "cache_manual_ttl"     : cfg.cache_manual_ttl,
                "cache_manual_msg"     : cfg.cache_manual_msg,
                "cache_auto_ttl"       : cfg.cache_auto_ttl,
                "cache_auto_msg"       : cfg.cache_auto_msg,
            },
            "cost_tracking" : {
                "model_cost_family"                                : cfg.model_cost_family,
                "input_token_cost_usd"                             : cfg.input_token_cost_usd,
                "output_token_cost_usd"                            : cfg.output_token_cost_usd,
                "cache_write_5m_cost_usd"                          : cfg.cache_write_5m_cost_usd,
                "cache_write_1h_cost_usd"                          : cfg.cache_write_1h_cost_usd,
                "cache_read_cost_usd"                              : cfg.cache_read_cost_usd,
                "session_total_spent_usd"                          : session["total_spent_usd"],
                "session_total_input_token_cost_usd"               : session["input_cost_usd"],
                "session_total_output_token_cost_usd"              : session["output_cost_usd"],
                "session_total_input_tokens"                       : session["input_tokens"],
                "session_total_output_tokens"                      : session["output_tokens"],
                "session_average_input_token_cost_usd_per_million" : session["average_input_cost_usd"]*1_000_000,
                "session_cache_net_cost_usd"                       : session["cache_net_cost_usd"],
            },
            "thinking" : {
                "thinking_enabled"          : cfg.thinking_enabled,
                "adaptive_thinking"         : cfg.use_adaptive,
                "thinking_budget"           : cfg.thinking_budget,
                "thinking_effort"           : cfg.thinking_effort,
                "preserve_thinking_blocks"  : "inf" if cfg.preserve_thinking_blocks == UINT64_MAX else str(cfg.preserve_thinking_blocks),
            }
        }
    )


@app.route("/chat/snapshot"   , methods=["GET"])
@app.route("/v1/chat/snapshot", methods=["GET"])
def chat_snapshot():
    with LATEST_CHAT_LOCK:
        snapshot = LATEST_CHAT_SNAPSHOT

    if not snapshot:
        return Response(json.dumps({"error": "No chat snapshot captured yet."}), status=404, content_type="application/json")

    return Response(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        content_type="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="mini-chat-latest.json"'},
    )


@app.route("/"                   , methods=["POST"])
@app.route("/chat/completions"   , methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def short_baseurl() : return handle_chat_completion()
def baseurl()       : return handle_chat_completion()
def v1_baseurl()    : return handle_chat_completion()


if __name__ == "__main__":
    load_dotenv()
    cfg.reload_from_env()
    refresh_anthropic_models(cfg.anthropic_api_key, cfg.model_list_timeout_seconds)
    open_ai.refresh_openai_models(cfg.model_list_timeout_seconds)
    # MODEL may name either an Anthropic model or a provider model ("glm-4.7" or "glm/glm-4.7").
    if not open_ai.apply_model_by_id(cfg.model):
        cfg.find_cfg(claude.ANTHROPIC_MODELS)

    print("Starting Claude proxy")
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
