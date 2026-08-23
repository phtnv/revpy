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
from werkzeug.exceptions import RequestEntityTooLarge

import image_orchestrator
import v1_messages
import providers
import v1_chat_completions
import v1_images
import v1_responses

from common import (
    IMAGE_BACKGROUNDS,
    IMAGE_FORMATS,
    IMAGE_QUALITIES,
    IMAGE_RESPONSE_FORMATS,
    IMAGE_SIZE_EDGE_MULTIPLE,
    IMAGE_SIZE_MAX_ASPECT,
    IMAGE_SIZE_MAX_EDGE,
    IMAGE_SIZE_MAX_PIXELS,
    IMAGE_SIZE_MIN_PIXELS,
    DISABLE_VALUES,
    ENABLE_VALUES,
    INF_VALUES,
    PREFILL_MODES,
    UINT64_MAX,
    cfg,
    check_proxy_key,
    content_to_plain_text,
    error_body,
    fmt_usd,
    image_cost_snapshot,
    print_error,
    session_cost_snapshot,
)

LATEST_CHAT_SNAPSHOT : Dict[str, Any] = {}
LATEST_CHAT_LOCK                      = threading.Lock()


# The backend module per wire protocol. All three expose the five entry points
# dispatched through here: generate_non_stream, generate_stream, after_model_switch,
# resolve_thinking and print_think_status. How each builds its request is its own business.
BACKENDS = {
    "messages"  : v1_messages,
    "chat"      : v1_chat_completions,
    "responses" : v1_responses,
}


def active_backend():
    """
    The backend module serving requests: the one for the protocol the selected model's
    provider was declared under (see providers.api_style).
    """
    return BACKENDS[providers.api_style()]


def model_label() -> str:
    return f"{cfg.backend}/{cfg.model}"

# Flask app
app = Flask(__name__)
CORS(app)


@app.before_request
def enforce_request_size() -> None:
    """
    Caps the body Flask will buffer. Without this any POST is read into memory whole,
    which uploaded image edits turn from a theoretical concern into a real one -- this
    proxy is meant to sit behind a public tunnel. Applied per request rather than once at
    startup so 'reload' can change it.
    """
    app.config["MAX_CONTENT_LENGTH"] = cfg.request_max_bytes


@app.errorhandler(413)
def request_too_large(exc: Exception):
    """A refused upload should say what the limit is, not just fail."""
    return Response(
        json.dumps({"error": {
            "message": f"Request body exceeds REQUEST_MAX_BYTES ({cfg.request_max_bytes:,} bytes).",
            "type"   : "request_too_large",
            "code"   : 413,
        }}),
        status=413,
        content_type="application/json",
    )

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

    providers.refresh_models(cfg.model_list_timeout_seconds)

    # Image generation is configured independently of the text model, so it is resolved
    # again here rather than riding along with the model switch below.
    v1_images.resolve_image_config()
    v1_images.refresh_image_models(cfg.model_list_timeout_seconds)
    # Idempotent: arms the poller if this reload switched it on, and does nothing if one
    # is already running (it re-reads the interval by itself).
    v1_images.start_batch_poller()

    print("Reloaded runtime configuration from .env.")
    print("HOST and PORT were not changed; restart the process to change bind address.")


def finish_model_switch(switched: bool) -> None:
    """
    Runs the newly selected model through its backend's post-switch hook, which
    validates the shared settings against it and reports how they land.

    providers.apply_model() does not do this itself: the registry serves all three wire
    modules and cannot import one to ask without a cycle. Nothing runs when no switch
    happened, which would report against a stale model.
    """
    if switched:
        active_backend().after_model_switch()


def cli_refresh_models() -> None:
    providers.refresh_models(cfg.model_list_timeout_seconds)
    finish_model_switch(providers.apply_model_by_id(f"{cfg.backend}/{cfg.model}"))



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

CLI_CMD_IMAGE_INFO = """\
  image command. Alias: img, i.
    image                     Show image generation status.
    image gen <prompt>        Generate an image now, using the configured defaults.
      Alias: g, generate
    image edit ...            Reference images for editing. 'image edit help' for more.
      Alias: e
    image model               List the image models of the image provider.
    image model <uint>        Select an image model. Does not change the chat model.
    image model refresh       Request the image model list again.
      Alias: r
    image batch <prompt>      Submit a one-request image batch.
    image batch get <uint>    Retrieve a batch by its number and save its images.
      Alias: g
    image batch list          List the batches submitted from this output directory.
      Alias: l
    image batch poll          Check every unsettled batch now, instead of waiting.
      Alias: p
    image cost                Show image session spending.
    image help                Display this message.
      Alias: ?
"""


def cli_print_image_cost() -> None:
    images = image_cost_snapshot()
    text   = session_cost_snapshot()
    print(f"  Text model cost:      {fmt_usd(text['total_spent_usd'])}")
    print(f"  Image immediate cost: {fmt_usd(images['immediate_cost_usd'])} ({images['images']} image(s))")
    print(f"  Image batch cost:     {fmt_usd(images['batch_cost_usd'])} ({images['batch_images']} image(s))")
    print(f"  Combined session:     {fmt_usd(text['total_spent_usd'] + images['total_cost_usd'])}")
    if images["contains_estimates"]:
        print("  Note: some image costs are estimates (the provider reported no usage).")


def cli_generate_image(prompt: str, batch: bool) -> None:
    """Runs one CLI image request. Errors are reported, never raised into the CLI loop."""
    try:
        req = v1_images.build_request({"prompt": prompt, "batch": batch}, source="cli")
        if batch:
            result = v1_images.submit_image_batch([req])
            print(f"Batch #{result.number} is {result.status}.")
            return
        result = v1_images.generate_image(req)
        for image in result.images:
            print(f"Saved {image.image_id} -> {image.path}")
        print(f"Cost: {fmt_usd(result.cost_usd)}{' (estimated)' if result.estimated else ''}")
    except Exception as exc:
        print_error(exc)
        write_error_log({"error": str(exc), "traceback": traceback.format_exc()})


def cli_retrieve_batch(token: str) -> None:
    """Retrieves a batch by its short number (or by a raw provider id)."""
    try:
        batch_id = v1_images.resolve_batch_id(token)
        result   = v1_images.retrieve_image_batch(batch_id)
        label    = f"#{result.number}" if result.number else result.batch_id
        print(f"Batch {label} is {result.status}. {result.counts}")
        for image in result.images:
            print(f"Saved {image.image_id} -> {image.path}")
        if result.images:
            print(f"Cost: {fmt_usd(result.cost_usd)}")
        for error in result.errors:
            print(f"  {error}")
    except Exception as exc:
        print_error(exc)
        write_error_log({"error": str(exc), "traceback": traceback.format_exc()})


CLI_CMD_IMAGE_EDIT_INFO = """\
  image edit command. Alias: e.
    image edit                  Show the reference image slots.
    image edit set <uint> <path>  Point slot <uint> at a file.
      Alias: s
    image edit add <path>       Fill the next free slot.
      Alias: a
    image edit mask <path>      Set the mask (PNG with alpha). 'mask clear' to remove.
      Alias: m
    image edit clear [uint]     Clear one slot, or every slot.
      Alias: c
    image edit <prompt>         Edit using the filled slots.
"""


def cli_show_slots() -> None:
    slots = v1_images.read_slots()
    if not slots:
        print("No reference image slots are filled. Use 'image edit set 1 <path>'.")
        return

    for number in v1_images.numbered_slots(slots):
        try:
            reference = v1_images.resolve_slot(number)
            geometry  = f"{reference.width}x{reference.height}" if reference.width else "?"
            print(f"  {number:>3}  {reference.format:<5} {geometry:>11}  {reference.size:>10,}B  {reference.path}")
        except Exception as exc:
            # Slots are re-validated on use, so a file that has since vanished or been
            # replaced shows up here rather than surprising the next chat turn.
            print(f"  {number:>3}  UNUSABLE  {slots[str(number)]}  ({exc})")

    if slots.get(v1_images.MASK_KEY):
        print(f"  mask  {slots[v1_images.MASK_KEY]}")


def cli_set_slot(number: int, path: str) -> None:
    try:
        reference = v1_images.set_slot(number, path)
        geometry  = f" {reference.width}x{reference.height}" if reference.width else ""
        print(f"Slot {number} -> {reference.path} ({reference.format},{geometry} {reference.size:,}B)")
    except Exception as exc:
        print_error(exc)


def cli_clear_slots(number: Optional[int]) -> None:
    try:
        cleared = v1_images.clear_slot(number)
        if number is None : print(f"Cleared {cleared} slot(s).")
        elif cleared      : print(f"Cleared slot {number}.")
        else              : print(f"Slot {number} was already empty.")
    except Exception as exc:
        print_error(exc)


def cli_edit_image(prompt: str) -> None:
    """Edits using whatever the slots hold, which is what 'edit: true' means in chat too."""
    try:
        request = v1_images.build_request({"prompt": prompt, "edit": True}, source="cli")
        print(f"Editing with {len(request.images)} reference image(s).")
        result = v1_images.generate_image(request)
        for image in result.images:
            print(f"Saved {image.image_id} -> {image.path}")
        print(f"Cost: {fmt_usd(result.cost_usd)}{' (estimated)' if result.estimated else ''}")
    except Exception as exc:
        print_error(exc)
        write_error_log({"error": str(exc), "traceback": traceback.format_exc()})


def handle_image_edit_command(line: str, parts: List[str]) -> None:
    """The 'image edit' subgroup. Paths are accepted here because the console is trusted."""
    parts_l = len(parts)

    if parts_l < 3:
        cli_show_slots()
        return

    arg2 = parts[2].lower()
    if arg2 in {"?", "help"}:
        print(CLI_CMD_IMAGE_EDIT_INFO)
        return

    if arg2 in {"c", "clear"}:
        if parts_l < 4:
            cli_clear_slots(None)
            return
        try: cli_clear_slots(int(parts[3]))
        except ValueError: print(CLI_CMD_IMAGE_EDIT_INFO)
        return

    if arg2 in {"s", "set"}:
        # 'image edit set 2 /a b/c.png' -- the path is the rest of the line, unsplit,
        # so a filename containing spaces survives.
        split_line = line.split(maxsplit=4)
        if len(split_line) < 5:
            print(CLI_CMD_IMAGE_EDIT_INFO)
            return
        try: number = int(split_line[3])
        except ValueError:
            print(CLI_CMD_IMAGE_EDIT_INFO)
            return
        cli_set_slot(number, split_line[4].strip().strip('"').strip("'"))
        return

    if arg2 in {"a", "add"}:
        split_line = line.split(maxsplit=3)
        if len(split_line) < 4:
            print(CLI_CMD_IMAGE_EDIT_INFO)
            return
        cli_set_slot(v1_images.next_free_slot(), split_line[3].strip().strip('"').strip("'"))
        return

    if arg2 in {"m", "mask"}:
        split_line = line.split(maxsplit=3)
        if len(split_line) < 4:
            print(CLI_CMD_IMAGE_EDIT_INFO)
            return
        value = split_line[3].strip().strip('"').strip("'")
        try:
            if value.lower() in {"clear", "none", "off"}:
                with v1_images.SLOT_LOCK:
                    slots = v1_images.read_slots()
                    slots.pop("mask", None)
                    v1_images.write_slots(slots)
                print("Mask cleared.")
                return
            reference = v1_images.load_reference(value, from_prompt=False)
            v1_images.validate_mask(reference, v1_images.filled_slots())
            with v1_images.SLOT_LOCK:
                slots = v1_images.read_slots()
                slots["mask"] = reference.path
                v1_images.write_slots(slots)
            print(f"Mask -> {reference.path} ({reference.width}x{reference.height})")
        except Exception as exc:
            print_error(exc)
        return

    # Anything else is the prompt for an edit against the filled slots.
    split_line = line.split(maxsplit=2)
    cli_edit_image(split_line[2])


def cli_poll_batches() -> None:
    """Runs the poller's work on demand, for when waiting out the interval is not the point."""
    pending = v1_images.pending_batches()
    if not pending:
        print("No image batches are waiting.")
        return

    print(f"Checking {len(pending)} batch(es): {', '.join('#' + str(number) for number, _ in pending)}")
    try:
        settled = v1_images.poll_batches_once()
    except Exception as exc:
        print_error(exc)
        write_error_log({"error": str(exc), "traceback": traceback.format_exc()})
        return

    if not settled:
        print("None of them have finished yet.")
        return
    for result in settled:
        print(f"  #{result.number} {result.status}, {len(result.images)} image(s) saved.")


def cli_list_batches() -> None:
    state = v1_images.read_batch_state()
    if not state:
        print("No image batches have been submitted from this output directory.")
        return

    rows = sorted(state.items(), key=lambda item: int(item[1].get("number") or 0))
    for batch_id, entry in rows:
        number = entry.get("number") or "?"
        status = entry.get("final_status", "") if entry.get("retrieved") else "pending"
        images = len(entry.get("images") or [])
        images_cell = f"{images} image(s)" if entry.get("retrieved") else ""
        print(f"  {str(number):>3}  {status or 'settled':<10}  {entry.get('model', ''):<16}  submitted {entry.get('submitted_at', '')}  {images_cell}")

    print("Retrieve one with 'image batch get <number>'.")


def handle_image_command(line: str, parts: List[str]) -> None:
    """The 'image' CLI command group. Nothing here touches the active chat model."""
    parts_l = len(parts)

    if parts_l < 2:
        cfg.print_image_status()
        return

    arg1 = parts[1].lower()
    if arg1 in {"?", "help"}:
        print(CLI_CMD_IMAGE_INFO)
        return
    if arg1 == "cost":
        cli_print_image_cost()
        return

    if arg1 in {"g", "gen", "generate"}:
        split_line = line.split(maxsplit=2)
        if len(split_line) < 3:
            print(CLI_CMD_IMAGE_INFO)
            return
        cli_generate_image(split_line[2], batch=False)
        return

    if arg1 in {"e", "edit"}:
        handle_image_edit_command(line, parts)
        return

    if arg1 in {"m", "model", "models"}:
        if parts_l < 3:
            v1_images.print_image_model_list()
            return
        arg2 = parts[2].lower()
        if arg2 in {"r", "refresh"}:
            v1_images.refresh_image_models(cfg.model_list_timeout_seconds)
            return
        try: index = int(arg2)
        except ValueError:
            print(CLI_CMD_IMAGE_INFO)
            return
        v1_images.select_image_model_by_number(index)
        return

    if arg1 in {"b", "batch"}:
        if parts_l < 3:
            print(CLI_CMD_IMAGE_INFO)
            return
        arg2 = parts[2].lower()
        if arg2 in {"l", "list"}:
            cli_list_batches()
            return
        if arg2 in {"p", "poll"}:
            cli_poll_batches()
            return
        if arg2 in {"g", "get"}:
            if parts_l < 4:
                print(CLI_CMD_IMAGE_INFO)
                return
            cli_retrieve_batch(parts[3])
            return
        split_line = line.split(maxsplit=2)
        cli_generate_image(split_line[2], batch=True)
        return

    print(CLI_CMD_IMAGE_INFO)


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
                if providers.api_style() != "messages":
                    print(f"Cache markers are an Anthropic-protocol feature. Backend '{cfg.backend}' has no explicit cache control (use EXTRA_BODY if the provider supports one).")
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
                if parts_l < 2:
                    active_backend().print_think_status()
                    continue

                arg1 = parts[1].lower()
                if arg1 in {"?", "help"} : print(CLI_CMD_THINK_INFO); continue
                if arg1 in DISABLE_VALUES | ENABLE_VALUES:
                    cfg.thinking_enabled = arg1 in ENABLE_VALUES
                    active_backend().resolve_thinking()
                    continue

                if parts_l < 3:
                    print(CLI_CMD_THINK_INFO)
                    continue
                arg2 = parts[2].lower()
                if arg1 in {"e", "effort"}:
                    # Report the new effort in the active backend's own terms: providers
                    # support different subsets, so the level sent is often not the one asked for.
                    if cfg.set_think_effort(arg2):
                        active_backend().resolve_thinking()
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
                    providers.print_model_list()
                    continue

                arg1 = parts[1].lower()
                if parts_l == 2:
                    try: model_id = int(arg1)
                    except Exception: pass
                    else:
                        finish_model_switch(providers.select_model_by_number(model_id))
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
                        providers.print_model_info(model_id)
                        continue
                print(CLI_CMD_MODEL_INFO)
                continue

            if cmd in {"i", "img", "image", "images"}:
                handle_image_command(line, parts)
                continue

            if cmd == "status":
                cfg.print_status()
                continue

            if cmd in {"d", "dump"}:
                fmt      = "json"
                raw_dump = False
                path     = ""
                opt_count = 0
                for tok in parts[1:]:
                    tok_l = tok.lower()
                    if   tok_l in {"n", "nat", "natural", "md", "markdown"} : fmt = "natural"
                    elif tok_l in {"j", "json"}                             : fmt = "json"
                    elif tok_l in {"r", "raw"}                              : raw_dump = True
                    else:
                        break
                    opt_count += 1
                split_line = line.split(maxsplit=1 + opt_count)
                if len(split_line) > 1 + opt_count:
                    path = split_line[1 + opt_count]
                if not path:
                    suffix = "_raw" if raw_dump else ""
                    path   = f"chat_snapshot{suffix}.md" if fmt == "natural" else f"chat_snapshot{suffix}.json"
                with LATEST_CHAT_LOCK:
                    snapshot = LATEST_CHAT_SNAPSHOT
                if not snapshot:
                    print("No chat snapshot captured yet.")
                    continue
                if not raw_dump:
                    snapshot = postprocess_chat_snapshot(snapshot)
                with open(path, "w", encoding="utf-8") as f:
                    if fmt == "natural":
                        f.write(snapshot_to_markdown(snapshot))
                    else:
                        json.dump(snapshot, f, indent=2, ensure_ascii=False)
                        f.write("\n")
                print(f"Wrote latest chat snapshot to {path} ({fmt}{', raw' if raw_dump else ''}).")
                continue

            if cmd in {"help", "?"}:
                print()
                print("Commands:")
                print(CLI_CMD_CACHE_INFO)
                print(CLI_CMD_IMAGE_INFO)
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
                print("    dump raw      Dump without summary block substitutions. Combines with a format, e.g. 'dump raw md'.")
                print("      Alias: r")
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
        content, thinking_blocks = v1_messages.extract_hidden_thinking_envelopes(content)

        msg_obj: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and v1_messages.thinking_preservation_enabled() and thinking_blocks:
            msg_obj["anthropic_thinking_blocks"] = thinking_blocks
        chat_messages.append(msg_obj)

    chat_messages, system_summary_text = apply_summary_blocks(chat_messages)

    if v1_messages.thinking_preservation_enabled():
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


def postprocess_chat_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies summary block substitutions to a raw snapshot, mirroring what
    split_system_and_messages() does to the outgoing request: control messages
    are removed, covered ranges collapse into their summaries, and role="system"
    summaries are appended after the system prompt.
    """
    messages = snapshot.get("messages", [])
    if not isinstance(messages, list):
        return snapshot

    chat = [dict(msg) for msg in messages if isinstance(msg, dict)]
    chat, system_summary_text = apply_summary_blocks(chat)

    if not chat or chat[0].get("role") != "user":
        chat = [{"role": "user", "content": OOC_SCENARIO_START}] + chat

    processed = dict(snapshot)
    processed["messages"] = chat
    if system_summary_text:
        system_prompt = str(processed.get("systemPrompt", "")).strip()
        processed["systemPrompt"] = f"{system_prompt}\n\n{system_summary_text}".strip()
    return processed


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


OOC_SCENARIO_START = "<OOC>\nBegin the scenario.\n</OOC>"

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
        chat_messages = [{"role": "user", "content": OOC_SCENARIO_START}] + chat_messages[1:]

    # Summary substitutions can leave the chat starting with an assistant message (or empty);
    # ensure the conversation always opens with a user message.
    if not chat_messages or chat_messages[0].get("role") != "user":
        chat_messages = [{"role": "user", "content": OOC_SCENARIO_START}] + chat_messages

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

    upstream_body = error_body(exc)
    if isinstance(upstream_body, dict):
        error_obj = upstream_body.get("error", {})
        if isinstance(error_obj, dict):
            message    = error_obj.get("message", message)
            error_type = error_obj.get("type", error_type)

    client_body = { "error": { "message": message, "type": error_type, "code": status_code } }
    return status_code, client_body


def make_error_response(exc: Exception, payload: Optional[Dict[str, Any]] = None) -> Response:
    status_code, client_body = build_error_body(exc)
    print_error(exc)

    log_body   = { "error": client_body, "request": payload, "traceback": traceback.format_exc() }
    write_error_log(log_body)

    return Response(json.dumps(client_body, ensure_ascii=False), status=status_code, content_type="application/json")


# Image generation
IMAGE_NOTE_MAX_CHARS = 240
HTML_TAG_RE          = re.compile(r"<[^>]+>")
HTML_DOC_RE          = re.compile(r"<!DOCTYPE\s+html|<html[\s>]|<body[\s>]", re.IGNORECASE)

def image_note_message(message: str) -> str:
    """
    An upstream error message fit to appear inside a chat reply.

    A gateway failure (Cloudflare's 520 page, say) arrives as HTML rather than JSON, and
    error_from_response falls back to the first 2000 characters of the body. Relaying that
    verbatim buries the model's actual reply under a wall of markup, so it is stripped to
    one short line here. The full body still reaches the error log.

    Tags are only stripped from something that is actually a markup document: the proxy's
    own messages use angle brackets for placeholders ("img edit set 1 <path>"), and an
    indiscriminate strip silently deletes the most useful part of the advice.
    """
    text = str(message or "")
    if HTML_DOC_RE.search(text):
        text = HTML_TAG_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > IMAGE_NOTE_MAX_CHARS:
        text = text[:IMAGE_NOTE_MAX_CHARS].rstrip() + "…"
    return text or "unknown error"


def run_image_requests(extraction: image_orchestrator.Extraction) -> str:
    """
    Runs the image requests a user turn asked for and returns the text to append to the
    reply. A failure here never costs the conversation its turn: the error is logged in
    full and reported inline, so the model's prose still reaches the client.
    """
    lines: List[str] = []

    for req in extraction.requests:
        try:
            if req.batch:
                batch = v1_images.submit_image_batch([req])
                if cfg.image_batch_auto_poll:
                    lines.append(f"[Image batch #{batch.number} submitted — it will be saved automatically when it completes]")
                else:
                    lines.append(f"[Image batch #{batch.number} submitted — retrieve with 'image batch get {batch.number}']")
                continue
            result = v1_images.generate_image(req)
            lines.append(v1_images.reference_line(result))
        except Exception as exc:
            _, client_body = build_error_body(exc)
            print_error(exc)
            write_error_log({"error": client_body, "image_request": req.prompt[:200], "traceback": traceback.format_exc()})
            lines.append(f"[Image generation failed: {image_note_message(client_body['error']['message'])}]")

    for error in extraction.errors:
        lines.append(f"[Image request rejected: {image_note_message(error)}]")

    return "\n".join(lines)


def make_image_only_response(note: str) -> Dict[str, Any]:
    """
    The chat completion for a turn that was nothing but image requests. No text backend
    was called, so the model label reports the image model that actually did the work.
    """
    return {
        "id"      : f"imggen-{int(time.time())}",
        "object"  : "chat.completion",
        "created" : int(time.time()),
        "model"   : f"{cfg.image_provider}/{cfg.image_model}",
        "choices" : [{
            "index"         : 0,
            "finish_reason" : "stop",
            "message"       : {"role": "assistant", "content": note},
        }],
        "usage"   : {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# Generation
def generate_non_stream(payload: Dict[str, Any]) -> Dict[str, Any]:
    extraction = image_orchestrator.extract(payload)
    if extraction.found:
        payload = {**payload, "messages": extraction.messages}

    # An image-only turn never reaches a text backend.
    if extraction.found and not extraction.needs_text:
        note     = run_image_requests(extraction)
        response = make_image_only_response(note)
        capture_chat_snapshot(payload, note)
        return response

    prepared = prepare_chat_request(payload)
    result   = active_backend().generate_non_stream(prepared)

    response = make_openai_non_stream_response(result)

    # Images run after the reply, so a slow generation does not delay the prose.
    if extraction.found:
        note = run_image_requests(extraction)
        if note:
            message = response["choices"][0]["message"]
            message["content"] = f"{message.get('content', '')}\n\n{note}".strip()

    capture_chat_snapshot(payload, response["choices"][0]["message"].get("content", ""))
    return response


def generate_stream(payload: Dict[str, Any]):
    try:
        extraction = image_orchestrator.extract(payload)
        if extraction.found:
            payload = {**payload, "messages": extraction.messages}

        if extraction.found and not extraction.needs_text:
            label = f"{cfg.image_provider}/{cfg.image_model}"
            note  = run_image_requests(extraction)
            capture_chat_snapshot(payload, note)
            yield openai_stream_chunk(label, {"role": "assistant", "content": note})
            yield openai_stream_chunk(label, {}, finish_reason="stop", message_id=f"imggen-{int(time.time())}")
            yield "data: [DONE]\n\n"
            return

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
                # Generated after the prose has streamed, so the reply is not held back
                # for the seconds an image takes, then appended as one last text delta.
                note = run_image_requests(extraction) if extraction.found else ""
                if note:
                    yield openai_stream_chunk(label, {"role": "assistant", "content": f"\n\n{note}"})

                snapshot_text = data["snapshot_text"]
                if note:
                    snapshot_text = f"{snapshot_text}\n\n{note}".strip()
                capture_chat_snapshot(payload, snapshot_text, data["snapshot_reasoning"])

                yield openai_stream_chunk(
                    label,
                    {},
                    finish_reason=data["stop_reason"],
                    usage=data["usage"],
                    message_id=data["id"],
                )

    except Exception as exc:
        _, client_body = build_error_body(exc)
        print_error(exc)
        log_body = { "error": client_body, "request": payload, "traceback": traceback.format_exc() }
        write_error_log(log_body)
        yield "data: " + json.dumps(client_body, ensure_ascii=False) + "\n\n"
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
    images  = image_cost_snapshot()

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
                # Text and image spending are tracked apart, then added up here.
                "session_image_spent_usd"                         : images["total_cost_usd"],
                "session_combined_spent_usd"                      : session["total_spent_usd"] + images["total_cost_usd"],
            },
            "images" : {
                "enabled"              : cfg.image_enabled,
                "chat_trigger_enabled" : cfg.image_chat_enabled,
                "request_tag"          : cfg.image_request_tag,
                "provider"             : cfg.image_provider,
                "model"                : cfg.image_model,
                "output_dir"           : cfg.image_output_dir,
                # The configured value is routinely relative, and relative to *this*
                # process's working directory, which a client cannot know. A local app
                # browsing the output folder needs the resolved one.
                "output_dir_abs"       : os.path.abspath(cfg.image_output_dir),
                "manifest_file"        : v1_images.MANIFEST_FILE,
                "cost_family"          : cfg.image_cost_family,
                "batch_auto_poll"      : cfg.image_batch_auto_poll,
                "batch_poll_seconds"   : cfg.image_batch_poll_seconds,
                "defaults"             : {
                    "size"       : cfg.image_default_size,
                    "quality"    : cfg.image_default_quality,
                    "format"     : cfg.image_default_format,
                    "background" : cfg.image_default_background,
                    "n"          : cfg.image_default_n,
                    "batch"      : cfg.image_default_batch,
                },
                # Enough for a client to reject a bad request before sending it, rather
                # than learning the rules one 400 at a time. Sizes are a constraint set
                # rather than a list because validate_size() checks bounds, not an enum.
                "limits"               : {
                    "max_n"            : cfg.image_max_n,
                    "max_prompt_chars" : cfg.image_max_prompt_chars,
                    "edit_enabled"     : cfg.image_edit_enabled,
                    "edit_max_images"  : cfg.image_edit_max_images,
                    "edit_max_bytes"   : cfg.image_edit_max_bytes,
                    "request_max_bytes": cfg.request_max_bytes,
                    "size"             : {
                        "edge_multiple" : IMAGE_SIZE_EDGE_MULTIPLE,
                        "max_edge"      : IMAGE_SIZE_MAX_EDGE,
                        "max_aspect"    : IMAGE_SIZE_MAX_ASPECT,
                        "min_pixels"    : IMAGE_SIZE_MIN_PIXELS,
                        "max_pixels"    : IMAGE_SIZE_MAX_PIXELS,
                    },
                },
                "options"              : {
                    "qualities"   : sorted(IMAGE_QUALITIES),
                    "formats"     : sorted(IMAGE_FORMATS),
                    "backgrounds" : sorted(IMAGE_BACKGROUNDS),
                },
                "session"              : images,
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

    raw_arg  = request.args.get("raw")
    raw_dump = raw_arg is not None and raw_arg.strip().lower() not in DISABLE_VALUES
    if not raw_dump:
        snapshot = postprocess_chat_snapshot(snapshot)

    filename = "mini-chat-latest-raw.json" if raw_dump else "mini-chat-latest.json"
    return Response(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        content_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Image HTTP surface: multipart uploads client bytes; JSON names slots, paths or file ids.
# Direct endpoints default to b64_json for OpenAI-client compatibility.
IMAGE_UPLOAD_FIELDS = ("image", "image[]", "image[0]")


def resolve_response_format(requested: Any) -> str:
    if requested:
        value = str(requested).strip().lower()
        if value not in IMAGE_RESPONSE_FORMATS:
            abort(400, description=f"response_format must be one of {sorted(IMAGE_RESPONSE_FORMATS)}, got {requested!r}.")
        return value
    return cfg.image_response_format


def make_image_response(result: v1_images.ImageResult, response_format: str = "path") -> Dict[str, Any]:
    """
    Direct image endpoint reply. 'path' is metadata-only; 'b64_json' also includes bytes.
    """
    usage: Dict[str, Any] = {"estimated_cost_usd": round(result.cost_usd, 6), "cost_is_estimate": result.estimated}
    if result.usage.get("reported"):
        usage.update({
            "text_input_tokens"   : result.usage["text_input"],
            "image_input_tokens"  : result.usage["image_input"],
            "image_output_tokens" : result.usage["output"],
        })

    data = []
    for image in result.images:
        entry: Dict[str, Any] = {"id": image.image_id, "path": image.path}
        if response_format == "b64_json":
            entry["b64_json"] = image.b64()
        data.append(entry)

    return {
        "created" : result.created,
        "model"   : result.model,
        "data"    : data,
        "usage"   : usage,
    }


EDIT_FIELDS = ("images", "edit", "mask", "file_ids")

def multipart_edit_fields() -> Dict[str, Any]:
    """Turn a multipart edit into the same override dict JSON requests use."""
    uploads: List[Any] = []
    try:
        for field_name in IMAGE_UPLOAD_FIELDS:
            uploads.extend(request.files.getlist(field_name))
    except RequestEntityTooLarge:
        # Parsing request.files is where Werkzeug surfaces oversized multipart bodies.
        abort(413, description=f"Upload exceeds REQUEST_MAX_BYTES ({cfg.request_max_bytes:,} bytes).")

    if not uploads:
        abort(400, description=f"a multipart edit must include at least one image file (field 'image' or 'image[]').")
    if len(uploads) > cfg.image_edit_max_images:
        abort(400, description=f"{len(uploads)} images uploaded; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")

    fields: Dict[str, Any] = {
        "images": [v1_images.load_uploaded_reference(item.filename or "", item.read()) for item in uploads]
    }

    mask = (request.files.get("mask"))
    if mask is not None:
        fields["mask"] = v1_images.load_uploaded_reference(mask.filename or "mask.png", mask.read())

    for name in ("prompt", "n", "size", "quality", "output_format", "background", "filename"):
        if name in request.form:
            fields[name] = request.form[name]

    # Structure has to travel as JSON here: a multipart body has no other way to carry it, which
    # is why validate_annotation reads a string as well as an object.
    for name in ("job_group", "job", "mask_region"):
        if name in request.form:
            fields[name] = request.form[name]

    return fields


def run_image_request(fields: Dict[str, Any], response_format: str):
    """The shared tail of both image routes: build, dispatch, reply."""
    req = v1_images.build_request(fields, source="direct")

    if req.batch:
        batch = v1_images.submit_image_batch([req])
        return jsonify({
            "number"            : batch.number,
            "batch_id"          : batch.batch_id,
            "status"            : batch.status,
            "model"             : batch.model,
            "completion_window" : cfg.image_batch_window,
            "auto_retrieved"    : cfg.image_batch_auto_poll,
        })

    return jsonify(make_image_response(v1_images.generate_image(req), response_format))


@app.route("/images/edits"   , methods=["POST"])
@app.route("/v1/images/edits", methods=["POST"])
def images_edits():
    """Image editing, via multipart uploads or JSON references."""
    uploaded = request.mimetype == "multipart/form-data"

    try:
        if uploaded:
            fields = multipart_edit_fields()
        else:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return Response(json.dumps({"error": {"message": "Invalid JSON body."}}), status=400, content_type="application/json")
            fields = dict(payload)

        response_format = resolve_response_format(fields.pop("response_format", None))

        # This route always edits, so a bare reference set with no explicit flag still
        # means an edit -- and a caller who posted to it expecting a generation is told so.
        if not any(fields.get(name) for name in EDIT_FIELDS):
            return Response(json.dumps({"error": {"message":
                "an edit needs reference images: upload them as multipart, or name slots/paths in 'images', "
                "or provider ids in 'file_ids'. For text-to-image use /v1/images/generations."}}),
                status=400, content_type="application/json")

        return run_image_request(fields, response_format)

    except Exception as exc:
        return make_error_response(exc, None if uploaded else request.get_json(silent=True))


@app.route("/images/generations"   , methods=["POST"])
@app.route("/v1/images/generations", methods=["POST"])
def images_generations():
    """
    Direct text-to-image generation, sharing every default, validator, storage rule and
    cost path with the chat-triggered route. Only 'prompt' is required.

    Editing lives at /v1/images/edits, mirroring the upstream API rather than overloading
    this URL, so a client that knows one knows the other.
    """
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return Response(json.dumps({"error": {"message": "Invalid JSON body."}}), status=400, content_type="application/json")

    present = [name for name in EDIT_FIELDS if payload.get(name)]
    if present:
        return Response(json.dumps({"error": {"message":
            f"{present} {'is' if len(present) == 1 else 'are'} an editing field; post to /v1/images/edits instead."}}),
            status=400, content_type="application/json")

    try:
        response_format = resolve_response_format(payload.pop("response_format", None))
        req = v1_images.build_request(payload, source="direct")

        if req.batch:
            batch = v1_images.submit_image_batch([req])
            return jsonify({
                # 'number' is the short reference the CLI and chat replies use; the
                # provider id is kept for callers that talk to the provider directly.
                "number"            : batch.number,
                "batch_id"          : batch.batch_id,
                "status"            : batch.status,
                "model"             : batch.model,
                "completion_window" : cfg.image_batch_window,
                "auto_retrieved"    : cfg.image_batch_auto_poll,
            })

        return jsonify(make_image_response(v1_images.generate_image(req), response_format))

    except Exception as exc:
        return make_error_response(exc, payload)


@app.route("/images/batches"   , methods=["GET"])
@app.route("/v1/images/batches", methods=["GET"])
def images_batch_list():
    """
    Every batch this proxy has submitted from its output directory.

    Reads recorded state only -- no provider call, and so no retrieval and no billing.
    A client rebuilding a job list after a restart wants exactly this, and wants it to
    stay cheap enough to poll.
    """
    try:
        return jsonify({"data": v1_images.list_batches()})
    except Exception as exc:
        return make_error_response(exc, None)


@app.route("/v1/images/manifest", methods=["PATCH"])
def images_manifest_patch():
    """
    Correct what some records say about the requests that made them.

    The one writable path into this proxy's manifest, held under the same lock that appending
    takes -- which is the reason it exists: a client rewriting the file itself would race a job
    landing and could drop a record.

    What may be corrected is testimony: the prompt, the request parameters, the mask's region, the
    lineage, the provider and model named, when it was made, what it cost, and which attempt it
    belongs to. What may not is measurement -- `file`, `image_id`, `bytes` and the provider's
    `usage` describe the file this proxy wrote.

    Match a record by image_id, or by file for one written before ids were recorded. A key left out
    is left alone; a null clears it, which for `job` is how an attempt is filed back out of a group.
    """
    check_proxy_key()
    payload = request.get_json(silent=True) or {}
    updates = payload.get("updates")
    if not isinstance(updates, list):
        abort(400, description="expected an object with an 'updates' array.")
    if len(updates) > 2000:
        abort(400, description=f"{len(updates)} updates in one request; the limit is 2000.")

    try:
        return jsonify(v1_images.patch_manifest(updates))
    except Exception as exc:
        return make_error_response(exc, None)


@app.route("/v1/images/rename", methods=["POST"])
def images_rename():
    """
    Rename one image this proxy wrote, and bring its record with it.

    Its own route rather than a field of the manifest patch, which refuses `file` because a
    record's filename describes the file on disk rather than what was asked for. Renaming does not
    correct that description -- it changes what is being described -- and it has to move the file
    and rewrite the manifest together, under the lock appending takes. A client doing it itself
    would leave the manifest naming something that is no longer there.

    Every other record that names the file follows it, as a source or as a mask, so an edit's
    lineage still points at a real file.

    Name the image by `image_id`, or by `file` for one written before ids were recorded. The
    extension is not the caller's to change: the bytes decide the format.
    """
    check_proxy_key()
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        abort(400, description="expected a JSON object.")

    try:
        return jsonify(v1_images.rename_image(
            payload.get("image_id", ""),
            payload.get("file", ""),
            payload.get("filename", ""),
        ))
    except Exception as exc:
        return make_error_response(exc, payload)


@app.route("/images/batches/<token>"   , methods=["GET"])
@app.route("/v1/images/batches/<token>", methods=["GET"])
def images_batch_status(token: str):
    """
    Batch status, saving the images once the batch has completed. Accepts either the
    short number the proxy assigned or the provider's own batch id.
    """
    try:
        result = v1_images.retrieve_image_batch(v1_images.resolve_batch_id(token))
        return jsonify({
            "number"   : result.number,
            "batch_id" : result.batch_id,
            "status"   : result.status,
            "model"    : result.model,
            "data"     : [{"id": image.image_id, "path": image.path} for image in result.images],
            "counts"   : result.counts,
            "errors"   : result.errors,
            "usage"    : {"estimated_cost_usd": round(result.cost_usd, 6)},
        })
    except Exception as exc:
        return make_error_response(exc, None)


@app.route("/"                   , methods=["POST"])
@app.route("/chat/completions"   , methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def short_baseurl() : return handle_chat_completion()
def baseurl()       : return handle_chat_completion()
def v1_baseurl()    : return handle_chat_completion()


def print_provider_table() -> None:
    """
    What the three provider lists resolved to. A provider declared under the wrong
    protocol is accepted here and only fails at the first request, so the resolved
    table is worth seeing before that.
    """
    print("Configured providers:")
    for name, provider in cfg.providers.items():
        print(f"  {name:<10}  {provider['api']:<10}  {provider['base_url']}")
    print()


if __name__ == "__main__":
    load_dotenv()
    cfg.reload_from_env()

    if not cfg.providers:
        print("No providers are configured.")
        print("Declare at least one in V1_MESSAGES_PROVIDERS, V1_CHAT_COMPLETIONS_PROVIDERS or")
        print("V1_RESPONSES_PROVIDERS in .env, then configure it through its <NAME>_* variables.")
        print("See env_example.ini for a working configuration.")
        raise SystemExit(1)

    print_provider_table()
    providers.refresh_models(cfg.model_list_timeout_seconds)

    # MODEL is a bare id or "provider/model-id". The prefixed form resolves without a
    # model list, which is the only thing that still works when a provider's /models
    # request failed -- so it is the form worth configuring.
    if not providers.apply_model_by_id(cfg.model):
        print()
        print(f"MODEL '{cfg.model}' matches no model of any configured provider.")
        print("Set MODEL=provider/model-id in .env, using one of the providers listed above.")
        raise SystemExit(1)
    finish_model_switch(True)

    v1_images.resolve_image_config()
    v1_images.refresh_image_models(cfg.model_list_timeout_seconds)
    v1_images.start_batch_poller()

    print("Starting proxy")
    print(f"Local URL: http://{cfg.host}:{cfg.port}")
    print(f"Chat completions: http://{cfg.host}:{cfg.port}/chat/completions")
    if cfg.image_enabled:
        print(f"Image generations: http://{cfg.host}:{cfg.port}/v1/images/generations")
    print("Cloudflare Tunnel service URL should point to this local address:")
    print(f"  http://{cfg.host}:{cfg.port}")
    print()

    if cfg.require_proxy_key and not cfg.proxy_key:
        print("WARNING: REQUIRE_PROXY_KEY=true but PROXY_KEY is missing.")
        print("Set PROXY_KEY in .env before exposing this through Cloudflare Tunnel.")
        print()

    if not cfg.providers[cfg.backend]["api_key"] and not cfg.allow_key_passthrough:
        key_name = cfg.providers[cfg.backend]["api_key_name"]
        print(f"WARNING: {key_name} is missing and ALLOW_KEY_PASSTHROUGH=false.")
        print("Requests will fail until you configure one of these modes.")
        print()

    thread = threading.Thread(target=admin_cli_loop, daemon=True)
    thread.start()
    serve(app, host=cfg.host, port=cfg.port)
