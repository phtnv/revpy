"""
The /v1/images/* backend: the low-level image API adapter.

Serves the whole image family -- /images/generations, /images/edits, and their Batch API
counterparts -- rather than one endpoint each. Unlike the three v1_* wire modules, which
are interchangeable implementations of one protocol selected per provider, generating and
editing are two operations over a single API that share almost everything downstream:
provider and price resolution, request validation, storage, the manifest, cost accounting
and the batch lifecycle. Only the transport differs, and that difference is confined to
post_generation() and post_edit().

This module is a secondary service, not a fourth wire protocol. It never writes to
cfg.backend, cfg.model or the text price fields, never appears in the conversational
model list, and never decides *whether* an image should be made -- that is
image_orchestrator's job.

Every entry point (the direct routes, the chat orchestrator and the CLI) builds an
ImageRequest and hands it here, so validation, transport, storage and accounting exist
exactly once.
"""

import base64
import binascii
import io
import json
import os
import re
import struct
import threading
import time
import uuid

from typing      import Any, Dict, List, Optional, Tuple

import httpx

from flask import has_request_context

from common import (
    IMAGE_BACKGROUNDS,
    IMAGE_FORMATS,
    IMAGE_QUALITIES,
    IMAGE_SIZE_EDGE_MULTIPLE,
    IMAGE_SIZE_MAX_ASPECT,
    IMAGE_SIZE_MAX_EDGE,
    IMAGE_SIZE_MAX_PIXELS,
    IMAGE_SIZE_MIN_PIXELS,
    cfg,
    fmt_usd,
    resolve_api_key,
    resolve_image_costs,
    track_image_usage,
)
from providers import (
    ProviderError,
    error_from_response,
    request_timeout,
)


# The fields a caller (or a model) may override. Everything else comes from configuration.
# Provider, model, base URL, credentials, headers and output *paths* are deliberately
# absent: those are proxy policy, not request content (see the plan, 7.3).
ALLOWED_OVERRIDES = {
    "prompt", "size", "quality", "output_format", "background", "n", "batch", "filename",
    # Editing. 'images' names local references (slot numbers, or paths when allowed);
    # 'file_ids' names references already on the provider, which is the only form a Batch
    # API request can carry. 'input_fidelity' is deliberately absent: gpt-image-2 always
    # processes inputs at high fidelity and rejects attempts to set it.
    "images", "mask", "edit", "file_ids",
    # Lineage the caller declares rather than the proxy derives: which file on this
    # machine each reference came from. Recorded in the manifest, never sent upstream.
    # See validate_source_files() for why it exists at all.
    "source_files",
}

# Fields an OpenAI client sends that this proxy accepts and then discards. They are not in
# ALLOWED_OVERRIDES because they are not honoured -- 'model' in particular is proxy policy,
# exactly as it is for chat, where the model chosen in the frontend has no effect either.
# Keeping them out of the unknown-field check is what lets client.images.generate() work
# unmodified; keeping them out of the override list is what stops a caller picking the
# model. A genuinely unknown field is still an error, so a typo is still reported.
IGNORED_FIELDS = {"model", "user", "moderation", "style", "output_compression", "partial_images"}

# What a reference image may be. Identified by magic bytes rather than by extension: a
# '.png' that is really something else must fail here, not at the provider.
IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png" , "image/png" ),
    (b"\xff\xd8\xff"     , "jpeg", "image/jpeg"),
)
WEBP_RIFF   = b"RIFF"
WEBP_TAG    = b"WEBP"

# Extension per output_format. 'jpeg' is the API's spelling; '.jpg' is the file's.
FORMAT_EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}

# What one 'source_files' entry may carry. 'file' is the basename, which is what joins a
# manifest record to a directory listing; 'path' is where it was at request time; and
# 'file_id' is the provider handle the job used, which outlives nothing.
SOURCE_FILE_KEYS = {"file_id", "path", "file"}

SIZE_RE     = re.compile(r"^(\d{1,5})\s*[x×]\s*(\d{1,5})$")
# A filename override is a *name*, never a path: no separators, no traversal, no
# leading dot, nothing outside this set. The extension is imposed from output_format.
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FILENAME_MAX_STEM = 120

BATCH_ENDPOINT      = "/v1/images/generations"
BATCH_EDIT_ENDPOINT = "/v1/images/edits"
BATCH_STATE_FILE = "img_batches.json"
MANIFEST_FILE    = "img_generation.json"
SLOTS_FILE       = "img_slots.json"
# The mask shares the slots file under a key no slot number can collide with.
MASK_KEY         = "mask"

# Guards the slot file the same way BATCH_LOCK guards the batch state.
SLOT_LOCK = threading.RLock()

# Statuses a batch never moves out of. Anything else is still worth polling.
BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# Serializes everything that reads-then-writes the batch state file. Two actors now touch
# it -- the background poller and whoever is at the CLI -- and a batch retrieved by both
# at once would save its images twice and bill the session twice. Re-entrant because
# retrieval resolves and labels batches while already holding it.
BATCH_LOCK = threading.RLock()

# Serializes filename allocation, file writes and manifest updates. Waitress serves
# requests on a thread pool, so two concurrent generations can otherwise pick the same
# free filename between the exists() check and the write.
STORAGE_LOCK = threading.Lock()


class ImageRequestError(ProviderError):
    """
    A request rejected before the provider was contacted. Subclasses ProviderError so
    server.build_error_body and common.error_body render it like any other backend
    failure, with no special cases at the call sites.
    """
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(status_code, {"error": {"message": message, "type": "invalid_request_error"}}, message)


class ImageConfigError(ProviderError):
    """A misconfiguration of the proxy itself, rather than a bad request."""
    def __init__(self, message: str):
        super().__init__(500, {"error": {"message": message, "type": "image_configuration_error"}}, message)


class ReferenceImage:
    """
    One validated image to edit from, held either as a path on this machine or as bytes
    uploaded with the request. The two arrive by very different routes -- a path is named
    from the console or an allowlisted prompt, an upload is posted by a client -- but from
    the point of validation onwards they are the same thing, so the rest of the module
    never has to care which it got.
    """
    def __init__(self, format: str, mime: str, size: int, path: str = "", data: Optional[bytes] = None,
                 name: str = "", width: int = 0, height: int = 0, has_alpha: bool = False, slot: int = 0):
        self.format    = format
        self.mime      = mime
        self.size      = size          # bytes
        self.path      = path          # empty for an upload
        self.data      = data          # None for a path on disk
        self.name      = name          # the client's filename, for uploads
        self.width     = width         # 0 when the header could not be read
        self.height    = height
        self.has_alpha = has_alpha     # PNG only; what a mask is required to have
        self.slot      = slot          # the slot it came from, 0 otherwise

    def filename(self) -> str:
        return os.path.basename(self.path) if self.path else (self.name or f"upload.{self.format}")

    def open(self) -> Any:
        """A fresh readable stream. Called once per attempt, since a retry re-reads it."""
        return open(self.path, "rb") if self.path else io.BytesIO(self.data or b"")

    def origin(self) -> str:
        """How this reference is recorded in the manifest."""
        return self.path if self.path else f"upload:{self.filename()}"


class ImageRequest:
    """
    One fully validated request. Every field has already been checked against the
    allowlists by build_request(), so anything downstream can use it without re-testing.
    """
    def __init__(self, prompt: str, size: str, quality: str, output_format: str, background: str,
                 n: int, batch: bool, filename: str = "", source: str = "direct",
                 images: Optional[List[ReferenceImage]] = None,
                 mask: Optional[ReferenceImage] = None,
                 file_ids: Optional[List[str]] = None,
                 source_files: Optional[List[Dict[str, str]]] = None):
        self.prompt        = prompt
        self.size          = size
        self.quality       = quality
        self.output_format = output_format
        self.background    = background
        self.n             = n
        self.batch         = batch
        self.filename      = filename   # sanitized stem, or "" for the timestamped default
        self.source        = source     # "direct", "chat" or "cli"; for logging only

        # Editing. 'images' and 'file_ids' are mutually exclusive: the first is a multipart
        # upload of local files, the second names references already held by the provider,
        # which is the only form the Batch API accepts.
        self.images   = images or []
        self.mask     = mask
        self.file_ids = file_ids or []

        # Declared lineage. Empty means "derive it from the references", which
        # manifest_source_files() does; it is only supplied explicitly when the
        # references are provider ids and the proxy has no other way to know.
        self.source_files = source_files or []

    @property
    def is_edit(self) -> bool:
        return bool(self.images or self.file_ids)


class SavedImage:
    """One image written to disk."""
    def __init__(self, image_id: str, path: str, data: Optional[bytes] = None):
        self.image_id = image_id
        self.path     = path
        # The decoded bytes, kept so a caller asking for b64_json is served without reading
        # the file back off disk. Saving happens either way; this is only about the reply.
        self.data     = data

    def b64(self) -> str:
        if self.data is not None:
            return base64.b64encode(self.data).decode("ascii")
        with open(self.path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")


class ImageResult:
    """One completed immediate request: what was made, where it went, what it cost."""
    def __init__(self, provider: str, model: str, created: int, images: List[SavedImage],
                 cost_usd: float, usage: Dict[str, Any], estimated: bool):
        self.provider  = provider
        self.model     = model
        self.created   = created
        self.images    = images
        self.cost_usd  = cost_usd
        self.usage     = usage
        self.estimated = estimated   # True when the cost came from the price table


class ImageBatchResult:
    """
    What is known about one batch. Unlike ImageResult this is filled in as retrieval
    proceeds rather than all at once -- the fields below the line accumulate while the
    output file is walked -- so the two halves are kept visibly apart.
    """
    def __init__(self, provider: str, model: str, batch_id: str, status: str,
                 number: int = 0, counts: Optional[Dict[str, int]] = None):
        self.provider = provider
        self.model    = model
        self.batch_id = batch_id
        self.status   = status
        # The short reference the user sees. 0 for a batch this proxy did not submit.
        self.number   = number
        self.counts   = counts or {}

        # Accumulated during retrieve_image_batch().
        self.images   : List[SavedImage] = []
        self.cost_usd : float            = 0.0
        self.errors   : List[str]        = []


# Provider and model resolution
def image_provider() -> Dict[str, Any]:
    """
    The provider entry image requests are served by.

    IMAGE_PROVIDER normally names a provider already declared in one of the three text
    lists, in which case its base URL and credentials are reused as-is. When it names one
    that is not declared anywhere, its <NAME>_* variables are parsed standalone -- which
    is what lets you generate images through OpenAI while chatting with Claude, without
    OpenAI's text models appearing in the conversational model list.

    The standalone entry is built with api='chat' because that only selects the auth
    header shape, and /images/generations is an OpenAI-style Bearer endpoint.
    """
    name = cfg.image_provider
    if not name:
        raise ImageConfigError("IMAGE_PROVIDER is not set. Name a provider in .env to enable image generation.")

    provider = cfg.providers.get(name)
    if provider is not None:
        return provider

    prefix   = re.sub(r"[^A-Z0-9]", "_", name.upper())
    base_url = os.getenv(f"{prefix}_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise ImageConfigError(
            f"IMAGE_PROVIDER is '{name}', but it is declared in no provider list and {prefix}_BASE_URL is missing."
        )

    return {
        "api"          : "chat",
        "base_url"     : base_url,
        "api_key"      : os.getenv(f"{prefix}_API_KEY", "").strip(),
        "api_key_name" : f"{prefix}_API_KEY",
        "extra_body"   : {},
    }


def image_headers(provider: Dict[str, Any]) -> Dict[str, str]:
    """
    Auth for an image request.

    Inside a Flask request this goes through common.resolve_api_key, so the PROXY_KEY
    check and ALLOW_KEY_PASSTHROUGH behave exactly as they do for chat. The CLI has no
    request to read a bearer token from, so it uses the configured key directly -- calling
    resolve_api_key there would fail on the missing request context rather than on
    anything to do with the key.
    """
    if has_request_context():
        key = resolve_api_key(provider["api_key"], provider["api_key_name"])
    else:
        key = provider["api_key"]
        if not key:
            raise ImageConfigError(f"{provider['api_key_name']} is not configured.")

    return {"Authorization": f"Bearer {key}"}


def apply_image_model(model_id: str) -> None:
    """
    Points the image price fields at a model. The counterpart of providers.apply_model,
    except that it binds nothing else: no backend, no wire protocol, and none of cfg's
    text state. Selecting an image model must leave the conversation exactly as it was.
    """
    prefix   = re.sub(r"[^A-Z0-9]", "_", (cfg.image_provider or "").upper())
    families = cfg.parse_image_cost_families(prefix)

    cost_source = resolve_image_costs({})
    cost_family = f"{cfg.image_provider}:default"
    for family in families:
        if family["regex"].search(model_id):
            cost_source = family
            cost_family = f"{cfg.image_provider}:{family['name']}"
            break

    cfg.image_model            = model_id
    cfg.image_cost_family      = cost_family
    cfg.image_text_input_cost   = cost_source["text_input_cost"]
    cfg.image_image_input_cost  = cost_source["image_input_cost"]
    cfg.image_cached_input_cost = cost_source["cached_input_cost"]
    cfg.image_output_cost       = cost_source["image_output_cost"]

    if not cost_source["image_output_cost"]:
        print(f"WARNING: no image prices are configured for '{cost_family}'. Generated images will be reported as free.")


def resolve_image_config() -> None:
    """
    Startup and post-reload check. Reports what image generation resolved to, and
    disables it rather than letting the first request fail with a configuration error.
    """
    if not cfg.image_enabled:
        print("Image generation is disabled (IMAGE_GENERATION_ENABLED).")
        return

    try:
        provider = image_provider()
    except ProviderError as exc:
        print(f"WARNING: {exc}. Disabling image generation.")
        cfg.image_enabled = False
        return

    if not cfg.image_model:
        print("WARNING: IMAGE_MODEL is not set. Disabling image generation.")
        cfg.image_enabled = False
        return

    # The configured default is checked once here rather than on every request, so a bad
    # IMAGE_DEFAULT_SIZE is a startup warning instead of an error blaming each request
    # for a field it never sent.
    try:
        cfg.image_default_size = validate_size(cfg.image_default_size)
    except ImageRequestError as exc:
        print(f"WARNING: IMAGE_DEFAULT_SIZE is invalid ({exc}). Falling back to 'auto'.")
        cfg.image_default_size = "auto"

    standalone = cfg.image_provider not in cfg.providers
    origin     = "standalone" if standalone else f"shared with the '{cfg.providers[cfg.image_provider]['api']}' provider"
    print(f"Image generation: {cfg.image_provider}/{cfg.image_model} at {provider['base_url']} ({origin}).")

    apply_image_model(cfg.image_model)

    if not provider["api_key"] and not cfg.allow_key_passthrough:
        print(f"WARNING: {provider['api_key_name']} is missing. Image requests will fail.")


# Validation
def validate_size(raw: Any) -> str:
    """
    A size the image model will actually accept.

    Checked against the model family's constraints rather than an allowlist of the four
    popular sizes: gpt-image-2 takes any resolution inside these bounds, so an enum would
    reject valid requests and rot as soon as the range changes.
    """
    size = str(raw).strip().lower()
    if size == "auto":
        return size

    match = SIZE_RE.match(size)
    if match is None:
        raise ImageRequestError(f"size must be 'auto' or WIDTHxHEIGHT, got {raw!r}.")

    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ImageRequestError("size must have positive dimensions.")
    if width % IMAGE_SIZE_EDGE_MULTIPLE or height % IMAGE_SIZE_EDGE_MULTIPLE:
        raise ImageRequestError(f"size edges must both be multiples of {IMAGE_SIZE_EDGE_MULTIPLE}, got {width}x{height}.")
    if max(width, height) > IMAGE_SIZE_MAX_EDGE:
        raise ImageRequestError(f"size edges must be at most {IMAGE_SIZE_MAX_EDGE}px, got {width}x{height}.")

    aspect = max(width, height)/min(width, height)
    if aspect > IMAGE_SIZE_MAX_ASPECT:
        raise ImageRequestError(f"size aspect ratio must be at most {IMAGE_SIZE_MAX_ASPECT:g}:1, got {aspect:.2f}:1.")

    pixels = width*height
    if not (IMAGE_SIZE_MIN_PIXELS <= pixels <= IMAGE_SIZE_MAX_PIXELS):
        raise ImageRequestError(
            f"size must total between {IMAGE_SIZE_MIN_PIXELS:,} and {IMAGE_SIZE_MAX_PIXELS:,} pixels, got {pixels:,}."
        )

    return f"{width}x{height}"


def validate_choice(field_name: str, raw: Any, allowed: set) -> str:
    value = str(raw).strip().lower()
    if value not in allowed:
        raise ImageRequestError(f"{field_name} must be one of {sorted(allowed)}, got {raw!r}.")
    return value


def validate_bool(field_name: str, raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}  : return True
    if value in {"0", "false", "no", "n", "off"} : return False
    raise ImageRequestError(f"{field_name} must be a boolean, got {raw!r}.")


def sanitize_filename_stem(raw: Any) -> str:
    """
    A caller-supplied filename reduced to a bare, safe stem.

    The caller names the file; the proxy decides where it goes and what it ends in. Any
    directory component, traversal segment, separator or unusual character is a rejection
    rather than something to quietly strip -- a request that meant to escape the output
    directory should fail loudly, and one that did not is unaffected.
    """
    name = str(raw).strip()
    if not name:
        raise ImageRequestError("filename must not be empty.")

    stem = os.path.splitext(name)[0]
    if not FILENAME_RE.match(stem) or ".." in stem:
        raise ImageRequestError(
            f"filename must be a bare name matching [A-Za-z0-9._-] with no path separators, got {raw!r}."
        )

    return stem[:FILENAME_MAX_STEM]


# Reference image slots
#
# The console's way of naming files without a prompt ever carrying a path. Slots persist
# so a restart does not lose them, and are re-validated every time they are used: the
# file behind a slot can be moved, replaced or truncated between being set and being sent.
def slots_path() -> str:
    return os.path.join(os.path.abspath(cfg.image_output_dir), SLOTS_FILE)


def read_slots() -> Dict[str, str]:
    path = slots_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            slots = json.load(handle)
        return {str(k): str(v) for k, v in slots.items()} if isinstance(slots, dict) else {}
    except Exception as exc:
        print(f"WARNING: could not read {path} ({exc}).")
        return {}


def write_slots(slots: Dict[str, str]) -> None:
    path     = os.path.join(output_dir_path(), SLOTS_FILE)
    tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(slots, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        try: os.unlink(tmp_path)
        except OSError: pass
        print(f"WARNING: could not write {path} ({exc}).")


def set_slot(number: int, path: str) -> ReferenceImage:
    """
    Points a slot at a file, validating it now so a mistake is reported at the console
    rather than three turns later inside a chat reply.
    """
    if number < 1:
        raise ImageRequestError("slot numbers start at 1.")

    # from_prompt=False: the console is trusted with paths, which is the entire reason
    # slots exist.
    reference = load_reference(path, from_prompt=False, slot=number)

    with SLOT_LOCK:
        slots = read_slots()
        slots[str(number)] = reference.path
        write_slots(slots)

    return reference


def clear_slot(number: Optional[int]) -> int:
    with SLOT_LOCK:
        slots = read_slots()
        if number is None:
            count = len(slots)
            write_slots({})
            return count
        removed = slots.pop(str(number), None)
        write_slots(slots)
        return 1 if removed else 0


def numbered_slots(slots: Dict[str, str]) -> List[int]:
    """
    Just the reference slots, in order. The mask lives in the same file under a
    non-numeric key, so every walk over the slots has to go through here.
    """
    numbers = []
    for key in slots:
        try: numbers.append(int(key))
        except ValueError:
            continue
    return sorted(numbers)


def next_free_slot() -> int:
    slots = read_slots()
    number = 1
    while str(number) in slots:
        number += 1
    return number


def stored_mask() -> Optional[ReferenceImage]:
    """The mask set from the console, re-validated, or None when none is set."""
    path = read_slots().get(MASK_KEY)
    if not path:
        return None
    return load_reference(path, from_prompt=False)


def resolve_slot(number: int) -> ReferenceImage:
    slots = read_slots()
    path  = slots.get(str(number))
    if not path:
        raise ImageRequestError(f"image slot {number} is empty. Fill it with 'img edit set {number} <path>'.")
    # Re-validated rather than trusted: the file may have changed since it was set.
    return load_reference(path, from_prompt=False, slot=number)


def filled_slots() -> List[ReferenceImage]:
    """Every filled slot, in number order. Unreadable ones raise rather than be skipped."""
    return [resolve_slot(number) for number in numbered_slots(read_slots())]


# Reference images
def sniff_image(head: bytes) -> Optional[Tuple[str, str]]:
    """
    (format, mime) from a file's leading bytes, or None when it is not an image this
    endpoint accepts. Content decides; the extension is never consulted.
    """
    for magic, fmt, mime in IMAGE_MAGIC:
        if head.startswith(magic):
            return fmt, mime
    if head[:4] == WEBP_RIFF and head[8:12] == WEBP_TAG:
        return "webp", "image/webp"
    return None


def read_png_geometry(head: bytes) -> Tuple[int, int, bool]:
    """
    (width, height, has_alpha) from a PNG's IHDR chunk.

    Colour types 4 and 6 carry an alpha channel; 3 (palette) can fake one with a tRNS
    chunk, which is why that is checked too. Enough to tell a usable mask from the JPEG
    that the provider's docs call the most common cause of edit failures, without taking
    on an imaging dependency for one header.
    """
    if len(head) < 26 or head[12:16] != b"IHDR":
        return 0, 0, False

    width, height = struct.unpack(">II", head[16:24])
    colour_type   = head[25]
    has_alpha     = colour_type in (4, 6) or (colour_type == 3 and b"tRNS" in head)
    return width, height, has_alpha


def path_is_allowed(resolved: str) -> bool:
    """
    Whether a prompt-supplied path may be read. Containment is tested after realpath, so
    a symlink sitting inside an allowed root but pointing outside it is refused -- the
    link, not its target, is what an attacker gets to place.
    """
    if not cfg.image_edit_roots:
        return False
    for root in cfg.image_edit_roots:
        try:
            if os.path.commonpath([root, resolved]) == root:
                return True
        except ValueError:
            continue
    return False


def describe_image(head: bytes, size: int, label: str) -> Tuple[str, str, int, int, bool]:
    """
    (format, mime, width, height, has_alpha) for a candidate image, or a rejection.

    The shared half of validation: everything that depends only on the bytes, so an
    uploaded image and one read off disk are judged by exactly the same rules.
    """
    if size <= 0:
        raise ImageRequestError(f"{label} is empty.")
    if size > cfg.image_edit_max_bytes:
        raise ImageRequestError(f"{label} is {size:,} bytes; the limit is {cfg.image_edit_max_bytes:,} (IMAGE_EDIT_MAX_BYTES).")

    sniffed = sniff_image(head)
    if sniffed is None:
        raise ImageRequestError(f"{label} is not a PNG, JPEG or WebP image.")

    fmt, mime = sniffed
    if fmt == "png":
        width, height, has_alpha = read_png_geometry(head)
        return fmt, mime, width, height, has_alpha
    return fmt, mime, 0, 0, False


def load_uploaded_reference(name: str, data: bytes) -> ReferenceImage:
    """
    One reference image from bytes posted with the request.

    Needs no filesystem access at all, so none of the path machinery applies: there is
    nothing to traverse, no allowlist to consult, and no way to reach a file the caller
    was not already holding. The content checks are the same ones a path gets.
    """
    label = f"uploaded file {name!r}" if name else "uploaded file"
    fmt, mime, width, height, has_alpha = describe_image(data[:4096], len(data), label)

    return ReferenceImage(format=fmt, mime=mime, size=len(data), data=data,
                          name=os.path.basename(name or ""), width=width, height=height,
                          has_alpha=has_alpha)


def load_reference(spec: Any, from_prompt: bool, slot: int = 0) -> ReferenceImage:
    """
    One validated reference image, from a slot number or a filesystem path.

    Rejects rather than skips, in this order: the path must resolve, be a regular file,
    be readable, be an image by its magic bytes, and fit the size cap. A caller that
    named something unusable should be told so, not quietly handed a shorter list.
    """
    # An upload arrives already validated, since its bytes never came from the filesystem
    # and there is nothing left to resolve.
    if isinstance(spec, ReferenceImage):
        return spec

    if isinstance(spec, bool):
        raise ImageRequestError(f"an image reference must be a slot number or a path, got {spec!r}.")

    # A bare integer is a slot; slots are filled from the console, never from a prompt.
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return resolve_slot(int(spec))

    if not isinstance(spec, str) or not spec.strip():
        raise ImageRequestError(f"an image reference must be a slot number or a path, got {spec!r}.")

    raw = os.path.expanduser(spec.strip())

    if from_prompt:
        if not cfg.image_edit_allow_prompt_paths:
            raise ImageRequestError(
                "image paths in a chat request are disabled. Use a slot number "
                "('img edit set 1 <path>'), or set IMAGE_EDIT_ALLOW_PROMPT_PATHS=true."
            )
        if not cfg.image_edit_roots:
            raise ImageRequestError("IMAGE_EDIT_ALLOW_PROMPT_PATHS is on but IMAGE_EDIT_ROOTS is empty, so no path is readable.")

    try: resolved = os.path.realpath(raw)
    except OSError as exc:
        raise ImageRequestError(f"could not resolve {spec!r}: {exc}")

    if from_prompt and not path_is_allowed(resolved):
        # Deliberately does not say whether the file exists: to an untrusted caller that
        # difference is a directory oracle.
        raise ImageRequestError(f"{spec!r} is outside every directory listed in IMAGE_EDIT_ROOTS.")

    if not os.path.exists(resolved)    : raise ImageRequestError(f"{spec!r} does not exist.")
    if not os.path.isfile(resolved)    : raise ImageRequestError(f"{spec!r} is not a regular file.")
    if not os.access(resolved, os.R_OK): raise ImageRequestError(f"{spec!r} is not readable.")

    try:
        with open(resolved, "rb") as handle:
            head = handle.read(4096)
    except OSError as exc:
        raise ImageRequestError(f"could not read {spec!r}: {exc}")

    fmt, mime, width, height, has_alpha = describe_image(head, os.path.getsize(resolved), repr(spec))

    return ReferenceImage(path=resolved, format=fmt, mime=mime, size=os.path.getsize(resolved),
                          width=width, height=height, has_alpha=has_alpha, slot=slot)


def load_references(specs: Any, from_prompt: bool) -> List[ReferenceImage]:
    if not isinstance(specs, list):
        specs = [specs]
    if not specs:
        return []
    if len(specs) > cfg.image_edit_max_images:
        raise ImageRequestError(f"{len(specs)} reference images requested; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")

    return [load_reference(spec, from_prompt) for spec in specs]


def validate_mask(mask: ReferenceImage, images: List[ReferenceImage]) -> None:
    """
    A mask has to be a PNG carrying an alpha channel; transparent pixels are the region
    the model may repaint. With several references the provider applies it to the first,
    so that is the one its geometry is checked against.
    """
    if mask.format != "png":
        raise ImageRequestError(f"the mask must be a PNG (alpha is what marks the editable region), got {mask.format}.")
    if not mask.has_alpha:
        raise ImageRequestError("the mask PNG has no alpha channel, so it marks nothing as editable.")
    if images and mask.width and images[0].width and (mask.width, mask.height) != (images[0].width, images[0].height):
        raise ImageRequestError(
            f"the mask is {mask.width}x{mask.height} but the first reference image is "
            f"{images[0].width}x{images[0].height}; they must match."
        )


def validate_file_ids(specs: Any) -> List[str]:
    """
    Provider-side reference ids for a batched edit. Accepts file ids and URLs, since
    input_reference takes either, and nothing else -- a local path here would silently
    never be uploaded.
    """
    if not isinstance(specs, list):
        specs = [specs]

    ids: List[str] = []
    for spec in specs:
        value = str(spec).strip()
        if not value:
            raise ImageRequestError("a file_ids entry must not be empty.")
        if not (value.startswith("file-") or value.startswith("http://") or value.startswith("https://")):
            raise ImageRequestError(f"file_ids entries must be a provider file id ('file-...') or a URL, got {spec!r}.")
        ids.append(value)

    if len(ids) > cfg.image_edit_max_images:
        raise ImageRequestError(f"{len(ids)} references requested; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")
    return ids


def validate_source_files(specs: Any) -> List[Dict[str, str]]:
    """
    Caller-declared lineage: the file on this machine each reference of this request came
    from. Recorded in the manifest and never sent to the provider.

    It exists because a batched edit names its references by provider file id, and those
    ids expire and are deleted -- the copy on this machine is what survives, so the
    manifest has to name it. The id is recorded too, but only as the handle the job
    happened to use.

    An entry is either an object with any of file_id / path / file, or a bare string,
    which is read as a path when it looks like one and as a name otherwise. A path that
    does not resolve is recorded as given rather than rejected: the file may have been
    moved since, and a stale link is worth more than no link at all.
    """
    if not isinstance(specs, list):
        specs = [specs]

    sources: List[Dict[str, str]] = []
    for spec in specs:
        if isinstance(spec, dict):
            unknown = sorted(set(spec) - SOURCE_FILE_KEYS)
            if unknown:
                raise ImageRequestError(f"unsupported source_files field(s) {unknown}; allowed: {sorted(SOURCE_FILE_KEYS)}.")
            file_id = str(spec.get("file_id") or "").strip()
            raw     = str(spec.get("path")    or "").strip()
            name    = str(spec.get("file")    or "").strip()
        elif isinstance(spec, str):
            value   = spec.strip()
            file_id = ""
            # A separator is the only thing distinguishing "D:/img/a.png" from "a.png",
            # and treating a bare name as a path would invent a location it never had.
            raw     = value if (os.sep in value or "/" in value) else ""
            name    = "" if raw else value
        else:
            raise ImageRequestError(f"a source_files entry must be a string or an object, got {spec!r}.")

        if not (file_id or raw or name):
            raise ImageRequestError("a source_files entry must name at least one of file_id, path or file.")

        entry: Dict[str, str] = {}
        if file_id:
            entry["file_id"] = file_id
        if raw:
            expanded = os.path.expanduser(raw)
            try:
                resolved = os.path.realpath(expanded) if os.path.exists(expanded) else raw
            except OSError:
                resolved = raw
            entry["path"] = resolved
            entry["file"] = name or os.path.basename(resolved)
        elif name:
            entry["file"] = name

        sources.append(entry)

    if len(sources) > cfg.image_edit_max_images:
        raise ImageRequestError(f"{len(sources)} source_files entries; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")
    return sources


def resolve_edit_inputs(fields: Dict[str, Any], source: str, batch: bool) -> Tuple[List[ReferenceImage], Optional[ReferenceImage], List[str]]:
    """
    Decides whether this request edits, and against what.

    'images' cannot be the sole signal, because slots are filled from the console and a
    block that means "edit what is loaded" carries no images field at all. So an explicit
    edit:true is a second trigger, and the combinations resolve like this:

        neither images nor edit     generation, exactly as before
        images: [...]               edit against those references
        edit: true, no images       edit against every filled slot
        edit: true + images         edit against those references
        edit: false + images        rejected -- a contradiction, not a preference
        edit: true, no slots filled rejected -- nothing to edit
        file_ids + batch: true      batched edit against provider-side references
        file_ids without batch      rejected -- file_ids exist for the Batch API
        images + batch: true        rejected -- local files cannot be batched
        images + file_ids           rejected -- mutually exclusive
    """
    has_images   = "images"   in fields and fields["images"] not in (None, [], "")
    has_file_ids = "file_ids" in fields and fields["file_ids"] not in (None, [], "")
    has_mask     = "mask"     in fields and fields["mask"] not in (None, "")

    edit_flag: Optional[bool] = None
    if "edit" in fields:
        edit_flag = validate_bool("edit", fields["edit"])

    if not (has_images or has_file_ids or edit_flag):
        if edit_flag is False and has_mask:
            raise ImageRequestError("mask was given with edit: false; a mask only applies to an edit.")
        return [], None, []

    if not cfg.image_edit_enabled:
        raise ImageConfigError("Image editing is disabled (IMAGE_EDIT_ENABLED=false).")

    if has_images and has_file_ids:
        raise ImageRequestError("images and file_ids are mutually exclusive: images are uploaded from this machine, file_ids already live on the provider.")
    if edit_flag is False and (has_images or has_file_ids):
        raise ImageRequestError("edit: false was given together with reference images. Remove one of them.")

    if has_file_ids:
        if not batch:
            raise ImageRequestError("file_ids is for Batch API edits only. Add batch: true, or use images for an immediate edit.")
        if has_mask:
            raise ImageRequestError("a mask cannot be sent with a batched edit, because it would have to be uploaded as a file.")
        return [], None, validate_file_ids(fields["file_ids"])

    if batch:
        raise ImageRequestError("local reference images cannot be batched. Upload them to the provider and pass file_ids, or drop batch: true.")

    from_prompt = source == "chat"
    if has_images:
        images = load_references(fields["images"], from_prompt)
    else:
        # edit: true with nothing named means "whatever the console has loaded".
        images = filled_slots()
        if not images:
            raise ImageRequestError("edit: true was requested but no image slots are filled. Use 'img edit set 1 <path>'.")
        if len(images) > cfg.image_edit_max_images:
            raise ImageRequestError(f"{len(images)} slots are filled; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")

    if not images:
        raise ImageRequestError("no usable reference images were given.")

    if has_mask:
        mask = load_reference(fields["mask"], from_prompt)
    else:
        # A mask set from the console applies to slot-driven edits the same way the slots
        # themselves do -- setting one and having it ignored would be the surprise.
        mask = stored_mask()

    if mask is not None:
        validate_mask(mask, images)

    return images, mask, []


def build_request(overrides: Optional[Dict[str, Any]] = None, source: str = "direct") -> ImageRequest:
    """
    One validated request, from the configured defaults plus whatever the caller
    explicitly overrode. Unknown fields are rejected rather than ignored, so a typo in a
    model-written block is reported instead of silently taking a default.
    """
    if not cfg.image_enabled:
        raise ImageConfigError("Image generation is disabled (IMAGE_GENERATION_ENABLED=false).")

    fields = dict(overrides or {})

    # Streaming image generation cannot be relayed, and silently returning a whole
    # response to a client waiting for events fails later and less clearly than saying so.
    if str(fields.get("stream", "")).strip().lower() in {"true", "1", "yes"}:
        raise ImageRequestError("streaming image generation is not supported by this proxy.")

    requested_model = fields.get("model")
    if requested_model and str(requested_model).strip() != cfg.image_model:
        print(f"NOTE: image request asked for model '{requested_model}'; using the configured '{cfg.image_model}'.")

    for name in IGNORED_FIELDS | {"stream"}:
        fields.pop(name, None)

    unknown = sorted(set(fields) - ALLOWED_OVERRIDES)
    if unknown:
        raise ImageRequestError(f"unsupported field(s) {unknown}; allowed: {sorted(ALLOWED_OVERRIDES)}.")

    prompt = str(fields.get("prompt", "") or "").strip()
    if not prompt:
        raise ImageRequestError("prompt is required and must not be empty.")
    if len(prompt) > cfg.image_max_prompt_chars:
        raise ImageRequestError(f"prompt is {len(prompt)} characters; the limit is {cfg.image_max_prompt_chars} (IMAGE_MAX_PROMPT_CHARS).")

    batch  = validate_bool("batch", fields.get("batch", cfg.image_default_batch))
    images, mask, file_ids = resolve_edit_inputs(fields, source, batch)
    is_edit = bool(images or file_ids)

    # An edit normally means "change this picture", so it defaults to the source geometry
    # rather than to the generation default, which would silently reframe it.
    default_size  = cfg.image_edit_default_size if is_edit else cfg.image_default_size
    size          = validate_size  (fields.get("size", default_size))
    quality       = validate_choice("quality"      , fields.get("quality"      , cfg.image_default_quality)   , IMAGE_QUALITIES)
    output_format = validate_choice("output_format", fields.get("output_format", cfg.image_default_format)    , IMAGE_FORMATS)
    background    = validate_choice("background"   , fields.get("background"   , cfg.image_default_background), IMAGE_BACKGROUNDS)

    raw_n = fields.get("n", cfg.image_default_n)
    try: count = int(raw_n)
    except (TypeError, ValueError):
        raise ImageRequestError(f"n must be an integer, got {raw_n!r}.")
    if count < 1 or count > cfg.image_max_n:
        raise ImageRequestError(f"n must be between 1 and {cfg.image_max_n} (IMAGE_MAX_N), got {count}.")

    filename = sanitize_filename_stem(fields["filename"]) if fields.get("filename") else ""

    source_files = validate_source_files(fields["source_files"]) if fields.get("source_files") else []

    # Positional pairing, so a caller listing the same references in both fields does not
    # have to repeat each id inside the lineage entry it already lines up with. Only done
    # when the counts match and no entry named an id itself, since either of those means
    # the caller had a pairing of its own in mind.
    if file_ids and len(source_files) == len(file_ids) and not any(entry.get("file_id") for entry in source_files):
        for entry, file_id in zip(source_files, file_ids):
            entry["file_id"] = file_id

    return ImageRequest(
        prompt        = prompt,
        size          = size,
        quality       = quality,
        output_format = output_format,
        background    = background,
        n             = count,
        batch         = batch,
        filename      = filename,
        source        = source,
        images        = images,
        mask          = mask,
        file_ids      = file_ids,
        source_files  = source_files,
    )


def build_body(req: ImageRequest) -> Dict[str, Any]:
    """
    The provider request. 'auto' is left out rather than sent, so the provider applies
    its own default instead of being told a literal it may not accept.
    """
    body: Dict[str, Any] = {
        "model"         : cfg.image_model,
        "prompt"        : req.prompt,
        "n"             : req.n,
        "output_format" : req.output_format,
        "background"    : req.background,
    }
    if req.size    != "auto" : body["size"]    = req.size
    if req.quality != "auto" : body["quality"] = req.quality

    # A batched edit carries its references inline, since the Batch API accepts no
    # multipart upload. Verified against the provider: the field is 'images', it is always
    # an array even for a single reference, and each entry must be an object -- a bare id
    # string is rejected with "expected an object", and 'input_reference' with "Missing
    # required parameter: 'images'". Several references work and bill exactly as they do
    # on the immediate endpoint (two 1024x1024 references measured 2048 image input
    # tokens). An image_url entry is accepted, but the provider must then fetch it, and a
    # host that refuses non-browser requests fails the line on the download rather than on
    # anything the proxy sent.
    if req.file_ids:
        body["images"] = [
            {"image_url": value} if value.startswith("http") else {"file_id": value}
            for value in req.file_ids
        ]

    return body


def build_edit_form(req: ImageRequest) -> Tuple[Dict[str, str], List[Tuple[str, Any]]]:
    """
    The multipart form for an immediate edit.

    Reference images go in as 'image[]' -- repeated, which is what the provider expects
    for several -- and every scalar becomes a form field, because multipart carries no
    types. Files are handed over as open handles so httpx streams them instead of the
    proxy holding every reference image in memory at once.

    'input_fidelity' is never sent: gpt-image-2 always processes inputs at high fidelity
    and rejects the parameter outright.
    """
    data: Dict[str, str] = {
        "model"         : cfg.image_model,
        "prompt"        : req.prompt,
        "n"             : str(req.n),
        "output_format" : req.output_format,
        "background"    : req.background,
    }
    if req.size    != "auto" : data["size"]    = req.size
    if req.quality != "auto" : data["quality"] = req.quality

    files: List[Tuple[str, Any]] = [
        ("image[]", (reference.filename(), reference.open(), reference.mime))
        for reference in req.images
    ]
    if req.mask is not None:
        files.append(("mask", (req.mask.filename(), req.mask.open(), req.mask.mime)))

    return data, files


def close_form_files(files: List[Tuple[str, Any]]) -> None:
    """Closes the handles build_edit_form opened, whether or not the request succeeded."""
    for _, spec in files:
        handle = spec[1]
        try: handle.close()
        except Exception: pass


# Storage
def output_dir_path() -> str:
    """The absolute output directory, created on first use."""
    path = os.path.abspath(cfg.image_output_dir)
    os.makedirs(path, exist_ok=True)
    return path


def allocate_path(directory: str, stem: str, extension: str) -> str:
    """
    A free path under `directory`. An existing file is never overwritten: a linear index
    is appended until the name is free. Call with STORAGE_LOCK held.

    The result is confined to the output directory even if the stem somehow survived
    sanitisation -- storage is the last place able to catch that, and a path escaping
    here would be the model writing wherever it liked.
    """
    candidate = os.path.join(directory, f"{stem}{extension}")
    index     = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{index}{extension}")
        index += 1

    root     = os.path.realpath(directory)
    resolved = os.path.realpath(candidate)
    try:
        contained = os.path.commonpath([root, resolved]) == root
    except ValueError:
        # Different drives on Windows have no common path at all, which is as far
        # outside the output directory as it gets.
        contained = False
    if not contained:
        raise ImageRequestError(f"refusing to write outside {cfg.image_output_dir}.")

    return candidate


def default_stem(image_id: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{image_id.removeprefix('img_')[:8]}"


def save_image_bytes(req: ImageRequest, data: bytes, image_id: str) -> SavedImage:
    """
    Writes one decoded image, atomically: the bytes land in a temporary file in the
    destination directory and are then renamed into place, so a reader never sees a
    half-written image and a failed write leaves no partial file behind.
    """
    directory = output_dir_path()
    extension = FORMAT_EXTENSIONS[req.output_format]
    stem      = req.filename or default_stem(image_id)

    with STORAGE_LOCK:
        path     = allocate_path(directory, stem, extension)
        tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise

    # Reported relative to the working directory, which is the short form worth reading in
    # a console. On Windows two drives share no common prefix at all and relpath raises
    # rather than returning something absolute, so the absolute path stands in -- an output
    # directory on another drive is a configuration, not an error.
    try: reported = os.path.relpath(path, os.getcwd())
    except ValueError:
        reported = path

    return SavedImage(image_id=image_id, path=reported, data=data)


def decode_image(b64_data: Any, index: int) -> bytes:
    if not isinstance(b64_data, str) or not b64_data:
        raise ProviderError(502, {"error": {"message": f"image {index} carried no b64_json data."}},
                            "image response carried no data")
    try:
        return base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(502, {"error": {"message": f"image {index} was not valid Base64: {exc}"}},
                            "invalid Base64 in image response")


def manifest_source_files(req: ImageRequest) -> List[Dict[str, str]]:
    """
    The lineage recorded for one request, in one shape whichever route produced it.

    A caller that declared its own wins, because only it knows which local file a
    provider file id stands for. Otherwise it is derived from the references the request
    actually carried, so an immediate edit gets the same field without anyone supplying
    it. An upload has a name but no location on this machine, which is exactly what the
    entry then says.
    """
    if req.source_files:
        return [dict(entry) for entry in req.source_files]

    sources: List[Dict[str, str]] = []
    for reference in req.images:
        if reference.path:
            sources.append({"path": reference.path, "file": os.path.basename(reference.path)})
        else:
            sources.append({"file": reference.filename()})
    return sources


def append_manifest(req: ImageRequest, saved: List[SavedImage], cost_usd: float, usage: Dict[str, Any], estimated: bool, batch_id: str = "") -> None:
    """
    Appends one record per image to the sidecar manifest, creating it only if absent.

    A corrupt or unreadable manifest is replaced rather than allowed to abort the
    request: the images are already on disk, and losing the sidecar is the lesser
    failure. IMAGE_MANIFEST_PROMPTS decides whether the prompt itself is recorded,
    since the manifest outlives the session that produced it.
    """
    if not cfg.image_manifest_enabled or not saved:
        return

    directory = output_dir_path()
    path      = os.path.join(directory, MANIFEST_FILE)
    created   = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    records = []
    for image in saved:
        record: Dict[str, Any] = {
            "image_id"           : image.image_id,
            "path"               : image.path,
            # 'path' is relative to the proxy's working directory, which a client reading
            # this file out of the output directory has no way to resolve. The basename
            # is what joins a record to a directory listing, and allocate_path() has
            # already made it unique within that directory.
            "file"               : os.path.basename(image.path),
            "bytes"              : len(image.data) if image.data is not None else 0,
            "provider"           : cfg.image_provider,
            "model"              : cfg.image_model,
            "request_parameters" : {
                "size"          : req.size,
                "quality"       : req.quality,
                "output_format" : req.output_format,
                "background"    : req.background,
                "n"             : req.n,
            },
            "source"             : req.source,
            "created_at"         : created,
            "operation"          : "edit" if req.is_edit else "generate",
            "estimated_cost_usd" : round(cost_usd/len(saved), 6),
            "cost_is_estimate"   : estimated,
            "usage"              : usage,
        }
        if batch_id:
            record["batch_id"] = batch_id
        # An edit driven by 'edit: true' resolves against whatever the slots held at the
        # time, so the inputs are recorded here or the result is unreproducible.
        if req.images:
            record["source_images"] = [reference.origin() for reference in req.images]
        if req.mask is not None:
            record["mask"] = req.mask.origin()
        if req.file_ids:
            record["source_references"] = list(req.file_ids)
        # The durable half of the above: 'source_references' names ids that expire, this
        # names the files on this machine they stood for.
        sources = manifest_source_files(req)
        if sources:
            record["source_files"] = sources
        if cfg.image_manifest_prompts:
            record["prompt"] = req.prompt
        records.append(record)

    with STORAGE_LOCK:
        existing: List[Any] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, list) : existing = loaded
                else                        : print(f"WARNING: {path} is not a JSON list. Starting a new manifest.")
            except Exception as exc:
                print(f"WARNING: could not read {path} ({exc}). Starting a new manifest.")

        existing.extend(records)
        tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(existing, handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
            os.replace(tmp_path, path)
        except Exception as exc:
            try: os.unlink(tmp_path)
            except OSError: pass
            print(f"WARNING: could not write {path} ({exc}).")


# Usage and cost
def parse_usage(usage: Any) -> Dict[str, Any]:
    """
    Pulls the token counts out of an image usage payload. The gpt-image family reports
    exact counts split by modality:
        {"input_tokens", "input_tokens_details": {"text_tokens", "image_tokens"},
         "output_tokens", "output_tokens_details": {"image_tokens", "text_tokens"}}
    Providers that report nothing get a zeroed dict, and the caller estimates instead.
    """
    usage = usage if isinstance(usage, dict) else {}
    if not usage:
        return {"text_input": 0, "image_input": 0, "output": 0, "reported": False}

    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}

    text_input  = max(0, int(input_details.get("text_tokens" , 0) or 0))
    image_input = max(0, int(input_details.get("image_tokens", 0) or 0))
    total_input = max(0, int(usage.get("input_tokens", 0) or 0))

    # Without the split, everything counts as text input: text-to-image sends no
    # reference images, so that is the correct bucket rather than a guess.
    if not text_input and not image_input:
        text_input = total_input

    return {
        "text_input"  : text_input,
        "image_input" : image_input,
        "output"      : max(0, int(usage.get("output_tokens", 0) or 0)),
        "reported"    : True,
    }


def estimate_image_cost(model: str, quality: str, size: str, n: int) -> float:
    """
    Fallback per-image pricing for providers that return no usage object, from
    IMAGE_PRICE_TABLE: {"model-regex": {quality: {size: usd}}}. Either key may be "*".
    Returns 0.0 when nothing matches, which is reported as free rather than invented.
    """
    table = cfg.image_price_table
    if not isinstance(table, dict):
        return 0.0

    for model_pattern, by_quality in table.items():
        try:
            if not re.search(str(model_pattern), model):
                continue
        except re.error:
            print(f"WARNING: IMAGE_PRICE_TABLE key {model_pattern!r} is not a valid regex. Skipping.")
            continue
        if not isinstance(by_quality, dict):
            continue

        by_size = by_quality.get(quality, by_quality.get("*"))
        if not isinstance(by_size, dict):
            continue

        price = by_size.get(size, by_size.get("*"))
        if price is None:
            return 0.0
        try: return float(price)*n
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def cost_for(req: ImageRequest, counts: Dict[str, Any], images: int, batch: bool) -> Tuple[float, bool]:
    """Costs one completed request, preferring the provider's own numbers."""
    estimated = not counts.get("reported")
    if estimated:
        counts = dict(counts)
        counts["estimated"]          = True
        counts["estimated_cost_usd"] = estimate_image_cost(cfg.image_model, req.quality, req.size, images)

    cost = track_image_usage(counts, images=images, batch=batch, model=cfg.image_model)
    return cost, estimated


# Immediate generation and editing
def post_generation(provider: Dict[str, Any], request: ImageRequest) -> Any:
    """Text-to-image: a JSON body to /images/generations."""
    body = build_body(request)

    if cfg.debug_log:
        print()
        print(f"=== image payload start ({cfg.image_provider}/{cfg.image_model}) ===")
        print(json.dumps(body, indent=2, ensure_ascii=False, default=str))
        print(f"=== image payload end ===")

    return httpx.post(
        f"{provider['base_url']}/images/generations",
        json=body,
        headers=image_headers(provider),
        timeout=request_timeout(),
    )


def post_edit(provider: Dict[str, Any], request: ImageRequest) -> Any:
    """
    Image-to-image: a multipart form to /images/edits. The only structural difference
    from generation is the transport, which is why it is isolated to this function --
    decoding, saving, the manifest and the cost accounting are shared downstream.
    """
    data, files = build_edit_form(request)

    if cfg.debug_log:
        references = ", ".join(f"{ref.filename()} ({ref.format}, {ref.size:,}B)" for ref in request.images)
        print()
        print(f"=== image edit payload start ({cfg.image_provider}/{cfg.image_model}) ===")
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        print(f"  references: {references}")
        if request.mask is not None:
            print(f"  mask      : {request.mask.filename()} ({request.mask.width}x{request.mask.height})")
        print(f"=== image edit payload end ===")

    try:
        return httpx.post(
            f"{provider['base_url']}/images/edits",
            data=data,
            files=files,
            headers=image_headers(provider),
            timeout=request_timeout(),
        )
    finally:
        close_form_files(files)


# Gateway-class failures: the request never reached or never completed at the origin, so
# no image was produced and nothing was billed. Retrying those is safe. A 4xx, or a 5xx
# the model itself produced, is not in here -- repeating a refusal just pays for it twice.
RETRYABLE_STATUSES  = {502, 503, 504, 520, 521, 522, 523, 524}
RETRYABLE_TRANSPORT = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError, httpx.WriteError)


def post_with_retry(send, provider: Dict[str, Any], request: ImageRequest) -> Any:
    """
    Runs one image call, retrying gateway failures.

    The callable is re-invoked rather than the response replayed, because an edit's
    multipart handles are read to EOF on the first attempt and a retry needs fresh ones.
    """
    attempts = max(1, cfg.image_retry_attempts)
    delay    = max(0.0, cfg.image_retry_backoff_seconds)

    for attempt in range(1, attempts + 1):
        last_error = ""
        try:
            response = send(provider, request)
            if response.status_code not in RETRYABLE_STATUSES:
                return response
            last_error = f"HTTP {response.status_code}"
        except RETRYABLE_TRANSPORT as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            response   = None

        if attempt >= attempts:
            if response is not None:
                return response
            raise ProviderError(502, {"error": {"message": f"image request failed after {attempts} attempts ({last_error})."}},
                                f"{cfg.image_provider}: {last_error}")

        print(f"WARNING: image request failed ({last_error}); retrying {attempt}/{attempts - 1} in {delay:.0f}s.")
        time.sleep(delay)
        delay *= 2

    raise RuntimeError("unreachable")


def generate_image(request: ImageRequest) -> ImageResult:
    """
    Run one immediate request -- a generation, or an edit when it carries references --
    and save every file it returns.
    """
    if request.batch:
        raise ImageRequestError("this request is marked batch=true; use submit_image_batch instead.")

    provider = image_provider()
    response = post_with_retry(post_edit if request.is_edit else post_generation, provider, request)

    if response.status_code != 200:
        raise error_from_response(cfg.image_provider, response)

    data = response.json()
    if not isinstance(data, dict):
        raise ProviderError(502, {"error": {"message": "image response was not a JSON object."}}, "malformed image response")

    entries = data.get("data")
    if not isinstance(entries, list) or not entries:
        raise ProviderError(502, {"error": {"message": "image response contained no data entries."}}, "empty image response")

    saved: List[SavedImage] = []
    for index, entry in enumerate(entries):
        entry     = entry if isinstance(entry, dict) else {}
        image_id  = f"img_{uuid.uuid4().hex[:16]}"
        raw_bytes = decode_image(entry.get("b64_json"), index)
        saved.append(save_image_bytes(request, raw_bytes, image_id))

    counts     = parse_usage(data.get("usage"))
    cost, estimated = cost_for(request, counts, len(saved), batch=False)
    append_manifest(request, saved, cost, counts, estimated)

    return ImageResult(
        provider  = cfg.image_provider,
        model     = cfg.image_model,
        created   = int(data.get("created") or time.time()),
        images    = saved,
        cost_usd  = cost,
        usage     = counts,
        estimated = estimated,
    )


# Batch API
#
# A batch is a fundamentally different thing from n>1: n asks for several images now,
# a batch trades latency (up to the completion window) for a lower rate. Nothing here
# can be awaited inside a chat turn, so submission and retrieval are separate calls and
# the CLI is what drives them.
def batch_state_path() -> str:
    """Deliberately does not create the output directory: reading state must not be what
    brings a directory into existence for a feature nobody has used yet."""
    return os.path.join(os.path.abspath(cfg.image_output_dir), BATCH_STATE_FILE)


def number_legacy_batches() -> None:
    """
    Gives a number to any batch recorded before numbering existed, so an older state file
    does not leave unreferenceable rows in the listing.
    """
    with BATCH_LOCK:
        state   = read_batch_state()
        missing = [batch_id for batch_id, entry in state.items() if not entry.get("number")]
        if not missing:
            return

        number = next_batch_number(state)
        for batch_id in missing:
            state[batch_id]["number"] = number
            number += 1
        write_batch_state(state)
        print(f"Assigned reference numbers to {len(missing)} previously recorded image batch(es).")


def next_batch_number(state: Dict[str, Any]) -> int:
    """
    The next short reference number. Numbers are assigned once, persisted, and never
    reused, so the number printed when a batch was submitted still names the same batch
    after a restart -- which is the whole point of having one instead of the provider's id.
    """
    highest = 0
    for entry in state.values():
        try: highest = max(highest, int(entry.get("number") or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


def resolve_batch_id(token: str) -> str:
    """
    Turns whatever the user typed into a provider batch id. A bare number is looked up
    among the submitted batches; anything else is taken as an id already, so a batch this
    proxy never submitted can still be retrieved.
    """
    token = str(token).strip()
    if not token.isdigit():
        return token

    number = int(token)
    with BATCH_LOCK:
        for batch_id, entry in read_batch_state().items():
            if int(entry.get("number") or 0) == number:
                return batch_id

    raise ImageRequestError(f"no batch numbered {number}. Use 'image batch list' to see them.")


def batch_label(batch_id: str, state: Optional[Dict[str, Any]] = None) -> str:
    """How a batch is named to the user: its number, not its provider hash."""
    entry  = (state or read_batch_state()).get(batch_id) or {}
    number = entry.get("number")
    return f"#{number}" if number else batch_id


def read_batch_state() -> Dict[str, Any]:
    path = batch_state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except Exception as exc:
        print(f"WARNING: could not read {path} ({exc}).")
        return {}


def write_batch_state(state: Dict[str, Any]) -> None:
    """
    Persists the request parameters behind each batch. The provider returns only the
    images and the custom_id it was given, so without this the output format, filename
    and prompt a batch was submitted with are gone by the time it completes.
    """
    path     = os.path.join(output_dir_path(), BATCH_STATE_FILE)
    tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        try: os.unlink(tmp_path)
        except OSError: pass
        print(f"WARNING: could not write {path} ({exc}).")


def request_to_state(req: ImageRequest) -> Dict[str, Any]:
    return {
        "prompt"        : req.prompt if cfg.image_manifest_prompts else "",
        "size"          : req.size,
        "quality"       : req.quality,
        "output_format" : req.output_format,
        "background"    : req.background,
        "n"             : req.n,
        "batch"         : True,
        "filename"      : req.filename,
        "source"        : req.source,
        "file_ids"      : list(req.file_ids),
        # Carried through the wait: the manifest is written at retrieval, by which time
        # the request that declared this lineage is long gone.
        "source_files"  : [dict(entry) for entry in req.source_files],
    }


def state_to_request(entry: Dict[str, Any]) -> ImageRequest:
    return ImageRequest(
        prompt        = str(entry.get("prompt", "")),
        size          = str(entry.get("size", "auto")),
        quality       = str(entry.get("quality", "auto")),
        output_format = str(entry.get("output_format", "png")),
        background    = str(entry.get("background", "opaque")),
        n             = max(1, int(entry.get("n", 1) or 1)),
        batch         = True,
        filename      = str(entry.get("filename", "")),
        source        = str(entry.get("source", "batch")),
        file_ids      = [str(value) for value in (entry.get("file_ids") or [])],
        source_files  = [
            {str(key): str(value) for key, value in item.items() if key in SOURCE_FILE_KEYS}
            for item in (entry.get("source_files") or []) if isinstance(item, dict)
        ],
    )


def upload_batch_input(provider: Dict[str, Any], lines: List[str]) -> str:
    """Uploads the JSONL input file and returns its file id."""
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    response = httpx.post(
        f"{provider['base_url']}/files",
        headers=image_headers(provider),
        files  = {
            "file"    : ("image_batch.jsonl", payload, "application/jsonl"),
            "purpose" : (None, "batch"),
        },
        timeout=request_timeout(),
    )
    if response.status_code not in (200, 201):
        raise error_from_response(cfg.image_provider, response)

    data    = response.json()
    file_id = str((data or {}).get("id") or "")
    if not file_id:
        raise ProviderError(502, {"error": {"message": "file upload returned no id."}}, "batch upload failed")
    return file_id


def submit_image_batch(requests: List[ImageRequest]) -> ImageBatchResult:
    """Create a Batch API input file and submit an image-generation batch."""
    if not requests:
        raise ImageRequestError("a batch must contain at least one request.")

    # A batch names one endpoint for all of its lines, so it cannot mix the two.
    if len({req.is_edit for req in requests}) > 1:
        raise ImageRequestError("a batch cannot mix edits and generations; submit them separately.")
    if any(req.images for req in requests):
        raise ImageRequestError("local reference images cannot be batched. Upload them to the provider and pass file_ids.")

    endpoint = BATCH_EDIT_ENDPOINT if requests[0].is_edit else BATCH_ENDPOINT
    provider = image_provider()

    lines   : List[str]            = []
    entries : Dict[str, Any]       = {}
    for index, req in enumerate(requests):
        custom_id = f"img_{index:04d}_{uuid.uuid4().hex[:8]}"
        lines.append(json.dumps({
            "custom_id" : custom_id,
            "method"    : "POST",
            "url"       : endpoint,
            "body"      : build_body(req),
        }, ensure_ascii=False))
        entries[custom_id] = request_to_state(req)

    file_id = upload_batch_input(provider, lines)

    response = httpx.post(
        f"{provider['base_url']}/batches",
        headers=image_headers(provider),
        json   = {
            "input_file_id"     : file_id,
            "endpoint"          : endpoint,
            "completion_window" : cfg.image_batch_window,
        },
        timeout=request_timeout(),
    )
    if response.status_code not in (200, 201):
        raise error_from_response(cfg.image_provider, response)

    data     = response.json() or {}
    batch_id = str(data.get("id") or "")
    if not batch_id:
        raise ProviderError(502, {"error": {"message": "batch creation returned no id."}}, "batch creation failed")

    with BATCH_LOCK:
        state  = read_batch_state()
        number = next_batch_number(state)
        state[batch_id] = {
            "number"        : number,
            "provider"      : cfg.image_provider,
            "model"         : cfg.image_model,
            "input_file_id" : file_id,
            "submitted_at"  : time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "retrieved"     : False,
            "requests"      : entries,
        }
        write_batch_state(state)

    auto = "It will be retrieved automatically when it completes." if cfg.image_batch_auto_poll else f"Retrieve it with: image batch get {number}"
    print(f"Submitted image batch #{number} ({len(requests)} request(s), window {cfg.image_batch_window}).")
    print(auto)

    return ImageBatchResult(
        provider = cfg.image_provider,
        model    = cfg.image_model,
        batch_id = batch_id,
        number   = number,
        status   = str(data.get("status") or "validating"),
        counts   = {"submitted": len(requests)},
    )


def fetch_batch(provider: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    response = httpx.get(
        f"{provider['base_url']}/batches/{batch_id}",
        headers=image_headers(provider),
        timeout=request_timeout(),
    )
    if response.status_code != 200:
        raise error_from_response(cfg.image_provider, response)
    data = response.json()
    return data if isinstance(data, dict) else {}


def fetch_file_content(provider: Dict[str, Any], file_id: str) -> str:
    response = httpx.get(
        f"{provider['base_url']}/files/{file_id}/content",
        headers=image_headers(provider),
        timeout=request_timeout(),
    )
    if response.status_code != 200:
        raise error_from_response(cfg.image_provider, response)
    return response.text


def retrieve_image_batch(batch_id: str) -> ImageBatchResult:
    """
    Read batch status and save completed image results when available.

    Saving is done once and recorded: re-running this on an already-retrieved batch
    reports it rather than writing a second copy of every image and billing the session
    for them again.
    """
    provider = image_provider()

    # The whole read-check-save-mark sequence is one critical section: the poller and the
    # CLI can both land on the same batch, and doing this twice would write a second copy
    # of every image and bill the session for it.
    with BATCH_LOCK:
        state = read_batch_state()
        entry = state.get(batch_id) or {}

        data   = fetch_batch(provider, batch_id)
        status = str(data.get("status") or "unknown")
        model  = str(entry.get("model") or cfg.image_model)

        result = ImageBatchResult(
            provider = cfg.image_provider,
            model    = model,
            batch_id = batch_id,
            number   = int(entry.get("number") or 0),
            status   = status,
            counts   = dict(data.get("request_counts") or {}),
        )

        if entry.get("retrieved"):
            result.errors.append("already retrieved; images were saved on the first retrieval.")
            return result

        if status != "completed":
            # A batch that failed, expired or was cancelled will never produce images.
            # Marking it settles it, so the poller stops asking about it every interval.
            if status in BATCH_TERMINAL_STATUSES and entry:
                entry["retrieved"]    = True
                entry["final_status"] = status
                state[batch_id]       = entry
                write_batch_state(state)
            return result

        output_file_id = str(data.get("output_file_id") or "")
        if not output_file_id:
            result.errors.append("batch completed without an output file.")
            return result

        requests_state = entry.get("requests") or {}
        total_counts   = {"text_input": 0, "image_input": 0, "output": 0, "reported": False}
        images_saved   = 0

        for line in fetch_file_content(provider, output_file_id).splitlines():
            line = line.strip()
            if not line:
                continue
            try: record = json.loads(line)
            except Exception:
                result.errors.append("could not parse an output line.")
                continue

            custom_id = str(record.get("custom_id") or "")
            response  = record.get("response") or {}
            body      = response.get("body") if isinstance(response, dict) else {}
            body      = body if isinstance(body, dict) else {}

            if int((response or {}).get("status_code") or 0) != 200:
                message = ((body.get("error") or {}) if isinstance(body.get("error"), dict) else {}).get("message")
                result.errors.append(f"{custom_id}: {message or 'request failed'}")
                continue

            req     = state_to_request(requests_state.get(custom_id) or {})
            entries = body.get("data")
            if not isinstance(entries, list):
                result.errors.append(f"{custom_id}: response carried no data entries.")
                continue

            saved: List[SavedImage] = []
            for index, item in enumerate(entries):
                item      = item if isinstance(item, dict) else {}
                image_id  = f"img_{uuid.uuid4().hex[:16]}"
                try:
                    raw_bytes = decode_image(item.get("b64_json"), index)
                except ProviderError as exc:
                    result.errors.append(f"{custom_id}: {exc}")
                    continue
                saved.append(save_image_bytes(req, raw_bytes, image_id))

            counts = parse_usage(body.get("usage"))
            for key in ("text_input", "image_input", "output"):
                total_counts[key] += counts[key]
            total_counts["reported"] = total_counts["reported"] or counts["reported"]

            append_manifest(req, saved, 0.0, counts, not counts["reported"], batch_id=batch_id)
            result.images.extend(saved)
            images_saved += len(saved)

        if images_saved:
            # Billed once for the whole batch, at the batch rate, rather than per output
            # line: the session report distinguishes batch from immediate spending.
            billing_req = state_to_request(next(iter(requests_state.values()), {}))
            result.cost_usd, _ = cost_for(billing_req, total_counts, images_saved, batch=True)

        entry["retrieved"]    = True
        entry["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        entry["final_status"] = status
        entry["images"]       = [image.path for image in result.images]
        state[batch_id]       = entry
        write_batch_state(state)

    return result


# Background retrieval
def pending_batches() -> List[Tuple[int, str]]:
    """Every submitted batch that has not settled yet, in submission order."""
    with BATCH_LOCK:
        state = read_batch_state()

    pending = [
        (int(entry.get("number") or 0), batch_id)
        for batch_id, entry in state.items()
        if not entry.get("retrieved")
    ]
    return sorted(pending)


def list_batches() -> List[Dict[str, Any]]:
    """
    Every batch submitted from this output directory, newest first.

    Reports the state file rather than asking the provider about each batch in turn: a
    listing is a cheap call a client makes often, and one that fanned out to the provider
    would not be. An unsettled batch therefore reports 'pending' -- its live status comes
    from retrieving it, which is what GET /v1/images/batches/<n> does.
    """
    with BATCH_LOCK:
        state = read_batch_state()

    listing: List[Dict[str, Any]] = []
    for batch_id, entry in state.items():
        entry     = entry if isinstance(entry, dict) else {}
        retrieved = bool(entry.get("retrieved"))
        requests  = entry.get("requests") or {}

        listing.append({
            "number"        : int(entry.get("number") or 0),
            "batch_id"      : batch_id,
            "provider"      : entry.get("provider") or "",
            "model"         : entry.get("model") or "",
            "submitted_at"  : entry.get("submitted_at") or "",
            "retrieved"     : retrieved,
            "retrieved_at"  : entry.get("retrieved_at") or "",
            # 'pending' is this proxy's word, not the provider's: nothing here has asked
            # the provider, so claiming one of its statuses would be inventing it.
            "status"        : str(entry.get("final_status") or ("settled" if retrieved else "pending")),
            "images"        : list(entry.get("images") or []),
            "requests"      : [
                {
                    "custom_id"     : custom_id,
                    "prompt"        : item.get("prompt", ""),
                    "size"          : item.get("size", ""),
                    "quality"       : item.get("quality", ""),
                    "output_format" : item.get("output_format", ""),
                    "n"             : item.get("n", 1),
                    "file_ids"      : list(item.get("file_ids") or []),
                    "source_files"  : list(item.get("source_files") or []),
                }
                for custom_id, item in requests.items() if isinstance(item, dict)
            ],
        })

    return sorted(listing, key=lambda item: item["number"], reverse=True)


def poll_batches_once() -> List[ImageBatchResult]:
    """
    Checks every unsettled batch once and retrieves the finished ones. Returns only the
    batches that actually settled, so a caller has nothing to report on a quiet pass.

    One batch failing must not stop the others from being checked, so each is caught
    separately -- a provider hiccup on one id is not a reason to strand the rest.
    """
    settled: List[ImageBatchResult] = []

    for number, batch_id in pending_batches():
        try:
            result = retrieve_image_batch(batch_id)
        except Exception as exc:
            print(f"WARNING: could not check image batch #{number}: {exc}")
            continue
        if result.status in BATCH_TERMINAL_STATUSES:
            settled.append(result)

    return settled


def report_settled_batch(result: ImageBatchResult) -> None:
    """Announces a batch that finished while nobody was watching."""
    label = f"#{result.number}" if result.number else result.batch_id
    print()
    print(f"=== image batch {label} {result.status} ===")
    for image in result.images:
        print(f"  saved {image.image_id} -> {image.path}")
    if result.images:
        print(f"  cost {fmt_usd(result.cost_usd)}")
    for error in result.errors:
        print(f"  {error}")
    print(f"=== image batch {label} end ===")
    print("> ", end="", flush=True)


def batch_poll_loop() -> None:
    """
    Retrieves completed batches in the background.

    A batch finishes on the provider's schedule, which is rarely the moment anyone is
    looking at the CLI. Without this its images stay on the provider until somebody
    remembers to ask, and the completion window can expire first.
    """
    while True:
        time.sleep(cfg.image_batch_poll_seconds)

        if not cfg.image_enabled or not cfg.image_batch_auto_poll:
            continue

        try:
            for result in poll_batches_once():
                report_settled_batch(result)
        except Exception as exc:
            # The poller outliving a bad pass matters more than the pass succeeding.
            print(f"WARNING: image batch poll failed: {exc}")


# One poller per process. 'reload' can turn auto-polling on after startup, so this is
# called again from there; without the flag that would leave two threads racing for the
# same batches. The thread itself re-reads the settings every pass, so a poller already
# running picks up a new interval (or a disabled switch) without being restarted.
POLLER_STARTED = False

def start_batch_poller() -> None:
    global POLLER_STARTED

    if not cfg.image_enabled:
        return

    number_legacy_batches()

    if not cfg.image_batch_auto_poll or POLLER_STARTED:
        return

    POLLER_STARTED = True
    threading.Thread(target=batch_poll_loop, daemon=True).start()
    waiting = len(pending_batches())
    print(f"Image batches are retrieved automatically every {cfg.image_batch_poll_seconds:.0f}s ({waiting} waiting).")


# Image model list. Fetched separately from the conversational one: <NAME>_MODELS_REGEX
# exists to keep image models out of that list, so they have to be found on their own.
IMAGE_MODELS : List[Dict[str, Any]] = []
IMAGE_MODEL_LOCK                    = threading.Lock()


def refresh_image_models(timeout_s: float) -> None:
    global IMAGE_MODELS

    if not cfg.image_enabled:
        return

    try:
        provider = image_provider()
        response = httpx.get(f"{provider['base_url']}/models", headers=image_headers(provider), timeout=timeout_s)
        if response.status_code != 200:
            raise error_from_response(cfg.image_provider, response)
        data = response.json()
    except Exception as exc:
        print(f"WARNING: could not retrieve an image model list from '{cfg.image_provider}'. {exc}")
        return

    if   isinstance(data, dict) : entries = data.get("data") or data.get("models")
    elif isinstance(data, list) : entries = data
    else                        : entries = None

    try: pattern = re.compile(cfg.image_models_regex)
    except re.error as exc:
        print(f"WARNING: IMAGE_MODELS_REGEX is not a valid regex ({exc}). Listing every model.")
        pattern = re.compile("")

    models = [
        {**entry, "id": str(entry["id"]), "provider": cfg.image_provider}
        for entry in entries or []
        if isinstance(entry, dict) and entry.get("id") and pattern.search(str(entry["id"]))
    ]

    with IMAGE_MODEL_LOCK:
        IMAGE_MODELS = models
    print(f"Retrieved {len(models)} image model(s) from provider '{cfg.image_provider}'.")


def print_image_model_list() -> None:
    with IMAGE_MODEL_LOCK:
        models = list(IMAGE_MODELS)
    if not models:
        print("No image models are listed. Use 'image model refresh', or set IMAGE_MODEL directly.")
        return

    width = len(str(len(models)))
    for index, entry in enumerate(models, start=1):
        selected = cfg.image_model == entry["id"]
        number   = f"[{str(index).rjust(width)}]" if selected else f" {str(index).rjust(width)} "
        print(f"{number}  {entry['id']:<42}  {entry['provider']}")


def select_image_model_by_number(index: int) -> bool:
    """
    Selects an image model. Deliberately does not touch cfg.backend, cfg.model or the
    text price fields -- the conversation carries on with whatever it was using.
    """
    with IMAGE_MODEL_LOCK:
        if not IMAGE_MODELS:
            print("No image models are listed. Use 'image model refresh'.")
            return False
        if index < 1 or index > len(IMAGE_MODELS):
            print(f"Image model number out of range [1:{len(IMAGE_MODELS)}].")
            return False
        entry = IMAGE_MODELS[index - 1]

    print(f"=== Selecting image model {entry['provider']}/{entry['id']} ===")
    apply_image_model(entry["id"])
    print(f"Using image cost family '{cfg.image_cost_family}'.")
    return True


def reference_line(result: ImageResult) -> str:
    """
    The one-line reference appended to a chat reply in place of the control block.
    """
    if len(result.images) == 1:
        image = result.images[0]
        return f"[Generated image: {image.image_id} — {image.path}]"
    parts = ", ".join(f"{image.image_id} — {image.path}" for image in result.images)
    return f"[Generated {len(result.images)} images: {parts}]"
