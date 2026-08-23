"""
Low-level adapter for /v1/images/*.

Generation, editing and batches share provider resolution, validation, storage,
manifests and accounting; only the upstream transport differs. This module never mutates
the active chat backend/model. Chat-side triggering lives in image_orchestrator.py.
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


# Request fields the caller may control; provider, model, auth and paths are proxy policy.
ALLOWED_OVERRIDES = {
    "prompt", "size", "quality", "output_format", "background", "n", "batch", "filename",
    # 'images' are local references; 'file_ids' are provider-side references for batches.
    "images", "mask", "edit", "file_ids",
    # The rectangles the mask was drawn as; recorded beside it, never sent upstream.
    "mask_region",
    # Local lineage for provider-side references; recorded, never sent upstream.
    "source_files",
    # Which line of work a request belongs to; recorded, never sent upstream.
    "job_group", "job",
}

# Known OpenAI-client fields accepted for compatibility but ignored.
IGNORED_FIELDS = {"model", "user", "moderation", "style", "output_compression", "partial_images"}

# Reference images are identified by magic bytes, not filename extension.
IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png" , "image/png" ),
    (b"\xff\xd8\xff"     , "jpeg", "image/jpeg"),
)
WEBP_RIFF   = b"RIFF"
WEBP_TAG    = b"WEBP"

# Extension per output_format. 'jpeg' is the API's spelling; '.jpg' is the file's.
FORMAT_EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}

# Allowed keys in manifest lineage entries.
SOURCE_FILE_KEYS = {"file_id", "path", "file"}

# Allowed keys in the job stamp: which group and which attempt produced an image. Recorded in the
# manifest so a folder of pictures can say what line of work it came out of; never sent upstream.
JOB_GROUP_KEYS = {"id", "name"}
JOB_KEYS       = {"id", "parent", "comment", "run"}

# A mask region: the working-pixel size the rectangles are measured in, and the rectangles. Held
# to a sane length because it is recorded verbatim, and a manifest is read far more often than
# it is written.
MASK_RECT_KEYS = {"x", "y", "w", "h", "mode"}
MASK_RECT_MAX  = 512

# What a client may correct in a record already written, and nothing else. The split is between
# testimony and measurement: what a request *asked for* is something the person who made it can
# know better than this proxy, while what the file is -- its name, its id, its bytes, the usage
# the provider reported -- is measured here and is not the caller's to revise.
PATCH_STRING_KEYS = {"prompt", "provider", "model", "created_at", "operation", "source", "batch_id"}
PATCH_PARAM_KEYS  = {"size", "quality", "output_format", "background", "n"}
PATCH_OTHER_KEYS  = {"image_id", "file", "job_group", "job", "mask", "request_parameters",
                     "source_files", "estimated_cost_usd", "cost_is_estimate"}
# A note on an attempt, not an essay. Long enough for a sentence about what was being tried.
JOB_VALUE_MAX  = 500

SIZE_RE     = re.compile(r"^(\d{1,5})\s*[x×]\s*(\d{1,5})$")
# `data:<mediatype>;base64,` -- the mediatype is advisory, since the bytes are sniffed.
DATA_URL_RE = re.compile(r"^data:([^,;]*)((?:;[^,]*)?),", re.IGNORECASE)
# A filename override is a *name*, never a path: no separators, no traversal, no
# leading dot, nothing outside this set. The extension is imposed from output_format.
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FILENAME_MAX_STEM = 120

# Windows device names are rejected so output directories stay portable.
RESERVED_STEMS = {"con", "prn", "aux", "nul"} | {f"com{n}" for n in range(1, 10)} | {f"lpt{n}" for n in range(1, 10)}

BATCH_ENDPOINT      = "/v1/images/generations"
BATCH_EDIT_ENDPOINT = "/v1/images/edits"
BATCH_STATE_FILE = "img_batches.json"
MANIFEST_FILE    = "img_generation.json"
SLOTS_FILE       = "img_slots.json"
# The mask shares the slots file under a key no slot number can collide with.
MASK_KEY         = "mask"
# Where a mask that arrived as bytes is kept. A sub-folder rather than the output
# directory itself, so masks do not crowd the listing of what was actually generated.
MASK_SUBDIR      = "masks"

# Guards the slot file the same way BATCH_LOCK guards the batch state.
SLOT_LOCK = threading.RLock()

# Statuses a batch never moves out of. Anything else is still worth polling.
BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# Batch state is read-modify-written by the CLI and the poller; keep retrieval idempotent.
BATCH_LOCK = threading.RLock()

# Filename allocation and manifest writes must stay atomic across request threads.
STORAGE_LOCK = threading.Lock()


class ImageRequestError(ProviderError):
    """A preflight rejection rendered like any other provider error."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(status_code, {"error": {"message": message, "type": "invalid_request_error"}}, message)


class ImageConfigError(ProviderError):
    """A misconfiguration of the proxy itself, rather than a bad request."""
    def __init__(self, message: str):
        super().__init__(500, {"error": {"message": message, "type": "image_configuration_error"}}, message)


class ReferenceImage:
    """One validated edit reference, either a local path or uploaded bytes."""
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
    """One fully validated image request."""
    def __init__(self, prompt: str, size: str, quality: str, output_format: str, background: str,
                 n: int, batch: bool, filename: str = "", source: str = "direct",
                 images: Optional[List[ReferenceImage]] = None,
                 mask: Optional[ReferenceImage] = None,
                 mask_region: Optional[Dict[str, Any]] = None,
                 file_ids: Optional[List[str]] = None,
                 source_files: Optional[List[Dict[str, str]]] = None,
                 job_group: Optional[Dict[str, str]] = None,
                 job: Optional[Dict[str, str]] = None):
        self.prompt        = prompt
        self.size          = size
        self.quality       = quality
        self.output_format = output_format
        self.background    = background
        self.n             = n
        self.batch         = batch
        self.filename      = filename   # sanitized stem, or "" for the timestamped default
        self.source        = source     # "direct", "chat" or "cli"; for logging only

        # Local references and provider-side references are mutually exclusive.
        self.images   = images or []
        self.mask     = mask
        # What the mask was drawn as. Recorded beside the picture it rasterised to and never sent
        # upstream; the provider edits the PNG, and this is the form anyone can read back.
        self.mask_region = mask_region or {}
        self.file_ids = file_ids or []

        # Empty means manifest_source_files() can derive lineage from local references.
        self.source_files = source_files or []

        # Which group and which attempt this belongs to, as the caller stated it. Copied into the
        # manifest verbatim and never sent upstream; this proxy does not interpret either.
        self.job_group = job_group or {}
        self.job       = job or {}

    @property
    def is_edit(self) -> bool:
        return bool(self.images or self.file_ids)


class SavedImage:
    """One image written to disk."""
    def __init__(self, image_id: str, path: str, data: Optional[bytes] = None):
        self.image_id = image_id
        self.path     = path
        # Kept only so b64_json replies need not reread the saved file.
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
    """What is known about one batch; images/cost/errors fill during retrieval."""
    def __init__(self, provider: str, model: str, batch_id: str, status: str,
                 number: int = 0, counts: Optional[Dict[str, int]] = None):
        self.provider = provider
        self.model    = model
        self.batch_id = batch_id
        self.status   = status
        # Short user-facing reference; 0 for foreign batches.
        self.number   = number
        self.counts   = counts or {}

        self.images   : List[SavedImage] = []
        self.cost_usd : float            = 0.0
        self.errors   : List[str]        = []


# Provider and model resolution
def image_provider() -> Dict[str, Any]:
    """
    Provider entry for image requests.

    IMAGE_PROVIDER may reuse a declared text provider or name a standalone <NAME>_*
    block. Standalone entries use api='chat' only for Bearer-auth header shape.
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
    Auth for an image request. HTTP requests follow the chat proxy-key rules; CLI calls
    use the configured provider key because there is no Flask request context.
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
    Bind only image-model pricing. Chat backend, model and text prices are untouched.
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
    """Validate image config at startup/reload and disable it if it cannot run."""
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

    # A bad default is a startup warning, not a per-request validation error.
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
    """A size accepted by the model family, not just a small enum of common sizes."""
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
    A caller-supplied filename reduced to a portable stem. Directories, traversal,
    unusual characters, device names and trailing dots are rejected rather than rewritten.
    """
    name = str(raw).strip()
    if not name:
        raise ImageRequestError("filename must not be empty.")

    stem = os.path.splitext(name)[0]
    if not FILENAME_RE.match(stem) or ".." in stem:
        raise ImageRequestError(
            f"filename must be a bare name matching [A-Za-z0-9._-] with no path separators, got {raw!r}."
        )

    stem = stem[:FILENAME_MAX_STEM]

    # Reached by 'name..'; splitext already treated one trailing dot as the extension.
    if stem.endswith("."):
        raise ImageRequestError(f"filename must not end in a dot, got {raw!r}.")

    if stem.lower() in RESERVED_STEMS:
        raise ImageRequestError(f"filename {raw!r} is a reserved device name on Windows; choose another.")

    return stem


# Reference image slots
#
# Persistent console-managed references; each use re-validates the current file.
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


def load_data_url(spec: str) -> ReferenceImage:
    """
    One reference from an inline `data:` URL.

    Base64 only: the percent-encoded alternative would have to be decoded before it could
    be sized, and nothing that holds an image inline emits it. Otherwise this is an upload
    by another route -- bytes the caller was already holding, reaching no filesystem -- so
    it is judged by exactly the checks an upload gets.

    It exists because a client can hold a picture without holding a file: a mask drawn in
    an editor has no path to name, and a batch accepts no multipart upload at all.
    """
    match = DATA_URL_RE.match(spec)
    if match is None:
        raise ImageRequestError("a data URL must look like 'data:image/png;base64,<payload>'.")
    if "base64" not in (match.group(2) or "").lower():
        raise ImageRequestError("only base64 data URLs are accepted as an image reference.")

    payload = spec[match.end():]
    # Sized before decoding: four base64 characters carry three bytes, so an oversized
    # payload is refused without ever being materialised.
    estimated = len(payload)//4*3
    if estimated > cfg.image_edit_max_bytes:
        raise ImageRequestError(
            f"the inline image is about {estimated:,} bytes; the limit is "
            f"{cfg.image_edit_max_bytes:,} (IMAGE_EDIT_MAX_BYTES)."
        )

    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRequestError(f"the inline image is not valid Base64: {exc}")

    # Advisory only -- describe_image sniffs the bytes, as it does for every other route.
    mediatype = (match.group(1) or "").strip().lower()
    name      = f"inline.{mediatype.split('/')[-1]}" if mediatype.startswith("image/") else "inline"
    return load_uploaded_reference(name, data)


def load_reference(spec: Any, from_prompt: bool, slot: int = 0) -> ReferenceImage:
    """
    One validated reference image, from a slot number, a data URL or a filesystem path.

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

    text = spec.strip()

    # Checked before the path machinery because a data URL reaches no filesystem: there is
    # nothing to resolve, no allowlist to consult, and no prompt-path setting to honour.
    if text[:5].lower() == "data:":
        return load_data_url(text)

    raw = os.path.expanduser(text)

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


def resolve_mask(fields: Dict[str, Any], from_prompt: bool, images: List[ReferenceImage]) -> Optional[ReferenceImage]:
    """
    The mask this request runs with: the one it named, none because it said so, or the one
    left set from the console.

    The console-set mask is a convenience for the CLI, but it is also a trap for every
    other caller -- it would otherwise apply to requests that never asked for it and have
    no way to say no. So `mask: false` (or "none"/"off") is a value in its own right,
    distinct from leaving the key out, which still inherits.

    With a batch there are no local references to measure against; validate_mask already
    guards its geometry check on having one, so it falls back to the format and alpha
    rules, which are the only ones that can be checked from here anyway.
    """
    given = fields.get("mask") if "mask" in fields else None
    if isinstance(given, str):
        given = given.strip()

    if given is False or (isinstance(given, str) and given.lower() in {"none", "off", "false"}):
        return None

    if given is not None and given != "":
        mask = load_reference(given, from_prompt)
    else:
        mask = stored_mask()

    if mask is not None:
        validate_mask(mask, images)
    return mask


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


def validate_mask_region(raw: Any) -> Dict[str, Any]:
    """
    The rectangles a mask was drawn as, as the client states them.

    Recorded verbatim and never sent upstream: what the provider edits is the PNG, and this is
    what the PNG *meant* -- the form that can be read back, corrected and asked for again. The
    proxy does not rasterise it, compare it against the mask, or act on it in any way.

    Accepts a JSON string as well as an object, because a multipart edit has no way to carry
    structure -- the same reason source_files and the job stamps are read that way there.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ImageRequestError(f"mask_region must be a JSON object: {exc}")

    if not isinstance(raw, dict):
        raise ImageRequestError(f"mask_region must be an object, got {type(raw).__name__}.")

    size  = raw.get("size")
    rects = raw.get("rects")
    if not isinstance(size, dict):
        raise ImageRequestError("mask_region.size must be an object with width and height.")
    if not isinstance(rects, list):
        raise ImageRequestError("mask_region.rects must be a list.")
    if len(rects) > MASK_RECT_MAX:
        raise ImageRequestError(f"{len(rects)} mask_region rectangles; the limit is {MASK_RECT_MAX}.")

    def whole(value: Any, where: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ImageRequestError(f"mask_region.{where} must be a number, got {value!r}.")

    clean_rects: List[Dict[str, Any]] = []
    for index, rect in enumerate(rects):
        if not isinstance(rect, dict):
            raise ImageRequestError(f"mask_region.rects[{index}] must be an object.")
        unknown = sorted(set(rect) - MASK_RECT_KEYS)
        if unknown:
            raise ImageRequestError(f"unsupported mask_region.rects[{index}] field(s) {unknown}; allowed: {sorted(MASK_RECT_KEYS)}.")
        mode = str(rect.get("mode", "add"))
        if mode not in ("add", "subtract"):
            raise ImageRequestError(f"mask_region.rects[{index}].mode must be 'add' or 'subtract', got {mode!r}.")
        clean_rects.append({
            "x"   : whole(rect.get("x", 0), f"rects[{index}].x"),
            "y"   : whole(rect.get("y", 0), f"rects[{index}].y"),
            "w"   : whole(rect.get("w", 0), f"rects[{index}].w"),
            "h"   : whole(rect.get("h", 0), f"rects[{index}].h"),
            "mode": mode,
        })

    # A region with no rectangles selects nothing, which is not a region worth recording.
    if not clean_rects:
        return {}
    return {
        "size" : {"width": whole(size.get("width", 0), "size.width"), "height": whole(size.get("height", 0), "size.height")},
        "rects": clean_rects,
    }


def validate_annotation(raw: Any, field_name: str, allowed: set) -> Dict[str, str]:
    """
    One job stamp: a flat map of short strings under known keys.

    Accepts a JSON string as well as an object, because a multipart edit has no way to carry
    structure -- the same reason source_files is read that way there.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ImageRequestError(f"{field_name} must be a JSON object: {exc}")

    if not isinstance(raw, dict):
        raise ImageRequestError(f"{field_name} must be an object, got {type(raw).__name__}.")

    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ImageRequestError(f"unsupported {field_name} field(s) {unknown}; allowed: {sorted(allowed)}.")

    stamp: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ImageRequestError(f"{field_name}.{key} must be a string, got {type(value).__name__}.")
        text = str(value).strip()
        if not text:
            continue
        if len(text) > JOB_VALUE_MAX:
            raise ImageRequestError(f"{field_name}.{key} is {len(text)} characters; the limit is {JOB_VALUE_MAX}.")
        stamp[key] = text
    return stamp


def resolve_edit_inputs(fields: Dict[str, Any], source: str, batch: bool) -> Tuple[List[ReferenceImage], Optional[ReferenceImage], List[str]]:
    """
    Decide whether this request edits, and against what:

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

    The mask is orthogonal to all of that and resolved by resolve_mask: named, refused
    with `mask: false`, or inherited from the console when the key is left out.
    """
    has_images   = "images"   in fields and fields["images"] not in (None, [], "")
    has_file_ids = "file_ids" in fields and fields["file_ids"] not in (None, [], "")
    has_mask     = "mask"     in fields and fields["mask"] not in (None, "", False)

    # Pure, so it can be settled here and used by either branch below.
    from_prompt = source == "chat"

    edit_given = "edit" in fields
    edit_flag  = validate_bool("edit", fields["edit"]) if edit_given else False

    if not (has_images or has_file_ids or edit_flag):
        if edit_given and has_mask:
            raise ImageRequestError("mask was given with edit: false; a mask only applies to an edit.")
        return [], None, []

    if not cfg.image_edit_enabled:
        raise ImageConfigError("Image editing is disabled (IMAGE_EDIT_ENABLED=false).")

    if has_images and has_file_ids:
        raise ImageRequestError("images and file_ids are mutually exclusive: images are uploaded from this machine, file_ids already live on the provider.")
    if edit_given and not edit_flag and (has_images or has_file_ids):
        raise ImageRequestError("edit: false was given together with reference images. Remove one of them.")

    if has_file_ids:
        if not batch:
            raise ImageRequestError("file_ids is for Batch API edits only. Add batch: true, or use images for an immediate edit.")
        # A batch accepts no multipart upload, but the JSON body takes a mask by reference
        # exactly as it takes the images -- so the bytes ride along as a data URL. Verified
        # against the provider: a batch line carrying one validates and runs.
        return [], resolve_mask(fields, from_prompt, []), validate_file_ids(fields["file_ids"])

    if batch:
        raise ImageRequestError("local reference images cannot be batched. Upload them to the provider and pass file_ids, or drop batch: true.")

    if has_images:
        images = load_references(fields["images"], from_prompt)
    else:
        images = filled_slots()
        if not images:
            raise ImageRequestError("edit: true was requested but no image slots are filled. Use 'img edit set 1 <path>'.")
        if len(images) > cfg.image_edit_max_images:
            raise ImageRequestError(f"{len(images)} slots are filled; the limit is {cfg.image_edit_max_images} (IMAGE_EDIT_MAX_IMAGES).")

    if not images:
        raise ImageRequestError("no usable reference images were given.")

    return images, resolve_mask(fields, from_prompt, images), []


def build_request(overrides: Optional[Dict[str, Any]] = None, source: str = "direct") -> ImageRequest:
    """
    One validated request, from configured defaults plus explicit caller overrides.
    """
    if not cfg.image_enabled:
        raise ImageConfigError("Image generation is disabled (IMAGE_GENERATION_ENABLED=false).")

    fields = dict(overrides or {})

    # Image responses are not streamed by this proxy.
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

    mask_region = validate_mask_region(fields["mask_region"]) if fields.get("mask_region") else {}
    # A region describes a mask, so one arriving without it describes nothing. Worth reporting
    # rather than recording: it means the caller sent the two halves of a mask down different paths.
    if mask_region and mask is None:
        raise ImageRequestError("mask_region was given without a mask; it describes one, and records nothing on its own.")

    job_group = validate_annotation(fields["job_group"], "job_group", JOB_GROUP_KEYS) if fields.get("job_group") else {}
    job       = validate_annotation(fields["job"]      , "job"      , JOB_KEYS      ) if fields.get("job")       else {}
    # The attempt's id is what joins the images of one request to each other and to a group. A
    # group named without one describes nothing that can be found again, so it is a mistake worth
    # reporting rather than a stamp worth writing.
    if job and not job.get("id"):
        raise ImageRequestError("job.id is required when a job is given.")
    if job_group and not job:
        raise ImageRequestError("job_group needs a job; a group with no attempt in it records nothing joinable.")

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
        mask_region   = mask_region,
        file_ids      = file_ids,
        source_files  = source_files,
        job_group     = job_group,
        job           = job,
    )


def reference_param(reference: ReferenceImage) -> Dict[str, str]:
    """
    One local reference as the JSON body wants it: the provider's image reference object,
    carrying the bytes inline.

    Read whole rather than streamed, unlike the multipart form, because a JSON body has to
    be materialised in full before it can be sent -- and because the only thing that takes
    this route is a mask, which is a flat two-colour PNG.
    """
    payload = reference.data
    if payload is None:
        with open(reference.path, "rb") as handle:
            payload = handle.read()
    return {"image_url": f"data:{reference.mime};base64,{base64.b64encode(payload).decode('ascii')}"}


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

    # Same reference shape as an entry of 'images', which is what lets a batch carry a
    # mask at all: there is no upload to make, so the bytes go inline as a data URL.
    if req.mask is not None:
        body["mask"] = reference_param(req.mask)

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


def confine(directory: str, candidate: str) -> str:
    """
    `candidate` back again if it really is inside `directory`, and a rejection if it is not.

    Storage is the last place able to catch a name that somehow survived sanitisation: everything
    above validates a *stem*, and this validates the path it turned into. A path escaping here
    would be the model, or a client, writing wherever it liked.
    """
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


def taken_names(directory: str) -> set:
    """
    Every name `directory` holds, lowercased -- so a name is judged taken case-insensitively,
    which is stricter than Linux requires and exactly what Windows enforces. Deciding it by
    os.path.exists would make the same directory behave differently on the two -- 'Cat.png'
    beside 'cat.png' on one, silently indexed on the other -- and the manifest is meant to
    describe a directory that can be copied between them unchanged. Call with STORAGE_LOCK held.
    """
    try:
        return {entry.lower() for entry in os.listdir(directory)}
    except OSError:
        # A directory we cannot list is one we are about to fail to write into anyway; let
        # that failure happen at the write, where it says something useful.
        return set()


def allocate_path(directory: str, stem: str, extension: str) -> str:
    """
    A free path under `directory`. An existing file is never overwritten: a linear index
    is appended until the name is free. Call with STORAGE_LOCK held.
    """
    taken = taken_names(directory)

    name  = f"{stem}{extension}"
    index = 1
    while name.lower() in taken:
        name = f"{stem}_{index}{extension}"
        index += 1

    return confine(directory, os.path.join(directory, name))


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


def persist_mask(req: ImageRequest, directory: str, stem: str) -> str:
    """
    Where the manifest should say this request's mask is, writing it out first if it only
    ever existed as bytes.

    A mask named by path already has somewhere to point at and is left where it is. One
    that arrived as an upload or a data URL has nowhere, and `upload:mask.png` records a
    name with nothing behind it -- so it is written beside the images it shaped, under
    `masks/`, and the manifest names that copy. One file per request, whatever `n` was:
    the mask belongs to the request, not to any one image it produced.

    Never fatal. The images are already on disk by the time this runs, and losing the
    record of a mask is not worth losing the record of the pictures.
    """
    mask = req.mask
    if mask is None:
        return ""
    if mask.path:
        return manifest_path(mask.path, directory)

    folder = os.path.join(directory, MASK_SUBDIR)
    try:
        os.makedirs(folder, exist_ok=True)
        with STORAGE_LOCK:
            path     = allocate_path(folder, stem, FORMAT_EXTENSIONS.get(mask.format, ".png"))
            tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
            try:
                with open(tmp_path, "wb") as handle:
                    handle.write(mask.data or b"")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, path)
            except Exception:
                try: os.unlink(tmp_path)
                except OSError: pass
                raise
    except Exception as exc:
        print(f"WARNING: could not write the mask into {folder} ({exc}).")
        return mask.origin()

    return manifest_path(path, directory)


def decode_image(b64_data: Any, index: int) -> bytes:
    if not isinstance(b64_data, str) or not b64_data:
        raise ProviderError(502, {"error": {"message": f"image {index} carried no b64_json data."}},
                            "image response carried no data")
    try:
        return base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(502, {"error": {"message": f"image {index} was not valid Base64: {exc}"}},
                            "invalid Base64 in image response")


def manifest_path(path: str, directory: str) -> str:
    """
    Manifest paths are relative to the manifest directory, with forward slashes.
    Absolute paths remain only when a relative path cannot be expressed.
    """
    if not path:
        return ""
    try:
        relative = os.path.relpath(path, directory)
    except ValueError:
        return path.replace("\\", "/")
    return relative.replace("\\", "/")


def manifest_source_files(req: ImageRequest) -> List[Dict[str, str]]:
    """
    Lineage for one request. Declared source_files win; otherwise derive from references.
    """
    directory = output_dir_path()

    if req.source_files:
        return [
            {key: (manifest_path(value, directory) if key == "path" else value) for key, value in entry.items()}
            for entry in req.source_files
        ]

    sources: List[Dict[str, str]] = []
    for reference in req.images:
        if reference.path:
            sources.append({"path": manifest_path(reference.path, directory), "file": os.path.basename(reference.path)})
        else:
            sources.append({"file": reference.filename()})
    return sources


def append_manifest(req: ImageRequest, saved: List[SavedImage], cost_usd: float, usage: Dict[str, Any], estimated: bool, batch_id: str = "") -> None:
    """
    Append one record per image. A corrupt manifest is replaced rather than aborting
    after the image has already been saved.
    """
    if not cfg.image_manifest_enabled or not saved:
        return

    directory = output_dir_path()
    path      = os.path.join(directory, MANIFEST_FILE)
    created   = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Once for the request, before the loop: every record of a batch of n points at the
    # same mask file, and writing it n times would be n different files saying one thing.
    mask_file   = persist_mask(req, directory, req.filename or default_stem(saved[0].image_id))
    # What the mask was, rather than merely where its picture went. The region is the durable half
    # -- it can be read back, corrected and asked for again, while the path only ever named a file
    # that a move or a rename could leave pointing at nothing.
    mask_record = dict(req.mask_region)
    if mask_file:
        mask_record["file"] = mask_file

    records = []
    for image in saved:
        record: Dict[str, Any] = {
            "image_id"           : image.image_id,
            "path"               : manifest_path(os.path.abspath(image.path), directory),
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
        if req.images:
            record["source_images"] = [
                origin if origin.startswith("upload:") else manifest_path(origin, directory)
                for origin in (reference.origin() for reference in req.images)
            ]
        if mask_record:
            record["mask"] = mask_record
        if req.file_ids:
            record["source_references"] = list(req.file_ids)
        sources = manifest_source_files(req)
        if sources:
            record["source_files"] = sources
        if req.job_group:
            record["job_group"] = dict(req.job_group)
        if req.job:
            record["job"] = dict(req.job)
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


def read_manifest(path: str) -> List[Any]:
    """
    The manifest as a list, for a route about to rewrite it. Call with STORAGE_LOCK held.

    Unreadable is an error here rather than the warning append_manifest settles for: appending
    has an image already on disk and must not lose it to a corrupt manifest, while a route that
    rewrites has nothing to salvage and no business overwriting what it could not read.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
    except Exception as exc:
        raise ImageRequestError(f"could not read {MANIFEST_FILE}: {exc}")
    if not isinstance(records, list):
        raise ImageRequestError(f"{MANIFEST_FILE} is not a JSON list.")
    return records


def write_manifest(path: str, records: List[Any]) -> None:
    """
    The manifest written whole through a temporary file, as appending is: a failure must leave
    the manifest that was there rather than half of a new one. Call with STORAGE_LOCK held.
    """
    tmp_path = f"{path}.{uuid.uuid4().hex[:8]}.part"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise ImageRequestError(f"could not write {MANIFEST_FILE}: {exc}")


def find_record(records: List[Any], image_id: str, file: str) -> Optional[Dict[str, Any]]:
    """
    The record a client is naming: by the proxy's own id, or by filename for one written before
    ids were recorded. An id is unique; a filename is only as unique as the directory holding it,
    so it is the fallback rather than the join.
    """
    for record in records:
        if not isinstance(record, dict):
            continue
        if image_id and str(record.get("image_id", "")) == image_id:
            return record
        if not image_id and file:
            named = str(record.get("file", "")) or os.path.basename(str(record.get("path", "")))
            if named == file:
                return record
    return None


def patch_manifest(updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Correct records already in the manifest: what the request asked for, and which attempt it
    belongs to.
    """
    if not cfg.image_manifest_enabled:
        raise ImageRequestError("manifests are disabled on this proxy (IMAGE_MANIFEST_ENABLED=false).")

    def as_text(value: Any, where: str) -> str:
        if not isinstance(value, (str, int, float)):
            raise ImageRequestError(f"{where} must be a string, got {type(value).__name__}.")
        return str(value).strip()

    def as_params(raw: Any, where: str) -> Dict[str, Any]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw.strip() or "{}")
            except ValueError as exc:
                raise ImageRequestError(f"{where} must be a JSON object: {exc}")
        if not isinstance(raw, dict):
            raise ImageRequestError(f"{where} must be an object, got {type(raw).__name__}.")
        unknown = sorted(set(raw) - PATCH_PARAM_KEYS)
        if unknown:
            raise ImageRequestError(f"unsupported {where} field(s) {unknown}; allowed: {sorted(PATCH_PARAM_KEYS)}.")
        clean: Dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                clean[key] = None
                continue
            if key == "n":
                try:
                    clean[key] = int(value)
                except (TypeError, ValueError):
                    raise ImageRequestError(f"{where}.n must be an integer, got {value!r}.")
            else:
                clean[key] = as_text(value, f"{where}.{key}")
        return clean

    wanted: List[Dict[str, Any]] = []
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            raise ImageRequestError(f"updates[{index}] must be an object.")

        unknown = sorted(set(update) - PATCH_STRING_KEYS - PATCH_OTHER_KEYS)
        if unknown:
            raise ImageRequestError(
                f"unsupported updates[{index}] field(s) {unknown}; this route records what a request "
                f"asked for, not what the file is. Allowed: {sorted(PATCH_STRING_KEYS | PATCH_OTHER_KEYS)}.")

        image_id = str(update.get("image_id", "") or "").strip()
        file     = os.path.basename(str(update.get("file", "") or "").strip())
        if not image_id and not file:
            raise ImageRequestError(f"updates[{index}] must name a record by image_id or file.")

        # Present-and-null is "clear it"; absent is "leave it as it is". The two are different
        # instructions, so which keys were given has to survive validation.
        entry: Dict[str, Any] = {"image_id": image_id, "file": file}

        for key in sorted(PATCH_STRING_KEYS):
            if key not in update:
                continue
            entry[key] = None if update[key] is None else as_text(update[key], f"updates[{index}].{key}")
        if entry.get("operation") not in (None, "", "generate", "edit") and "operation" in entry:
            raise ImageRequestError(f"updates[{index}].operation must be 'generate' or 'edit', got {entry['operation']!r}.")

        if "request_parameters" in update:
            entry["request_parameters"] = (
                None if update["request_parameters"] is None
                else as_params(update["request_parameters"], f"updates[{index}].request_parameters"))
        if "mask" in update:
            entry["mask"] = None if update["mask"] is None else validate_mask_region(update["mask"])
        if "source_files" in update:
            entry["source_files"] = validate_source_files(update["source_files"]) if update["source_files"] else None
        if "estimated_cost_usd" in update:
            if update["estimated_cost_usd"] is None:
                entry["estimated_cost_usd"] = None
            else:
                try:
                    entry["estimated_cost_usd"] = float(update["estimated_cost_usd"])
                except (TypeError, ValueError):
                    raise ImageRequestError(f"updates[{index}].estimated_cost_usd must be a number.")
        if "cost_is_estimate" in update:
            entry["cost_is_estimate"] = None if update["cost_is_estimate"] is None else bool(update["cost_is_estimate"])

        if "job_group" in update:
            entry["job_group"] = validate_annotation(update["job_group"], "job_group", JOB_GROUP_KEYS) if update["job_group"] else None
        if "job" in update:
            entry["job"] = validate_annotation(update["job"], "job", JOB_KEYS) if update["job"] else None
            if entry["job"] and not entry["job"].get("id"):
                raise ImageRequestError(f"updates[{index}].job.id is required when a job is given.")
        wanted.append(entry)

    directory = output_dir_path()
    path      = os.path.join(directory, MANIFEST_FILE)

    with STORAGE_LOCK:
        if not os.path.exists(path):
            return {"changed": 0, "missing": [entry["image_id"] or entry["file"] for entry in wanted]}
        records = read_manifest(path)

        changed = 0
        missing: List[str] = []
        for entry in wanted:
            found = find_record(records, entry["image_id"], entry["file"])
            if found is None:
                missing.append(entry["image_id"] or entry["file"])
                continue

            # Compared over the whole record rather than key by key: several kinds of field are
            # being written now, and "did this actually change anything" is one question, not nine.
            before = json.dumps(found, sort_keys=True, default=str)

            for key in sorted(PATCH_STRING_KEYS):
                if key not in entry:
                    continue
                if entry[key]:
                    found[key] = entry[key]
                else:
                    found.pop(key, None)

            for key in ("job_group", "job", "source_files"):
                if key not in entry:
                    continue
                if entry[key]:
                    found[key] = [dict(item) for item in entry[key]] if key == "source_files" else dict(entry[key])
                else:
                    found.pop(key, None)

            # The region merges over what the record already said, so correcting the rectangles
            # keeps the picture this proxy rasterised -- which it cannot render again.
            if "mask" in entry:
                if entry["mask"]:
                    was = found.get("mask") if isinstance(found.get("mask"), dict) else {}
                    found["mask"] = {**was, **entry["mask"]}
                else:
                    found.pop("mask", None)

            # Merged for the reason the mask is: a proxy records parameters this route does not
            # model, and a moderation setting must not be lost to a correction of the size beside it.
            if "request_parameters" in entry:
                if entry["request_parameters"] is None:
                    found.pop("request_parameters", None)
                else:
                    params = dict(found.get("request_parameters") or {}) if isinstance(found.get("request_parameters"), dict) else {}
                    for key, value in entry["request_parameters"].items():
                        if value in (None, "", 0):
                            params.pop(key, None)
                        else:
                            params[key] = value
                    if params:
                        found["request_parameters"] = params
                    else:
                        found.pop("request_parameters", None)

            # The flag says how to read the number, so it goes when there is no number to read.
            if "estimated_cost_usd" in entry:
                if entry["estimated_cost_usd"]:
                    found["estimated_cost_usd"] = entry["estimated_cost_usd"]
                else:
                    found.pop("estimated_cost_usd", None)
                    found.pop("cost_is_estimate", None)
            if "cost_is_estimate" in entry and found.get("estimated_cost_usd"):
                if entry["cost_is_estimate"] is None:
                    found.pop("cost_is_estimate", None)
                else:
                    found["cost_is_estimate"] = entry["cost_is_estimate"]

            if json.dumps(found, sort_keys=True, default=str) != before:
                changed += 1

        if changed:
            write_manifest(path, records)

    return {"changed": changed, "missing": missing}


def rename_in_records(records: List[Any], directory: str, was: str, now: str) -> int:
    """
    Every mention of one file rewritten to its new name, and how many records had to change.

    The record *of* it, and every record naming it as a source, as a source image or as a mask --
    an edit's lineage points at a file on this disk precisely because a provider id expires and a
    path does not, so a rename that only fixed the record of the file itself would quietly break
    the history of everything made from it.

    The mirror of `renameInRecords` in mini-img's src/manifest.ts, which does exactly this for the
    manifests that app owns. Records are rewritten in place; the caller writes the list back.
    """
    def here(value: str) -> str:
        return os.path.normcase(os.path.abspath(os.path.join(directory, value)))

    before   = here(was)
    recorded = manifest_path(os.path.join(directory, now), directory)

    def is_target(value: Any) -> bool:
        text = str(value or "")
        # An upload never had a name on this disk to keep up to date.
        if not text or text.startswith("upload:"):
            return False
        return here(text) == before

    changed = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        touched = False

        if is_target(record.get("path") or record.get("file")):
            record["file"] = now
            record["path"] = recorded
            touched = True

        sources = record.get("source_files")
        if isinstance(sources, list):
            for entry in sources:
                if isinstance(entry, dict) and is_target(entry.get("path")):
                    entry["path"] = recorded
                    # `file` follows only where it was recorded: absent means the reference went
                    # up as bytes, and never had a name here to keep up to date.
                    if entry.get("file"):
                        entry["file"] = now
                    touched = True

        images = record.get("source_images")
        if isinstance(images, list):
            for index, origin in enumerate(images):
                if isinstance(origin, str) and is_target(origin):
                    images[index] = recorded
                    touched = True

        mask = record.get("mask")
        if isinstance(mask, dict) and is_target(mask.get("file")):
            mask["file"] = recorded
            touched = True

        if touched:
            changed += 1

    return changed


def rename_image(image_id: str, file: str, filename: str) -> Dict[str, Any]:
    """
    Rename one image this proxy wrote, and follow the new name through the manifest.

    Deliberately not part of PATCH /v1/images/manifest, which refuses `file` on the grounds that a
    record's filename is measurement rather than testimony -- it describes the file this proxy
    wrote. That still holds: a rename does not correct the measurement, it changes the thing being
    measured, and only the process holding STORAGE_LOCK can move the file and rewrite its record
    without racing a job landing beside it.

    The extension is imposed rather than chosen, as it is on the way in: the bytes decide the
    format, and a rename from .png to .webp would leave the manifest describing a file that is
    not what it says it is.
    """
    if not cfg.image_manifest_enabled:
        raise ImageRequestError("manifests are disabled on this proxy (IMAGE_MANIFEST_ENABLED=false).")

    image_id = str(image_id or "").strip()
    file     = os.path.basename(str(file or "").strip())
    if not image_id and not file:
        raise ImageRequestError("name the image to rename by image_id or file.")

    wanted    = str(filename or "").strip()
    stem      = sanitize_filename_stem(wanted)
    asked     = os.path.splitext(wanted)[1]

    directory = output_dir_path()
    path      = os.path.join(directory, MANIFEST_FILE)

    with STORAGE_LOCK:
        if not os.path.exists(path):
            raise ImageRequestError(f"there is no {MANIFEST_FILE} in the output directory.")

        records = read_manifest(path)
        found   = find_record(records, image_id, file)
        if found is None:
            raise ImageRequestError(f"no record of {image_id or file} in {MANIFEST_FILE}.")

        was = str(found.get("file", "")) or os.path.basename(str(found.get("path", "")))
        if not was:
            raise ImageRequestError(f"the record of {image_id or file} does not name a file to rename.")

        held = os.path.splitext(was)[1]
        if asked.lower() != held.lower():
            raise ImageRequestError(
                f"keep the {held} -- a rename does not change the format, and {wanted!r} asks for "
                f"{asked or 'no extension'}.")

        name = f"{stem}{held}"
        if name == was:
            return {"name": was, "changed": 0}

        old_path = confine(directory, os.path.join(directory, was))
        new_path = confine(directory, os.path.join(directory, name))
        if not os.path.exists(old_path):
            raise ImageRequestError(f"{was} is recorded but is no longer in the output directory.")

        # A change of case alone is the file being renamed to itself, which Windows performs
        # happily and which `taken_names` would otherwise read as a collision with itself.
        itself = os.path.normcase(old_path) == os.path.normcase(new_path)
        if not itself and name.lower() in taken_names(directory):
            raise ImageRequestError(f"{name} is already taken in the output directory.")

        os.replace(old_path, new_path)

        changed = rename_in_records(records, directory, was, name)
        write_manifest(path, records)

    return {"name": name, "changed": changed}


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


# Batch API. Submission and retrieval are separate because chat turns cannot wait on it.
def batch_state_path() -> str:
    """Reading state must not create an unused output directory."""
    return os.path.join(os.path.abspath(cfg.image_output_dir), BATCH_STATE_FILE)


def number_legacy_batches() -> None:
    """Number batches recorded before short references existed."""
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
    """Next short reference number; numbers are never reused."""
    highest = 0
    for entry in state.values():
        try: highest = max(highest, int(entry.get("number") or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


def resolve_batch_id(token: str) -> str:
    """Resolve a short number, or pass through a raw provider batch id."""
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
    """Persist request parameters the provider will not return with completed images."""
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
        "source_files"  : [dict(entry) for entry in req.source_files],
        # Carried so a batch retrieved days later is stamped as this proxy writes its record,
        # whether or not the client that submitted it is still running.
        "mask_region"   : dict(req.mask_region),
        "job_group"     : dict(req.job_group),
        "job"           : dict(req.job),
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
        # Re-validated rather than trusted: this comes off disk, where a hand-edited state file is
        # as likely as one we wrote.
        mask_region   = validate_mask_region(entry.get("mask_region")) if entry.get("mask_region") else {},
        job_group     = {str(key): str(value) for key, value in (entry.get("job_group") or {}).items() if key in JOB_GROUP_KEYS},
        job           = {str(key): str(value) for key, value in (entry.get("job")       or {}).items() if key in JOB_KEYS},
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

        if status not in BATCH_TERMINAL_STATUSES:
            # Still validating, running or finalizing: there is nothing to read yet.
            return result

        # From here the batch is settled below whatever it produced, which is not always an
        # output file. A batch whose every request was rejected still reports 'completed',
        # with only an error file to its name; a cancelled or expired one can carry results
        # for the requests that finished before it stopped. Both files are read the same
        # way -- the error lines carry the provider's reason for each failure, which is the
        # only place the user can learn why nothing came back.
        output_file_id = str(data.get("output_file_id") or "")
        error_file_id  = str(data.get("error_file_id") or "")

        if not output_file_id and not error_file_id:
            result.errors.append(f"batch {status} without an output file.")

        requests_state = entry.get("requests") or {}
        total_counts   = {"text_input": 0, "image_input": 0, "output": 0, "reported": False, "estimated_cost_usd": 0.0}
        images_saved   = 0

        # A fetch that fails here raises rather than settling the batch on a half-read
        # result: the poller retries it next pass, where losing the images would be final.
        result_lines: List[str] = []
        for file_id in (output_file_id, error_file_id):
            if file_id:
                result_lines.extend(fetch_file_content(provider, file_id).splitlines())

        for line in result_lines:
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

            # Accumulated per line, against this line's own request, because a batch may
            # mix sizes and qualities and need not come back whole: pricing the images
            # that did arrive off one arbitrary request bills the wrong rate as soon as
            # that request is not representative -- or was itself one of the rejected ones.
            if not counts["reported"]:
                total_counts["estimated_cost_usd"] += estimate_image_cost(cfg.image_model, req.quality, req.size, len(saved))

            append_manifest(req, saved, 0.0, counts, not counts["reported"], batch_id=batch_id)
            result.images.extend(saved)
            images_saved += len(saved)

        if images_saved:
            # Billed once for the whole batch, at the batch rate, rather than per output
            # line: the session report distinguishes batch from immediate spending. Only
            # the lines that came back are counted, so a batch the provider fulfilled in
            # part costs what it delivered rather than what it was asked for.
            total_counts["estimated"] = not total_counts["reported"]
            result.cost_usd = track_image_usage(total_counts, images=images_saved, batch=True, model=cfg.image_model)

        entry["retrieved"]    = True
        entry["retrieved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        entry["final_status"] = status
        entry["images"]       = [image.path for image in result.images]
        # Kept so a batch that produced nothing can still say why long after the poller
        # printed it once and moved on.
        entry["errors"]       = list(result.errors)
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
            "errors"        : list(entry.get("errors") or []),
            "requests"      : [
                {
                    "custom_id"     : custom_id,
                    # The name the user gave the job, which is what they will look for in
                    # the folder -- more use in a listing than either id.
                    "filename"      : item.get("filename", ""),
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
