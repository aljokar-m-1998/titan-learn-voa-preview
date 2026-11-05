"""Read-only access to the Titan Learn VOA dataset at one pinned commit.

Everything the Learn library serves comes from a single immutable revision, named by
VOA_LEARN_PIN. Nothing in this module takes a repository, a revision or a file path from a
caller: a request asks for a record or one media object, and the answer always comes from that
commit - so a rebuild can never change an answer underneath somebody mid-lesson.

Four layers are kept apart on purpose:

  manifest   the build description: counts, shard list, media reconciliation, and the metadata
             index this revision publishes, with its checksum
  index      the browsable metadata over all 71,291 records. It travels inside the dataset, next
             to the records it describes, and is proven to belong here before a single row is
             served. It holds no record bodies and no media bytes.
  record     one full record, read from the single shard whose id range can contain it - and
             only that shard is fetched, so booting never needs the whole 113 MB of records
  media      one archived object, content-addressed, fetched on demand with resume and hashing

If the declared index is missing, fails its checksum, or is linked to a different set of shards,
the library says so and stops: it does not read "latest", and it does not serve an index left
over from an earlier boot. Rebuilding the index from the shards is a maintenance action, asked
for by setting VOA_LEARN_MAINTENANCE_REBUILD=1, and its product stays local.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

REPO_DEFAULT = "M-hv1/Titan-Learn"
PIN_RE = re.compile(r"^[0-9a-f]{40}$")
MEDIA_PATH_RE = re.compile(r"media/[0-9a-f]{2}/[0-9a-f]{64}\.[A-Za-z0-9]{1,5}")
USER_AGENT = "TitanLearnGateway/1.0 (pinned dataset reader)"
MANIFEST_FILE = "manifest.json"

# The metadata index is only usable if it can be tied back to the records it describes. Those
# five fields are the tie: the shard digest, the two versions, the row count and the commit the
# rows were read from.
INDEX_KIND = "titan-learn-metadata-index"
INDEX_BUILDER = "voa-adapter/1"
INDEX_FIELD = "metadata_index"
MAINTENANCE_ENV = "VOA_LEARN_MAINTENANCE_REBUILD"
_LINK_FIELDS = ("records_sha256", "schema_version", "dataset_version", "rows",
                "built_from_revision")

# The columns the index is built from. Every leaf is a scalar or a short list: the transcript and
# the paragraph bodies stay in Parquet, because the index only needs to know that a body exists
# (word_count > 0 - measured equal to "paragraphs non-empty" on all 71,291 records of the pinned
# build). Reading the bodies is what raised the cold-start peak from 500 MB to 640 MB.
_INDEX_FIELDS = (
    "id", "media_status", "media_refs", "media_outcomes", "quality.flags", "arabic.status",
    "source.item_id", "source.program", "source.url", "source.published", "source.keywords",
    "source.tags", "content.kind", "content.title", "content.description",
    "content.word_count", "content.audio.url", "content.video",
    "learning.proposed_cefr", "learning.confidence", "learning.needs_human_review",
)
_INDEX_NAMES = (
    "id", "media_status", "media_refs", "media_outcomes", "flags", "arabic_status",
    "item_id", "program", "page_url", "published", "keywords",
    "tags", "kind", "title", "description", "words", "audio_url",
    "video", "cefr", "confidence", "review",
)
_REF_KEYS = ("kind", "object", "sha256", "bytes", "mime")
_OUTCOME_KEYS = ("kind", "status", "reason_class", "attempts")
assert len(_INDEX_FIELDS) == len(_INDEX_NAMES), "index projection drifted from its fields"

_lock = threading.RLock()
_build_lock = threading.Lock()  # one index load at a time, so a burst of cold requests shares it
_verified: set[str] = set()
_resolved: dict[str, Path] = {}  # repo path -> local file: a shard is fetched at most once
_state: dict[str, Any] = {
    "pin": "", "manifest": {}, "rows": {}, "order": [], "shard_ranges": [],
    "loaded_at": 0.0, "cold_start_seconds": None, "index_loaded": False,
    "refs_count": 0, "last_error": "", "index_source": "", "index_reason": "",
    "index_bytes": None,
}


class DatasetUnavailable(RuntimeError):
    """The pinned revision cannot be read - the router turns this into a 503."""


class IndexNotReady(DatasetUnavailable):
    """No usable metadata index at this commit: undeclared, unchecked, or not linked."""


def pin() -> str:
    """The pinned commit id, or "" when the environment does not name a valid one."""
    value = os.environ.get("VOA_LEARN_PIN", "").strip().lower()
    return value if PIN_RE.match(value) else ""


def repo() -> str:
    return os.environ.get("VOA_LEARN_DATASET_REPO", "").strip() or REPO_DEFAULT


def maintenance_allowed() -> bool:
    """Rebuilding from the shards is opt-in, so a broken revision cannot pass unnoticed."""
    return os.environ.get(MAINTENANCE_ENV, "").strip().lower() in ("1", "true", "yes")


def _cache_root() -> Path:
    configured = os.environ.get("VOA_LEARN_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / ".cache" / "voa-learn"


def _cache_dir() -> Path:
    current = pin()
    if not current:
        raise DatasetUnavailable("VOA_LEARN_PIN is not set to a 40-character commit id")
    path = _cache_root() / current[:12]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _token() -> str | None:
    return os.environ.get("HF_TOKEN", "").strip() or None


def _media_cache_limit() -> int:
    try:
        megabytes = int(os.environ.get("VOA_LEARN_MEDIA_CACHE_MB", "512"))
    except ValueError:
        megabytes = 512
    return max(32, megabytes) * 1_000_000


def resolve_url(repo_path: str) -> str:
    """The hub URL for one file of the pinned commit. Built here, never taken from a client."""
    return f"https://huggingface.co/datasets/{repo()}/resolve/{pin()}/{repo_path}"


def reset() -> None:
    """Forget everything derived from a revision. Used when the pin changes and by tests."""
    with _lock:
        _state.update({"pin": "", "manifest": {}, "rows": {}, "order": [], "shard_ranges": [],
                       "loaded_at": 0.0, "cold_start_seconds": None, "index_loaded": False,
                       "refs_count": 0, "last_error": "", "index_source": "",
                       "index_reason": "", "index_bytes": None})
        _verified.clear()
        _resolved.clear()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo_path: str, sha256: str = "") -> Path:
    """Fetch one file of the pinned commit, repairing a cache entry that fails its checksum.

    A file that does not hash to what the manifest recorded is almost always a damaged local
    copy, not a damaged dataset: the commit is immutable. It is dropped and fetched once more; a
    second mismatch is reported rather than served.
    """
    from huggingface_hub import hf_hub_download

    attempts = 0
    while True:
        attempts += 1
        try:
            path = Path(hf_hub_download(repo_id=repo(), repo_type="dataset", filename=repo_path,
                                        revision=pin(), token=_token()))
        except Exception as error:  # the hub raises a family of network/API errors
            raise DatasetUnavailable(
                f"{repo_path} is unreadable at {pin()[:12]}: {type(error).__name__}") from error
        if not path.is_file():
            raise DatasetUnavailable(f"{repo_path} did not materialise at {pin()[:12]}")
        if not sha256 or repo_path in _verified:
            return path
        found = _digest(path)
        if found == sha256:
            _verified.add(repo_path)
            return path
        if attempts > 1:
            raise DatasetUnavailable(
                f"{repo_path} hashes to {found[:12]}, the manifest says {sha256[:12]}")
        target = path.resolve() if path.is_symlink() else path
        if target != path:
            target.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def manifest() -> dict:
    with _lock:
        if _state["manifest"] and _state["pin"] == pin():
            return _state["manifest"]
    raw = download(MANIFEST_FILE).read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetUnavailable(f"manifest.json at {pin()[:12]} is not readable: {error}") from error
    if not parsed.get("shards"):
        raise DatasetUnavailable("manifest.json lists no record shards")
    with _lock:
        _state.update({"manifest": parsed, "pin": pin()})
    return parsed


def _index_file() -> Path:
    """Where a maintenance rebuild is kept. It is never a serving source by itself."""
    build = manifest()
    return (_cache_dir() / f"index-{build.get('accepted', 0)}-"
            f"{str(build.get('records_sha256', ''))[:12]}.json.gz")


def _haystack(record: dict) -> str:
    """The one searchable string per item: what a learner sees in a list, never the body."""
    parts = [record.get("title"), (record.get("description") or "")[:160], record.get("program"),
             record.get("keywords"), record.get("kind")]
    return " ".join(str(part) for part in parts if part).lower()


def _row_from(record: dict) -> dict:
    """The compact index row: enough to browse, filter, page and resolve media - no bodies."""
    return {
        "id": record["id"],
        "item_id": record["item_id"],
        "program": record["program"],
        "url": record["page_url"],
        "published": record["published"],
        "kind": record["kind"],
        "title": record["title"],
        "cefr": record["cefr"],
        "confidence": record["confidence"],
        "review": record["review"],
        "words": int(record["words"] or 0),
        "audio": bool(record["audio_url"]),
        "video": bool(record["video"]),
        "topics": tuple(record["tags"] or ()),
        "flags": tuple(record["flags"] or ()),
        "arabic": record["arabic_status"],
        "media_status": record["media_status"],
        "outcomes": tuple(tuple(entry.get(key) for key in _OUTCOME_KEYS)
                          for entry in (record["media_outcomes"] or [])),
        "refs": tuple(tuple((entry[key] if key != "bytes" else int(entry["bytes"] or 0))
                            for key in _REF_KEYS)
                      for entry in (record["media_refs"] or []) if entry.get("sha256")),
        "hay": _haystack(record),
    }


def _read_shard_rows(path: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(_INDEX_FIELDS))
    for record in table.rename_columns(list(_INDEX_NAMES)).to_pylist():
        yield _row_from(record)


def _shard_ranges() -> list[tuple[str, str, str]]:
    """Id range per shard, read off the manifest: no download, so booting stays cheap."""
    return [(str(shard.get("first_id", "")), str(shard.get("last_id", "")), shard["path"])
            for shard in manifest().get("shards", [])]


def _shard_file(shard: dict) -> Path:
    """Resolve one shard of the pinned commit once per process."""
    path = shard["path"]
    if path not in _resolved:
        _resolved[path] = download(path, shard.get("sha256", ""))
    return _resolved[path]


def _all_shard_files() -> list[tuple[str, Path]]:
    return [(shard["path"], _shard_file(shard)) for shard in manifest().get("shards", [])]


def index_repo_path(build: dict) -> str:
    return f"index/metadata-{str(build.get('records_sha256', 'unknown'))[:12]}.json.gz"


def index_declaration() -> dict:
    """What the manifest declares about the index - a cheap read, no download, no content."""
    build = manifest()
    declared = build.get(INDEX_FIELD) or {}
    reason = ""
    if not declared.get("path") or not declared.get("sha256"):
        reason = "this revision declares no metadata index"
    elif str(declared.get("records_sha256")) != str(build.get("records_sha256")):
        reason = "the declared index is linked to a different set of record shards"
    return {"declared": bool(declared.get("path")), "ready": not reason, "reason": reason,
            "path": declared.get("path"), "bytes": declared.get("bytes"),
            "records_sha256": str(declared.get("records_sha256", ""))[:12] or None,
            "built_from_revision": declared.get("built_from_revision"),
            "rows": declared.get("rows"),
            "maintenance_rebuild_allowed": maintenance_allowed()}


def descriptor_for(build: dict, rows: int, built_from: str) -> dict:
    """The link block that ties an index payload to the records it was built from."""
    return {
        "kind": INDEX_KIND, "builder": INDEX_BUILDER, "path": index_repo_path(build),
        "rows": rows, "records_sha256": build.get("records_sha256"),
        "schema_version": build.get("schema_version"),
        "dataset_version": build.get("dataset_version"),
        "built_from_revision": built_from,
    }


def index_payload(rows: dict, order: list, descriptor: dict) -> bytes:
    """The gzip file that gets published: the linkage fields first, then the rows themselves."""
    envelope = {key: descriptor.get(key) for key in ("kind", "builder", "records_sha256",
                                                    "schema_version", "dataset_version",
                                                    "rows", "built_from_revision")}
    # "rows" is the declared count; the data itself lives under "records" so the two can never
    # be confused - an index that overwrote its own count with a dict would pass for a wrong one
    envelope.update({"built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "order": order, "records": rows})
    return gzip.compress(json.dumps(envelope, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8"), 6)


def _normalize(stored: dict) -> dict:
    """JSON turns every tuple into a list; the router is handed the same shapes either way."""
    row = dict(stored)
    row["topics"] = tuple(row.get("topics") or ())
    row["flags"] = tuple(row.get("flags") or ())
    row["outcomes"] = tuple(tuple(entry) for entry in (row.get("outcomes") or ()))
    row["refs"] = tuple(tuple(entry) for entry in (row.get("refs") or ()))
    row["words"] = int(row.get("words") or 0)
    return row


def load_dataset_index() -> tuple[dict[str, dict], list[str], dict]:
    """Fetch the index this revision declares, and prove it belongs here before using a row of it.

    Three faults, three reasons: the bytes do not match the declared checksum, the payload is not
    linked to these records, or the counts do not add up. None of them is answered by another
    revision or by an index left in the cache from an earlier boot.
    """
    build = manifest()
    declared = index_declaration()
    if not declared["ready"]:
        raise IndexNotReady(f"{declared['reason']} at {pin()[:12]}")
    descriptor = build[INDEX_FIELD]
    path = download(descriptor["path"], descriptor["sha256"])
    try:
        envelope = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, EOFError, ValueError, UnicodeDecodeError) as error:
        raise IndexNotReady(
            f"the metadata index at {pin()[:12]} is not readable gzip/json: {type(error).__name__}")
    problems = []
    if envelope.get("kind") != INDEX_KIND:
        problems.append(f"kind is {envelope.get('kind')!r}, expected {INDEX_KIND!r}")
    for field in _LINK_FIELDS:
        if str(envelope.get(field)) != str(descriptor.get(field)):
            problems.append(f"{field} says {envelope.get(field)!r} but the manifest says "
                            f"{descriptor.get(field)!r}")
    if problems:
        raise IndexNotReady("the metadata index at "
                            f"{pin()[:12]} is not linked to these records: {'; '.join(problems)}")
    rows = {key: _normalize(value) for key, value in envelope["records"].items()}
    order = list(envelope["order"])
    if len(rows) != int(descriptor["rows"]) or len(order) != len(rows):
        raise IndexNotReady(
            f"the metadata index at {pin()[:12]} holds {len(rows)} rows "
            f"({len(order)} ordered) for {descriptor['rows']} declared")
    return rows, order, descriptor


def build_index(force: bool = False) -> None:
    """Serve the index the dataset declares; rebuild from the shards only under maintenance."""
    current = pin()
    if not force and _loaded_for(current):
        return
    with _build_lock:
        if not force and _loaded_for(current):
            return
        started = time.time()
        descriptor: dict | None
        try:
            rows, order, descriptor = load_dataset_index()
            source = "dataset"
        except DatasetUnavailable as error:
            if not maintenance_allowed():
                with _lock:
                    _state.update({"last_error": str(error), "index_reason": str(error),
                                   "index_source": "", "index_loaded": False})
                raise
            rows, order = _rebuild_from_shards(current)
            descriptor, source = None, "rebuilt-local"
        with _lock:
            _state.update({
                "pin": current, "rows": rows, "order": order,
                "shard_ranges": _shard_ranges(),
                "refs_count": sum(len(row["refs"]) for row in rows.values()),
                "loaded_at": time.time(),
                "cold_start_seconds": round(time.time() - started, 2),
                "index_loaded": True, "index_source": source, "index_reason": "",
                "index_bytes": (descriptor or {}).get("bytes"),
                "last_error": "",
            })


def _loaded_for(current: str) -> bool:
    with _lock:
        return bool(_state["index_loaded"] and _state["pin"] == current)


def _rows_from_shards(current: str) -> tuple[dict[str, dict], list[str]]:
    rows: dict[str, dict] = {}
    order: list[str] = []
    for path, origin in _all_shard_files():
        try:
            found = list(_read_shard_rows(Path(origin)))
        except DatasetUnavailable:
            raise
        except Exception as error:  # an unprojectable shard is a dataset fault, not a bug to hide
            raise DatasetUnavailable(
                f"{path} could not be indexed: {type(error).__name__}: {error}") from error
        for row in found:
            if row["id"] not in rows:
                rows[row["id"]] = row
                order.append(row["id"])
    return rows, order


def build_for_publishing() -> tuple[dict[str, dict], list[str]]:
    """The publisher's entry point: index every shard of the pinned commit, cache aside."""
    current = pin()
    if not current:
        raise DatasetUnavailable("VOA_LEARN_PIN is not set to a 40-character commit id")
    return _rows_from_shards(current)


def _rebuild_from_shards(current: str) -> tuple[dict[str, dict], list[str]]:
    """Maintenance path: index every shard and leave the product on disk where an operator can
    read it - it is not a serving source, and it is never pushed anywhere."""
    rows, order = _rows_from_shards(current)
    payload = gzip.compress(json.dumps(
        {"pin": current, "maintenance": True, "order": order, "records": rows},
        ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
    _atomic(_index_file(), payload)
    return rows, order


def _atomic(path: Path, payload: bytes) -> None:
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(payload)
    if temp.stat().st_size != len(payload):
        temp.unlink(missing_ok=True)
        raise DatasetUnavailable(f"{path.name}: short write")
    temp.replace(path)


def order() -> list[str]:
    build_index()
    with _lock:
        return list(_state["order"])


def rows() -> list[dict]:
    build_index()
    with _lock:
        return list(_state["rows"].values())


def row(item_id: str) -> dict | None:
    build_index()
    with _lock:
        return _state["rows"].get(item_id)


def record(item_id: str) -> dict | None:
    """One full record, from the single shard that can hold it - and only that shard is fetched."""
    build_index()
    import pyarrow.parquet as pq

    by_path = {shard["path"]: shard for shard in manifest().get("shards", [])}
    for first, last, path in _shard_ranges():
        if first and last and not (first <= item_id <= last):
            continue
        table = pq.read_table(_shard_file(by_path[path]), filters=[("id", "=", item_id)])
        if table.num_rows:
            return table.to_pylist()[0]
    return None


def summary() -> dict:
    """Health numbers for the pinned revision.

    Build facts come from the manifest, so this stays cheap before the index is loaded. It
    carries no content and no credentials: repository, commit, counts and the index checksum
    prefix are properties of a public dataset.
    """
    build = manifest()
    media = build.get("media") or {}
    with _lock:
        loaded = bool(_state["index_loaded"]) and _state["pin"] == pin()
        loaded_at = _state["loaded_at"]
        return {
            "dataset_repo": repo(),
            "pinned_revision": pin(),
            "dataset_version": build.get("dataset_version"),
            "schema_version": build.get("schema_version"),
            "records_count": len(_state["rows"]) if loaded else build.get("accepted"),
            "media_objects_count": media.get("unique_objects"),
            "shards": len(build.get("shards", [])),
            "last_successful_load": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(loaded_at))
                                     if loaded_at else None),
            "index_loaded": loaded,
            "index_rows_in_memory": len(_state["rows"]) if loaded else 0,
            "media_refs_indexed": _state["refs_count"] if loaded else 0,
            "cold_start_seconds": _state["cold_start_seconds"],
            "built_at": build.get("built_at"),
            "metadata_index": index_declaration(),
            "index_source": _state["index_source"] if loaded else None,
        }


def media_ref(item_id: str, kind: str) -> dict | None:
    """The dataset object holding one asset of one item, as {kind, object, sha256, bytes, mime}."""
    entry = row(item_id)
    if not entry:
        return None
    for ref in entry["refs"]:
        if ref[0] == kind:
            return dict(zip(_REF_KEYS, ref))
    return None


def _trim_media_cache(keep: Path) -> None:
    """Keep the streamed media inside its byte budget, least-recently-used first."""
    limit = _media_cache_limit()
    root = keep.parent.parent
    files = [path for path in root.rglob("*") if path.is_file() and path != keep]
    total = keep.stat().st_size + sum(path.stat().st_size for path in files)
    if total <= limit:
        return
    files.sort(key=lambda path: path.stat().st_atime)
    for path in files:
        if total <= limit:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size


def _open(request: urllib.request.Request):
    """One indirection so the resume path can be exercised without a network."""
    return urllib.request.urlopen(request, timeout=300)


def media_local(ref: dict) -> Path:
    """Bring one media object into the cache from the pinned commit and hand back the file.

    A half-written object is resumed with a Range request instead of starting over, the bytes are
    hashed while they stream, and nothing is served until the digest matches the one the dataset
    recorded for it.
    """
    object_path = ref["object"]
    if not MEDIA_PATH_RE.fullmatch(object_path):
        raise DatasetUnavailable(f"{object_path} is not a bucketed dataset media path")
    directory = _cache_dir() / "media" / object_path.split("/")[1]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(object_path).name
    expected = int(ref.get("bytes") or 0)
    if target.is_file():
        if expected and target.stat().st_size != expected:
            target.unlink(missing_ok=True)  # the wrong length cannot be the right object
        else:
            return target
    part = directory / (target.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    digest = hashlib.sha256()
    try:
        if have:
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
        with _open(urllib.request.Request(resolve_url(object_path), headers=headers)) as response:
            status = getattr(response, "status", 200) or 200
            resumed = bool(have) and status == 206
            if have and not resumed:
                have, digest = 0, hashlib.sha256()  # the range was ignored: start the part over
            with part.open("ab" if resumed else "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
    except (urllib.error.URLError, OSError) as error:
        raise DatasetUnavailable(
            f"media {object_path} could not be retrieved: {type(error).__name__}")
    if expected and part.stat().st_size != expected:
        raise DatasetUnavailable(
            f"media {object_path}: got {part.stat().st_size} of {expected} bytes")
    found = digest.hexdigest()
    if found != ref["sha256"]:
        part.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"media {object_path} hashes to {found[:12]}, the dataset says {ref['sha256'][:12]}")
    part.replace(target)
    _trim_media_cache(target)
    return target
