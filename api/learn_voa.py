"""Serving for the VOA Learning English library - read-only, pinned to one dataset commit.

The library is the Titan-Learn dataset at one immutable revision, named by VOA_LEARN_PIN. No
route here accepts a repository, a revision or a file path from a client: a request names an
item, and the answer is read from the pinned commit through core.voa_dataset. That is what
makes an answer reproducible - the same commit always produces the same payload, and a rebuild
of the dataset cannot change a screen underneath somebody mid-lesson.

What is served:

  /health                          build facts of the pinned revision (no content, no secret)
  /manifest                        the dataset's own build description
  /catalog                         totals + every value a filter can take
  /programs /levels /kinds /topics
  /items /search /recommendations  paginated browsing over the metadata index
  /items/{id}                      one full record
  /items/{id}/transcript|vocabulary|media
  /media/{id}/{kind}               the archived object, streamed with Range support
  /compat/*                        the temporary shape the current client already reads

Routes live under /api/v1/learn/voa, so they cannot collide with the legacy bundle router in
api/learn.py - that one keeps serving the old A1 client and is untouched here.

Environment:
  VOA_LEARN_PIN             required: a 40-character commit id. Without it every route 503s.
  VOA_LEARN_PREFIX          where the library is mounted; default /api/v1/learn/voa. An isolated
                            trial sets it to a preview path, and the media URLs the library hands
                            back follow it, so a trial never points at production.
  VOA_LEARN_PREVIEW_UNAUTH  1 only with a non-default prefix: lets an isolated deployment be
                            exercised without a Firebase token. On the default prefix it does
                            nothing, so the library cannot accidentally go live unauthenticated.
  VOA_LEARN_DATASET_REPO    default M-hv1/Titan-Learn
  VOA_LEARN_CACHE_DIR       default <app root>/.cache/voa-learn
  VOA_LEARN_MEDIA_CACHE_MB  disk budget for streamed media, default 512
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from core import voa_dataset
from core.security import require_user

DEFAULT_PREFIX = "/api/v1/learn/voa"
PREVIEW_UNAUTH_ENV = "VOA_LEARN_PREVIEW_UNAUTH"


def _configured_prefix() -> str:
    """Where this router is mounted. One source of truth: the mount and the URLs it emits.

    A missing leading slash is a typo in an environment variable, not a reason for the container
    to fail at import - it is repaired. Anything still not absolute falls back to the default
    mount, so a bad value can never expose the library on an unintended path.
    """
    configured = os.environ.get("VOA_LEARN_PREFIX", "").strip().rstrip("/")
    if configured and not configured.startswith("/"):
        configured = "/" + configured
    return configured or DEFAULT_PREFIX


PREFIX = _configured_prefix()

router = APIRouter(prefix=PREFIX, tags=["Titan Learn VOA"])


def preview_unauthenticated() -> bool:
    """A tokenless preview exists only off the default prefix - production cannot inherit it."""
    enabled = os.environ.get(PREVIEW_UNAUTH_ENV, "").strip().lower() in ("1", "true", "yes")
    return enabled and PREFIX != DEFAULT_PREFIX

# Two id shapes exist across builds of this dataset: the pinned revision uses voa:<digits>,
# the earlier fetcher wrote voa:<program>:<digits>. Both are accepted; neither may carry a
# path separator, because an id also appears in the stream URL the library hands back.
ITEM_ID_RE = re.compile(r"^voa:[0-9A-Za-z][0-9A-Za-z ()&',._:@-]{0,79}$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?]) +")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LADDER = ["A1", "A2", "B1", "B2", "C1"]
UNCLASSIFIED = "(not classified)"
SEARCH_FIELDS = ["title", "description", "program", "keywords", "kind"]
SORTS = ["dataset", "newest", "oldest", "shortest", "longest", "level"]
MEDIA_KINDS = ("audio", "video")
RATE_WINDOW = 60
RATE_LIMIT = 180
MAX_PAGE = 100
_RATE: dict[str, list[float]] = defaultdict(list)

_facets_lock = threading.Lock()
_facets: dict[str, Any] = {"key": "", "data": {}}


# ---------------------------------------------------------------- request plumbing


def _revision() -> str:
    pinned = voa_dataset.pin()
    if not pinned:
        raise HTTPException(503, "VOA_LEARN_PIN is not set to a 40-character commit id")
    return pinned


def _check_rate(uid: str) -> None:
    now = time.time()
    recent = [stamp for stamp in _RATE[uid] if now - stamp < RATE_WINDOW]
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(429, "Too many VOA library requests")
    recent.append(now)
    _RATE[uid] = recent


async def voa_user(authorization: str | None = Header(default=None)) -> str:
    """Content routes stay behind the Firebase check the rest of the gateway uses.

    The single exception is an explicitly isolated preview mount, where the dataset is public and
    the point is to exercise the route from outside. It cannot apply to the default prefix, and
    the rate limit still runs.
    """
    if preview_unauthenticated():
        _check_rate("preview-anonymous")
        return "preview-anonymous"
    uid = await require_user(authorization)
    _check_rate(uid)
    return uid


def _etag(value: str) -> str:
    return f'"{hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]}"'


def _respond(data: Any, request: Request, etag_seed: str, status_code: int = 200) -> Response:
    quoted = _etag(etag_seed)
    headers = {"ETag": quoted, "Cache-Control": "private, max-age=300",
               "X-Learn-Revision": _revision()}
    if request.headers.get("if-none-match") == quoted and status_code == 200:
        return Response(status_code=304, headers=headers)
    return Response(content=json.dumps(data, ensure_ascii=False),
                    media_type="application/json; charset=utf-8", headers=headers,
                    status_code=status_code)


async def _offload(function: Callable[..., Any], *args: Any) -> Any:
    """Parquet and hub reads block: keep them off the event loop, and 503 when pinned reads fail."""
    try:
        return await run_in_threadpool(function, *args)
    except voa_dataset.DatasetUnavailable as error:
        raise HTTPException(503, str(error)[:300]) from error
    except HTTPException:
        raise


def _resource_sample() -> dict:
    """What this process holds, reported instead of guessed. No content and no credentials."""
    sample = {"rows_in_memory": voa_dataset.summary()["index_rows_in_memory"],
              "prefix": PREFIX}
    try:
        import resource

        sample["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except (ImportError, ValueError):
        sample["peak_rss_mb"] = None      # ru_maxrss is unix-only; the container has it, Windows has not
    try:
        root = os.environ.get("VOA_LEARN_CACHE_DIR", "").strip() or str(
            Path(__file__).resolve().parents[1] / ".cache" / "voa-learn")
        Path(root).mkdir(parents=True, exist_ok=True)
        sample["cache_free_mb"] = round(__import__("shutil").disk_usage(root)[2] / 1e6)
    except OSError:
        sample["cache_free_mb"] = None
    return sample


def _valid_id(item_id: str) -> str:
    if not ITEM_ID_RE.fullmatch(item_id) or ".." in item_id:
        raise HTTPException(400, "Item id must look like voa:<identifier>")
    return item_id


def _require_row(item_id: str) -> dict:
    entry = voa_dataset.row(item_id)
    if not entry:
        raise HTTPException(404, "Item not in the pinned revision")
    return entry


def _require_record(item_id: str) -> dict:
    record = voa_dataset.record(item_id)
    if not record:
        raise HTTPException(404, "Item not in the pinned revision")
    return record


# ------------------------------------------------------------------- list machinery


def _has_arabic(row: dict) -> bool:
    return bool(row["arabic"]) and row["arabic"] != "not_started"


def _media_index(row: dict) -> dict:
    """What a list row says about playability: sizes and the route that streams the object."""
    assets = {}
    for kind, obj, _sha, size, mime in row["refs"]:
        assets.setdefault(kind, {"bytes": size, "mime": mime, "object": obj,
                                 "stream": f"{PREFIX}/media/{row['id']}/{kind}"})
    return assets


def _summary(row: dict) -> dict:
    """The list shape: enough to browse and to reach the media, without any body text."""
    return {
        "id": row["id"],
        "program": row["program"],
        "title": row["title"],
        "url": row["url"],
        "published": row["published"],
        "kind": row["kind"],
        "cefr": row["cefr"],
        "confidence": row["confidence"],
        "needs_human_review": row["review"],
        "word_count": row["words"],
        "audio": row["audio"],
        "video": row["video"],
        "has_transcript": row["words"] > 0,
        "has_arabic": _has_arabic(row),
        "media_status": row["media_status"],
        "quality_flags": list(row["flags"]),
        "topics": list(row["topics"]),
        "media": _media_index(row),
    }


def _matches(row: dict, filters: dict) -> bool:
    program = filters["program"]
    if program is not None:
        if program == "none":
            if row["program"]:
                return False
        elif row["program"] != program:
            return False
    cefr = filters["cefr"]
    if cefr is not None:
        level = row["cefr"] or UNCLASSIFIED
        if cefr == "unclassified":
            if level != UNCLASSIFIED:
                return False
        elif level != cefr:
            return False
    if filters["kind"] and row["kind"] != filters["kind"]:
        return False
    if filters["topic"] and filters["topic"].lower() not in [t.lower() for t in row["topics"]]:
        return False
    if filters["media"] and filters["media"] != "any" and row["media_status"] != filters["media"]:
        return False
    if filters["flag"] and filters["flag"] not in row["flags"]:
        return False
    if filters["audio"] is not None and row["audio"] is not filters["audio"]:
        return False
    if filters["video"] is not None and row["video"] is not filters["video"]:
        return False
    if filters["transcript"] is not None and (row["words"] > 0) is not filters["transcript"]:
        return False
    if filters["arabic"] is not None and _has_arabic(row) is not filters["arabic"]:
        return False
    if filters["min_words"] and row["words"] < filters["min_words"]:
        return False
    if filters["max_words"] and row["words"] > filters["max_words"]:
        return False
    published = (row["published"] or "")[:10]
    if filters["since"] and published < filters["since"]:
        return False
    if filters["until"] and published > filters["until"]:
        return False
    needle = filters["q"]
    if needle and not all(token in row["hay"] for token in needle.split()):
        return False
    return True


def _sorted(found: list[dict], sort: str, level: str) -> list[dict]:
    if sort == "newest":
        return sorted(found, key=lambda row: row["published"] or "", reverse=True)
    if sort == "oldest":
        return sorted(found, key=lambda row: row["published"] or "")
    if sort == "shortest":
        return sorted(found, key=lambda row: row["words"])
    if sort == "longest":
        return sorted(found, key=lambda row: -(row["words"] or 0))
    if sort == "level":
        ladder = level if level in LADDER else "A1"
        rank = {name: abs(index - LADDER.index(ladder)) for index, name in enumerate(LADDER)}
        return sorted(found, key=lambda row: (rank.get(row["cefr"] or "", 9), row["words"]))
    return found  # "dataset": shard order, id by id - the order the build wrote


def _page(found: list[dict], offset: int, limit: int) -> dict:
    total = len(found)
    return {"revision": _revision(), "total": total, "offset": offset, "limit": limit,
            "returned": min(limit, max(total - offset, 0)),
            "items": [_summary(row) for row in found[offset:offset + limit]]}


async def _query(filters: dict, sort: str, level: str, offset: int, limit: int) -> dict:
    await _offload(voa_dataset.build_index)
    found = _sorted([row for row in voa_dataset.rows() if _matches(row, filters)], sort, level)
    payload = _page(found, offset, limit)
    payload["filter"] = {key: value for key, value in filters.items() if value not in (None, "", 0)}
    return payload


def _clean_date(value: str, label: str) -> str:
    if value and not DATE_RE.fullmatch(value):
        raise HTTPException(400, f"{label} must look like 2014-01-17")
    return value


def _filters(program: str | None, cefr: str | None, kind: str | None, topic: str | None,
             media: str | None, flag: str | None, audio: bool | None, video: bool | None,
             transcript: bool | None, arabic: bool | None, min_words: int, max_words: int,
             since: str, until: str, q: str | None) -> dict:
    if cefr and cefr not in LADDER + ["unclassified", UNCLASSIFIED]:
        raise HTTPException(400, f"cefr must be one of {LADDER + ['unclassified']}")
    if media and media != "any" and media not in ("archived", "not_downloaded", "no_audio",
                                                 "terminal_failed"):
        raise HTTPException(400, "media must be archived, not_downloaded, no_audio, "
                                 "terminal_failed or any")
    return {
        "program": program, "cefr": cefr, "kind": kind, "topic": topic, "media": media,
        "flag": flag, "audio": audio, "video": video, "transcript": transcript,
        "arabic": arabic, "min_words": min_words, "max_words": max_words,
        "since": _clean_date(since, "since"), "until": _clean_date(until, "until"),
        "q": q.lower().strip() if q else None,
    }


# ------------------------------------------------------------------------- facets


def _compute_facets(stats: dict) -> dict:
    programs: dict[str, dict] = {}
    levels: dict[str, int] = {}
    confidence: dict[str, dict[str, int]] = {}
    kinds: dict[str, int] = {}
    topics: dict[str, int] = {}
    media: dict[str, int] = {}
    flags: dict[str, int] = {}
    objects: set[str] = set()
    totals = {"items": 0, "with_audio": 0, "with_video": 0, "with_transcript": 0,
              "with_arabic": 0, "archived_media": 0}
    for row in voa_dataset.rows():
        totals["items"] += 1
        totals["with_audio"] += int(row["audio"])
        totals["with_video"] += int(row["video"])
        totals["with_transcript"] += int(row["words"] > 0)
        totals["with_arabic"] += int(_has_arabic(row))
        totals["archived_media"] += int(row["media_status"] == "archived")
        name = row["program"] or "none"
        bucket = programs.setdefault(name, {"program": name, "items": 0, "audio": 0, "video": 0,
                                            "archived": 0, "cefr": {}})
        bucket["items"] += 1
        bucket["audio"] += int(row["audio"])
        bucket["video"] += int(row["video"])
        bucket["archived"] += int(row["media_status"] == "archived")
        level = row["cefr"] or UNCLASSIFIED
        bucket["cefr"][level] = bucket["cefr"].get(level, 0) + 1
        levels[level] = levels.get(level, 0) + 1
        row_conf = confidence.setdefault(level, {})
        label = row["confidence"] or "none"
        row_conf[label] = row_conf.get(label, 0) + 1
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        media[row["media_status"] or "unknown"] = media.get(row["media_status"] or "unknown", 0) + 1
        for topic in row["topics"]:
            topics[topic] = topics.get(topic, 0) + 1
        for flag in row["flags"]:
            flags[flag] = flags.get(flag, 0) + 1
        objects.update(entry[1] for entry in row["refs"])
    return {
        "revision": stats["pinned_revision"],
        "totals": {**totals, "media_objects_indexed": len(objects)},
        "programs": sorted(programs.values(), key=lambda row: -row["items"]),
        "levels": dict(sorted(levels.items(), key=lambda kv: -kv[1])),
        "confidence": confidence,
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "topics": dict(sorted(topics.items(), key=lambda kv: -kv[1])[:120]),
        "topics_total": len(topics),
        "media_status": dict(sorted(media.items(), key=lambda kv: -kv[1])),
        "quality_flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
    }


def _facets_data() -> dict:
    """Counts over the whole index, memoised until the index is rebuilt for another revision."""
    stats = voa_dataset.summary()
    key = f"{stats['pinned_revision']}:{stats['index_rows_in_memory']}:{stats['last_successful_load']}"
    with _facets_lock:
        if _facets["key"] != key or not _facets["data"]:
            _facets["data"] = _compute_facets(stats)
            _facets["key"] = key
        return _facets["data"]


# --------------------------------------------------------------------------- routes


@router.get("/health")
async def health(request: Request) -> Response:
    """Build facts of the pinned revision. Open, cheap, no content, no credentials.

    It says plainly whether the revision declares a metadata index, whether that index is linked
    to these records, and whether it is loaded - so an unusable library is visible before a
    client discovers it by failing.
    """
    try:
        stats = await run_in_threadpool(voa_dataset.summary)
    except voa_dataset.DatasetUnavailable as error:
        payload = {"status": "unavailable", "reason": str(error)[:300]}
        return Response(content=json.dumps(payload, ensure_ascii=False),
                        media_type="application/json; charset=utf-8", status_code=503)
    declared = stats["metadata_index"]
    status = "ok" if declared["ready"] else "index_unavailable"
    payload = {"status": status, **stats, "search_fields": SEARCH_FIELDS,
               "auth": "preview-unauthenticated" if preview_unauthenticated() else "firebase",
               "resources": await run_in_threadpool(_resource_sample)}
    code = 200 if declared["ready"] else 503
    if not declared["ready"]:
        payload["reason"] = declared["reason"]
    return _respond(payload, request, json.dumps(payload, sort_keys=True), status_code=code)


@router.get("/manifest")
async def manifest(request: Request, uid: str = Depends(voa_user)):
    stats = await _offload(voa_dataset.summary)
    build = await _offload(voa_dataset.manifest)
    payload = {"revision": stats["pinned_revision"], "records": stats["records_count"], **build}
    return _respond(payload, request,
                    json.dumps([stats["pinned_revision"], stats["records_count"],
                                sorted(build)], sort_keys=True))


@router.get("/catalog")
async def catalog(request: Request, uid: str = Depends(voa_user)):
    """One call that opens the library: what it holds, and every value a filter accepts."""
    await _offload(voa_dataset.build_index)
    stats = await _offload(voa_dataset.summary)
    facets = _facets_data()
    payload = {**stats, **facets,
               "search_fields": SEARCH_FIELDS,
               "filter_values": {
                   "program": [row["program"] for row in facets["programs"]],
                   "cefr": list(facets["levels"]) + ["unclassified"],
                   "kind": list(facets["kinds"]),
                   "media": list(facets["media_status"]),
                   "flag": list(facets["quality_flags"]),
                   "topic_count": facets["topics_total"],
                   "sort": SORTS,
               }}
    return _respond(payload, request,
                    json.dumps([stats["pinned_revision"], facets["totals"],
                                len(facets["programs"])], sort_keys=True))


@router.get("/programs")
async def programs(request: Request, uid: str = Depends(voa_user)):
    await _offload(voa_dataset.build_index)
    facets = _facets_data()
    payload = {"revision": facets["revision"], "programs": facets["programs"],
               "note": "the 'none' row is items whose source page carried no program evidence"}
    return _respond(payload, request, json.dumps([row["program"] for row in facets["programs"]]))


@router.get("/levels")
async def levels(request: Request, uid: str = Depends(voa_user)):
    await _offload(voa_dataset.build_index)
    facets = _facets_data()
    payload = {"revision": facets["revision"], "levels": facets["levels"],
               "confidence": facets["confidence"],
               "note": "CEFR is measured from the text, not copied from the VOA program name; "
                       f"'unclassified' ({UNCLASSIFIED}) means the text gave no confident band"}
    return _respond(payload, request,
                    json.dumps([facets["levels"], facets["confidence"]], sort_keys=True))


@router.get("/kinds")
async def kinds(request: Request, uid: str = Depends(voa_user)):
    await _offload(voa_dataset.build_index)
    facets = _facets_data()
    payload = {"revision": facets["revision"], "kinds": facets["kinds"]}
    return _respond(payload, request, json.dumps(facets["kinds"], sort_keys=True))


@router.get("/topics")
async def topics(request: Request, uid: str = Depends(voa_user)):
    await _offload(voa_dataset.build_index)
    facets = _facets_data()
    payload = {"revision": facets["revision"], "topics": facets["topics"],
               "topics_total": facets["topics_total"],
               "note": "tags as published by VOA, not invented categories"}
    return _respond(payload, request, json.dumps(facets["topics"], sort_keys=True))


@router.get("/items")
async def items(
    request: Request,
    uid: str = Depends(voa_user),
    program: str | None = Query(default=None, max_length=80),
    cefr: str | None = Query(default=None, max_length=20),
    kind: str | None = Query(default=None, max_length=40),
    topic: str | None = Query(default=None, max_length=80),
    media: str | None = Query(default=None, max_length=20),
    flag: str | None = Query(default=None, max_length=40),
    audio: bool | None = Query(default=None),
    video: bool | None = Query(default=None),
    transcript: bool | None = Query(default=None),
    arabic: bool | None = Query(default=None),
    min_words: int = Query(default=0, ge=0, le=20000),
    max_words: int = Query(default=0, ge=0, le=20000),
    since: str = Query(default="", max_length=10),
    until: str = Query(default="", max_length=10),
    q: str | None = Query(default=None, max_length=120),
    sort: str = Query(default="dataset"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=MAX_PAGE),
):
    if sort not in SORTS:
        raise HTTPException(400, f"sort must be one of {SORTS}")
    filters = _filters(program, cefr, kind, topic, media, flag, audio, video, transcript,
                       arabic, min_words, max_words, since, until, q)
    payload = await _query(filters, sort, cefr or "", offset, limit)
    payload["sort"] = sort
    return _respond(payload, request,
                    json.dumps([payload["total"], offset, limit, sort,
                                [item["id"] for item in payload["items"]]]))


@router.get("/search")
async def search(
    request: Request,
    uid: str = Depends(voa_user),
    q: str = Query(min_length=2, max_length=120),
    program: str | None = Query(default=None, max_length=80),
    cefr: str | None = Query(default=None, max_length=20),
    kind: str | None = Query(default=None, max_length=40),
    topic: str | None = Query(default=None, max_length=80),
    media: str | None = Query(default=None, max_length=20),
    flag: str | None = Query(default=None, max_length=40),
    audio: bool | None = Query(default=None),
    video: bool | None = Query(default=None),
    transcript: bool | None = Query(default=None),
    arabic: bool | None = Query(default=None),
    min_words: int = Query(default=0, ge=0, le=20000),
    max_words: int = Query(default=0, ge=0, le=20000),
    since: str = Query(default="", max_length=10),
    until: str = Query(default="", max_length=10),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=MAX_PAGE),
):
    """Every word must appear in title, description, program, keywords or kind."""
    filters = _filters(program, cefr, kind, topic, media, flag, audio, video, transcript,
                       arabic, min_words, max_words, since, until, q)
    await _offload(voa_dataset.build_index)
    found = [row for row in voa_dataset.rows() if _matches(row, filters)]
    payload = _page(found, offset, limit)
    payload["query"] = q
    payload["filter"] = {key: value for key, value in filters.items() if value not in (None, "", 0)}
    payload["coverage"] = {"indexed": SEARCH_FIELDS,
                           "not_indexed": ["transcript", "paragraphs", "arabic translation"],
                           "why": "the metadata index carries no bodies by design; searching "
                                  "71k transcripts needs a text index, not a full scan per query"}
    tokens = q.lower().split()
    for row, item in zip(found[offset:offset + limit], payload["items"]):
        item["title_match"] = all(token in (row["title"] or "").lower() for token in tokens)
    payload["title_matches"] = sum(1 for item in payload["items"] if item["title_match"])
    return _respond(payload, request,
                    json.dumps([payload["total"], offset, limit, q], sort_keys=True))


@router.get("/recommendations")
async def recommendations(
    request: Request,
    uid: str = Depends(voa_user),
    level: str = Query(default="A1", max_length=20),
    done: str = Query(default="", max_length=8000),
    count: int = Query(default=6, ge=1, le=25),
    audio_only: bool = Query(default=True),
):
    """Next-up list: the learner's level first, playable items before everything else.

    `done` is the client's progress - this route stores nothing, so it cannot disagree with the
    dataset.
    """
    await _offload(voa_dataset.build_index)
    finished = {value for value in done.split(",") if value}
    rank = {name: abs(index - LADDER.index(level)) if name in LADDER else 9
            for index, name in enumerate(LADDER)}
    found = [row for row in voa_dataset.rows()
             if row["id"] not in finished and row["words"] > 0
             and (not audio_only or row["media_status"] == "archived")]
    found.sort(key=lambda row: (rank.get(row["cefr"] or "", 9),
                                0 if row["cefr"] == level else 1,
                                row["words"]))
    payload = {"revision": _revision(), "level": level,
               "requested": {"audio_only": audio_only, "already_done": len(finished),
                             "count": count},
               "pool": len(found), "items": [_summary(row) for row in found[:count]]}
    return _respond(payload, request,
                    json.dumps([level, audio_only, sorted(finished)[:20], count], sort_keys=True))


@router.get("/items/{item_id}")
async def item(item_id: str, request: Request, uid: str = Depends(voa_user)):
    clean_id = _valid_id(item_id)
    entry = await _offload(_require_row, clean_id)
    record = await _offload(voa_dataset.record, clean_id)
    if not record:  # the index and the shards are the same commit, so this is an integrity fault
        raise HTTPException(500, f"{clean_id} is indexed but not present in the shards")
    payload = {"revision": _revision(), "summary": _summary(entry), "item": record}
    return _respond(payload, request, json.dumps(record, sort_keys=True))


@router.get("/items/{item_id}/transcript")
async def transcript(item_id: str, request: Request, uid: str = Depends(voa_user)):
    clean_id = _valid_id(item_id)
    record = await _offload(_require_record, clean_id)
    content = record["content"]
    paragraphs = content.get("paragraphs") or []
    sentences = []
    for index, paragraph in enumerate(paragraphs):
        for chunk in SENTENCE_SPLIT.split(paragraph):
            text = chunk.strip()
            if len(text) > 12:
                sentences.append({"id": f"{clean_id}:s{len(sentences)}", "text": text,
                                  "paragraph": index})
    payload = {
        "revision": _revision(), "id": clean_id, "kind": content.get("kind"),
        "title": content.get("title"), "word_count": content.get("word_count"),
        "url": record["source"].get("url"), "attribution": record["source"].get("attribution"),
        "paragraphs": paragraphs, "sentences": sentences,
        "transcript_en": content.get("transcript_en"), "arabic": record.get("arabic"),
        "evidence": "text exactly as VOA published it; nothing was rewritten or re-levelled",
    }
    return _respond(payload, request, json.dumps(payload, sort_keys=True))


@router.get("/items/{item_id}/vocabulary")
async def vocabulary(item_id: str, request: Request, uid: str = Depends(voa_user)):
    clean_id = _valid_id(item_id)
    record = await _offload(_require_record, clean_id)
    source_rows = (record.get("vocabulary") or {}).get("source_definitions") or []
    payload = {
        "revision": _revision(), "id": clean_id, "title": record["content"].get("title"),
        "source_definitions": source_rows,
        "site_vocabulary": (record.get("original_educational_assets") or {})
                           .get("vocabulary_definitions") or [],
        "our_entries": (record.get("vocabulary") or {}).get("our_entries") or [],
        "grammar": record.get("grammar") or {},
        "count": len(source_rows),
        "evidence": "these are the VOA page's own lists, carried through unreviewed; some rows "
                    "split word and definition on the wrong boundary, so treat them as raw",
    }
    return _respond(payload, request, json.dumps(payload, sort_keys=True))


@router.get("/items/{item_id}/media")
async def item_media(item_id: str, request: Request, uid: str = Depends(voa_user)):
    """Audio/video metadata: the dataset object, its checksum and size, and how to stream it."""
    clean_id = _valid_id(item_id)
    entry = await _offload(_require_row, clean_id)
    record = await _offload(voa_dataset.record, clean_id)
    content = (record or {}).get("content") or {}
    audio_url = (content.get("audio") or {}).get("url")
    video_urls = content.get("video") or []
    assets = {}
    for kind, obj, sha, size, mime in entry["refs"]:
        assets[kind] = {
            "object": obj, "sha256": sha, "bytes": size, "mime": mime,
            "stream": f"{PREFIX}/media/{clean_id}/{kind}",
            "dataset_url": voa_dataset.resolve_url(obj),
            "source_url": audio_url if kind == "audio" else next(iter(video_urls), None),
        }
    payload = {
        "revision": _revision(), "id": clean_id, "media_status": entry["media_status"],
        "assets": assets,
        "outcomes": [dict(zip(("kind", "status", "reason_class", "attempts"), row))
                     for row in entry["outcomes"]],
        "unarchived_urls": {"audio": (content.get("audio") or {}).get("all_urls") or [],
                            "video": video_urls},
        "note": "assets are content-addressed objects inside the pinned dataset; "
                "unarchived_urls are dead or unmirrored sources, kept as evidence only",
    }
    return _respond(payload, request,
                    json.dumps([clean_id, sorted(assets), entry["media_status"]], sort_keys=True))


# ------------------------------------------------------------------ media streaming


def _read_slice(path, start: int, end: int):
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


@router.api_route("/media/{item_id}/{kind}", methods=["GET", "HEAD"])
async def media_stream(item_id: str, kind: str, request: Request,
                       uid: str = Depends(voa_user)):
    if kind not in MEDIA_KINDS:
        raise HTTPException(404, "Unknown media kind")
    clean_id = _valid_id(item_id)
    ref = await _offload(voa_dataset.media_ref, clean_id, kind)
    if not ref or not ref.get("object"):
        raise HTTPException(404, f"No archived {kind} for {clean_id} in the pinned revision")
    path = await _offload(voa_dataset.media_local, ref)
    size = path.stat().st_size
    etag = f'"{ref["sha256"]}"'  # already verified against the dataset when the object was cached
    headers = {"Accept-Ranges": "bytes", "ETag": etag, "X-Learn-Revision": _revision(),
               "Cache-Control": "private, max-age=31536000, immutable",
               "Content-Disposition": f'inline; filename="{path.name}"'}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    start, end, status_code = 0, size - 1, 200
    range_header = (request.headers.get("range") or "").strip()
    if range_header:
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
        if not match:
            raise HTTPException(416, "Invalid range",
                                headers={"Content-Range": f"bytes */{size}"})
        start = int(match.group(1))
        end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
        if start > end or start >= size:
            raise HTTPException(416, "Range not satisfiable",
                                headers={"Content-Range": f"bytes */{size}"})
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    if request.method == "HEAD":
        return Response(status_code=status_code, headers=headers)
    return StreamingResponse(_read_slice(path, start, end), status_code=status_code,
                             media_type=ref.get("mime") or "application/octet-stream",
                             headers=headers)


# ------------------------------------------------------ temporary compatibility layer


@router.get("/compat/course")
async def compat_course(request: Request, uid: str = Depends(voa_user),
                        limit_per_program: int = Query(default=300, ge=1, le=2000)):
    """The shape the current client already reads: programs become groups, items become units.

    It maps, it does not convert. A VOA item stays one item - nothing is poured into the old
    A1 unit/lesson template - and these are the same index rows every other route serves, so
    no second copy of the content exists to drift.
    """
    await _offload(voa_dataset.build_index)
    grouped: dict[str, list[dict]] = defaultdict(list)
    unassigned = 0
    for row in voa_dataset.rows():
        if not row["program"]:
            unassigned += 1
            continue
        grouped[row["program"]].append(row)
    levels = []
    for name, found in sorted(grouped.items()):
        levels.append({
            "id": f"voa:{name}",
            "label": name,
            "unit_count": len(found),
            "truncated": len(found) > limit_per_program,
            "units": [{"id": row["id"], "title": row["title"], "lesson_count": 1,
                       "cefr": row["cefr"], "audio": row["media_status"] == "archived"}
                      for row in found[:limit_per_program]],
        })
    payload = {"revision": _revision(), "compatibility": "legacy client shape", "levels": levels,
               "items_without_program": unassigned, "limit_per_program": limit_per_program,
               "note": "units are VOA items, not lessons: one item = one unit, and progress "
                       "for the old template has no meaning here"}
    return _respond(payload, request,
                    json.dumps([level["id"] for level in levels], sort_keys=True))


@router.get("/compat/unit/{item_id}")
async def compat_unit(item_id: str, request: Request, uid: str = Depends(voa_user)):
    """One VOA item inside the old unit envelope, so the current UI can render it unchanged."""
    clean_id = _valid_id(item_id)
    record = await _offload(_require_record, clean_id)
    entry = voa_dataset.row(clean_id)
    content = record["content"]
    paragraphs = content.get("paragraphs") or []
    sentences = []
    for index, paragraph in enumerate(paragraphs):
        for chunk in SENTENCE_SPLIT.split(paragraph):
            text = chunk.strip()
            if len(text) > 12:
                sentences.append({"id": f"{clean_id}:s{index}", "text": text})
    payload = {
        "revision": _revision(),
        "unit": {
            "id": clean_id,
            "title": content.get("title"),
            "source_url": record["source"].get("url"),
            "attribution": record["source"].get("attribution"),
            "audio": next((f"{PREFIX}/media/{clean_id}/audio"
                           for kind, *_ in entry["refs"] if kind == "audio"), None),
            "sentences": sentences,
            "vocabulary": (record.get("vocabulary") or {}).get("source_definitions") or [],
            "arabic": record.get("arabic"),
            "exercises": record.get("exercises") or [],
            "learning": record.get("learning"),
        },
    }
    return _respond(payload, request, json.dumps(payload, sort_keys=True))
