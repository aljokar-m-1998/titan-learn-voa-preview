"""The VOA adapter, tested as a running server in a hosted Linux container.

Two boot phases against the same image, because they prove different things:

  phase A  VOA_LEARN_PREVIEW_UNAUTH=1 - the functional matrix: the published index, every route,
           Range/206/416, ETag/304, byte-exact media against the dataset's own checksum, the
           cache-damage repair, and two cold boots (a killed process, then a wiped cache).
  phase B  no such flag - the authentication matrix: every content route answers 401 while
           /health stays readable, which is the state the gateway will be deployed in.

No credential is used anywhere: the pinned revision of M-hv1/Titan-Learn is public, and this run
shows the reader works anonymously. Results are written to hosted-matrix.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent
PORT = 0
REPORT: dict = {"phases": {}, "notes": []}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def base(server: "Server", path: str = "") -> str:
    return f"http://127.0.0.1:{PORT}{server.prefix}{path}"


class Server:
    """One uvicorn process, with the timings a cold container is judged on."""

    def __init__(self, env: dict, prefix: str):
        self.env = env
        self.prefix = prefix
        self.proc: subprocess.Popen | None = None
        self.output = io.StringIO()
        self.log = ROOT / f"uvicorn-{int(time.time() * 1000) % 100000}.log"

    def start(self) -> dict:
        global PORT
        PORT = free_port()
        started = time.time()
        # the log goes to a file, not a pipe: a chatty uvicorn must never fill a buffer and
        # block the very process whose startup time we are measuring
        self.handle = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
             "--port", str(PORT), "--log-level", "warning"],
            cwd=ROOT, env=self.env, stdout=self.handle, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 180
        first = None
        while time.time() < deadline:
            try:
                response = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
                if response.status_code == 200:
                    first = time.time() - started
                    break
            except requests.RequestException:
                time.sleep(0.4)
        if first is None:
            raise RuntimeError("the server never answered /health:\n" + self.output.getvalue())
        return {"process_startup_seconds": round(first, 2)}

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if getattr(self, "handle", None):
            self.handle.close()
            self.handle = None


def run(server: Server, path: str, *, expect=(200,), **kwargs) -> requests.Response:
    url = base(server, path)
    started = time.time()
    response = requests.get(url, timeout=300, **kwargs)
    response.elapsed_seconds = round(time.time() - started, 3)
    if expect and response.status_code not in expect:
        raise AssertionError(f"{path} -> {response.status_code}: {response.text[:300]}")
    return response


def check(name: str, value) -> None:
    REPORT["checks"][name] = value
    print(f"  {name:44} {value}", flush=True)


def find_cached_index(hf_home: str) -> Path | None:
    for path in Path(hf_home).rglob("metadata-*"):
        if path.is_file() and path.stat().st_size > 1000:
            return path
    return None


def functional_matrix(server: Server, env: dict, pin: str, prefix: str) -> None:
    phase: dict = {"label": "A - preview switch on (functional)"}
    timings = {}
    health = run(server, "/health")
    body = health.json()
    mounted = body["resources"]["prefix"]
    if mounted != server.prefix:
        # a shell or an input field can rewrite a path; never test a mount that is not the one asked
        raise AssertionError(f"asked to mount {server.prefix}, the app mounted {mounted}")
    check("health status", (health.status_code, body["status"], body["auth"]))
    check("mounted prefix", mounted)
    check("health pinned_revision", body["pinned_revision"])
    check("health records_count (manifest)", body["records_count"])
    check("health media_objects_count", body["media_objects_count"])
    check("index declared+ready", (body["metadata_index"]["declared"], body["metadata_index"]["ready"]))
    check("index bytes", body["metadata_index"]["bytes"])
    check("index records_sha256 link", body["metadata_index"]["records_sha256"])
    check("index built_from_revision", body["metadata_index"]["built_from_revision"][:12])
    check("default mount absent", requests.get(
        f"http://127.0.0.1:{PORT}/api/v1/learn/voa/health", timeout=20).status_code)
    check("resource sample at boot", body["resources"])

    catalog = run(server, "/catalog")
    cold = catalog.json()
    timings["first_catalog_seconds"] = catalog.elapsed_seconds
    check("cold /catalog index_source", cold["index_source"])
    check("cold /catalog cold_start_seconds", cold["cold_start_seconds"])
    check("cold /catalog rows", cold["records_count"])
    check("catalog totals", cold["totals"])
    check("catalog facet sums agree", {
        "programs": sum(row["items"] for row in cold["programs"]),
        "levels": sum(cold["levels"].values()),
        "kinds": sum(cold["kinds"].values()),
        "media_status": sum(cold["media_status"].values())} ==
        {"programs": cold["totals"]["items"], "levels": cold["totals"]["items"],
         "kinds": cold["totals"]["items"], "media_status": cold["totals"]["items"]})
    again = run(server, "/catalog")
    timings["warm_catalog_seconds"] = again.elapsed_seconds
    check("warm /catalog rows match", again.json()["totals"] == cold["totals"])

    for path in ("/programs", "/levels", "/topics", "/kinds", "/manifest"):
        response = run(server, path)
        check(f"{path} status", (response.status_code, response.elapsed_seconds))

    page = run(server, "/items", params={"limit": 5})
    check("items page", (page.json()["total"], [row["id"] for row in page.json()["items"]]))
    second = run(server, "/items", params={"limit": 5, "offset": 5})
    ids = {row["id"] for row in page.json()["items"]} | {row["id"] for row in second.json()["items"]}
    check("pages disjoint", len(ids) == 10)
    filtered_params = {"program": "As It Is", "cefr": "B1", "media": "archived",
                       "sort": "newest"}
    filtered = run(server, "/items", params=dict(filtered_params, limit=3))
    check("filtered items", (filtered.json()["total"],
                             [row["id"] for row in filtered.json()["items"]]))
    total_all = run(server, "/items", params={"limit": 1}).json()["total"]
    tail = run(server, "/items", params={"limit": 5, "offset": total_all - 2}).json()
    check("tail page is the remainder", (tail["offset"], tail["returned"], tail["total"],
                                         tail["offset"] + tail["returned"] == tail["total"]))
    filtered_tail = run(server, "/items",
                        params=dict(filtered_params, limit=5,
                                    offset=filtered.json()["total"] - 1)).json()
    check("filter survives paging", (filtered_tail["total"], filtered_tail["returned"]))
    check("bad filter rejected", run(server, "/items?cefr=C2", expect=(400,)).status_code)
    check("oversized page rejected", run(server, "/items?limit=500", expect=(422,)).status_code)

    search = run(server, "/search", params={"q": "water"})
    check("search", (search.json()["total"], search.json()["coverage"]["indexed"],
                     search.elapsed_seconds))
    same = run(server, "/items", params={"q": "water", "limit": 1})
    check("search and items?q agree", (search.json()["total"], same.json()["total"],
                                       search.json()["total"] == same.json()["total"]))
    check("search hits are pageable", all(
        row["id"] for row in run(server, "/search", params={"q": "water", "limit": 2,
                                                            "offset": 2}).json()["items"]))

    item_id = filtered.json()["items"][0]["id"]
    detail = run(server, f"/items/{item_id}")
    check("item detail", (detail.status_code, detail.elapsed_seconds,
                          detail.json()["item"]["content"]["title"][:60],
                          detail.json()["revision"] == pin))
    check("item revision header", detail.headers.get("x-learn-revision"))
    transcript = run(server, f"/items/{item_id}/transcript")
    check("transcript", (len(transcript.json()["paragraphs"]), len(transcript.json()["sentences"]),
                         transcript.json()["attribution"][:40]))
    vocabulary = run(server, f"/items/{item_id}/vocabulary")
    check("vocabulary", vocabulary.json()["count"])
    media = run(server, f"/items/{item_id}/media")
    assets = media.json()["assets"]
    check("media assets", {k: (v["bytes"], v["object"][:14]) for k, v in assets.items()})
    check("dataset_url is pinned", all(pin in v["dataset_url"] for v in assets.values()))

    audio_stream = assets["audio"]["stream"]
    head = requests.head(f"http://127.0.0.1:{PORT}{audio_stream}", timeout=60)
    check("HEAD audio", (head.status_code, head.headers.get("content-length"),
                         len(head.content) == 0, head.headers.get("accept-ranges")))
    full = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}", timeout=300)
    digest = hashlib.sha256(full.content).hexdigest()
    check("GET audio byte-exact vs dataset sha", (full.status_code, len(full.content),
                                                  digest == assets["audio"]["sha256"]))
    etag = full.headers["etag"]
    again304 = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}",
                          headers={"If-None-Match": etag}, timeout=60)
    check("audio 304 on ETag", (again304.status_code, len(again304.content)))
    slice_ = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}",
                          headers={"Range": "bytes=1024-65535"}, timeout=120)
    check("audio 206 slice", (slice_.status_code, len(slice_.content),
                              slice_.headers.get("content-range"),
                              slice_.content == full.content[1024:65536]))
    open_end = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}",
                            headers={"Range": f"bytes={len(full.content) - 16}-"}, timeout=120)
    check("audio open-ended 206", (open_end.status_code, open_end.content == full.content[-16:]))
    bad = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}",
                       headers={"Range": "bytes=abc-def"}, timeout=60)
    check("audio 416 malformed range", (bad.status_code, bad.headers.get("content-range")))
    over = requests.get(f"http://127.0.0.1:{PORT}{audio_stream}",
                        headers={"Range": f"bytes={assets['audio']['bytes'] + 10}-"}, timeout=60)
    check("audio 416 past the end", (over.status_code, over.headers.get("content-range")))

    video_row = run(server, "/items", params={"video": True, "media": "archived", "limit": 20})
    with_video = [row for row in video_row.json()["items"] if "video" in row["media"]]
    if with_video:
        vid = with_video[0]["id"]
        vmedia = run(server, f"/items/{vid}/media").json()["assets"]["video"]
        vstream = vmedia["stream"]
        vhead = requests.get(f"http://127.0.0.1:{PORT}{vstream}",
                             headers={"Range": "bytes=0-65535"}, timeout=300)
        check("video 206 head", (vhead.status_code, len(vhead.content),
                                 vhead.headers.get("content-range") ==
                                 f"bytes 0-65535/{vmedia['bytes']}"))
        sha_ok = hashlib.sha256(vhead.content).hexdigest() == hashlib.sha256(
            requests.get(f"http://127.0.0.1:{PORT}{vstream}", timeout=600).content[0:65536]).hexdigest()
        check("video slice matches its object prefix", sha_ok)
    else:
        check("video 206 head", "no archived video in the sampled page")

    json304 = requests.get(base(server, "/levels"),
                           headers={"If-None-Match": run(server, "/levels").headers["etag"]},
                           timeout=60)
    check("JSON 304 on ETag", (json304.status_code, len(json304.content)))

    after = run(server, "/health").json()["resources"]
    check("peak RSS MB after reads and media", after["peak_rss_mb"])
    check("rows in memory after reads", after["rows_in_memory"])
    check("cache free MB", after["cache_free_mb"])
    REPORT["phases"]["A"] = {**phase, "timings": timings}
    return timings


def cache_damage(server: Server, env: dict, pin: str, prefix: str) -> None:
    """Damage the downloaded index on disk, restart, and require the reader to repair it."""
    hf_home = env["HF_HOME"]
    cached = find_cached_index(hf_home)
    if not cached:
        raise AssertionError("the published index was never cached - where did the rows come from?")
    before = cached.stat().st_size
    # the hub keeps a snapshot entry as a read-only link into blobs/, so damaging the cache
    # means writing the bytes it resolves to - which is what a torn download looks like on disk
    target = cached.resolve() if cached.is_symlink() else cached
    mode = target.stat().st_mode
    target.chmod(0o600)
    target.write_bytes(b"garbage from a half-written download")
    check("cache damaged at", (target.name, target.stat().st_size))
    server.stop()
    boot = server.start()
    catalog = run(server, "/catalog")
    check("after cache damage: boot", boot)
    check("after cache damage: rows", catalog.json()["records_count"])
    check("after cache damage: re-downloaded size", find_cached_index(hf_home).stat().st_size == before)


def cold_boot_again(env: dict, pin: str, prefix: str, label: str) -> None:
    """Wipe every cache: this is a fresh container pulling from the hub."""
    for key in ("HF_HOME", "VOA_LEARN_CACHE_DIR"):
        shutil.rmtree(env[key], ignore_errors=True)
    server = Server(dict(env), prefix)
    boot = server.start()
    catalog = run(server, "/catalog")
    check(f"{label}: /health up after", boot["process_startup_seconds"])
    check(f"{label}: cold /catalog seconds", catalog.elapsed_seconds)
    check(f"{label}: cold_start_seconds reported", catalog.json()["cold_start_seconds"])
    check(f"{label}: index source", catalog.json()["index_source"])
    after = run(server, "/health").json()
    check(f"{label}: peak RSS MB", after["resources"]["peak_rss_mb"])
    check(f"{label}: rows in memory", after["resources"]["rows_in_memory"])
    check(f"{label}: index bytes pulled", after["metadata_index"]["bytes"])
    server.stop()


def auth_matrix(env: dict, pin: str, prefix: str) -> None:
    """Without the preview switch the same image must refuse everything but /health."""
    clean = {k: v for k, v in env.items() if k != "VOA_LEARN_PREVIEW_UNAUTH"}
    server = Server(clean, prefix)
    server.start()
    health = requests.get(base(server) + "/health", timeout=60)
    check("health open without a token", health.status_code)
    check("health says auth=firebase", health.json()["auth"])
    for path in ("/items", "/catalog", "/programs", "/levels", "/topics", "/search?q=water",
                 "/items/voa:7980858", "/items/voa:7980858/transcript", "/items/voa:7980858/media",
                 "/compat/course", "/recommendations"):
        response = requests.get(base(server) + path, timeout=60)
        check(f"401 {path}", response.status_code)
    forged = requests.get(base(server) + "/items",
                          headers={"Authorization": "Bearer not-a-firebase-token"}, timeout=60)
    check("401 with a forged bearer", forged.status_code)
    server.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", required=True)
    parser.add_argument("--prefix", default="/api/v1/preview/learn/voa")
    args = parser.parse_args()
    REPORT["checks"] = {}
    work = ROOT / ".runtime"
    shutil.rmtree(work, ignore_errors=True)
    (work / "hf").mkdir(parents=True)
    env = dict(os.environ,
               VOA_LEARN_PIN=args.pin,
               VOA_LEARN_PREFIX=args.prefix,
               VOA_LEARN_PREVIEW_UNAUTH="1",
               VOA_LEARN_MEDIA_CACHE_MB="256",
               HF_HOME=str(work / "hf"),
               VOA_LEARN_CACHE_DIR=str(work / "library-cache"))
    env.pop("VOA_LEARN_MAINTENANCE_REBUILD", None)
    print(f"pin {args.pin}\nprefix {args.prefix}\nHF_HOME {env['HF_HOME']} (empty: a cold container)\n",
          flush=True)

    failure = None
    server = Server(env, args.prefix)
    try:
        boot = server.start()
        check("cold process startup seconds", boot["process_startup_seconds"])
        functional_matrix(server, env, args.pin, args.prefix)
        cache_damage(server, env, args.pin, args.prefix)
        server.stop()
        cold_boot_again(env, args.pin, args.prefix, "fresh cache (new container)")
        auth_matrix(env, args.pin, args.prefix)
    except BaseException as error:            # a failed phase still has to produce its evidence
        failure = f"{type(error).__name__}: {error}"
        REPORT["failure"] = failure
        traceback.print_exc()
        print(f"FAILED: {failure}", flush=True)
    finally:
        server.stop()
        out = ROOT / "hosted-matrix.json"
        out.write_text(json.dumps(REPORT, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size} bytes, {len(REPORT['checks'])} checks)",
              flush=True)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
