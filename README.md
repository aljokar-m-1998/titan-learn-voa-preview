# titan-learn-voa-preview

Hosted test bed for the Titan Learn **VOA adapter**. This repo is not the gateway and not the
product: it exists to run the adapter as a real server on real Linux against the pinned
Titan-Learn revision, and to publish the measurements.

```text
pin      0cf163c23c2ee9777c93c9e9c70afceeb837c6a3   an immutable commit, never a branch
dataset  M-hv1/Titan-Learn                          public; no token is used anywhere here
mount    /api/v1/preview/learn/voa                  the default path is deliberately absent
```

## What is in it

| file | origin |
| --- | --- |
| `api/learn_voa.py`, `core/voa_dataset.py` | copied unmodified from the `staging-voa-adapter` branch |
| `core/security.py` | the gateway's own Firebase check, unmodified - it fails closed here because no credential is configured |
| `main.py` | mounts the library only when `VOA_LEARN_PREFIX` is set, and then only at that path |
| `Dockerfile` | the same base image, cache layout (`HF_HOME=/code/.cache/huggingface`), non-root user and port as the gateway |
| `hosted_matrix.py` | the two-phase matrix described below |
| `.github/workflows/hosted-matrix.yml` | `workflow_dispatch`, ubuntu-24.04, no secrets |

## The matrix

**Phase A** runs with `VOA_LEARN_PREVIEW_UNAUTH=1` - a preview-only switch that cannot apply to
the default mount. It reads health, the published index and its checksum link, catalog, programs,
levels, topics, kinds, manifest, pagination and filters, search, one full record, transcript,
vocabulary, media metadata, then audio and video through `HEAD`, a whole-object `GET` whose bytes
are hashed and compared with the SHA-256 the dataset recorded, `Range`/`206`, `416`, and
`ETag`/`304`. Afterwards it damages the cached index on disk and restarts the process: the reader
must re-download, re-verify and serve the same 71,291 rows. Finally it wipes every cache, which is
what a new container looks like.

**Phase B** starts the same image without that switch: `/health` stays 200, and every content
route - including a forged bearer token - answers 401. That is the state the gateway ships in.

Results are written to `hosted-matrix.json` and uploaded as a workflow artifact.

## Delete it

```bash
gh repo delete aljokar-m-1998/titan-learn-voa-preview --yes
```

This repo holds no state worth keeping: the dataset revision, the published index and the
container configuration all live on Hugging Face, and the adapter code lives in the gateway
source. Nothing here is referenced by the app.
