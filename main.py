"""Titan Learn VOA adapter - isolated preview container.

This Space exists to exercise one thing: api/learn_voa.py reading the pinned Titan-Learn
revision through core/voa_dataset.py, in a real container with a real restart. It is not the
gateway. The production Space (M-hv1/Titan-Gateway) keeps its own main.py, its A1 bundle routes
at /api/v1/learn/*, and every other router - none of that code lives or runs here.

The library is mounted only when VOA_LEARN_PREFIX names a path, and then only at that path.
With VOA_LEARN_PREFIX=/api/v1/preview/learn/voa the whole app answers under the preview prefix
and nothing at /api/v1/learn/voa, which is what makes the trial provably separate.

Firebase: this container gets no service-account secret, so require_user fails closed and every
content route answers 401. That is the intended state - it is a genuine authentication test, not
a mocked one. VOA_LEARN_PREVIEW_UNAUTH=1 opens the read routes only while a non-default prefix is
mounted, and /health reports which of the two states is live.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import learn_voa

app = FastAPI(
    title="Titan Learn VOA adapter (preview)",
    version="preview-1",
    description="Isolated trial of the pinned-revision VOA library. Not the Titan Gateway.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def index() -> dict:
    return {
        "space": "titan-gateway-voa-preview",
        "mounted_at": learn_voa.PREFIX or None,
        "pinned_revision": learn_voa.voa_dataset.pin() or None,
        "health": "/api/v1/preview/learn/voa/health",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict:
    """Container liveness. Library state, including the pinned commit, is on the library route."""
    return {"status": "ok", "space": "titan-gateway-voa-preview",
            "voa_prefix": learn_voa.PREFIX or "not mounted"}


# exactly the gateway's rule: the trial exists at its own prefix, or not at all
if os.environ.get("VOA_LEARN_PREFIX", "").strip():
    app.include_router(learn_voa.router)
