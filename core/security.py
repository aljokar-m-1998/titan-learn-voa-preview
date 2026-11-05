# ============================================================
# core/security.py — Titan Gateway Auth V2.0 (Firebase-based)
# ============================================================
# SINGLE SOURCE OF TRUTH for authenticating requests to the Gateway.
#
# There are TWO distinct kinds of caller, and they must NOT share
# one secret:
#
#   1. Real app users (browser / Android app), calling through the
#      Nexus proxy (/api/gateway-proxy/*) or directly.
#      → require_user: verifies `Authorization: Bearer <Firebase ID
#        token>` and returns the verified uid. NEVER trust a uid
#        that arrives in a request body/query param instead.
#
#   2. Trusted internal callers only — the Immersion scheduler's
#      self-call, Nexus's admin buttons, your own maintenance
#      scripts. These have no Firebase user behind them.
#      → require_internal_key: checks `X-Titan-Key` against
#        TITAN_INTERNAL_KEY (env var). This value must NEVER be
#        shipped in any client bundle (PWA JS, Android app, etc).
#
# Old state (do not repeat): a single static string
# ("Titan2026_Admin") was hardcoded directly in chat.py and ALSO
# embedded in the PWA's config.js, so every "protected" endpoint
# was really only as secure as view-source. That value is
# considered burned — rotate it out of every server's env vars
# even after adopting this module.
#
# HOW TO USE IN A ROUTER:
#   from core.security import require_user, require_internal_key
#
#   # a user-facing endpoint:
#   @router.post("/generate-lesson")
#   async def generate_lesson(body: LessonRequest, uid: str = Depends(require_user)):
#       ...
#
#   # an internal/admin-only endpoint:
#   @router.post("/reprocess-all-stories", dependencies=[Depends(require_internal_key)])
#   async def reprocess_all_stories():
#       ...
#
#   # a whole router that's entirely internal (e.g. cinema, most of immersion):
#   router = APIRouter(prefix="/api/v1/cinema", dependencies=[Depends(require_user)])
# ============================================================

from __future__ import annotations

import json
import logging
import os

from fastapi import Header, HTTPException

logger = logging.getLogger("titan.security")

# ============================================================
# Firebase Admin SDK — initialised once, reused everywhere
# ============================================================
try:
    import firebase_admin
    from firebase_admin import credentials as fb_credentials
    from firebase_admin import auth as fb_auth
    _FIREBASE_AVAILABLE = True
except ImportError:  # pragma: no cover
    firebase_admin = None   # type: ignore[assignment]
    fb_credentials = None   # type: ignore[assignment]
    fb_auth = None          # type: ignore[assignment]
    _FIREBASE_AVAILABLE = False

_FIREBASE_INITIALIZED = False

if _FIREBASE_AVAILABLE:
    try:
        if not firebase_admin._apps:
            # Reuse whichever env var your other services already use for
            # the service-account JSON. Both names are accepted so you don't
            # have to duplicate the secret under two different names across
            # your three servers.
            _cred_raw = (
                os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
                or os.environ.get("FIREBASE_ADMIN_CRED", "").strip()
            )
            if _cred_raw:
                _cert = fb_credentials.Certificate(json.loads(_cred_raw))
                firebase_admin.initialize_app(_cert)
                _FIREBASE_INITIALIZED = True
                logger.info("[Security] Firebase Admin initialized.")
            else:
                logger.error(
                    "[Security] Neither FIREBASE_SERVICE_ACCOUNT nor "
                    "FIREBASE_ADMIN_CRED is set — require_user will reject "
                    "every request until this is fixed."
                )
        else:
            _FIREBASE_INITIALIZED = True
            logger.info("[Security] Firebase Admin already initialized — reusing.")
    except Exception:
        logger.exception("[Security] Firebase Admin initialization failed.")
        _FIREBASE_INITIALIZED = False
else:
    logger.warning("[Security] firebase-admin not installed — require_user will fail closed.")


# ============================================================
# Internal-only shared secret (server-to-server / admin tools)
# ============================================================
# Set TITAN_INTERNAL_KEY to a brand-new random value in every server's
# secrets. Do NOT reuse the old TITAN_API_KEY / TITAN_GATEWAY_KEY /
# TITAN_RELAY_KEY value here if there's any chance it matches the
# already-leaked "Titan2026_Admin" string — generate a fresh one
# (e.g. `openssl rand -hex 32`) and update it in all three servers'
# HF Space secrets at the same time.
_INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

if not _INTERNAL_KEY:
    logger.warning(
        "[Security] TITAN_INTERNAL_KEY is not set — require_internal_key "
        "will reject every request until this is fixed."
    )


# ============================================================
# Core token verification (reusable outside FastAPI Depends too —
# e.g. from a raw WebSocket handler, see api/chat.py live proxy)
# ============================================================
async def verify_firebase_token(id_token: str) -> str:
    """
    Verifies a raw Firebase ID token string and returns the uid.
    Raises ValueError on any failure. Use this directly when you're
    not in a normal FastAPI Header/Depends context (e.g. inside a
    WebSocket handler reading the token from a message frame).
    """
    if not _FIREBASE_INITIALIZED:
        raise ValueError("Firebase Admin is not initialized on this server")
    if not id_token:
        raise ValueError("Empty token")
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded["uid"]
    except Exception as exc:
        raise ValueError(f"Invalid or expired Firebase token: {exc}") from exc


# ============================================================
# FastAPI dependencies
# ============================================================
async def require_user(authorization: str | None = Header(default=None)) -> str:
    """
    Verifies `Authorization: Bearer <Firebase ID token>` and returns the
    caller's uid. Use this on every endpoint a real signed-in app user
    calls directly — chat, translate, image generation, transcription,
    TTS, per-user lesson generation, etc.

    The returned uid is the ONLY uid you should trust for this request —
    never read a uid out of the request body/query string instead.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization[7:].strip()
    try:
        return await verify_firebase_token(token)
    except ValueError as exc:
        logger.warning("[Security] Token rejected: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


async def require_internal_key(
    x_titan_key: str | None = Header(default=None, alias="X-Titan-Key"),
) -> None:
    """
    For trusted server-to-server / automation callers ONLY (the Immersion
    scheduler's self-call, Nexus's admin dashboard buttons, your own ops
    scripts). This key must never be sent by the PWA or the Android app —
    if you find yourself wanting to add this to client-side JS, use
    require_user instead.
    """
    if not _INTERNAL_KEY or x_titan_key != _INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def require_user_or_internal(
    authorization: str | None = Header(default=None),
    x_titan_key: str | None = Header(default=None, alias="X-Titan-Key"),
) -> str | None:
    """
    Accepts EITHER a valid Firebase user token OR the internal key.
    Returns the uid if a user token was used, otherwise None.
    Useful for endpoints like /status that make sense for both a real
    user checking their own state and your own admin/monitoring tools.
    """
    if authorization and authorization.startswith("Bearer "):
        try:
            return await verify_firebase_token(authorization[7:].strip())
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if _INTERNAL_KEY and x_titan_key == _INTERNAL_KEY:
        return None

    raise HTTPException(status_code=401, detail="Unauthorized")


# Backwards-compatible alias — old routers imported `require_titan_key`.
# Points at the internal-key check so nothing silently becomes
# user-open during the migration; re-point call sites to require_user
# or require_internal_key explicitly as you touch each file.
require_titan_key = require_internal_key
