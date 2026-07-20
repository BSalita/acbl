"""ACBL event strata (MP-limit) normalization for Elo board rows.

Used by ``acbl_elo_ratings_create.py`` to write ``mp_limit`` + ``strata_bucket``
onto Elo parquets. Keep in sync with ``Elo_Ratings/acbl_strata.py`` (API/UI).

K-weighting of Open vs restricted updates lives in
``mlBridge.mlBridgeAugmentLib.EVENT_MP_LIMIT_K_SCALE``.
"""
from __future__ import annotations

import re
from typing import Optional

BUCKET_OPEN = "open"
BUCKET_0_299 = "0-299"
BUCKET_0_499_NLM = "0-499-nlm"
BUCKET_0_749 = "0-749"
BUCKET_0_1500 = "0-1500"
BUCKET_OTHER = "other-restricted"

_CLUB_PREFIX_RE = re.compile(r"^MP\s*Limits:\s*", re.IGNORECASE)
_INT_RE = re.compile(r"[\d,]+")


def _parse_int_token(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token or token.lower() == "none":
        return None
    m = _INT_RE.search(token.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def club_top_mp_limit(mp_limits_raw: str | None) -> Optional[int]:
    """Top (first) MP limit from club ``mpLimits`` / ``mp_limit``, or None if unlimited."""
    if mp_limits_raw is None:
        return None
    text = str(mp_limits_raw).strip()
    if not text:
        return None
    text = _CLUB_PREFIX_RE.sub("", text).strip()
    if not text:
        return None
    return _parse_int_token(text.split("/")[0].strip())


def tournament_top_mp_limit(mp_limit_raw: str | None) -> tuple[Optional[int], bool]:
    """Return (top_limit, is_nlm) for tournament ``mp_limit`` strings."""
    if mp_limit_raw is None:
        return (None, False)
    text = str(mp_limit_raw).strip()
    if not text or text.lower() == "none":
        return (None, False)
    is_nlm = text.upper().startswith("NLM")
    if "/" in text:
        parts = text.split("/")
        for part in reversed(parts):
            if part.upper() == "NLM":
                continue
            if part.upper().startswith("0-"):
                return (_parse_int_token(part[2:]), is_nlm)
            n = _parse_int_token(part)
            if n is not None:
                return (n, is_nlm)
        return (None, is_nlm)
    if text.upper() == "NLM":
        return (500, True)
    if text.upper().startswith("0-"):
        return (_parse_int_token(text[2:]), is_nlm)
    return (_parse_int_token(text), is_nlm)


def bucket_from_top_limit(top: Optional[int], *, is_nlm: bool = False) -> str:
    if top is None and not is_nlm:
        return BUCKET_OPEN
    if is_nlm and (top is None or top <= 500):
        return BUCKET_0_499_NLM
    if top is None:
        return BUCKET_OTHER
    if top <= 300:
        return BUCKET_0_299
    if top <= 500:
        return BUCKET_0_499_NLM
    if top <= 750:
        return BUCKET_0_749
    if top <= 1500:
        return BUCKET_0_1500
    return BUCKET_OTHER


def club_strata_bucket(mp_limits_raw: str | None) -> str:
    return bucket_from_top_limit(club_top_mp_limit(mp_limits_raw))


def tournament_strata_bucket(mp_limit_raw: str | None) -> str:
    top, is_nlm = tournament_top_mp_limit(mp_limit_raw)
    return bucket_from_top_limit(top, is_nlm=is_nlm)
