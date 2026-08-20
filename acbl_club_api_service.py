"""Streamlit-free core for the ACBL club-results API.

Cache-first reads of club-results/<club_id>/details/<session_id>.data.json,
with Playwright fetches of my.acbl.org when the cache misses or refresh=True.
Does not import the downloader CLI (it uses a process-global browser and
os._exit). Playwright helpers come from mlBridge.mlBridgeAcblLib.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
import threading
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import polars as pl
from bs4 import BeautifulSoup

_APP_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mlBridge.mlBridgeAcblLib import (  # noqa: E402
    _goto_with_diagnostics,
    _run_in_thread_with_new_loop,
    create_acbl_browser_context,
    create_club_dfs,
    get_club_results_details_data_playwright,
    parse_acbl_events_from_html,
    resolve_acbl_browser_profile_dir,
)

ACBL_ORIGIN = "https://my.acbl.org"
DEFAULT_ROW_LIMIT = 200
MAX_ROW_LIMIT = 2000
NAV_SLEEP_SECONDS = 2.0
# Live pagination depth when parquet history already covers the past: only the
# boundary gap (games newer than the last stage-1b run) needs scraping.
GAP_SCRAPE_LIMIT = 50
CLOUDFLARE_HINT = (
    "my.acbl.org is behind Cloudflare. Warm the persistent Chrome profile "
    "(ACBL_BROWSER_PROFILE_DIR) with solve_acbl_postmortem.ps1 or "
    "python acbl_solve_challenge.py, then retry."
)

_CHATBOT_CACHE = _SRC_DIR / "Bridge_Game_Postmortem_Chatbot" / "club-results"
_LOCAL_CACHE = _APP_DIR / "club-results"


def _default_cache_dir() -> pathlib.Path:
    env = os.environ.get("ACBL_CLUB_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    if _CHATBOT_CACHE.is_dir():
        return _CHATBOT_CACHE
    return _LOCAL_CACHE


CACHE_DIR = _default_cache_dir()

# Master archive maintained by acbl_all.bat stage 1a (acbl_club_download_to_json.py):
# same <club>/details/<session>.data.json layout, ~1.3M sessions. Live scrapes are
# written here so stage 1a skips them later. Never enumerated wholesale; access is
# by direct path or per-club glob only.
ARCHIVE_CACHE_DIR = pathlib.Path(
    os.environ.get("ACBL_CLUB_ARCHIVE_DIR", "e:/bridge/data/acbl/club-results")
)
# Stage 1b outputs (acbl_club_json_to_sql.py): normalized per-table parquets
# (events, players, pair_summaries, sections, sessions) covering the full archive.
PARQUET_DIR = pathlib.Path(
    os.environ.get("ACBL_CLUB_PARQUET_DIR", "e:/bridge/data/acbl/club_results_parquet")
)

# Ensure the cache dir exists up front: CACHE_ROOTS is fixed at import, and a
# missing dir would otherwise be excluded from reads even after writes create it.
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ROOTS: List[pathlib.Path] = [
    root for root in (CACHE_DIR, ARCHIVE_CACHE_DIR) if root.is_dir()
] or [CACHE_DIR]
# Write scrapes into the archive so stage 1a skips them -- but only when it is
# writable. In production the archive is a read-only mount of the OneDrive
# "recent slice" (see src/acbl/u.bat), so writes go to the local cache instead.
WRITE_CACHE_DIR = (
    ARCHIVE_CACHE_DIR
    if ARCHIVE_CACHE_DIR.is_dir() and os.access(str(ARCHIVE_CACHE_DIR), os.W_OK)
    else CACHE_DIR
)

_PLAYER_INFO_CANDIDATES = (
    os.environ.get("ACBL_PLAYER_INFO_PARQUET"),
    str(_SRC_DIR / "bridgestats" / "data" / "acbl_player_info.parquet"),
    str(_APP_DIR / "data" / "acbl_player_info.parquet"),
)

_SCRAPE_LOCK = threading.Lock()
_SESSION_DF_CACHE: Dict[Tuple[str, float], Dict[str, pl.DataFrame]] = {}
_SESSION_DF_LOCK = threading.Lock()
_SESSION_DF_MAX = 8


class ClubApiError(Exception):
    """Structured error for the FastAPI layer."""

    def __init__(self, detail: str, status_code: int = 400, hint: Optional[str] = None):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.hint = hint


def clamp_limit(limit: Optional[int], default: int = DEFAULT_ROW_LIMIT) -> int:
    if limit is None:
        return default
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, MAX_ROW_LIMIT))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _as_pl(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise ClubApiError(f"Unsupported dataframe type: {type(df)!r}", status_code=500)


def dataframe_to_table(
    df: Any,
    limit: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a frame to {columns, rows, row_count, truncated, meta}."""
    cap = clamp_limit(limit)
    frame = _as_pl(df)
    truncated = frame.height > cap
    frame = frame.head(cap)
    rows = [{k: _jsonable(v) for k, v in rec.items()} for rec in frame.to_dicts()]
    return {
        "columns": list(frame.columns),
        "rows": rows,
        "row_count": frame.height,
        "truncated": truncated,
        "meta": _jsonable(meta or {}),
    }


def rows_to_table(
    rows: List[Dict[str, Any]],
    limit: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cap = clamp_limit(limit)
    truncated = len(rows) > cap
    sliced = rows[:cap]
    columns: List[str] = []
    seen = set()
    for row in sliced:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return {
        "columns": columns,
        "rows": [{k: _jsonable(v) for k, v in row.items()} for row in sliced],
        "row_count": len(sliced),
        "truncated": truncated,
        "meta": _jsonable(meta or {}),
    }


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_mtime_iso(path: pathlib.Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def _newest_mtime_iso(paths: Iterable[pathlib.Path]) -> Optional[str]:
    newest: Optional[float] = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return datetime.fromtimestamp(newest).isoformat(timespec="seconds")


def _parse_date_any(value: Any) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _parquet_file(name: str) -> Optional[pathlib.Path]:
    path = PARQUET_DIR / f"{name}.parquet"
    return path if path.is_file() else None


def _parquet_scan(name: str) -> Optional[pl.LazyFrame]:
    path = _parquet_file(name)
    if path is None:
        return None
    return pl.scan_parquet(path)


def _collect_retry(lazy: pl.LazyFrame) -> Optional[pl.DataFrame]:
    """Collect with one retry: stage 1b rewrites the parquets in place, so a
    scan can transiently fail mid-replace.

    Streaming engine: identical results, but chunked execution roughly halves
    peak memory on the 5-table join queries (measured 2.7 GB vs 4.1 GB worst
    case), which matters inside the memory-capped acbl-club-api container."""
    for attempt in (0, 1):
        try:
            return lazy.collect(engine="streaming")
        except Exception:
            if attempt:
                return None
            time.sleep(0.5)
    return None


def _parquet_updated_iso() -> Optional[str]:
    path = _parquet_file("events")
    return _file_mtime_iso(path) if path else None


def _session_cache_path(club_id: str, session_id: str) -> pathlib.Path:
    return WRITE_CACHE_DIR / str(club_id) / "details" / f"{session_id}.data.json"


def _club_of_session_from_parquet(session_id: str) -> Optional[str]:
    events = _parquet_scan("events")
    if events is None:
        return None
    df = _collect_retry(
        events.filter(pl.col("id") == pl.lit(str(session_id))).select("club_id_number").head(1)
    )
    if df is None or df.is_empty():
        return None
    club = df[0, "club_id_number"]
    return str(club) if club else None


def find_session_cache(session_id: str) -> Optional[pathlib.Path]:
    sid = str(session_id)
    # Direct path via the events parquet (session -> club), avoiding globs
    # over the 2,883-club archive.
    club = _club_of_session_from_parquet(sid)
    if club:
        for root in CACHE_ROOTS:
            path = root / club / "details" / f"{sid}.data.json"
            if path.is_file():
                return path
    # Fallback glob per root, e.g. sessions newer than the last stage-1b run
    # that a live fetch already cached. Local root first (it is tiny).
    for root in CACHE_ROOTS:
        hits = list(root.glob(f"*/details/{sid}.data.json"))
        if hits:
            return hits[0]
    return None


def _iter_session_cache_files() -> Iterable[pathlib.Path]:
    """Session JSONs in the local cache only. The 1.3M-file archive is
    deliberately excluded: enumerate it never, access it by direct path."""
    if not CACHE_DIR.is_dir():
        return []
    return CACHE_DIR.glob("*/details/*.data.json")


def cached_session_count() -> int:
    return sum(1 for _ in _iter_session_cache_files())


def _player_info_path() -> Optional[pathlib.Path]:
    for candidate in _PLAYER_INFO_CANDIDATES:
        if not candidate:
            continue
        path = pathlib.Path(candidate)
        if path.is_file():
            return path
    return None


_DATASET_INFO_TTL_S = 15.0
_dataset_info_cache: Optional[Tuple[float, Dict[str, Any]]] = None


def dataset_info() -> Dict[str, Any]:
    global _dataset_info_cache
    now = time.monotonic()
    if _dataset_info_cache is not None and now - _dataset_info_cache[0] < _DATASET_INFO_TTL_S:
        return _dataset_info_cache[1]
    profile = resolve_acbl_browser_profile_dir()
    parquet = _player_info_path()
    clubs = sorted(
        p.name
        for p in CACHE_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ) if CACHE_DIR.is_dir() else []
    archive_clubs = (
        sum(1 for p in ARCHIVE_CACHE_DIR.iterdir() if p.is_dir() and p.name.isdigit())
        if ARCHIVE_CACHE_DIR.is_dir()
        else 0
    )
    info = {
        "cache_dir": str(CACHE_DIR),
        "cached_sessions": cached_session_count(),
        "cached_clubs": clubs,
        "archive_dir": str(ARCHIVE_CACHE_DIR) if ARCHIVE_CACHE_DIR.is_dir() else None,
        "archive_clubs": archive_clubs,
        "parquet_dir": str(PARQUET_DIR) if PARQUET_DIR.is_dir() else None,
        "parquet_updated_at": _parquet_updated_iso(),
        "write_cache_dir": str(WRITE_CACHE_DIR),
        "chrome_profile": str(profile) if profile else None,
        "player_info_parquet": str(parquet) if parquet else None,
        "note": (
            "Cache tiers: club_results_parquet (listings/lookups, refreshed by "
            "acbl_all.bat stage 1b), club-results JSON archive (session details, "
            "stage 1a), then live Playwright for anything newer. This is raw "
            "club-page data, not the DD/Elo-augmented postmortem parquet cache."
        ),
    }
    _dataset_info_cache = (now, info)
    return info


def _save_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _playwright_paginated_html(
    url: str,
    limit: int = 0,
    count_items: Optional[Callable[[str], int]] = None,
) -> Tuple[str, bool]:
    """Fetch url and follow DataTables 'Next' until last page or limit items.

    Returns (html, complete). complete is False when pagination stopped early
    because the item limit was reached, so callers can mark caches partial.
    """

    def _sync() -> Tuple[str, bool]:
        from playwright.sync_api import sync_playwright

        pages_html: List[str] = []
        complete = True
        with sync_playwright() as p:
            browser, context = create_acbl_browser_context(p, headless=True)
            page = context.new_page()
            try:
                response = _goto_with_diagnostics(page, url, verbose=False)
                if response is None or response.status != 200:
                    status = getattr(response, "status", None)
                    raise ClubApiError(
                        f"Failed to load {url} (status {status})",
                        status_code=502,
                        hint=CLOUDFLARE_HINT,
                    )
                page_num = 1
                while True:
                    try:
                        page.wait_for_selector(".dataTables_paginate, table", timeout=5000)
                    except Exception:
                        pass
                    html = page.content()
                    pages_html.append(html)
                    if limit > 0 and count_items is not None:
                        combined = "\n".join(pages_html)
                        if count_items(combined) >= limit:
                            complete = False
                            break
                    next_clicked = False
                    selectors = [
                        "a.paginate_button.next",
                        "#DataTables_Table_0_next",
                        'a.page-link:has-text("Next")',
                        'a[rel="next"]',
                        "a.next:not(.disabled)",
                    ]
                    for selector in selectors:
                        try:
                            next_button = page.locator(selector).first
                            if next_button.count() == 0 or not next_button.is_visible():
                                continue
                            class_attr = next_button.get_attribute("class") or ""
                            if "disabled" in class_attr:
                                continue
                            next_button.click()
                            page.wait_for_load_state("networkidle", timeout=60000)
                            time.sleep(0.5)
                            page_num += 1
                            next_clicked = True
                            break
                        except Exception:
                            continue
                    if not next_clicked:
                        break
            finally:
                browser.close()
        return "\n".join(pages_html), complete

    try:
        with _SCRAPE_LOCK:
            time.sleep(NAV_SLEEP_SECONDS)
            return _run_in_thread_with_new_loop(_sync)
    except ClubApiError:
        raise
    except Exception as exc:
        raise ClubApiError(
            f"Playwright fetch failed for {url}: {exc}",
            status_code=502,
            hint=CLOUDFLARE_HINT,
        ) from exc


def _count_detail_links(html: str) -> int:
    return len(set(re.findall(r"/club-results/details/(\d+)", html)))


def parse_club_page(html: str, club_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Club header plus game rows from a club-results HTML listing."""
    soup = BeautifulSoup(html, "html.parser")
    info: Dict[str, Any] = {"club_id": str(club_id)}
    club_info = soup.find("div", class_="col-md-8")
    if club_info:
        h1 = club_info.find("h1")
        h5 = club_info.find("h5")
        link = club_info.find("a")
        if h1:
            info["club_name"] = h1.get_text(strip=True)
        if h5:
            info["location"] = h5.get_text(strip=True)
        if link and link.get("href"):
            info["website"] = link["href"]

    rows: List[Dict[str, Any]] = []
    seen = set()
    for anchor in soup.find_all("a", href=re.compile(r"/club-results/details/\d+$")):
        href = anchor.get("href") or ""
        session_id = href.rstrip("/").split("/")[-1]
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        tr = anchor.find_parent("tr")
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")] if tr else []
        date_text = cells[0] if cells else None
        rows.append(
            {
                "session_id": session_id,
                "date": date_text,
                "event_name": anchor.get_text(strip=True) or (cells[1] if len(cells) > 1 else None),
                "club_id": str(club_id),
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
            }
        )
    return info, rows


def parse_player_games_html(html: str, player_id: str) -> List[Dict[str, Any]]:
    parsed = parse_acbl_events_from_html(html, f"{ACBL_ORIGIN}/club-results/my-results/{player_id}")
    rows: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "html.parser")
    by_id = {}
    for anchor in soup.find_all("a", href=re.compile(r"/club-results/details/\d+$")):
        sid = anchor["href"].rstrip("/").split("/")[-1]
        tr = anchor.find_parent("tr")
        tds = tr.find_all("td") if tr else []
        texts = [td.get_text(" ", strip=True) for td in tds]
        club_id = None
        club_name = None
        if tr:
            club_link = tr.find("a", href=re.compile(r"/club-results/\d+$"))
            if club_link and club_link.get("href"):
                m = re.search(r"/club-results/(\d+)$", club_link["href"])
                if m:
                    club_id = m.group(1)
                club_name = club_link.get_text(strip=True)
        # my-results row cells: Date, Club Name, Event, Session, Last Updated,
        # Score, mps, Color, Links, Personal Scores
        by_id[sid] = {
            "date": texts[0] if texts else None,
            "club_name": club_name or (texts[1] if len(texts) > 1 else None),
            "event": texts[2] if len(texts) > 2 else None,
            "session": texts[3] if len(texts) > 3 else None,
            "score": texts[5] if len(texts) > 5 else None,
            "club_id": club_id,
        }
    for event_id, (_url, detail_url, msg, club_id) in parsed.items():
        extra = by_id.get(str(event_id), {})
        rows.append(
            {
                "session_id": str(event_id),
                "date": extra.get("date"),
                "club_id": extra.get("club_id") or club_id,
                "club_name": extra.get("club_name"),
                "event": extra.get("event") or msg,
                "session": extra.get("session"),
                "score": extra.get("score"),
                "details_url": detail_url,
                "player_id": str(player_id),
            }
        )
    return rows


def parse_club_directory_html(html: str) -> List[Dict[str, Any]]:
    match = re.search(r"let\s+address\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if match:
        try:
            address_data = json.loads(match.group(1))
            rows = []
            for item in address_data:
                if not isinstance(item, dict):
                    continue
                if item.get("type") and item.get("type") != "club":
                    continue
                if not item.get("club_id"):
                    continue
                rows.append(item)
            if rows:
                return rows
        except json.JSONDecodeError:
            pass
    match = re.search(r"clubs:JSON\.stringify\(\[([\d,\s]+)\]", html)
    if match:
        return [{"club_id": cid.strip()} for cid in match.group(1).split(",") if cid.strip()]
    return []


def _filter_games_by_date(
    rows: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    if not start_date and not end_date:
        return rows

    start = _parse_date_any(start_date)
    end = _parse_date_any(end_date)
    out = []
    for row in rows:
        parsed = _parse_date_any(row.get("date") or row.get("start_date"))
        if parsed is None:
            out.append(row)
            continue
        if start and parsed < start:
            continue
        if end and parsed > end:
            continue
        out.append(row)
    return out


def _games_from_cached_details(club_id: str) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for root in CACHE_ROOTS:
        details_dir = root / str(club_id) / "details"
        if not details_dir.is_dir():
            continue
        for path in details_dir.glob("*.data.json"):
            session_id = path.name.replace(".data.json", "")
            if session_id in seen:
                continue
            try:
                data = _load_json(path)
            except Exception:
                continue
            session_id = str(data.get("id") or session_id)
            seen.add(session_id)
            rows.append(
                {
                    "session_id": session_id,
                    "date": data.get("start_date"),
                    "event_name": data.get("name"),
                    "club_id": str(data.get("club_id_number") or club_id),
                    "club_name": data.get("club_name"),
                    "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
                }
            )
    rows.sort(key=_date_sort_key, reverse=True)
    return rows


def _date_sort_key(row: Dict[str, Any]) -> Tuple[date, str]:
    text = str(row.get("date") or "")
    return (_parse_date_any(text) or date.min, text)


def _player_games_from_cached_details(player_id: str) -> List[Dict[str, Any]]:
    pid = str(player_id)
    rows = []
    seen = set()
    for path in _iter_session_cache_files():
        try:
            data = _load_json(path)
        except Exception:
            continue
        if not _session_has_player(data, pid):
            continue
        session_id = str(data.get("id") or path.name.replace(".data.json", ""))
        if session_id in seen:
            continue
        seen.add(session_id)
        rows.append(
            {
                "session_id": session_id,
                "date": data.get("start_date"),
                "club_id": str(data.get("club_id_number") or path.parent.parent.name),
                "club_name": data.get("club_name"),
                "event": data.get("name"),
                "score": None,
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
                "player_id": pid,
            }
        )
    rows.sort(key=_date_sort_key, reverse=True)
    return rows


def _session_has_player(data: Dict[str, Any], player_id: str) -> bool:
    pid = str(player_id)
    for session in data.get("sessions") or []:
        for section in session.get("sections") or []:
            for pair in section.get("pair_summaries") or []:
                for player in pair.get("players") or []:
                    if str(player.get("id_number") or "") == pid:
                        return True
    return False


def _extract_players_from_session(data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    sid = str(session_id or data.get("id") or "")
    club_id = str(data.get("club_id_number") or "")
    rows = []
    seen = set()
    for session in data.get("sessions") or []:
        for section in session.get("sections") or []:
            section_name = section.get("name")
            for pair in section.get("pair_summaries") or []:
                direction = pair.get("direction")
                pair_number = pair.get("pair_number")
                for player in pair.get("players") or []:
                    number = str(player.get("id_number") or "")
                    name = player.get("name")
                    key = (number, name, sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "player_number": number,
                            "player_name": name,
                            "session_id": sid,
                            "club_id": club_id,
                            "club_name": data.get("club_name"),
                            "section": section_name,
                            "direction": direction,
                            "pair_number": pair_number,
                            "city": player.get("city"),
                            "state": player.get("state"),
                        }
                    )
    return rows


def _club_games_from_parquet(club_id: str) -> List[Dict[str, Any]]:
    events = _parquet_scan("events")
    if events is None:
        return []
    df = _collect_retry(
        events.filter(pl.col("club_id_number") == pl.lit(str(club_id))).select(
            "id", "start_date", "name", "club_name", "type", "board_scoring_method"
        )
    )
    if df is None or df.is_empty():
        return []
    rows = [
        {
            "session_id": rec["id"],
            "date": rec["start_date"],
            "event_name": rec["name"],
            "club_id": str(club_id),
            "club_name": rec["club_name"],
            "event_type": rec["type"],
            "board_scoring_method": rec["board_scoring_method"],
            "details_url": f"{ACBL_ORIGIN}/club-results/details/{rec['id']}",
        }
        for rec in df.to_dicts()
        if rec.get("id")
    ]
    # Event ids increase over time; start_date strings (MM/DD/YYYY) do not sort.
    rows.sort(key=lambda r: int(r["session_id"]), reverse=True)
    return rows


def _merge_game_rows(
    fresh: List[Dict[str, Any]], history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fresh (live-scraped) rows first, then history rows they don't cover."""
    seen = {str(r.get("session_id")) for r in fresh}
    return fresh + [r for r in history if str(r.get("session_id")) not in seen]


def club_games(
    club_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    cid = str(club_id).strip()
    if not cid.isdigit():
        raise ClubApiError("club_id must be a numeric ACBL club number", status_code=400)
    cap = clamp_limit(limit)
    source_url = f"{ACBL_ORIGIN}/club-results/{cid}"
    html_cache = CACHE_DIR / cid / f"{cid}.html"
    games_cache = CACHE_DIR / cid / f"{cid}.games.json"
    cached = True
    complete = False
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"club_id": cid}

    parquet_rows = _club_games_from_parquet(cid)
    if parquet_rows:
        rows = parquet_rows
        info["club_name"] = parquet_rows[0].get("club_name")
        source = "parquet"
        complete = True  # full history through the last stage-1b run
        fetched_at = _parquet_updated_iso()
    elif not refresh:
        if games_cache.is_file():
            payload = _load_json(games_cache)
            info = payload.get("info") or info
            rows = payload.get("games") or []
            complete = bool(payload.get("complete"))
            fetched_at = _file_mtime_iso(games_cache)
        elif html_cache.is_file() and html_cache.stat().st_size > 2048:
            # Legacy HTML cache; completeness unknown, so treat as partial.
            info, rows = parse_club_page(html_cache.read_text(encoding="utf-8"), cid)
            fetched_at = _file_mtime_iso(html_cache)
        else:
            rows = _games_from_cached_details(cid)
            if rows:
                info["club_name"] = rows[0].get("club_name")
                fetched_at = _newest_mtime_iso(
                    p for root in CACHE_ROOTS for p in (root / cid / "details").glob("*.data.json")
                )

    # Scrape when asked to refresh, when nothing local answers, or when a
    # partial cache cannot satisfy a request for more rows than it holds.
    if refresh or not rows or (not complete and len(rows) < cap):
        scrape_limit = min(cap, GAP_SCRAPE_LIMIT) if parquet_rows else cap
        try:
            html, live_complete = _playwright_paginated_html(
                source_url, limit=scrape_limit, count_items=_count_detail_links
            )
        except ClubApiError:
            if not rows:
                raise
            # Scrape failed (e.g. Cloudflare); serve what the caches have.
        else:
            html_cache.parent.mkdir(parents=True, exist_ok=True)
            html_cache.write_text(html, encoding="utf-8")
            info, live_rows = parse_club_page(html, cid)
            _save_json(games_cache, {"info": info, "games": live_rows, "complete": live_complete})
            # Parquet history extends a limit-truncated live listing.
            rows = _merge_game_rows(live_rows, parquet_rows)
            complete = live_complete or bool(parquet_rows)
            cached = False
            source = "live+parquet" if parquet_rows else "live"
            fetched_at = _now_iso()

    rows = _filter_games_by_date(rows, start_date, end_date)
    meta = {
        "source_url": source_url,
        "source": source,
        "cached": cached,
        "complete": complete,
        "fetched_at": fetched_at,
        "club": info,
    }
    if rows:
        # rows_to_table: parquet and live rows have slightly different keys.
        return rows_to_table(rows, limit=cap, meta=meta)
    return dataframe_to_table(
        pl.DataFrame(
            schema={"session_id": pl.Utf8, "date": pl.Utf8, "event_name": pl.Utf8, "club_id": pl.Utf8, "details_url": pl.Utf8}
        ),
        limit=cap,
        meta=meta,
    )


def club_info(club_id: str, refresh: bool = False) -> Dict[str, Any]:
    table = club_games(club_id, limit=1, refresh=refresh)
    inner_meta = table.get("meta") or {}
    info = inner_meta.get("club") or {"club_id": str(club_id)}
    if not info.get("club_name"):
        cached = _games_from_cached_details(str(club_id))
        if cached:
            info["club_name"] = cached[0].get("club_name")
    rows = [info]
    return rows_to_table(
        rows,
        limit=1,
        meta={
            "source_url": f"{ACBL_ORIGIN}/club-results/{club_id}",
            "cached": inner_meta.get("cached"),
            "fetched_at": inner_meta.get("fetched_at"),
        },
    )


def _player_games_from_parquet(player_id: str) -> List[Dict[str, Any]]:
    """Full club-game history for a player from the stage-1b parquets:
    players -> pair_summaries -> sections -> sessions -> events."""
    players = _parquet_scan("players")
    pairs = _parquet_scan("pair_summaries")
    sections = _parquet_scan("sections")
    sessions = _parquet_scan("sessions")
    events = _parquet_scan("events")
    if any(lf is None for lf in (players, pairs, sections, sessions, events)):
        return []
    lazy = (
        players.filter(pl.col("id_number") == pl.lit(str(player_id)))
        .select("pair_summary_id")
        .join(
            pairs.select(pl.col("id").alias("pair_summary_id"), "section_id", "percentage"),
            on="pair_summary_id",
        )
        .join(
            sections.select(pl.col("id").alias("section_id"), "session_id"),
            on="section_id",
        )
        .join(
            sessions.select(pl.col("id").alias("session_id"), "event_id", "game_date"),
            on="session_id",
        )
        .join(
            events.select(
                pl.col("id").alias("event_id"),
                pl.col("name").alias("event_name"),
                "club_id_number",
                "club_name",
            ),
            on="event_id",
        )
        .unique(subset="event_id", keep="first")
    )
    df = _collect_retry(lazy)
    if df is None or df.is_empty():
        return []
    rows = []
    for rec in df.to_dicts():
        eid = rec.get("event_id")
        if not eid:
            continue
        pct = rec.get("percentage")
        rows.append(
            {
                "session_id": str(eid),
                "date": str(rec.get("game_date") or "")[:10] or None,
                "club_id": rec.get("club_id_number"),
                "club_name": rec.get("club_name"),
                "event": rec.get("event_name"),
                "score": f"{pct}%" if pct not in (None, "") else None,
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{eid}",
                "player_id": str(player_id),
            }
        )
    rows.sort(key=lambda r: int(r["session_id"]), reverse=True)
    return rows


def player_games(
    player_id: str,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    pid = str(player_id).strip()
    if not pid.isdigit():
        raise ClubApiError("player_id must be a numeric ACBL player number", status_code=400)
    cap = clamp_limit(limit)
    source_url = f"{ACBL_ORIGIN}/club-results/my-results/{pid}"
    cache_file = CACHE_DIR / "_players" / f"{pid}.games.json"
    cached = True
    complete = False
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []

    parquet_rows = _player_games_from_parquet(pid)
    if parquet_rows:
        rows = parquet_rows
        source = "parquet"
        complete = True  # full history through the last stage-1b run
        fetched_at = _parquet_updated_iso()
    elif not refresh:
        if cache_file.is_file():
            payload = _load_json(cache_file)
            if isinstance(payload, dict):
                rows = payload.get("games") or []
                complete = bool(payload.get("complete"))
            else:
                # Legacy cache: bare list, completeness unknown.
                rows = payload or []
            fetched_at = _file_mtime_iso(cache_file)
        if not rows:
            rows = _player_games_from_cached_details(pid)
            if rows:
                fetched_at = _newest_mtime_iso(_iter_session_cache_files())

    # Scrape when asked to refresh, when nothing local answers, or when a
    # partial cache cannot satisfy a request for more rows than it holds.
    if refresh or not rows or (not complete and len(rows) < cap):
        scrape_limit = min(cap, GAP_SCRAPE_LIMIT) if parquet_rows else cap
        try:
            html, live_complete = _playwright_paginated_html(
                source_url, limit=scrape_limit, count_items=_count_detail_links
            )
        except ClubApiError:
            if not rows:
                raise
            # Scrape failed (e.g. Cloudflare); serve what the caches have.
        else:
            live_rows = parse_player_games_html(html, pid)
            if live_rows:
                _save_json(cache_file, {"games": live_rows, "complete": live_complete})
                # Parquet history extends a limit-truncated live listing.
                rows = _merge_game_rows(live_rows, parquet_rows)
                complete = live_complete or bool(parquet_rows)
                cached = False
                source = "live+parquet" if parquet_rows else "live"
                fetched_at = _now_iso()

    if not rows:
        raise ClubApiError(
            f"No club games found for player {pid}",
            status_code=404,
            hint="Confirm the ACBL number, or generate cache by loading a postmortem for that player.",
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "source_url": source_url,
            "source": source,
            "cached": cached,
            "complete": complete,
            "fetched_at": fetched_at,
        },
    )


def _fetch_session_json(session_id: str, refresh: bool = False) -> Tuple[Dict[str, Any], bool, pathlib.Path]:
    sid = str(session_id).strip()
    if not sid.isdigit():
        raise ClubApiError("session_id must be a numeric club event id", status_code=400)
    cached_path = find_session_cache(sid)
    if cached_path is not None and not refresh:
        return _load_json(cached_path), True, cached_path

    url = f"{ACBL_ORIGIN}/club-results/details/{sid}"
    try:
        with _SCRAPE_LOCK:
            time.sleep(NAV_SLEEP_SECONDS)
            data = get_club_results_details_data_playwright(url, headless=True, verbose=False)
    except Exception as exc:
        raise ClubApiError(
            f"Failed to fetch session {sid}: {exc}",
            status_code=502,
            hint=CLOUDFLARE_HINT,
        ) from exc
    if not data:
        raise ClubApiError(
            f"No details JSON for session {sid} (team events are not supported)",
            status_code=404,
            hint="Pair matchpoint games embed var data = {...} on the details page.",
        )
    club_id = str(data.get("club_id_number") or data.get("club_id") or "unknown")
    path = _session_cache_path(club_id, sid)
    _save_json(path, data)
    return data, False, path


def session_raw(session_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Verbatim details JSON for one session.

    Unlike the table endpoints this preserves the nested structure, which the
    postmortem app needs for create_club_dfs / merge_clean_augment_club_dfs.
    """
    data, _cached, _path = _fetch_session_json(session_id, refresh=refresh)
    return data


def _standings_df(dfs: Dict[str, Any]) -> pl.DataFrame:
    pairs = _as_pl(dfs["pair_summaries"])
    players = _as_pl(dfs["players"])
    if "id" in pairs.columns and "pair_summary_id" not in pairs.columns:
        pairs = pairs.rename({"id": "pair_summary_id"})
    player_cols = [c for c in ("id_number", "name", "mp_total", "city", "state") if c in players.columns]
    aggs = []
    if "id_number" in player_cols:
        aggs.extend(
            [
                pl.first("id_number").alias("player_number_1"),
                pl.last("id_number").alias("player_number_2"),
            ]
        )
    if "name" in player_cols:
        aggs.extend(
            [
                pl.first("name").alias("player_name_1"),
                pl.last("name").alias("player_name_2"),
            ]
        )
    for col in ("mp_total", "city", "state"):
        if col in player_cols:
            aggs.extend(
                [
                    pl.first(col).alias(f"{col}_1"),
                    pl.last(col).alias(f"{col}_2"),
                ]
            )
    grouped = players.group_by("pair_summary_id").agg(aggs) if aggs else players
    return pairs.join(grouped, on="pair_summary_id", how="left")


_NESTED_SESSION_TABLES = {
    "hand_records": ["sessions", "hand_records"],
    "strat_place": ["sessions", "sections", "pair_summaries", "strat_place"],
    "sections": ["sessions", "sections"],
    "boards": ["sessions", "sections", "boards"],
    "pair_summaries": ["sessions", "sections", "pair_summaries"],
    "players": ["sessions", "sections", "pair_summaries", "players"],
    "board_results": ["sessions", "sections", "boards", "board_results"],
}


def _drop_nested_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in frame.columns:
        sample = frame[col].iloc[0] if len(frame) else None
        if isinstance(sample, (dict, list)):
            continue
        keep.append(col)
    return frame[keep] if keep else frame.iloc[:, 0:0]


def _normalize_path(data: Dict[str, Any], path: List[str]) -> pl.DataFrame:
    try:
        frame = pd.json_normalize(data, path)
    except Exception:
        return pl.DataFrame()
    if frame.empty:
        return pl.from_pandas(frame)
    frame = _drop_nested_columns(frame)
    try:
        return pl.from_pandas(frame)
    except Exception:
        return pl.from_pandas(frame.astype("string"))


def session_frames_from_json(data: Dict[str, Any]) -> Dict[str, pl.DataFrame]:
    """Build the same tables as create_club_dfs, without its nested-scalar crash."""
    try:
        raw = create_club_dfs(data)
        return {name: _as_pl(frame) for name, frame in raw.items()}
    except Exception:
        pass

    dfs: Dict[str, pl.DataFrame] = {}
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    dfs["event"] = pl.from_pandas(pd.DataFrame([scalars]).astype("string"))
    club = data.get("club")
    if isinstance(club, dict):
        club_df = pd.json_normalize(club)
        dfs["club"] = pl.from_pandas(_drop_nested_columns(club_df).astype("string"))
    sessions = data.get("sessions")
    if isinstance(sessions, list) and sessions:
        sess_df = pd.json_normalize(sessions, max_level=0)
        dfs["sessions"] = pl.from_pandas(_drop_nested_columns(sess_df).astype("string"))
    for name, path in _NESTED_SESSION_TABLES.items():
        dfs[name] = _normalize_path(data, path)
    if "board_results" not in dfs or dfs["board_results"].is_empty():
        raise ClubApiError(
            "Could not build tables for this session",
            status_code=422,
            hint="Session must be a pair game with embedded details JSON (Mitchell-style club games).",
        )
    return dfs


def session_dataframes(session_id: str, refresh: bool = False) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Any]]:
    data, cached, path = _fetch_session_json(session_id, refresh=refresh)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    key = (str(path), mtime)
    with _SESSION_DF_LOCK:
        if key in _SESSION_DF_CACHE and not refresh:
            dfs = _SESSION_DF_CACHE[key]
        else:
            try:
                dfs = session_frames_from_json(data)
            except ClubApiError:
                raise
            except Exception as exc:
                raise ClubApiError(
                    f"Could not build tables for session {session_id}: {exc}",
                    status_code=422,
                    hint="Session must be a pair game with embedded details JSON (Mitchell-style club games).",
                ) from exc
            try:
                dfs["standings"] = _standings_df(dfs)
            except Exception:
                pass
            if len(_SESSION_DF_CACHE) >= _SESSION_DF_MAX:
                _SESSION_DF_CACHE.pop(next(iter(_SESSION_DF_CACHE)))
            _SESSION_DF_CACHE[key] = dfs
    if not cached:
        session_source = "live"
    elif ARCHIVE_CACHE_DIR.is_dir() and str(path).startswith(str(ARCHIVE_CACHE_DIR)):
        session_source = "archive"
    else:
        session_source = "cache"
    meta = {
        "source_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
        "source": session_source,
        "cached": cached,
        "fetched_at": _file_mtime_iso(path) if cached else _now_iso(),
        "session_id": str(session_id),
        "club_id": str(data.get("club_id_number") or ""),
        "club_name": data.get("club_name"),
        "event_name": data.get("name"),
        "start_date": data.get("start_date"),
        "cache_file": str(path),
    }
    return dfs, meta


def session_tables(session_id: str, refresh: bool = False) -> Dict[str, Any]:
    dfs, meta = session_dataframes(session_id, refresh=refresh)
    rows = [
        {
            "table": name,
            "row_count": frame.height,
            "columns": ", ".join(frame.columns),
            "column_count": len(frame.columns),
        }
        for name, frame in dfs.items()
    ]
    return rows_to_table(rows, limit=MAX_ROW_LIMIT, meta=meta)


def session_results(
    session_id: str,
    table: str = "board_results",
    columns: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    dfs, meta = session_dataframes(session_id, refresh=refresh)
    name = (table or "board_results").strip()
    if name not in dfs:
        raise ClubApiError(
            f"Unknown table '{name}' for session {session_id}",
            status_code=404,
            hint=f"Valid tables: {', '.join(sorted(dfs))}",
        )
    frame = dfs[name]
    if columns:
        wanted = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise ClubApiError(
                f"Unknown columns: {', '.join(missing)}",
                status_code=400,
                hint=f"Available columns: {', '.join(frame.columns)}",
            )
        frame = frame.select(wanted)
    meta = dict(meta)
    meta["table"] = name
    return dataframe_to_table(frame, limit=limit, meta=meta)


def session_sql(session_id: str, sql: str, limit: Optional[int] = None, refresh: bool = False) -> Dict[str, Any]:
    if not sql or not sql.strip():
        raise ClubApiError("sql is required", status_code=400)
    dfs, meta = session_dataframes(session_id, refresh=refresh)
    cap = clamp_limit(limit)
    query = sql.strip().rstrip(";")
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        for name, frame in dfs.items():
            con.register(name, frame)
        result = con.execute(query).pl()
    except Exception as exc:
        raise ClubApiError(
            f"SQL failed: {exc}",
            status_code=400,
            hint=f"Registered tables: {', '.join(sorted(dfs))}",
        ) from exc
    finally:
        con.close()
    meta = dict(meta)
    meta["sql"] = query
    table = dataframe_to_table(result, limit=cap, meta=meta)
    table["sql"] = query
    return table


def club_list(query: Optional[str] = None, limit: Optional[int] = None, refresh: bool = False) -> Dict[str, Any]:
    cap = clamp_limit(limit)
    cache_file = CACHE_DIR / "_clubs.json"
    source_url = f"{ACBL_ORIGIN}/club-results"
    cached = True
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []

    if not refresh:
        events = _parquet_scan("events")
        if events is not None:
            df = _collect_retry(
                events.group_by("club_id_number")
                .agg(
                    pl.col("club_name").last().alias("club_name"),
                    pl.len().alias("events"),
                )
                .sort("club_id_number")
            )
            if df is not None and not df.is_empty():
                rows = [
                    {
                        "club_id": rec.get("club_id_number"),
                        "club_name": rec.get("club_name"),
                        "events": rec.get("events"),
                        "type": "club",
                    }
                    for rec in df.to_dicts()
                    if rec.get("club_id_number")
                ]
                source = "parquet"
                fetched_at = _parquet_updated_iso()
        if not rows and cache_file.is_file():
            rows = _load_json(cache_file)
            fetched_at = _file_mtime_iso(cache_file)
        if not rows and CACHE_DIR.is_dir():
            for club_dir in sorted(CACHE_DIR.iterdir()):
                if not (club_dir.is_dir() and club_dir.name.isdigit()):
                    continue
                games = _games_from_cached_details(club_dir.name)
                rows.append(
                    {
                        "club_id": club_dir.name,
                        "club_name": games[0].get("club_name") if games else None,
                        "cached_sessions": len(games),
                        "type": "club",
                    }
                )
            if rows:
                fetched_at = _newest_mtime_iso(_iter_session_cache_files())

    if refresh or not rows:
        html, _complete = _playwright_paginated_html(source_url, limit=0)
        rows = parse_club_directory_html(html)
        if not rows:
            raise ClubApiError(
                "Could not parse club directory from my.acbl.org/club-results",
                status_code=502,
                hint=CLOUDFLARE_HINT,
            )
        _save_json(cache_file, rows)
        cached = False
        source = "live"
        fetched_at = _now_iso()

    q = (query or "").strip().lower()
    if q:
        filtered = []
        for row in rows:
            blob = " ".join(str(v) for v in row.values() if v is not None).lower()
            if q in blob:
                filtered.append(row)
        rows = filtered
    return rows_to_table(
        rows,
        limit=cap,
        meta={"source_url": source_url, "source": source, "cached": cached, "fetched_at": fetched_at},
    )


def _player_lookup_from_parquet(
    query: str, by_number: bool, club_id: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    """Name <-> number lookup against the stage-1b players parquet
    (151k players, all club sessions), optionally scoped to one club."""
    players = _parquet_scan("players")
    if players is None:
        return []
    if by_number:
        base = players.filter(pl.col("id_number") == pl.lit(query))
    else:
        base = players.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower(), literal=True)
        )
    base = base.select("id_number", "name", "city", "state", "mp_total", "pair_summary_id", "transaction_date")
    if club_id:
        pairs = _parquet_scan("pair_summaries")
        sections = _parquet_scan("sections")
        sessions = _parquet_scan("sessions")
        events = _parquet_scan("events")
        if any(lf is None for lf in (pairs, sections, sessions, events)):
            return []
        base = (
            base.join(pairs.select(pl.col("id").alias("pair_summary_id"), "section_id"), on="pair_summary_id")
            .join(sections.select(pl.col("id").alias("section_id"), "session_id"), on="section_id")
            .join(sessions.select(pl.col("id").alias("session_id"), "event_id"), on="session_id")
            .join(events.select(pl.col("id").alias("event_id"), "club_id_number"), on="event_id")
            .filter(pl.col("club_id_number") == pl.lit(str(club_id)))
        )
    lazy = (
        base.sort("transaction_date")
        .group_by("id_number")
        .agg(
            pl.col("name").last(),
            pl.col("city").last(),
            pl.col("state").last(),
            pl.col("mp_total").last(),
            pl.len().alias("club_sessions"),
        )
        .sort("name")
        .head(limit)
    )
    df = _collect_retry(lazy)
    if df is None or df.is_empty():
        return []
    return [
        {
            "player_number": rec.get("id_number"),
            "player_name": rec.get("name"),
            "city": rec.get("city"),
            "state": rec.get("state"),
            "mp_total": rec.get("mp_total"),
            "club_sessions": rec.get("club_sessions"),
            "club_id": str(club_id) if club_id else None,
            "source": "club_results_parquet",
        }
        for rec in df.to_dicts()
    ]


def _lookup_from_parquet(query: str, by_number: bool, limit: int) -> List[Dict[str, Any]]:
    path = _player_info_path()
    if path is None:
        return []
    df = pl.read_parquet(path)
    number_col = next((c for c in ("acbl_number", "player_number", "id_number") if c in df.columns), None)
    name_cols = [c for c in ("player_name", "name", "last_name", "first_name") if c in df.columns]
    if by_number and number_col:
        df = df.filter(pl.col(number_col).cast(pl.Utf8) == query)
    elif name_cols:
        expr = None
        for col in name_cols:
            piece = pl.col(col).cast(pl.Utf8).str.to_lowercase().str.contains(query.lower(), literal=True)
            expr = piece if expr is None else (expr | piece)
        df = df.filter(expr)
    else:
        return []
    keep = [c for c in df.columns if not str(c).startswith("mp_")]
    df = df.select(keep).head(limit)
    rows = []
    for rec in df.to_dicts():
        rec["source"] = "parquet"
        rows.append(rec)
    return rows


def player_lookup(
    query: str,
    session_id: Optional[str] = None,
    club_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ClubApiError("q is required (player name or ACBL number)", status_code=400)
    cap = clamp_limit(limit)
    by_number = q.isdigit()
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    cached = True

    if session_id:
        data, cached, _path = _fetch_session_json(session_id, refresh=False)
        rows.extend(_extract_players_from_session(data, session_id))
        sources.append(f"session:{session_id}{' (cache)' if cached else ''}")
    else:
        parquet_rows = _player_lookup_from_parquet(q, by_number=by_number, club_id=club_id, limit=cap)
        if parquet_rows:
            rows.extend(parquet_rows)
            sources.append("club_results_parquet")
        else:
            files: List[pathlib.Path]
            if club_id:
                files = [
                    p
                    for root in CACHE_ROOTS
                    for p in (root / str(club_id) / "details").glob("*.data.json")
                ]
            else:
                files = list(_iter_session_cache_files())
            for path in files:
                try:
                    data = _load_json(path)
                except Exception:
                    continue
                sid = str(data.get("id") or path.name.replace(".data.json", ""))
                rows.extend(_extract_players_from_session(data, sid))
            sources.append("session-cache")

    if by_number:
        rows = [r for r in rows if str(r.get("player_number") or "") == q]
    else:
        needle = q.lower()
        rows = [r for r in rows if needle in str(r.get("player_name") or "").lower()]

    if not rows:
        parquet_rows = _lookup_from_parquet(q, by_number=by_number, limit=cap)
        if parquet_rows:
            sources.append("player_info_parquet")
            for rec in parquet_rows:
                number = str(rec.get("acbl_number") or rec.get("player_number") or rec.get("id_number") or "")
                name = rec.get("player_name") or rec.get("name") or rec.get("last_name")
                rows.append(
                    {
                        "player_number": number,
                        "player_name": name,
                        "session_id": None,
                        "club_id": rec.get("club"),
                        "source": "parquet",
                    }
                )

    if not rows:
        raise ClubApiError(
            f"No players matching '{q}'",
            status_code=404,
            hint="Pass session_id for a game roster, club_id to scan that club's cached sessions, or configure ACBL_PLAYER_INFO_PARQUET.",
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "query": q,
            "by_number": by_number,
            "sources": sources,
            "cached": cached,
            "fetched_at": _now_iso(),
        },
    )
