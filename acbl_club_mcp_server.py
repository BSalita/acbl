"""MCP server that proxies the ACBL club-results FastAPI.

Transport: streamable HTTP (endpoint /mcp) on ACBL_CLUB_MCP_PORT
(default 8515), stateless with JSON responses. Same pattern as
Elo_Ratings/elo_mcp_server.py ACBL tools: this process never scrapes
my.acbl.org; it GET-s ACBL_CLUB_API_BASE_URL (default http://127.0.0.1:8508).

Public hostname (Cloudflare tunnel): https://acbl-club-mcp.7nt.info
  /mcp     MCP endpoint
  /health  liveness (proxies the club API)

  python acbl_club_api_server.py
  python acbl_club_mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

ACBL_CLUB_API_BASE_URL = os.environ.get("ACBL_CLUB_API_BASE_URL", "http://127.0.0.1:8508").rstrip("/")
ACBL_CLUB_MCP_PORT = int(os.environ.get("ACBL_CLUB_MCP_PORT", "8515"))
_TIMEOUT_S = 300

mcp = MCPServer("acbl-club")


def _club_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{ACBL_CLUB_API_BASE_URL}{path}"
    try:
        resp = requests.get(
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return {
            "error": str(exc),
            "hint": f"Is acbl_club_api_server running at {ACBL_CLUB_API_BASE_URL}?",
        }
    if not resp.ok:
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if isinstance(body, dict):
            return {
                "error": body.get("detail", resp.text),
                "hint": body.get("hint"),
                "status_code": resp.status_code,
            }
        return {"error": str(body), "status_code": resp.status_code}
    return resp.json()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    payload = _club_get("/health")
    status = "ok" if payload.get("status") == "ok" else "error"
    return JSONResponse({"status": status, "service": "acbl-club-mcp", "api": payload})


@mcp.tool()
def acbl_dataset_info() -> Dict[str, Any]:
    """Cache directory, cached club-session count, and whether a Chrome
    profile is configured on the ACBL club-results API."""
    return _club_get("/health")


@mcp.tool()
def acbl_club_list(q: Optional[str] = None, limit: int = 200, refresh: bool = False) -> Dict[str, Any]:
    """Directory of ACBL clubs (club_id, name, location). q is an optional
    substring filter. refresh=True re-scrapes https://my.acbl.org/club-results."""
    return _club_get("/clubs", {"q": q, "limit": limit, "refresh": refresh})


@mcp.tool()
def acbl_club_info(club_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Name, location, and website for one ACBL club number."""
    return _club_get(f"/clubs/{club_id}", {"refresh": refresh})


@mcp.tool()
def acbl_club_games(
    club_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Club games at https://my.acbl.org/club-results/{club_id}. Columns:
    session_id, date, event_name, club_id, details_url. Date filters are
    YYYY-MM-DD inclusive. refresh=True bypasses the HTML/JSON cache."""
    return _club_get(
        f"/clubs/{club_id}/games",
        {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "refresh": refresh,
        },
    )


@mcp.tool()
def acbl_player_games(player_id: str, limit: int = 200, refresh: bool = False) -> Dict[str, Any]:
    """Club games played by one ACBL player number (my-results page).
    Columns: session_id, date, club_id, club_name, event, session, score, details_url."""
    return _club_get(
        f"/players/{player_id}/games",
        {"limit": limit, "refresh": refresh},
    )


@mcp.tool()
def acbl_player_lookup(
    q: str,
    session_id: Optional[str] = None,
    club_id: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """Bidirectional player name <-> ACBL number lookup. Digits match
    player_number exactly; otherwise a case-insensitive name substring.
    Scope: that session's roster if session_id is set; else cached
    club-results JSON (optionally one club_id); plus ACBL_PLAYER_INFO_PARQUET
    when that file exists."""
    return _club_get(
        "/players/lookup",
        {"q": q, "session_id": session_id, "club_id": club_id, "limit": limit},
    )


@mcp.tool()
def acbl_session_tables(session_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Table names, row counts, and columns for one club session
    (event, club, sessions, sections, boards, board_results, pair_summaries,
    players, hand_records, strat_place, standings)."""
    return _club_get(f"/sessions/{session_id}/tables", {"refresh": refresh})


@mcp.tool()
def acbl_session_results(
    session_id: str,
    table: str = "board_results",
    columns: Optional[str] = None,
    limit: int = 200,
    refresh: bool = False,
) -> Dict[str, Any]:
    """One table from a club session details page. table names come from
    acbl_session_tables; default board_results. columns is an optional
    comma-separated subset."""
    return _club_get(
        f"/sessions/{session_id}/tables/{table}",
        {"columns": columns, "limit": limit, "refresh": refresh},
    )


@mcp.tool()
def acbl_session_sql(session_id: str, sql: str, limit: int = 200, refresh: bool = False) -> Dict[str, Any]:
    """DuckDB SQL against all tables of one club session (external access
    off). Example: SELECT pair_number, percentage FROM pair_summaries ORDER BY percentage DESC."""
    return _club_get(
        f"/sessions/{session_id}/sql",
        {"sql": sql, "limit": limit, "refresh": refresh},
    )


if __name__ == "__main__":
    print(
        f"[acbl-club-mcp] starting on :{ACBL_CLUB_MCP_PORT} "
        f"(endpoint /mcp, health /health); api -> {ACBL_CLUB_API_BASE_URL}",
        flush=True,
    )
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=ACBL_CLUB_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
