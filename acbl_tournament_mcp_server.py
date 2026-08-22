"""MCP server for ACBL tournament discovery and postmortem analysis.

All tools proxy the unified ACBL Results API. The MCP never holds the ACBL
bearer token, reads Streamlit state, or drives a browser.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

ACBL_API_BASE_URL = os.environ.get(
    "ACBL_CLUB_API_BASE_URL", "http://127.0.0.1:8508").rstrip("/")
ACBL_TOURNAMENT_MCP_PORT = int(
    os.environ.get("ACBL_TOURNAMENT_MCP_PORT", "8516"))
_TIMEOUT_S = 300

mcp = MCPServer("acbl-tournament")


def _api_get(
    path: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{ACBL_API_BASE_URL}{path}",
            params={
                key: value for key, value in (params or {}).items()
                if value is not None
            },
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return {
            "error": str(exc),
            "hint": f"Is the ACBL Results API running at {ACBL_API_BASE_URL}?",
        }
    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text}
        return {
            "error": body.get("detail", response.text),
            "hint": body.get("hint"),
            "status_code": response.status_code,
        }
    return response.json()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    payload = _api_get("/health")
    status = "ok" if payload.get("status") == "ok" else "error"
    return JSONResponse(
        {"status": status, "service": "acbl-tournament-mcp", "api": payload})


@mcp.tool()
def acbl_tournament_dataset_info() -> Dict[str, Any]:
    """Tournament parquet, API cache, and live-source availability."""
    return _api_get("/health")


@mcp.tool()
def acbl_tournament_player_sessions(
    player_id: str,
    limit: int = 200,
    refresh: bool = True,
) -> Dict[str, Any]:
    """Tournament sessions for a player, merging historical parquet with the
    latest official ACBL API listing."""
    return _api_get(
        f"/tournaments/players/{player_id}/sessions",
        {"limit": limit, "refresh": refresh},
    )


@mcp.tool()
def acbl_tournament_postmortem_boards(
    player_id: str,
    session_id: str,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: int = 100,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Augmented board results for a tournament session. The API resolves
    historical parquet, API parquet cache, then a live headless build."""
    return _api_get(
        f"/postmortems/{session_id}/boards",
        {
            "player_id": player_id,
            "only_my_boards": only_my_boards,
            "columns": columns,
            "limit": limit,
            "refresh": refresh,
        },
    )


@mcp.tool()
def acbl_tournament_postmortem_sql(
    player_id: str,
    session_id: str,
    sql: str,
    limit: int = 500,
    refresh: bool = False,
) -> Dict[str, Any]:
    """DuckDB SQL against a tournament postmortem registered as `self`."""
    return _api_get(
        f"/postmortems/{session_id}/sql",
        {
            "player_id": player_id,
            "sql": sql,
            "limit": limit,
            "refresh": refresh,
        },
    )


@mcp.tool()
def acbl_tournament_postmortem_schema(
    player_id: str,
    session_id: str,
    pattern: Optional[str] = None,
    limit: int = 200,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Search names and dtypes in an augmented tournament postmortem."""
    return _api_get(
        f"/postmortems/{session_id}/schema",
        {
            "player_id": player_id,
            "pattern": pattern,
            "limit": limit,
            "refresh": refresh,
        },
    )


if __name__ == "__main__":
    print(
        f"[acbl-tournament-mcp] starting on :{ACBL_TOURNAMENT_MCP_PORT} "
        f"(endpoint /mcp, health /health); api -> {ACBL_API_BASE_URL}",
        flush=True,
    )
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=ACBL_TOURNAMENT_MCP_PORT,
        stateless_http=True,
        json_response=True,
    )
