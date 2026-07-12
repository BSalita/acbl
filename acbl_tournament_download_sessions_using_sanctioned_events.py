"""
ACBL Tournament Sessions Downloader (from sanctioned events)

Script version of `acbl_tournament_download_sessions_using_sanctioned_events.ipynb`.

Reads `tournaments/events/*.sanction.json` files (downloaded previously) and derives
session ids of the form:

  <sanction>-<event_code>-<session_number>

Then downloads each session payload from:
  https://api.acbl.org/v1/tournament/session?id=<session_id>&full_monty=1

and writes:
  tournaments/sessions/<session_id>.session.json

Behavior notes (matches notebook):
  - Skips if `<session_id>.session.json` already exists
  - Also skips if `<session_id>.session.sql` exists (downstream pipeline may have converted it)
  - Skips 400/404/500 responses, stops on 504
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.parse
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")


def _iter_files_sorted(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    return sorted(paths, key=lambda p: p.as_posix())


def _load_sanction_event_files(events_dir: pathlib.Path) -> list[pathlib.Path]:
    return _iter_files_sorted(events_dir.rglob("*.sanction.json"))


def build_session_ids_from_sanctioned_events(events_dir: pathlib.Path) -> list[str]:
    """
    Read `*.sanction.json` files and return unique session ids (sorted desc).
    """
    event_files = _load_sanction_event_files(events_dir)
    sessions: list[str] = []

    for fp in event_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                event = json.load(f)
        except Exception:
            # Keep going; corrupted file should not block the run.
            continue

        sanction = event.get("sanction")
        if sanction is None:
            continue
        event_code = event.get("event_code")
        session_count = event.get("session_count")
        if event_code is None or session_count is None:
            continue

        try:
            sc = int(session_count)
        except Exception:
            continue

        for c in range(1, sc + 1):
            sessions.append("-".join([str(sanction), str(event_code), str(c)]))

    # Notebook sorts reverse to start with newest-ish ids first.
    return sorted(set(sessions), reverse=True)


def download_tournament_sessions(
    session_ids: list[str],
    api_key: str,
    output_dir: pathlib.Path,
    full_monty: int = 1,
    timeout: int = 10,
    sleep_seconds: float = 0.0,
    starting_nfile: int = 0,
    ending_nfile: int = 0,
    skip_if_json_exists: bool = True,
    skip_if_sql_exists: bool = True,
    delete_partial_json_on_error: bool = True,
) -> tuple[int, int, int]:
    """
    Download tournament session JSON payloads.

    Returns:
        (written_count, skipped_count, error_count)
    """
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + api_key,
        # User-Agent is required (May 2025+).
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    base_url = "https://api.acbl.org/v1/tournament/session"
    output_dir.mkdir(parents=True, exist_ok=True)

    if ending_nfile == 0:
        ending_nfile = len(session_ids)
    filtered = session_ids[starting_nfile:ending_nfile]
    total = len(filtered)

    written = 0
    skipped = 0
    errors = 0

    except_count = 0
    start_run = time.time()

    for idx, session_id in enumerate(filtered):
        file_json = output_dir / f"{session_id}.session.json"
        file_sql = output_dir / f"{session_id}.session.sql"

        if skip_if_sql_exists and file_sql.exists():
            print(f"{idx}/{total}: File exists: {file_sql}: skipping")
            skipped += 1
            continue
        if skip_if_json_exists and file_json.exists():
            print(f"{idx}/{total}: File exists: {file_json}: skipping")
            skipped += 1
            continue

        query = {"id": session_id, "full_monty": full_monty}
        url = base_url + "?" + urllib.parse.urlencode(query)
        t0 = time.time()
        print(f"{idx}/{total} url:{url}")

        if sleep_seconds > 0 and idx > 0:
            time.sleep(sleep_seconds)

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            errors += 1
            print(f"Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}")
            if except_count > 5:
                print("Except count exceeded; stopping")
                break
            except_count += 1
            time.sleep(1)  # transient retry delay (notebook behavior)
            continue
        else:
            except_count = 0

        # Notebook behavior
        if response.status_code in (400, 404, 500):
            print(f"Status Code:{response.status_code}: skipping")
            skipped += 1
            continue
        if response.status_code == 504:
            print(f"Status Code:{response.status_code}: stopping")
            break
        if response.status_code == 429:
            print("RATE LIMITED: HTTP 429 - waiting 60 seconds...")
            time.sleep(60)
            continue

        if response.status_code != 200:
            errors += 1
            print(f"ERROR: HTTP {response.status_code}: url:{url}")
            continue

        try:
            json_response = response.json()
            json_pretty = json.dumps(json_response, indent=4)
            print(
                f"{idx}/{total} rate:{round(time.time() - t0, 2)} "
                f"elapsed:{round(time.time() - start_run, 1)} url:{url}"
            )
            print(f"{idx}/{total}: Writing:{file_json} size:{len(json_pretty)}")
            with open(file_json, "w", encoding="utf-8") as f:
                f.write(json_pretty)
            written += 1
        except KeyboardInterrupt:
            raise
        except Exception:
            errors += 1
            if delete_partial_json_on_error:
                file_json.unlink(missing_ok=True)
            print("ERROR writing response JSON")
            print(response.text[:500])
            continue

    return (written, skipped, errors)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download ACBL tournament sessions using tournaments/events/*.sanction.json"
    )
    parser.add_argument(
        "--events-dir",
        default="tournaments/events",
        help=f"Input subdirectory relative to {acblPath} (default: tournaments/events)",
    )
    parser.add_argument(
        "--sessions-dir",
        default="tournaments/sessions",
        help=f"Output subdirectory relative to {acblPath} (default: tournaments/sessions)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="ACBL API key (default: from ACBL_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout seconds (default: 10)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests (default: 0.0)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting session index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=0,
        help="Ending session index, 0 means all (default: 0)",
    )
    parser.add_argument(
        "--full-monty",
        type=int,
        default=1,
        help="full_monty query parameter (default: 1)",
    )
    parser.add_argument(
        "--no-skip-existing-json",
        action="store_false",
        dest="skip_if_json_exists",
        help="Do not skip when .session.json exists",
    )
    parser.set_defaults(skip_if_json_exists=True)
    parser.add_argument(
        "--no-skip-existing-sql",
        action="store_false",
        dest="skip_if_sql_exists",
        help="Do not skip when .session.sql exists",
    )
    parser.set_defaults(skip_if_sql_exists=True)

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ACBL_API_KEY")
    if not api_key:
        print("ERROR: ACBL_API_KEY environment variable not set")
        return 1

    events_dir = acblPath.joinpath(args.events_dir)
    sessions_dir = acblPath.joinpath(args.sessions_dir)

    if not events_dir.exists():
        print(f"ERROR: events directory does not exist: {events_dir}")
        return 1

    print("=" * 70)
    print("ACBL Tournament Sessions Downloader (from sanctioned events)")
    print("=" * 70)
    print(f"Events dir:   {events_dir}")
    print(f"Sessions dir: {sessions_dir}")
    print(f"Start/end:    {args.start}/{args.end or 'all'}")
    print(f"Sleep:        {args.sleep}s")
    print()

    session_ids = build_session_ids_from_sanctioned_events(events_dir)
    print(f"Derived {len(session_ids):,} unique sessions from sanctioned events")

    written, skipped, errors = download_tournament_sessions(
        session_ids=session_ids,
        api_key=api_key,
        output_dir=sessions_dir,
        full_monty=args.full_monty,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        starting_nfile=args.start,
        ending_nfile=args.end,
        skip_if_json_exists=args.skip_if_json_exists,
        skip_if_sql_exists=args.skip_if_sql_exists,
    )

    print()
    print("=" * 70)
    print(f"COMPLETE: written:{written} skipped:{skipped} errors:{errors}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    from mlBridge import print_started, print_ended
    program_start_time = print_started()
    try:
        sys.exit(main())
    finally:
        print_ended(program_start_time)

