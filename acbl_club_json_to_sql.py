"""
Converted from `acbl_club_json_to_sql.ipynb`.

Takes 4h to process 1.2m json files into 150GB sql file. Use Windows version of Python to save 1h. C:/Users/bsali/AppData/Local/Microsoft/WindowsApps/python.exe

Creates per-session `.data.sql` files from `club-results/*/details/*.data.json`, then
builds a SQLite database from the SQL files.

Notes from the original notebook:
- Creating an in-memory DB, executing scripts, then writing to disk is ~100x faster
  than directly writing to a DB on disk.
- `PRAGMA journal_mode=WAL;` can dramatically improve SQLite performance.
"""

import argparse
import os
import pathlib
import sys
import time
import traceback  # kept for parity with the notebook

# ---------------------------------------------------------------------------
# Graceful Ctrl+C handling on Windows
# ---------------------------------------------------------------------------
# The Intel Fortran runtime (libifcoremd.dll, used by NumPy/Pandas MKL builds)
# installs its own console control handler via SetConsoleCtrlHandler. When Ctrl+C
# arrives, Fortran's handler runs and prints:
#   forrtl: error (200): program aborting due to control-C event
#
# To suppress this, we install our *own* handler using the Windows API that runs
# BEFORE Fortran's (handlers are called LIFO) and immediately terminates via
# os._exit(), bypassing all other handlers.
# ---------------------------------------------------------------------------

_ctrl_c_pressed = False
_handler_refs = []  # Keep references to prevent garbage collection


def _install_windows_ctrl_handler() -> None:
    """Install a Windows console control handler to gracefully handle Ctrl+C."""
    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1

    # Handler type: BOOL WINAPI HandlerRoutine(DWORD dwCtrlType)
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    kernel32 = ctypes.windll.kernel32

    def handler(ctrl_type: int) -> bool:
        global _ctrl_c_pressed
        if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
            if not _ctrl_c_pressed:
                _ctrl_c_pressed = True
                # Print message to stderr (stdout may be buffered)
                sys.stderr.write("\n\nCtrl+C detected. Exiting gracefully.\n")
                sys.stderr.flush()
            # Exit immediately, bypassing Fortran runtime's handler
            os._exit(130)
        return False  # Let other handlers process unknown events

    # Keep a reference so the callback isn't garbage collected
    _handler_func = HANDLER_ROUTINE(handler)
    # Add=True means add to handler list (called before existing handlers)
    kernel32.SetConsoleCtrlHandler(_handler_func, True)
    # Store reference to prevent GC (append to list to handle multiple calls)
    _handler_refs.append(_handler_func)


# Import mlBridge libraries (these may trigger numpy/pandas/MKL imports which
# install the Fortran console handler)
import mlBridge.mlBridgeLib as mlBridgeLib
import mlBridge.mlBridgeAcblLib as mlBridgeAcblLib

# Install our handler AFTER all imports. Windows console handlers are called
# in LIFO order (last registered = first called), so by registering last,
# our handler runs first and calls os._exit() before Fortran's handler can run.
_install_windows_ctrl_handler()

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")


def main(argv: list[str] | None = None) -> int:
    # Reinstall handler in case any lazy imports added new handlers
    _install_windows_ctrl_handler()

    parser = argparse.ArgumentParser(
        description="Create per-session .data.sql files from club results JSON and build a SQLite database."
    )
    parser.add_argument(
        "--keep-bad-sql",
        action="store_true",
        help="On SQL execution errors, keep the offending *.data.sql file (default: delete it).",
    )
    args = parser.parse_args(argv)

    from mlBridge import print_started
    program_start_time = print_started()

    # override pandas display options
    mlBridgeLib.pd_options_display()

    # Generate a SQL file for each JSON file; skip if SQL already exists.
    # JSON files were created by downloading HTML files and extracting JSON.

    # Get all the json files in the club-results/*/details/ directory
    json_unstemed = {
        str(file).removesuffix(".data.json")
        for file in acblPath.joinpath("club-results").rglob("*/details/*.json")
    }
    # Get all the sql files in the club-results/*/details/ directory
    sql_unstemed = {
        str(file).removesuffix(".data.sql")
        for file in acblPath.joinpath("club-results").rglob("*/details/*.sql")
    }

    # Find the difference between the json stems and sql stems
    json_files_without_sql_files = [
        pathlib.Path(fn).with_suffix(".data.json") for fn in (json_unstemed - sql_unstemed)
    ]
    print(f"json_files_without_sql_files: {len(json_files_without_sql_files)}")

    total_urls, total_files_written = mlBridgeAcblLib.club_results_json_to_sql(
        json_files_without_sql_files
    )

    # Build the SQLite DB from the generated SQL files
    create_tables_sql_file = "acbl_club_results_schema.sql"
    db_file = "acbl_club_results.sqlite"
    db_file_connection_string = "sqlite:///" + acblPath.joinpath(db_file).as_posix()
    db_file_path = acblPath.joinpath(db_file)

    mlBridgeAcblLib.club_results_create_sql_db(
        db_file_connection_string,
        create_tables_sql_file,
        db_file_path,
        acblPath,
        "club-results",
        delete_bad_sql_files=not args.keep_bad_sql,
    )

    from mlBridge import print_ended
    print_ended(program_start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
