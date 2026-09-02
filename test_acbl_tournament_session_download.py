from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import acbl_elo_ratings_create
from acbl_tournament_download_sessions_using_sanctioned_events import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DownloadStats,
    audit_session_artifacts,
    build_session_ids_from_sanctioned_events,
    download_tournament_sessions,
    is_unavailable_http,
    session_request_timeout,
)


class TournamentSessionDiscoveryTests(unittest.TestCase):
    def test_uses_canonical_event_id_for_nabc_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            event = {
                "id": "NABC262-OSHL",
                "sanction": "2607001",
                "event_code": "OSHL",
                "session_count": 4,
            }
            (events_dir / "NABC262-OSHL.sanction.json").write_text(
                json.dumps(event),
                encoding="utf-8",
            )

            self.assertEqual(
                build_session_ids_from_sanctioned_events(events_dir),
                [
                    "NABC262-OSHL-4",
                    "NABC262-OSHL-3",
                    "NABC262-OSHL-2",
                    "NABC262-OSHL-1",
                ],
            )

    def test_audit_reports_only_sessions_without_json_or_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            (sessions_dir / "event-1.session.json").write_text("{}", encoding="utf-8")
            (sessions_dir / "event-2.session.sql").write_text("-- ok", encoding="utf-8")

            missing = audit_session_artifacts(
                ["event-1", "event-2", "event-3"],
                sessions_dir,
            )

            self.assertEqual(missing, ["event-3"])
            audit = json.loads(
                (sessions_dir / "_session_download_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["expected_sessions"], 3)
            self.assertEqual(audit["complete_sessions"], 2)
            self.assertEqual(audit["unavailable_sessions"], [])
            self.assertEqual(audit["unresolved_sessions"], ["event-3"])

    def test_audit_separates_unavailable_from_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            missing = audit_session_artifacts(
                ["ok", "gone", "later"],
                sessions_dir,
                unavailable_ids=["gone"],
            )
            self.assertEqual(missing, ["ok", "gone", "later"])
            audit = json.loads(
                (sessions_dir / "_session_download_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["unavailable_sessions"], ["gone"])
            self.assertEqual(audit["unresolved_sessions"], ["ok", "later"])

    def test_session_request_timeout_splits_connect_and_read(self) -> None:
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 90)
        self.assertEqual(
            session_request_timeout(90),
            (float(DEFAULT_CONNECT_TIMEOUT_SECONDS), 90.0),
        )
        self.assertEqual(session_request_timeout(10), (10.0, 10.0))
        with self.assertRaises(ValueError):
            session_request_timeout(0)

    def test_unavailable_http_is_only_400_and_404(self) -> None:
        self.assertTrue(is_unavailable_http(400))
        self.assertTrue(is_unavailable_http(404))
        self.assertFalse(is_unavailable_http(429))
        self.assertFalse(is_unavailable_http(500))
        self.assertFalse(is_unavailable_http(200))

    def test_download_stats_fail_only_on_hard_errors(self) -> None:
        self.assertFalse(DownloadStats(unavailable=12).failed())
        self.assertTrue(DownloadStats(errors=1).failed())
        self.assertTrue(DownloadStats(aborted=True).failed())

    def test_download_treats_404_as_unavailable_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            response = Mock()
            response.status_code = 404
            response.headers = {}
            with patch(
                "acbl_tournament_download_sessions_using_sanctioned_events.requests.get",
                return_value=response,
            ) as get:
                stats = download_tournament_sessions(
                    session_ids=["NABC262-SPIN-1"],
                    api_key="test-key",
                    output_dir=output_dir,
                    max_attempts=1,
                )
            get.assert_called_once()
            _args, kwargs = get.call_args
            self.assertEqual(
                kwargs["timeout"],
                session_request_timeout(DEFAULT_TIMEOUT_SECONDS),
            )
            self.assertEqual(stats.unavailable, 1)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(stats.unavailable_ids, ["NABC262-SPIN-1"])
            self.assertFalse(stats.failed())
            self.assertFalse((output_dir / "NABC262-SPIN-1.session.json").exists())

    def test_download_exhausted_timeout_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch(
                "acbl_tournament_download_sessions_using_sanctioned_events.requests.get",
                side_effect=requests.Timeout("read timed out"),
            ):
                stats = download_tournament_sessions(
                    session_ids=["NABC262-WERN-1"],
                    api_key="test-key",
                    output_dir=output_dir,
                    max_attempts=1,
                )
            self.assertEqual(stats.errors, 1)
            self.assertEqual(stats.unavailable, 0)
            self.assertTrue(stats.failed())

    def test_acbl_builder_persists_elo_from_first_session(self) -> None:
        self.assertEqual(
            acbl_elo_ratings_create.PERSISTED_ELO_MINIMUM_SESSIONS,
            1,
        )


if __name__ == "__main__":
    unittest.main()
