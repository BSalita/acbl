from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import acbl_elo_ratings_create
from acbl_tournament_download_sessions_using_sanctioned_events import (
    audit_session_artifacts,
    build_session_ids_from_sanctioned_events,
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

    def test_acbl_builder_persists_elo_from_first_session(self) -> None:
        self.assertEqual(
            acbl_elo_ratings_create.PERSISTED_ELO_MINIMUM_SESSIONS,
            1,
        )


if __name__ == "__main__":
    unittest.main()
