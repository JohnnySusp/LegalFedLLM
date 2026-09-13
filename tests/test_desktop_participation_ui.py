from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from desktop.app import (
    _desktop_icon_path,
    _participation_allowed,
    _participation_inflight_resolved,
)


class DesktopParticipationUiTests(unittest.TestCase):
    def _round(self, **overrides):
        payload = {
            "round_id": "round-000003",
            "state": "COLLECTING",
            "selected": True,
            "participated": False,
        }
        payload.update(overrides)
        return payload

    def test_participation_is_available_only_for_selected_collecting_client(self) -> None:
        self.assertTrue(
            _participation_allowed(
                self._round(),
                compatible=True,
                coordinator_connected=True,
            )
        )
        self.assertFalse(
            _participation_allowed(
                self._round(participated=True),
                compatible=True,
                coordinator_connected=True,
            )
        )
        self.assertFalse(
            _participation_allowed(
                self._round(state="DISTILLING"),
                compatible=True,
                coordinator_connected=True,
            )
        )

    def test_inflight_request_stays_guarded_while_same_round_remains_retryable(self) -> None:
        self.assertFalse(
            _participation_inflight_resolved(
                self._round(),
                "round-000003",
            )
        )

    def test_inflight_request_resolves_when_client_participated(self) -> None:
        self.assertTrue(
            _participation_inflight_resolved(
                self._round(participated=True),
                "round-000003",
            )
        )

    def test_inflight_request_resolves_when_round_advances(self) -> None:
        self.assertTrue(
            _participation_inflight_resolved(
                self._round(round_id="round-000004"),
                "round-000003",
            )
        )

    def test_icon_prefers_ico_on_windows_and_png_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "legalfedllm.ico").write_bytes(b"ico")
            (root / "legalfedllm.png").write_bytes(b"png")

            self.assertEqual(
                _desktop_icon_path(root, platform="win32"),
                root / "legalfedllm.ico",
            )
            self.assertEqual(
                _desktop_icon_path(root, platform="linux"),
                root / "legalfedllm.png",
            )

    def test_icon_path_returns_none_when_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_desktop_icon_path(Path(directory), platform="win32"))


if __name__ == "__main__":
    unittest.main()
