from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from client.model_profiles import QWEN_PROFILE_ID
from desktop.app import DIAGNOSTIC_MODES, MonitorLoop
from desktop.profiles import DesktopProfile


class DesktopDiagnosticsTests(unittest.TestCase):
    def _profile(self) -> DesktopProfile:
        return DesktopProfile(
            profile_id="profile-test",
            display_name="Test",
            client_id="client-test",
            model_profile_id=QWEN_PROFILE_ID,
            ssh_target="user@example-host",
            ssh_port=22,
            coordinator_local_port=8000,
            agent_port=8001,
            created_at="2026-09-07T00:00:00Z",
        )

    def test_default_diagnostics_are_state_and_gpu_only(self) -> None:
        self.assertEqual(DIAGNOSTIC_MODES, ("state", "gpu"))

    def test_state_monitor_prints_http_ok_and_state_payload(self) -> None:
        monitor = MonitorLoop(self._profile(), "admin-token", "state")
        monitor.api = mock.Mock()
        monitor.api.status.return_value = {"client_id": "client-test", "state": "Idle"}

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            monitor._state()

        text = output.getvalue()
        self.assertTrue(text.startswith("HTTP 200 OK\n"))
        payload = json.loads(text.split("\n", 1)[1])
        self.assertEqual(payload["client_id"], "client-test")
        self.assertEqual(payload["state"], "Idle")


if __name__ == "__main__":
    unittest.main()
