from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime
from desktop.app import (
    _diagnostics_ready,
    _enrollment_ready,
    _host_preview_finished,
    _host_preview_should_start,
    _poll_failure_state,
)
from desktop.profiles import PortableProfileManager


class PortableDesktopProfileTests(unittest.TestCase):
    def _create(self, manager: PortableProfileManager, name: str):
        return manager.create(
            display_name=name,
            model_profile_id=QWEN_PROFILE_ID,
            ssh_target="user@example-host",
            ssh_port=22,
        )

    def test_profiles_have_independent_identity_state_and_no_persisted_enrollment_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            first = self._create(manager, "First")
            second = self._create(manager, "Second")
            self.assertNotEqual(first.profile_id, second.profile_id)
            self.assertNotEqual(first.client_id, second.client_id)

            first_paths = manager.profile_paths(first.profile_id)
            second_paths = manager.profile_paths(second.profile_id)
            self.assertNotEqual(first_paths.client_data, second_paths.client_data)
            self.assertNotIn("REGISTRATION_TOKEN", first_paths.env_file.read_text(encoding="utf-8"))

            env = manager.agent_environment(first, enrollment_token="one-time-token")
            self.assertEqual(env["REGISTRATION_TOKEN"], "one-time-token")
            self.assertEqual(env["CLIENT_ID"], first.client_id)
            self.assertEqual(Path(env["CLIENT_DATA_DIR"]), first_paths.client_data)
            self.assertEqual(Path(env["CLIENT_PRIVATE_DATA_PATH"]), first_paths.private_train)
            self.assertEqual(env["CLIENT_SERVING_BACKEND"], "transformers")
            self.assertEqual(env["CLIENT_SSH_TUNNEL_ENABLED"], "true")
            self.assertNotIn("REGISTRATION_TOKEN", first_paths.env_file.read_text(encoding="utf-8"))

            first_runtime = ClientRuntime(
                data_dir=first_paths.client_data,
                client_id=first.client_id,
                model_profile=pinned_client_profile(QWEN_PROFILE_ID, serving_backend="transformers"),
                private_data_path=first_paths.private_train,
            )
            second_runtime = ClientRuntime(
                data_dir=second_paths.client_data,
                client_id=second.client_id,
                model_profile=pinned_client_profile(QWEN_PROFILE_ID, serving_backend="transformers"),
                private_data_path=second_paths.private_train,
            )
            self.assertNotEqual(
                first_runtime.identity.public_key_b64,
                second_runtime.identity.public_key_b64,
            )

    def test_last_used_profile_is_persisted_in_portable_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            first = self._create(manager, "First")
            second = self._create(manager, "Second")
            manager.set_active(first.profile_id)

            restarted = PortableProfileManager(directory)
            self.assertEqual(restarted.active_profile_id(), first.profile_id)
            self.assertEqual(restarted.active_profile(), first)
            self.assertEqual(len(restarted.list_profiles()), 2)
            self.assertEqual(Path(directory).resolve(), restarted.data_root)
            self.assertNotEqual(first.profile_id, second.profile_id)

    def test_enrollment_waits_for_reachable_ssh_forward(self) -> None:
        waiting = {"tunnel": {"forward_reachable": False}}
        connected = {"tunnel": {"forward_reachable": True}}

        self.assertFalse(_enrollment_ready(waiting, "one-time-token", False))
        self.assertTrue(_enrollment_ready(connected, "one-time-token", False))
        self.assertFalse(_enrollment_ready(connected, None, False))
        self.assertFalse(_enrollment_ready(connected, "one-time-token", True))

    def test_poll_failure_after_agent_was_healthy_does_not_request_ssh_password(self) -> None:
        connection, message = _poll_failure_state(
            controller_running=True,
            agent_has_been_healthy=True,
        )
        self.assertEqual(connection, "Client Agent busy/unresponsive")
        self.assertIn("Do not re-enter the SSH password", message)
        self.assertIn("unless OpenSSH itself prompts", message)

    def test_initial_poll_failure_still_requests_ssh_password(self) -> None:
        connection, message = _poll_failure_state(
            controller_running=True,
            agent_has_been_healthy=False,
        )
        self.assertEqual(connection, "Waiting for SSH authentication")
        self.assertIn("Enter the Host SSH password", message)


    def test_host_preview_is_retryable_after_failure_but_not_duplicated_inflight(self) -> None:
        previewed: set[str] = set()
        inflight: set[str] = set()
        round_id = "round-000001"

        self.assertTrue(_host_preview_should_start(round_id, previewed, inflight))
        inflight.add(round_id)
        self.assertFalse(_host_preview_should_start(round_id, previewed, inflight))

        _host_preview_finished(
            round_id,
            previewed,
            inflight,
            succeeded=False,
        )
        self.assertTrue(_host_preview_should_start(round_id, previewed, inflight))

        inflight.add(round_id)
        _host_preview_finished(
            round_id,
            previewed,
            inflight,
            succeeded=True,
        )
        self.assertFalse(_host_preview_should_start(round_id, previewed, inflight))
        self.assertIn(round_id, previewed)

    def test_diagnostics_wait_for_initial_ssh_forward(self) -> None:
        self.assertFalse(
            _diagnostics_ready(
                {"tunnel": {"enabled": True, "forward_reachable": False}}
            )
        )
        self.assertTrue(
            _diagnostics_ready(
                {"tunnel": {"enabled": True, "forward_reachable": True}}
            )
        )
        self.assertTrue(
            _diagnostics_ready(
                {"tunnel": {"enabled": False, "forward_reachable": False}}
            )
        )


if __name__ == "__main__":
    unittest.main()
