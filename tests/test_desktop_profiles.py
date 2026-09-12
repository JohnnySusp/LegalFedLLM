from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime
from desktop.app import (
    _anythingllm_browser_url,
    _browser_launch_ready,
    _desktop_restart_command,
    _diagnostics_ready,
    _local_ai_ready_message,
    _local_ai_start_ready,
    _provider_details_text,
    open_default_browser,
    _enrollment_ready,
    _host_preview_finished,
    _host_preview_should_start,
    _poll_failure_state,
)
from desktop.local_ai import LocalAiStack
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

    def test_desktop_settings_default_and_persist_globally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            first = self._create(manager, "First")
            second = self._create(manager, "Second")

            self.assertEqual(
                manager.desktop_settings(),
                {"constant_learning": True, "debug_mode": False, "low_vram_mode": False},
            )
            manager.set_desktop_setting("constant_learning", False)
            manager.set_desktop_setting("debug_mode", True)
            manager.set_desktop_setting("low_vram_mode", True)
            manager.set_active(first.profile_id)
            manager.set_active(second.profile_id)

            restarted = PortableProfileManager(directory)
            self.assertEqual(
                restarted.desktop_settings(),
                {"constant_learning": False, "debug_mode": True, "low_vram_mode": True},
            )
            self.assertEqual(restarted.active_profile_id(), second.profile_id)

    def test_reset_desktop_settings_restores_defaults_without_changing_active_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "First")
            manager.set_desktop_setting("constant_learning", False)
            manager.set_desktop_setting("debug_mode", True)
            manager.set_desktop_setting("low_vram_mode", True)

            settings = manager.reset_desktop_settings()

            self.assertEqual(
                settings,
                {"constant_learning": True, "debug_mode": False, "low_vram_mode": False},
            )
            self.assertEqual(manager.desktop_settings(), settings)
            self.assertEqual(manager.active_profile_id(), profile.profile_id)

    def test_desktop_restart_command_preserves_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory).resolve()
            with mock.patch.dict(os.environ, {"APPIMAGE": ""}, clear=False), mock.patch.object(
                __import__("desktop.app", fromlist=["sys"]).sys, "frozen", False, create=True
            ):
                command = _desktop_restart_command(data_root)
            self.assertEqual(command[:3], [os.sys.executable, "-m", "desktop.app"])
            self.assertEqual(command[3:], ["--data-root", str(data_root)])

    def test_low_vram_mode_controls_agent_cuda_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "First")

            inherited = {
                "CLIENT_GRADIENT_CHECKPOINTING": "true",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
            }
            with mock.patch.dict(os.environ, inherited, clear=False):
                normal_env = manager.agent_environment(profile)
            self.assertEqual(normal_env["CLIENT_GRADIENT_CHECKPOINTING"], "false")
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", normal_env)

            manager.set_desktop_setting("low_vram_mode", True)
            with mock.patch.dict(os.environ, inherited, clear=False):
                low_vram_env = manager.agent_environment(profile)
            self.assertEqual(low_vram_env["CLIENT_GRADIENT_CHECKPOINTING"], "true")
            self.assertEqual(
                low_vram_env["PYTORCH_CUDA_ALLOC_CONF"],
                "expandable_segments:True",
            )

    def test_unknown_desktop_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            with self.assertRaisesRegex(ValueError, "unsupported desktop setting"):
                manager.set_desktop_setting("unknown", True)

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


    def test_windows_native_local_ai_does_not_use_backend_as_browser_target(self) -> None:
        self.assertIsNone(
            _anythingllm_browser_url(
                {
                    "mode": "windows-native",
                    "openai_base_url": "http://127.0.0.1:8001/v1",
                    "anythingllm_url": "http://127.0.0.1:3001",
                    "anythingllm_configured": True,
                    "anythingllm_managed": False,
                }
            )
        )
        self.assertEqual(
            _anythingllm_browser_url({"anythingllm_url": "http://127.0.0.1:3001"}),
            "http://127.0.0.1:3001",
        )

    def test_windows_native_ready_message_preserves_anythingllm_default_provider(self) -> None:
        message = _local_ai_ready_message(
            {
                "mode": "windows-native",
                "anythingllm_url": "http://127.0.0.1:3001",
                "anythingllm_settings": {
                    "onboarding_complete": True,
                    "default_provider": "ollama",
                },
            }
        )
        self.assertIn("Native Ollama", message)
        self.assertIn("Generic OpenAI connection is configured", message)
        self.assertIn("default LLM provider was left unchanged", message)
        self.assertNotIn("http://127.0.0.1:3001", message)

    def test_windows_native_ready_message_leaves_fresh_anythingllm_onboarding_to_user(self) -> None:
        message = _local_ai_ready_message(
            {
                "mode": "windows-native",
                "anythingllm_settings": {
                    "onboarding_complete": False,
                    "default_provider": None,
                },
            }
        )
        self.assertIn("Generic OpenAI connection is preconfigured", message)
        self.assertIn("Complete AnythingLLM Desktop's one-time setup", message)
        self.assertIn("default LLM provider remains your choice", message)

    def test_windows_provider_details_expose_only_local_profile_connection_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "Windows")
            stack = LocalAiStack(manager.data_root, platform="win32")
            text = _provider_details_text(
                profile,
                admin_token="local-profile-secret",
                local_ai=stack,
            )

        self.assertIn("AnythingLLM Desktop is external", text)
        self.assertIn("reserves", text)
        self.assertIn("does not change AnythingLLM's default LLM provider", text)
        self.assertIn("choose Generic OpenAI", text)
        self.assertIn("manual fallback", text)
        self.assertIn(f"http://127.0.0.1:{profile.agent_port}/v1", text)
        self.assertIn("local-profile-secret", text)
        self.assertIn("legalfedllm-local", text)
        self.assertIn("legalfedllm-host", text)
        self.assertIn("4096", text)
        self.assertIn("1024", text)
        self.assertIn("not a Host credential", text)
        self.assertNotIn("Runtime files", text)

    def test_anythingllm_browser_launch_waits_for_agent_and_local_stack(self) -> None:
        self.assertFalse(
            _browser_launch_ready(
                agent_healthy=False,
                local_ai_ready=True,
                already_attempted=False,
            )
        )
        self.assertFalse(
            _browser_launch_ready(
                agent_healthy=True,
                local_ai_ready=False,
                already_attempted=False,
            )
        )
        self.assertTrue(
            _browser_launch_ready(
                agent_healthy=True,
                local_ai_ready=True,
                already_attempted=False,
            )
        )
        self.assertFalse(
            _browser_launch_ready(
                agent_healthy=True,
                local_ai_ready=True,
                already_attempted=True,
            )
        )

    def test_local_ai_waits_for_agent_health_and_starts_only_once(self) -> None:
        self.assertFalse(
            _local_ai_start_ready(
                agent_healthy=False,
                already_attempted=False,
            )
        )
        self.assertTrue(
            _local_ai_start_ready(
                agent_healthy=True,
                already_attempted=False,
            )
        )
        self.assertFalse(
            _local_ai_start_ready(
                agent_healthy=True,
                already_attempted=True,
            )
        )

    def test_default_browser_helper_uses_system_webbrowser(self) -> None:
        from unittest import mock

        with mock.patch("desktop.app.webbrowser.open", return_value=True) as opened:
            self.assertTrue(open_default_browser("http://127.0.0.1:3001/"))
        opened.assert_called_once_with(
            "http://127.0.0.1:3001/",
            new=2,
            autoraise=True,
        )

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
