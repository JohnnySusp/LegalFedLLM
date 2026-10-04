from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime
from desktop.app import (
    _anythingllm_browser_url,
    _anythingllm_host_available,
    _browser_launch_ready,
    _desktop_restart_command,
    _desktop_restart_environment,
    _diagnostics_ready,
    _local_ai_ready_message,
    _local_ai_start_ready,
    _provider_detail_fields,
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

    def test_fresh_profile_starts_local_only_without_enrollment_token_or_ssh_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = manager.create(
                display_name="Local only",
                model_profile_id=QWEN_PROFILE_ID,
                ssh_target="",
                ssh_port=22,
            )

            env = manager.agent_environment(profile)

            self.assertEqual(env["CLIENT_SSH_TUNNEL_ENABLED"], "false")
            self.assertEqual(env["CLIENT_SSH_TARGET"], "")
            self.assertNotIn("REGISTRATION_TOKEN", env)

    def test_persisted_registration_reenables_federation_on_later_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "Enrolled")
            registration = manager.profile_paths(profile.profile_id).client_data / "identity" / "registration.json"
            registration.parent.mkdir(parents=True, exist_ok=True)
            registration.write_text("{}", encoding="utf-8")

            env = manager.agent_environment(profile)

            self.assertEqual(env["CLIENT_SSH_TUNNEL_ENABLED"], "true")
            self.assertNotIn("REGISTRATION_TOKEN", env)

    def test_profile_connection_settings_can_be_added_after_local_only_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = manager.create(
                display_name="Local only",
                model_profile_id=QWEN_PROFILE_ID,
                ssh_target="",
                ssh_port=22,
            )

            updated = manager.update_connection(
                profile.profile_id,
                display_name="Local then federated",
                ssh_target="user@example-host",
                ssh_port=2222,
            )

            self.assertEqual(updated.display_name, "Local then federated")
            self.assertEqual(updated.ssh_target, "user@example-host")
            self.assertEqual(updated.ssh_port, 2222)
            self.assertEqual(manager.load(profile.profile_id), updated)

    def test_desktop_settings_default_and_persist_globally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            first = self._create(manager, "First")
            second = self._create(manager, "Second")

            self.assertEqual(
                manager.desktop_settings(),
                {"constant_learning": True, "debug_mode": False, "low_vram_mode": True},
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
                {"constant_learning": True, "debug_mode": False, "low_vram_mode": True},
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
    def test_windows_frozen_restart_resets_pyinstaller_environment(self) -> None:
        app_module = __import__("desktop.app", fromlist=["sys"])
        with mock.patch.dict(
            os.environ,
            {"PYINSTALLER_RESET_ENVIRONMENT": "0"},
            clear=False,
        ), mock.patch.object(app_module.sys, "platform", "win32"), mock.patch.object(
            app_module.sys, "frozen", True, create=True
        ):
            environment = _desktop_restart_environment()
        self.assertEqual(environment["PYINSTALLER_RESET_ENVIRONMENT"], "1")

    def test_source_restart_does_not_force_pyinstaller_reset(self) -> None:
        app_module = __import__("desktop.app", fromlist=["sys"])
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            app_module.sys, "platform", "win32"
        ), mock.patch.object(app_module.sys, "frozen", False, create=True):
            environment = _desktop_restart_environment()
        self.assertNotIn("PYINSTALLER_RESET_ENVIRONMENT", environment)

    def test_low_vram_mode_controls_agent_cuda_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "First")

            manager.set_desktop_setting("low_vram_mode", False)
            inherited = {
                "CLIENT_GRADIENT_CHECKPOINTING": "true",
                "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE": "128",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
            }
            with mock.patch.dict(os.environ, inherited, clear=False):
                normal_env = manager.agent_environment(profile)
            self.assertEqual(normal_env["CLIENT_GRADIENT_CHECKPOINTING"], "true")
            self.assertEqual(normal_env["CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE"], "64")
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", normal_env)

            manager.set_desktop_setting("low_vram_mode", True)
            with mock.patch.dict(os.environ, inherited, clear=False):
                low_vram_env = manager.agent_environment(profile)
            self.assertEqual(low_vram_env["CLIENT_GRADIENT_CHECKPOINTING"], "true")
            self.assertEqual(low_vram_env["CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE"], "32")
            if sys.platform == "win32":
                self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", low_vram_env)
            else:
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

    def test_initial_local_only_poll_failure_does_not_request_ssh_password(self) -> None:
        connection, message = _poll_failure_state(
            controller_running=True,
            agent_has_been_healthy=False,
            waiting_for_ssh=False,
        )
        self.assertEqual(connection, "Starting local Client Agent")
        self.assertIn("Federation is not connected", message)
        self.assertNotIn("SSH password", message)


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

    def test_windows_native_ready_message_preserves_existing_anythingllm_provider(self) -> None:
        message = _local_ai_ready_message(
            {
                "mode": "windows-native",
                "anythingllm_url": "http://127.0.0.1:3001",
                "anythingllm_settings": {
                    "onboarding_complete": True,
                    "default_provider": "ollama",
                    "active_provider": "ollama",
                    "model": "legalfedllm-local",
                    "onboarding_completed_by_legalfedllm": False,
                },
            }
        )
        self.assertIn("Native Ollama", message)
        self.assertIn("Generic OpenAI connection is configured", message)
        self.assertIn("existing provider was preserved", message)
        self.assertIn("choose Local or Host in LegalFedLLM", message)
        self.assertNotIn("http://127.0.0.1:3001", message)

    def test_windows_native_ready_message_reports_automatic_fresh_onboarding(self) -> None:
        message = _local_ai_ready_message(
            {
                "mode": "windows-native",
                "anythingllm_settings": {
                    "onboarding_complete": True,
                    "default_provider": None,
                    "active_provider": "generic-openai",
                    "model": "legalfedllm-local",
                    "onboarding_completed_by_legalfedllm": True,
                },
            }
        )
        self.assertIn("AnythingLLM Desktop was initialized", message)
        self.assertIn("legalfedllm-local selected", message)
        self.assertNotIn("Complete AnythingLLM", message)

    def test_windows_provider_details_are_manual_fallback_for_automatic_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "Windows")
            stack = LocalAiStack(manager.data_root, platform="win32")
            text = _provider_details_text(
                profile,
                admin_token="local-profile-secret",
                local_ai=stack,
            )

        self.assertIn("configures a fresh AnythingLLM Desktop installation automatically", text)
        self.assertIn("selects legalfedllm-local", text)
        self.assertIn("Existing AnythingLLM installations keep their current provider", text)
        self.assertIn("manual fallback", text)
        self.assertIn(f"http://127.0.0.1:{profile.agent_port}/v1", text)
        self.assertIn("local-profile-secret", text)
        self.assertIn("Model:\nlegalfedllm-local", text)
        self.assertIn("4096", text)
        self.assertIn("1024", text)
        self.assertIn("not a Host credential", text)
        self.assertNotIn("Runtime files", text)

    def test_anythingllm_integration_fields_match_single_model_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "Windows")
            stack = LocalAiStack(manager.data_root, platform="win32")
            fields = dict(
                _provider_detail_fields(
                    profile,
                    admin_token="local-profile-secret",
                    local_ai=stack,
                )
            )

        self.assertEqual(fields["Provider"], "Generic OpenAI")
        self.assertEqual(
            fields["OpenAI-compatible base URL"],
            f"http://127.0.0.1:{profile.agent_port}/v1",
        )
        self.assertEqual(fields["API key for this local profile"], "local-profile-secret")
        self.assertEqual(fields["Model"], "legalfedllm-local")
        self.assertNotIn("Local model", fields)
        self.assertNotIn("Host model", fields)
        self.assertNotIn("Runtime files", fields)

    def test_linux_anythingllm_integration_details_do_not_expose_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PortableProfileManager(directory)
            profile = self._create(manager, "Linux")
            stack = LocalAiStack(manager.data_root, platform="linux")
            fields = dict(
                _provider_detail_fields(
                    profile,
                    admin_token="local-profile-secret",
                    local_ai=stack,
                )
            )
            text = _provider_details_text(
                profile,
                admin_token="local-profile-secret",
                local_ai=stack,
            )

        self.assertNotIn("Runtime files", fields)
        self.assertNotIn("Runtime files", text)
        self.assertEqual(fields["Model"], "legalfedllm-local")

    def test_anythingllm_host_selection_requires_enrollment_and_live_coordinator(self) -> None:
        self.assertFalse(_anythingllm_host_available({}, {}))
        self.assertFalse(
            _anythingllm_host_available(
                {"enrolled": True},
                {"coordinator_connected": False},
            )
        )
        self.assertFalse(
            _anythingllm_host_available(
                {"enrolled": False},
                {"coordinator_connected": True},
            )
        )
        self.assertTrue(
            _anythingllm_host_available(
                {"enrolled": True},
                {"coordinator_connected": True},
            )
        )

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
