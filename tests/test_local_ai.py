from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.model_profiles import QWEN_PROFILE_ID
from desktop.local_ai import BUNDLE_FILES, LocalAiStack, _read_env
from desktop.profiles import DesktopProfile


class LocalAiStackTests(unittest.TestCase):
    def _profile(self, *, agent_port: int = 8001) -> DesktopProfile:
        return DesktopProfile(
            profile_id="profile-test",
            display_name="Test",
            client_id="client-test",
            model_profile_id=QWEN_PROFILE_ID,
            ssh_target="user@example-host",
            ssh_port=22,
            coordinator_local_port=8000,
            coordinator_remote_host="127.0.0.1",
            coordinator_remote_port=8000,
            agent_port=agent_port,
            created_at="2026-09-08T00:00:00Z",
        )

    def _bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / "compose.yaml").write_text("name: legalfed-ai\n", encoding="utf-8")
        (bundle / "legalfedllm.network.yaml").write_text("services: {}\n", encoding="utf-8")
        env = (
            "STORAGE_DIR=/app/server/storage\n"
            "JWT_SECRET=GENERATED_BY_LEGALFEDLLM\n"
            "LLM_PROVIDER=ollama\n"
            "OLLAMA_BASE_PATH=http://ollama:11434\n"
            "EMBEDDING_ENGINE=native\n"
            "EMBEDDING_MODEL_PREF=Xenova/all-MiniLM-L6-v2\n"
            "VECTOR_DB=lancedb\n"
        )
        (bundle / "anythingllm.env").write_text(env, encoding="utf-8")
        (bundle / "anythingllm.docker.env").write_text(env, encoding="utf-8")
        return bundle

    def test_first_seed_prefers_existing_legacy_four_file_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            legacy = root / "legacy"
            legacy.mkdir()
            for name in BUNDLE_FILES:
                (legacy / name).write_text(f"legacy:{name}\n", encoding="utf-8")

            stack = LocalAiStack(root / "data", bundle_root=bundle, legacy_root=legacy)
            runtime = stack.ensure_runtime_files()

            self.assertEqual(
                sorted(path.name for path in runtime.iterdir() if path.is_file()),
                sorted(BUNDLE_FILES),
            )
            for name in BUNDLE_FILES:
                self.assertEqual(
                    (runtime / name).read_text(encoding="utf-8"),
                    f"legacy:{name}\n",
                )

    def test_provider_configuration_preserves_rag_and_secret_but_targets_active_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            stack = LocalAiStack(
                root / "data",
                bundle_root=bundle,
                legacy_root=root / "missing-legacy",
            )
            stack.ensure_runtime_files()
            docker_env = stack.runtime_root / "anythingllm.docker.env"
            docker_env.write_text(
                "JWT_SECRET=existing-secret\n"
                "LLM_PROVIDER=ollama\n"
                "OLLAMA_BASE_PATH=http://ollama:11434\n"
                "ANYTHINGLLM_FETCH_TIMEOUT=1000\n"
                "ANYTHINGLLM_MAX_RETRIES=5\n"
                "EMBEDDING_ENGINE=native\n"
                "EMBEDDING_MODEL_PREF=custom-embedder\n"
                "VECTOR_DB=lancedb\n",
                encoding="utf-8",
            )

            stack.configure_anythingllm(self._profile(agent_port=8123), "client-admin-secret")
            values = _read_env(docker_env)

            self.assertEqual(values["JWT_SECRET"], "existing-secret")
            self.assertEqual(values["LLM_PROVIDER"], "generic-openai")
            self.assertEqual(
                values["GENERIC_OPEN_AI_BASE_PATH"],
                "http://127.0.0.1:8123/v1",
            )
            self.assertEqual(values["GENERIC_OPEN_AI_MODEL_PREF"], "legalfedllm-local")
            self.assertEqual(values["GENERIC_OPEN_AI_API_KEY"], "client-admin-secret")
            self.assertEqual(values["GENERIC_OPENAI_STREAMING_DISABLED"], "true")
            self.assertEqual(
                values["PROVIDER_DISABLE_NATIVE_TOOL_CALLING"],
                "generic-openai",
            )
            self.assertEqual(values["ANYTHINGLLM_FETCH_TIMEOUT"], "1800000")
            self.assertEqual(values["ANYTHINGLLM_MAX_RETRIES"], "0")
            self.assertEqual(values["EMBEDDING_MODEL_PREF"], "custom-embedder")
            self.assertEqual(values["VECTOR_DB"], "lancedb")
            self.assertNotIn("OLLAMA_BASE_PATH", values)
            self.assertEqual(
                _read_env(stack.runtime_root / "anythingllm.env"),
                values,
            )

    def test_placeholder_jwt_is_replaced_without_changing_on_reconfigure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
            )
            profile = self._profile()
            stack.configure_anythingllm(profile, "first-token")
            first = _read_env(stack.runtime_root / "anythingllm.docker.env")["JWT_SECRET"]
            self.assertNotEqual(first, "GENERATED_BY_LEGALFEDLLM")
            self.assertGreaterEqual(len(first), 32)

            stack.configure_anythingllm(profile, "second-token")
            values = _read_env(stack.runtime_root / "anythingllm.docker.env")
            self.assertEqual(values["JWT_SECRET"], first)
            self.assertEqual(values["GENERIC_OPEN_AI_API_KEY"], "second-token")


    def test_windows_prepare_uses_native_ollama_without_docker_or_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="win32",
            )
            profile = self._profile(agent_port=8123)
            with (
                mock.patch("desktop.local_ai.shutil.which", return_value=r"C:\Program Files\Ollama\ollama.exe") as which,
                mock.patch.object(stack, "configure_anythingllm") as configure,
                mock.patch.object(stack, "_run") as run,
                mock.patch.object(stack, "_wait_for_ollama") as wait,
                mock.patch.object(stack, "_verify_ollama_model") as verify,
            ):
                result = stack.prepare(profile, "client-admin-secret")

            which.assert_called_once_with("ollama")
            configure.assert_not_called()
            run.assert_not_called()
            wait.assert_called_once_with()
            verify.assert_called_once_with("qwen3:1.7b")
            self.assertFalse(stack.runtime_root.exists())
            self.assertEqual(result["mode"], "windows-native")
            self.assertEqual(result["openai_base_url"], "http://127.0.0.1:8123/v1")
            self.assertFalse(result["anythingllm_managed"])
            self.assertNotIn("anythingllm_url", result)

    def test_windows_prepare_requires_native_ollama_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="win32",
            )
            with mock.patch("desktop.local_ai.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Native Ollama for Windows"):
                    stack.prepare(self._profile(), "client-admin-secret")
            self.assertFalse(stack.runtime_root.exists())

    def test_windows_stack_does_not_claim_or_stop_external_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="win32",
            )
            with (
                mock.patch("desktop.local_ai.shutil.which") as which,
                mock.patch.object(stack, "_run") as run,
            ):
                self.assertEqual(stack.running_services(), set())
                self.assertFalse(stack.is_running())
                stack.stop()
            which.assert_not_called()
            run.assert_not_called()

    def test_prepare_creates_missing_network_starts_ollama_then_anythingllm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="linux",
            )
            calls: list[list[str]] = []

            def run(command: list[str], *, allow_failure: bool = False):
                calls.append(command)
                return subprocess.CompletedProcess(
                    command,
                    1 if command[1:3] == ["network", "inspect"] else 0,
                    "",
                    "",
                )

            with (
                mock.patch("desktop.local_ai.shutil.which", return_value="/usr/bin/docker"),
                mock.patch.object(stack, "_run", side_effect=run),
                mock.patch.object(stack, "_wait_for_ollama") as wait,
                mock.patch.object(stack, "_verify_ollama_model") as verify,
            ):
                result = stack.prepare(self._profile(), "client-admin-secret")

            self.assertEqual(calls[0], ["/usr/bin/docker", "network", "inspect", "legalfed-ai-net"])
            self.assertEqual(calls[1], ["/usr/bin/docker", "network", "create", "legalfed-ai-net"])
            self.assertEqual(
                calls[2],
                ["/usr/bin/docker", "compose", "-f", "compose.yaml", "up", "-d", "ollama"],
            )
            self.assertEqual(
                calls[3],
                [
                    "/usr/bin/docker",
                    "compose",
                    "-f",
                    "compose.yaml",
                    "up",
                    "-d",
                    "--force-recreate",
                    "anythingllm",
                ],
            )
            wait.assert_called_once_with()
            verify.assert_called_once_with("qwen3:1.7b")
            self.assertEqual(result["anythingllm_url"], "http://127.0.0.1:3001")

    def test_running_services_detects_only_managed_running_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="linux",
            )
            stack.ensure_runtime_files()
            result = subprocess.CompletedProcess(
                ["docker"],
                0,
                "anythingllm\nollama\nunrelated-service\n",
                "",
            )
            with (
                mock.patch("desktop.local_ai.shutil.which", return_value="/usr/bin/docker"),
                mock.patch.object(stack, "_run", return_value=result) as run,
            ):
                self.assertEqual(stack.running_services(), {"anythingllm", "ollama"})
                self.assertTrue(stack.is_running())

            run.assert_called_with(
                [
                    "/usr/bin/docker",
                    "compose",
                    "-f",
                    "compose.yaml",
                    "ps",
                    "--services",
                    "--status",
                    "running",
                ],
                allow_failure=True,
            )

    def test_running_services_is_empty_when_runtime_or_docker_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="linux",
            )
            self.assertEqual(stack.running_services(), set())
            stack.ensure_runtime_files()
            with mock.patch("desktop.local_ai.shutil.which", return_value=None):
                self.assertEqual(stack.running_services(), set())
                self.assertFalse(stack.is_running())

    def test_stop_preserves_runtime_and_uses_compose_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
                platform="linux",
            )
            stack.ensure_runtime_files()
            calls: list[list[str]] = []

            def run(command: list[str], *, allow_failure: bool = False):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch("desktop.local_ai.shutil.which", return_value="/usr/bin/docker"),
                mock.patch.object(stack, "_run", side_effect=run),
            ):
                stack.stop()

            self.assertEqual(
                calls,
                [[
                    "/usr/bin/docker",
                    "compose",
                    "-f",
                    "compose.yaml",
                    "stop",
                    "anythingllm",
                    "ollama",
                ]],
            )
            self.assertTrue(stack.runtime_root.is_dir())

    def test_missing_required_ollama_model_fails_without_pulling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = LocalAiStack(
                root / "data",
                bundle_root=self._bundle(root),
                legacy_root=root / "missing-legacy",
            )
            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"models": [{"name": "granite3.3:2b"}]}
            with mock.patch("desktop.local_ai.httpx.get", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "will not download"):
                    stack._verify_ollama_model("qwen3:1.7b")


if __name__ == "__main__":
    unittest.main()
