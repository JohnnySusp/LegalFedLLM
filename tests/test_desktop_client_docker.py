from __future__ import annotations

import signal
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from desktop import agent_entry

from client.tunnel import SshTunnelConfig, SshTunnelError, SshTunnelManager
from desktop.client_docker import ClientDockerStack, client_runtime_bundle_hash


def make_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
    (root / "compose.yaml").write_text("services:\n  client: {}\n", encoding="utf-8")
    (root / "client").mkdir()
    (root / "client/main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "shared").mkdir()
    (root / "shared/protocol.py").write_text("VALUE = 2\n", encoding="utf-8")
    return root


class DesktopClientDockerTests(unittest.TestCase):
    def _environment(self, data_root: Path) -> dict[str, str]:
        profile = data_root / "profiles/profile-test"
        return {
            "CLIENT_DATA_DIR": str(profile / "client-data"),
            "CLIENT_PRIVATE_DATA_PATH": str(profile / "private/train.jsonl"),
            "HF_HOME": str(data_root / "models/huggingface"),
            "CLIENT_ID": "client-test",
            "CLIENT_MODEL_PROFILE": "qwen3-1.7b-lora-v1",
            "CLIENT_ADMIN_TOKEN": "secret",
            "CLIENT_PRIVATE_DATASET_ID": "client-test-local-learning-v1",
            "CLIENT_AGENT_PORT": "8001",
            "CLIENT_COORDINATOR_LOCAL_PORT": "8000",
            "CLIENT_SSH_TARGET": "user@example-host",
            "CLIENT_SSH_PORT": "22",
        }

    def test_bundle_hash_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = make_bundle(Path(directory) / "bundle")
            first = client_runtime_bundle_hash(bundle)
            second = client_runtime_bundle_hash(bundle)
            self.assertEqual(first, second)
            (bundle / "client/main.py").write_text("VALUE = 3\n", encoding="utf-8")
            self.assertNotEqual(first, client_runtime_bundle_hash(bundle))

    def test_runtime_is_materialized_under_portable_data_root_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = make_bundle(root / "bundle")
            data_root = root / "LegalFedLLM-data"
            stack = ClientDockerStack(self._environment(data_root), bundle_root=bundle)
            first = stack.prepare_runtime()
            second = stack.prepare_runtime()
            self.assertEqual(first, second)
            self.assertEqual(first.parent, data_root / "client-runtime")
            self.assertTrue((first / "client/main.py").is_file())
            self.assertTrue((first / "shared/protocol.py").is_file())
            self.assertEqual(
                (first / ".bundle-sha256").read_text(encoding="utf-8").strip(),
                client_runtime_bundle_hash(bundle),
            )

    def test_docker_environment_keeps_profile_state_and_model_cache_on_host(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = make_bundle(root / "bundle")
            data_root = root / "LegalFedLLM-data"
            stack = ClientDockerStack(self._environment(data_root), bundle_root=bundle)
            stack.prepare_runtime()
            values = stack._docker_environment()
            self.assertEqual(values["LEGALFEDLLM_CLIENT_DATA_DIR"], str((data_root / "profiles/profile-test/client-data").resolve()))
            self.assertEqual(values["LEGALFEDLLM_PRIVATE_DATA_DIR"], str((data_root / "profiles/profile-test/private").resolve()))
            self.assertEqual(values["LEGALFEDLLM_HF_HOME"], str((data_root / "models/huggingface").resolve()))
            self.assertTrue(values["LEGALFEDLLM_CLIENT_IMAGE"].startswith("legalfedllm-client:"))

    def test_appimage_client_runtime_forwards_sequence_chunk_size(self) -> None:
        compose_path = (
            Path(__file__).resolve().parents[1]
            / "desktop"
            / "client-runtime"
            / "compose.yaml"
        )
        compose_text = compose_path.read_text(encoding="utf-8")
        self.assertIn(
            "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE: ${CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE:-0}",
            compose_text,
        )

    def test_appimage_private_learning_directory_is_writable(self) -> None:
        compose_path = (
            Path(__file__).resolve().parents[1]
            / "desktop"
            / "client-runtime"
            / "compose.yaml"
        )
        compose_text = compose_path.read_text(encoding="utf-8")
        self.assertIn(
            '- "${LEGALFEDLLM_PRIVATE_DATA_DIR}:/private"',
            compose_text,
        )
        self.assertNotIn(
            '- "${LEGALFEDLLM_PRIVATE_DATA_DIR}:/private:ro"',
            compose_text,
        )

    def test_external_tunnel_reports_existing_host_forward_without_spawning_ssh(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            config = SshTunnelConfig(
                enabled=True,
                target="",
                local_port=port,
                external=True,
                external_forward_host="127.0.0.1",
            )
            manager = SshTunnelManager(config)
            with mock.patch("client.tunnel.subprocess.Popen") as popen:
                manager.start()
                status = manager.status()
            popen.assert_not_called()
            self.assertTrue(status["enabled"])
            self.assertTrue(status["external"])
            self.assertTrue(status["running"])
            self.assertTrue(status["forward_reachable"])
        finally:
            listener.close()

    def test_external_tunnel_requires_enabled_contract(self) -> None:
        with self.assertRaisesRegex(SshTunnelError, "CLIENT_SSH_TUNNEL_EXTERNAL"):
            SshTunnelConfig(enabled=False, target="", external=True).validate()

    def test_docker_agent_signal_runs_full_stack_cleanup(self) -> None:
        tunnel = mock.Mock()
        stack = mock.Mock()
        handlers: dict[int, object] = {}

        def install_signal(signum: int, handler: object) -> object:
            if callable(handler):
                handlers[signum] = handler
            return signal.SIG_DFL

        def run_foreground() -> int:
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            return 0

        stack.run_foreground.side_effect = run_foreground

        with (
            mock.patch(
                "desktop.agent_entry.SshTunnelConfig.from_environment",
                return_value=mock.Mock(),
            ),
            mock.patch(
                "desktop.agent_entry.SshTunnelManager",
                return_value=tunnel,
            ),
            mock.patch(
                "desktop.agent_entry.ClientDockerStack",
                return_value=stack,
            ),
            mock.patch(
                "desktop.agent_entry.signal.signal",
                side_effect=install_signal,
            ),
        ):
            result = agent_entry._run_docker_agent()

        self.assertEqual(result, 0)
        stack.request_stop.assert_not_called()
        self.assertEqual(stack.stop.call_count, 2)
        tunnel.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
