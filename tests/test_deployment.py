from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
)
from scripts.create_remote_round import (
    RoundClientSlot,
    alignment_version_for_profiles,
    build_round_request,
)
from scripts.run_host_stack import _uvicorn_command, validate_environment
from scripts.run_remote_round import enabled_client_slots
from shared.alignment_profiles import (
    GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION,
    MISTRAL_NEMO_DTW_PROFILE_VERSION,
)


class HostStackConfigurationTests(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        reference = root / "datasets/reference.jsonl"
        validation = root / "datasets/validation.jsonl"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text("{}\n", encoding="utf-8")
        validation.write_text("{}\n", encoding="utf-8")
        return {
            "LEGALFEDLLM_RUNTIME_ROOT": str(root),
            "HOST_MODEL_PROFILE": "mistral-nemo-instruct-2407-host-lora-v1",
            "HOST_TRAINING_BACKEND": "transformers",
            "HOST_SERVING_BACKEND": "mock",
            "ADMIN_TOKEN": "test-secret-admin",
            "REGISTRATION_TOKEN": "test-secret-registration",
            "INTERNAL_API_TOKEN": "test-secret-internal",
            "COORDINATOR_QUORUM_POLICY": "majority",
            "COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM": "2",
            "COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE": "1",
            "HOST_DATA_DIR": str(root / "artifacts/host"),
            "COORDINATOR_DATA_DIR": str(root / "artifacts/coordinator"),
            "HF_HOME": str(root / "cache/huggingface"),
            "PIP_CACHE_DIR": str(root / "cache/pip"),
            "TORCH_HOME": str(root / "cache/torch"),
            "TORCH_EXTENSIONS_DIR": str(root / "cache/torch-extensions"),
            "TRITON_HOME": str(root / "cache/triton"),
            "XDG_CACHE_HOME": str(root / "cache/xdg"),
            "CUDA_CACHE_PATH": str(root / "cache/cuda"),
            "TMPDIR": str(root / "tmp"),
            "COORDINATOR_REFERENCE_DATASET_PATH": str(reference),
            "COORDINATOR_VALIDATION_DATASET_PATH": str(validation),
        }

    def test_valid_configuration_is_loopback_and_scratch_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self._environment(root), clear=True):
                configuration = validate_environment()

            self.assertEqual(configuration["root"], root.resolve())
            self.assertEqual(configuration["host_bind"], "127.0.0.1")
            self.assertEqual(configuration["coordinator_bind"], "127.0.0.1")
            self.assertTrue((root / "cache/triton").is_dir())

    def test_placeholder_secrets_and_external_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = self._environment(root)
            environment["ADMIN_TOKEN"] = "placeholder-admin-token"
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, "ADMIN_TOKEN"):
                    validate_environment()

            environment = self._environment(root)
            environment["HOST_DATA_DIR"] = str(root.parent / "outside")
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, "HOST_DATA_DIR"):
                    validate_environment()

    def test_uvicorn_command_uses_the_active_python(self) -> None:
        command = _uvicorn_command("host.main:app", "127.0.0.1", 8002)
        self.assertEqual(command[1:4], ["-m", "uvicorn", "host.main:app"])
        self.assertEqual(command[-2:], ["--port", "8002"])


class RemoteRoundConfigurationTests(unittest.TestCase):
    def test_only_selected_client_requires_an_admin_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ROUND_CLIENT_SLOTS": "client-1",
                "QWEN_CLIENT_ADMIN_TOKEN": "qwen-secret",
                "GRANITE_CLIENT_ADMIN_TOKEN": "",
            },
            clear=True,
        ):
            slots = enabled_client_slots()

        self.assertEqual([slot.name for slot in slots], ["client-1"])

    def test_selected_client_rejects_a_placeholder_admin_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ROUND_CLIENT_SLOTS": "client-1",
                "QWEN_CLIENT_ADMIN_TOKEN": "placeholder-qwen-token",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "client-1"):
                enabled_client_slots()

    def test_split_round_request_signs_answer_only_private_label_format(self) -> None:
        request = build_round_request(
            slots=[
                RoundClientSlot(
                    name="client-1",
                    client_id="legal-client-1",
                    profile_id=QWEN_PROFILE_ID,
                )
            ],
            alignment_version=MISTRAL_NEMO_DTW_PROFILE_VERSION,
            expected_quorum=1,
        )

        self.assertEqual(request["label_format"], "chat_sft_answer_only_v1")

    def test_profile_selects_the_exact_nemo_alignment(self) -> None:
        self.assertEqual(
            alignment_version_for_profiles({QWEN_PROFILE_ID}),
            MISTRAL_NEMO_DTW_PROFILE_VERSION,
        )
        self.assertEqual(
            alignment_version_for_profiles(
                {GRANITE_3_3_2B_CLIENT_PROFILE_ID}
            ),
            GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION,
        )

    def test_mixed_profiles_fail_until_per_client_manifests_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "one alignment profile"):
            alignment_version_for_profiles(
                {QWEN_PROFILE_ID, GRANITE_3_3_2B_CLIENT_PROFILE_ID}
            )


class DeploymentFileContractTests(unittest.TestCase):
    def test_standalone_clients_are_profiled_and_loopback_only(self) -> None:
        compose = Path("compose.clients.yaml").read_text(encoding="utf-8")

        self.assertIn('profiles: ["qwen"]', compose)
        self.assertIn('profiles: ["granite"]', compose)
        self.assertIn("network_mode: host", compose)
        self.assertIn('"--host", "127.0.0.1"', compose)
        self.assertNotIn("host.docker.internal", compose)

    def test_client_deployment_does_not_receive_coordinator_admin_authority(self) -> None:
        template = Path("config/clients.env.example").read_text(encoding="utf-8")
        compose = Path("compose.clients.yaml").read_text(encoding="utf-8")
        runner = Path("scripts/run_remote_round.py").read_text(encoding="utf-8")

        self.assertNotIn("\nADMIN_TOKEN=", "\n" + template)
        self.assertNotIn("\n      ADMIN_TOKEN:", "\n" + compose)
        self.assertNotIn("${ADMIN_TOKEN", compose)
        self.assertNotIn("X-Admin-Token", runner)

    def test_host_template_uses_normal_five_epoch_profile(self) -> None:
        template = Path("config/container.env.example").read_text(encoding="utf-8")
        self.assertIn("HOST_PUBLIC_DATA_EPOCHS=5", template)
        self.assertIn("HOST_WARMUP_RATIO=0.008", template)
        self.assertIn("ROUND_HOST_PUBLIC_DATA_EPOCHS=5", template)
        self.assertIn("COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE=", template)

    def test_role_environment_templates_do_not_contain_secrets(self) -> None:
        for path in (
            Path("config/container.env.example"),
            Path("config/clients.env.example"),
        ):
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                for prefix in (
                    "ADMIN_TOKEN=",
                    "REGISTRATION_TOKEN=",
                    "INTERNAL_API_TOKEN=",
                    "QWEN_CLIENT_ADMIN_TOKEN=",
                    "GRANITE_CLIENT_ADMIN_TOKEN=",
                ):
                    matching = [
                        line for line in content.splitlines()
                        if line.startswith(prefix)
                    ]
                    if matching:
                        self.assertEqual(matching, [prefix])

    def test_docker_context_excludes_local_state(self) -> None:
        ignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        for required in (
            ".git",
            ".env",
            "config/*.env",
            "data",
            "artifacts",
            "*.pem",
            "*.key",
        ):
            self.assertIn(required, ignore)


if __name__ == "__main__":
    unittest.main()
