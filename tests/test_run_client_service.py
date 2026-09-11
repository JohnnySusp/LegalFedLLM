from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_client_service import project_slot_environment


class NativeClientServiceConfigurationTests(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        return {
            "QWEN_CLIENT_ID": "legal-client-1",
            "QWEN_CLIENT_PORT": "8001",
            "QWEN_CLIENT_MODEL_PROFILE": "qwen3-1.7b-lora-v1",
            "QWEN_CLIENT_MODEL_ID": "legalfedllm/client-qwen",
            "QWEN_CLIENT_SERVING_BACKEND": "ollama",
            "QWEN_CLIENT_OLLAMA_MODEL": "qwen3:1.7b",
            "QWEN_CLIENT_PRIVATE_DATA_DIR": str(root / "qwen"),
            "QWEN_CLIENT_PRIVATE_DATASET_ID": "qwen-client-private-v1",
            "QWEN_CLIENT_ADMIN_TOKEN": "qwen-secret",
            "GRANITE_CLIENT_ID": "legal-client-2",
            "GRANITE_CLIENT_PORT": "8002",
            "GRANITE_CLIENT_MODEL_PROFILE": (
                "granite-3.3-2b-instruct-client-lora-v1"
            ),
            "GRANITE_CLIENT_MODEL_ID": "legalfedllm/client-granite",
            "GRANITE_CLIENT_SERVING_BACKEND": "ollama",
            "GRANITE_CLIENT_OLLAMA_MODEL": "granite3.3:2b",
            "GRANITE_CLIENT_PRIVATE_DATA_DIR": str(root / "granite"),
            "GRANITE_CLIENT_PRIVATE_DATASET_ID": "granite-client-private-v1",
            "GRANITE_CLIENT_ADMIN_TOKEN": "granite-secret",
            "REGISTRATION_TOKEN": "registration-secret",
            "CLIENT_TRAINING_DEVICE": "cuda",
        }

    def test_projects_qwen_slot_into_generic_client_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, port = project_slot_environment(
                self._environment(root),
                "client-1",
            )

        self.assertEqual(port, 8001)
        self.assertEqual(environment["CLIENT_ID"], "legal-client-1")
        self.assertEqual(
            environment["CLIENT_MODEL_PROFILE"],
            "qwen3-1.7b-lora-v1",
        )
        self.assertEqual(environment["CLIENT_ADMIN_TOKEN"], "qwen-secret")
        self.assertEqual(
            Path(environment["CLIENT_PRIVATE_DATA_PATH"]),
            root / "qwen" / "train.jsonl",
        )
        self.assertEqual(environment["REGISTRATION_TOKEN"], "registration-secret")
        self.assertEqual(environment["CLIENT_TRAINING_DEVICE"], "cuda")

    def test_projects_granite_slot_into_generic_client_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, port = project_slot_environment(
                self._environment(root),
                "client-2",
            )

        self.assertEqual(port, 8002)
        self.assertEqual(environment["CLIENT_ID"], "legal-client-2")
        self.assertEqual(
            environment["CLIENT_MODEL_PROFILE"],
            "granite-3.3-2b-instruct-client-lora-v1",
        )
        self.assertEqual(environment["CLIENT_ADMIN_TOKEN"], "granite-secret")
        self.assertEqual(
            Path(environment["CLIENT_PRIVATE_DATA_PATH"]),
            root / "granite" / "train.jsonl",
        )

    def test_selected_slot_rejects_missing_or_placeholder_admin_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = self._environment(Path(directory))
            environment["GRANITE_CLIENT_ADMIN_TOKEN"] = ""
            with self.assertRaisesRegex(
                ValueError,
                "GRANITE_CLIENT_ADMIN_TOKEN",
            ):
                project_slot_environment(environment, "client-2")

            environment["GRANITE_CLIENT_ADMIN_TOKEN"] = (
                "placeholder-granite-client-admin-token"
            )
            with self.assertRaisesRegex(
                ValueError,
                "GRANITE_CLIENT_ADMIN_TOKEN",
            ):
                project_slot_environment(environment, "client-2")

    def test_unknown_slot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown Client slot"):
                project_slot_environment(
                    self._environment(Path(directory)),
                    "client-3",
                )


if __name__ == "__main__":
    unittest.main()
