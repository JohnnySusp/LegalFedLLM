from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime
from client.training import (
    TrainingExecutionProfile,
    execution_profile_from_environment,
)
from shared.crypto import Ed25519Identity, sha256_hex
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import (
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)


RUN_REAL_MODEL_TESTS = os.getenv(
    "LEGALFEDLLM_RUN_REAL_MODEL_TESTS", "false"
).lower() in {"1", "true", "yes"}


def mock_host_profile() -> ModelProfile:
    return ModelProfile(
        profile_id="acceptance-host",
        role="host",
        model_id="legalfedllm/mock-host",
        model_revision="mock-v1",
        tokenizer_id="legalfedllm/mock-tokenizer",
        tokenizer_revision="mock-v1",
        tokenizer_class="MockTokenizer",
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        lora=LoraProfile(rank=8),
    )


@unittest.skipUnless(
    RUN_REAL_MODEL_TESTS,
    "set LEGALFEDLLM_RUN_REAL_MODEL_TESTS=true for the model download test",
)
class RealClientModelAcceptanceTests(unittest.TestCase):
    def test_one_peft_step_save_reload_promotion_and_restart(self) -> None:
        profile_id = os.getenv(
            "LEGALFEDLLM_REAL_MODEL_PROFILE",
            QWEN_PROFILE_ID,
        )
        profile = pinned_client_profile(profile_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_path = root / "private.jsonl"
            rows = [
                {
                    "schema_version": "1.0",
                    "example_id": "acceptance-1",
                    "prompt": "Ποιος είναι ο σκοπός μιας σύμβασης;",
                    "answer": "Η σύμβαση καταγράφει τους συμφωνημένους όρους.",
                },
                {
                    "schema_version": "1.0",
                    "example_id": "acceptance-2",
                    "prompt": "Give a short legal drafting precaution.",
                    "answer": "Define material terms consistently and review the final text.",
                },
            ]
            private_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            identity = Ed25519Identity.load_or_create(root / "coordinator.pem")
            request = RoundCreateRequest(
                selected_client_ids=["client-a"],
                trusted_client_quorum=1,
                reference_dataset_id="acceptance-reference",
                reference_dataset_hash=sha256_hex(b"acceptance-reference"),
                sample_ids=["acceptance-sample"],
                prompt_template=PROMPT_TEMPLATE,
                label_format="chat_sft_answer_only_v1",
                maximum_sequence_length=512,
                truncation_policy="reject",
                top_k=4,
                training_epochs=1,
            )
            manifest = RoundManifest.create_signed(
                identity=identity,
                round_id="round-real-acceptance",
                coordinator_id="coordinator",
                current_host_adapter_version=0,
                host_model_profile=mock_host_profile(),
                selected_client_profile_hashes={
                    "client-a": profile.profile_hash()
                },
                request=request,
                submission_deadline=utc_text(utc_now() + timedelta(hours=2)),
            )
            execution_values = execution_profile_from_environment(
                "transformers"
            ).model_dump(mode="json")
            execution_values["verify_frozen_base_checksum"] = True
            runtime = ClientRuntime(
                data_dir=root / "client",
                client_id="client-a",
                model_profile=profile,
                private_data_path=private_path,
                training_execution_profile=(
                    TrainingExecutionProfile.model_validate(execution_values)
                ),
            )
            record = runtime.local_train_round(manifest)
            print(
                json.dumps(
                    {
                        "adapter_version": record["result_adapter_version"],
                        "checkpoint_hash": record["result_checkpoint_hash"],
                        "optimizer_step_count": record["optimizer_step_count"],
                        "training_loss": record["training_loss"],
                        "trainable_parameter_count": record[
                            "trainable_parameter_count"
                        ],
                        "total_parameter_count": record[
                            "total_parameter_count"
                        ],
                        "frozen_base_checksum_verified": record[
                            "training_execution_profile"
                        ]["verify_frozen_base_checksum"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            self.assertEqual(record["schema_version"], "1.1")
            self.assertEqual(record["checkpoint_format"], "peft-safetensors")
            self.assertGreaterEqual(record["optimizer_step_count"], 1)
            self.assertGreaterEqual(record["training_loss"], 0)
            self.assertEqual(
                record["training_execution_profile"][
                    "learning_rate_scheduler"
                ],
                "linear",
            )
            self.assertTrue(
                record["training_execution_profile"][
                    "verify_frozen_base_checksum"
                ]
            )
            self.assertGreater(record["trainable_parameter_count"], 0)
            self.assertGreater(
                record["total_parameter_count"],
                record["trainable_parameter_count"],
            )
            metadata, checkpoint_path = runtime.adapter_store.current()
            self.assertEqual(metadata.version, record["result_adapter_version"])
            self.assertTrue(
                (checkpoint_path / "adapter_model.safetensors").is_file()
            )
            self.assertFalse((checkpoint_path / "model.safetensors").exists())

            restarted = ClientRuntime(
                data_dir=root / "client",
                client_id="client-a",
                model_profile=profile,
                private_data_path=private_path,
            )
            restarted.require_round_training(manifest)
            self.assertEqual(
                restarted.state()["training_checkpoint_hash"],
                record["result_checkpoint_hash"],
            )


if __name__ == "__main__":
    unittest.main()
