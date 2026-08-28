from __future__ import annotations

import json
import math
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
from shared.fedmkt_core.safety import inspect_knowledge_package
from shared.knowledge_artifact import load_package_samples
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import (
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
    write_reference_jsonl,
)


RUN_REAL_MODEL_TESTS = os.getenv(
    "LEGALFEDLLM_RUN_REAL_MODEL_TESTS", "false"
).lower() in {"1", "true", "yes"}
REAL_REFERENCE_DATASET_PATH = os.getenv(
    "LEGALFEDLLM_REAL_REFERENCE_DATASET_PATH", ""
).strip()


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
    def exercise_real_client(
        self,
        reference_samples: list[ReferenceSample],
        *,
        round_id: str,
    ) -> dict[str, object]:
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
            reference_path = root / "reference.jsonl"
            write_reference_jsonl(reference_path, reference_samples)
            reference_identity = reference_dataset_identity(reference_samples)
            identity = Ed25519Identity.load_or_create(root / "coordinator.pem")
            request = RoundCreateRequest(
                selected_client_ids=["client-a"],
                trusted_client_quorum=1,
                reference_dataset_id=reference_identity.dataset_id,
                reference_dataset_hash=reference_identity.dataset_hash,
                sample_ids=[sample.sample_id for sample in reference_samples],
                prompt_template=PROMPT_TEMPLATE,
                label_format="chat_sft_answer_only_v1",
                maximum_sequence_length=int(
                    os.getenv("LEGALFEDLLM_REAL_MAX_SEQUENCE_LENGTH", "512")
                ),
                truncation_policy="reject",
                top_k=4,
                training_epochs=1,
            )
            manifest = RoundManifest.create_signed(
                identity=identity,
                round_id=round_id,
                coordinator_id="coordinator",
                current_host_adapter_version=0,
                host_model_profile=mock_host_profile(),
                selected_client_profile_hashes={"client-a": profile.profile_hash()},
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
            self.assertEqual(record["schema_version"], "1.1")
            self.assertEqual(record["checkpoint_format"], "peft-safetensors")
            self.assertGreaterEqual(record["optimizer_step_count"], 1)
            self.assertGreaterEqual(record["training_loss"], 0)
            self.assertEqual(
                record["training_execution_profile"]["learning_rate_scheduler"],
                "linear",
            )
            self.assertTrue(
                record["training_execution_profile"]["verify_frozen_base_checksum"]
            )
            self.assertGreater(record["trainable_parameter_count"], 0)
            self.assertGreater(
                record["total_parameter_count"],
                record["trainable_parameter_count"],
            )
            metadata, checkpoint_path = runtime.adapter_store.current()
            self.assertEqual(metadata.version, record["result_adapter_version"])
            self.assertTrue((checkpoint_path / "adapter_model.safetensors").is_file())
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
            restarted.cache_reference_dataset(
                manifest=manifest,
                content=reference_path.read_bytes(),
            )
            package = restarted.create_knowledge_package(manifest)
            artifact_path = restarted.package_artifact_path(package)
            artifact_before_retry = artifact_path.read_bytes()
            knowledge = load_package_samples(
                artifact_path,
                package,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            retry = restarted.create_knowledge_package(manifest)
            self.assertEqual(retry, package)
            self.assertEqual(
                restarted.package_artifact_path(retry).read_bytes(),
                artifact_before_retry,
            )
            self.assertEqual(package.round_id, manifest.round_id)
            self.assertEqual(package.manifest_hash, manifest.manifest_hash)
            self.assertEqual(package.sender_id, "client-a")
            self.assertEqual(package.sender_role, "client")
            self.assertEqual(package.model_profile, profile)
            self.assertEqual(
                package.adapter_version,
                record["result_adapter_version"],
            )
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertEqual(package.top_k, manifest.top_k)
            self.assertTrue(package.verify_signature(restarted.identity.public_key_b64))
            self.assertEqual(
                package.artifact.sha256,
                sha256_hex(artifact_before_retry),
            )
            self.assertLessEqual(
                package.artifact.byte_size,
                manifest.maximum_knowledge_package_bytes,
            )
            exported_bytes = (
                json.dumps(
                    package.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
                + artifact_before_retry
            )
            for row in rows:
                self.assertNotIn(row["prompt"].encode("utf-8"), exported_bytes)
                self.assertNotIn(row["answer"].encode("utf-8"), exported_bytes)
            self.assertEqual(
                [sample.sample_id for sample in knowledge],
                [sample.sample_id for sample in reference_samples],
            )
            safety = inspect_knowledge_package(package, knowledge)
            self.assertTrue(safety.accepted, safety.reasons)
            self.assertEqual(safety.probe_stage, "pre_alignment")
            for sample in knowledge:
                self.assertEqual(
                    sample.attention_length,
                    len(sample.source_input_ids),
                )
                self.assertEqual(
                    len(sample.top_k_token_ids),
                    sample.attention_length,
                )
                self.assertEqual(
                    len(sample.top_k_logits),
                    sample.attention_length,
                )
                self.assertTrue(math.isfinite(sample.ce_loss))
                self.assertGreaterEqual(sample.ce_loss, 0)
                self.assertTrue(
                    all(len(row) == manifest.top_k for row in sample.top_k_token_ids)
                )
                self.assertTrue(
                    all(len(row) == manifest.top_k for row in sample.top_k_logits)
                )

            summary = {
                "adapter_version": record["result_adapter_version"],
                "checkpoint_hash": record["result_checkpoint_hash"],
                "optimizer_step_count": record["optimizer_step_count"],
                "training_loss": record["training_loss"],
                "trainable_parameter_count": record["trainable_parameter_count"],
                "total_parameter_count": record["total_parameter_count"],
                "frozen_base_checksum_verified": record["training_execution_profile"][
                    "verify_frozen_base_checksum"
                ],
                "knowledge_sample_count": len(knowledge),
                "knowledge_total_token_count": sum(
                    sample.attention_length for sample in knowledge
                ),
                "package_hash": package.package_hash,
                "artifact_sha256": package.artifact.sha256,
                "artifact_byte_size": package.artifact.byte_size,
                "minimum_ce_loss": min(sample.ce_loss for sample in knowledge),
                "maximum_ce_loss": max(sample.ce_loss for sample in knowledge),
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return summary

    def test_one_peft_step_save_reload_and_real_knowledge(self) -> None:
        samples = [
            ReferenceSample(
                schema_version=1,
                dataset_id="acceptance-reference",
                dataset_version="v1",
                sample_id="acceptance-sample-1",
                chapter="Contract Law",
                section="Purpose",
                question="What is the purpose of a contract?",
                gold_answer="It records the terms agreed by the parties.",
            ),
            ReferenceSample(
                schema_version=1,
                dataset_id="acceptance-reference",
                dataset_version="v1",
                sample_id="acceptance-sample-2",
                chapter="Legal Drafting",
                section="Definitions",
                question="Why define material terms?",
                gold_answer="Definitions keep the document consistent.",
            ),
        ]
        summary = self.exercise_real_client(
            samples,
            round_id="round-real-acceptance",
        )
        self.assertEqual(summary["knowledge_sample_count"], 2)

    @unittest.skipUnless(
        REAL_REFERENCE_DATASET_PATH,
        "set LEGALFEDLLM_REAL_REFERENCE_DATASET_PATH for the full D^P test",
    )
    def test_full_565_sample_reference_dataset(self) -> None:
        samples = load_reference_jsonl(REAL_REFERENCE_DATASET_PATH)
        self.assertEqual(
            len(samples),
            565,
            "the authoritative D^P acceptance input must contain 565 samples",
        )
        self.assertEqual(
            reference_dataset_identity(samples).dataset_hash,
            "5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231",
            "the full acceptance input must be the accepted D^P",
        )
        summary = self.exercise_real_client(
            samples,
            round_id="round-real-full-reference",
        )
        self.assertEqual(summary["knowledge_sample_count"], 565)


if __name__ == "__main__":
    unittest.main()
