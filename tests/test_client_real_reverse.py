from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.reverse_training import ClientReverseCandidateResult
from client.runtime import ClientRuntime
from client.safety_probe import ClientSafetyProbeReport
from client.training import (
    TrainingExecutionProfile,
    execution_profile_from_environment,
)
from shared.alignment_profiles import (
    POC_DTW_PROFILE_VERSION,
    resolve_alignment_profile,
)
from shared.client_reverse_artifact import write_client_reverse_training_artifact
from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_core.reverse_integration import integrate_reverse_distillation
from shared.knowledge_artifact import serialize_knowledge_artifact
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import (
    ClientReverseTrainingJob,
    KnowledgePackage,
    KnowledgeSample,
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceSample,
    client_public_data_partition,
    reference_dataset_identity,
    write_reference_jsonl,
)
from shared.reference_knowledge import encode_reference_samples
from shared.tokenizer_validation import load_pinned_tokenizer


RUN_REAL_CLIENT_REVERSE_TESTS = os.getenv(
    "LEGALFEDLLM_RUN_REAL_CLIENT_REVERSE_TESTS",
    "false",
).lower() in {"1", "true", "yes"}


def _host_fixture_profile() -> ModelProfile:
    endpoint = resolve_alignment_profile(
        f"dtw:{POC_DTW_PROFILE_VERSION}"
    ).host
    return ModelProfile(
        profile_id=endpoint.profile_id,
        role="host",
        model_id=endpoint.model_id,
        model_revision=endpoint.model_revision,
        model_class=endpoint.model_class,
        model_type=endpoint.model_type,
        tokenizer_id=endpoint.tokenizer_id,
        tokenizer_revision=endpoint.tokenizer_revision,
        tokenizer_class=endpoint.tokenizer_class,
        vocabulary_size=endpoint.vocabulary_size,
        training_backend="transformers",
        serving_backend="mock",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        tokenizer_chat_template_hash=endpoint.tokenizer_chat_template_hash,
        chat_template_mode=endpoint.chat_template_mode,
        lora=LoraProfile(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=(),
        ),
    )


class _AlwaysBenignTestProbe:
    def __init__(self, model_profile_hash: str):
        self.model_profile_hash = model_profile_hash
        self.manifest = SimpleNamespace(
            artifact_hash=sha256_hex(
                {
                    "fixture": "real-client-reverse-benign-probe",
                    "model_profile_hash": model_profile_hash,
                }
            )
        )

    def evaluate(self, **values):
        return ClientSafetyProbeReport.create(
            round_id=values["round_id"],
            manifest_hash=values["manifest_hash"],
            job_hash=values["job_hash"],
            probe_id="real-client-reverse-test-probe",
            probe_version="v1",
            probe_artifact_hash=self.manifest.artifact_hash,
            model_profile_hash=self.model_profile_hash,
            parent_checkpoint_hash=values["parent_checkpoint_hash"],
            candidate_checkpoint_hash=values["candidate_checkpoint_hash"],
            delta_sha256=sha256_hex(
                {
                    "parent": values["parent_checkpoint_hash"],
                    "candidate": values["candidate_checkpoint_hash"],
                }
            ),
            maliciousness_probability=0.01,
            harmful_threshold=0.8,
            safety_gate_passed=True,
            created_at=values["created_at"],
        )


def _reference_samples() -> list[ReferenceSample]:
    return [
        ReferenceSample(
            schema_version=1,
            dataset_id="real-reverse-reference",
            dataset_version="v1",
            sample_id=f"real-reverse-{index:02d}",
            chapter="Contract Law",
            section="Formation",
            question=f"State contract drafting precaution number {index}.",
            gold_answer="Use clear terms and verify that the parties agree.",
        )
        for index in range(10)
    ]


def _knowledge_samples(
    encoded,
    *,
    ce_loss: float,
    vocabulary_size: int,
) -> list[KnowledgeSample]:
    values: list[KnowledgeSample] = []
    for sample in encoded:
        token_rows = [
            [token_id, (token_id + 1) % vocabulary_size]
            for token_id in sample.input_ids
        ]
        values.append(
            KnowledgeSample(
                sample_id=sample.sample_id,
                source_input_ids=sample.input_ids,
                attention_length=len(sample.input_ids),
                top_k_token_ids=token_rows,
                top_k_logits=[[4.0, 1.0] for _ in token_rows],
                full_logsumexp=[
                    4.0 + ce_loss,
                    *([6.0] * (len(token_rows) - 1)),
                ],
                gold_token_ids=[token_rows[0][0], *([-100] * (len(token_rows) - 1))],
                gold_token_logits=[4.0, *([0.0] * (len(token_rows) - 1))],
                gold_token_nll=[ce_loss, *([0.0] * (len(token_rows) - 1))],
                ce_loss=ce_loss,
            )
        )
    return values


@unittest.skipUnless(
    RUN_REAL_CLIENT_REVERSE_TESTS,
    "set LEGALFEDLLM_RUN_REAL_CLIENT_REVERSE_TESTS=true for real Qwen reverse training",
)
class RealClientReverseAcceptanceTests(unittest.TestCase):
    def _exercise(self, *, force_rejection: bool, round_id: str):
        profile = pinned_client_profile(QWEN_PROFILE_ID)
        samples = _reference_samples()
        identity = reference_dataset_identity(samples)
        request = RoundCreateRequest(
            selected_client_ids=["client-a"],
            trusted_client_quorum=1,
            reference_dataset_id=identity.dataset_id,
            reference_dataset_hash=identity.dataset_hash,
            sample_ids=[sample.sample_id for sample in samples],
            prompt_template=PROMPT_TEMPLATE,
            label_format="chat_sft_answer_only_v1",
            maximum_sequence_length=int(
                os.getenv("LEGALFEDLLM_REAL_REVERSE_MAX_SEQUENCE_LENGTH", "256")
            ),
            truncation_policy="reject",
            top_k=2,
            training_epochs=1,
            host_public_data_epochs=1,
            client_public_data_epochs=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = Ed25519Identity.load_or_create(root / "coordinator.pem")
            manifest = RoundManifest.create_signed(
                identity=coordinator,
                round_id=round_id,
                coordinator_id="coordinator",
                current_host_adapter_version=0,
                host_model_profile=_host_fixture_profile(),
                selected_client_profile_hashes={
                    "client-a": profile.profile_hash()
                },
                selected_client_alignment_profiles={
                    "client-a": f"dtw:{POC_DTW_PROFILE_VERSION}"
                },
                request=request,
                submission_deadline=utc_text(utc_now() + timedelta(hours=2)),
            )
            private_path = root / "private.jsonl"
            private_rows = [
                {
                    "schema_version": "1.0",
                    "example_id": "private-1",
                    "prompt": "What should a contract define?",
                    "answer": "It should define material terms consistently.",
                },
                {
                    "schema_version": "1.0",
                    "example_id": "private-2",
                    "prompt": "Why review a final agreement?",
                    "answer": "Review confirms that the text matches the agreement.",
                },
            ]
            private_path.write_text(
                "".join(json.dumps(row) + "\n" for row in private_rows),
                encoding="utf-8",
            )
            execution_values = execution_profile_from_environment(
                "transformers"
            ).model_dump(mode="json")
            execution_values["learning_rate"] = float(
                os.getenv("LEGALFEDLLM_REAL_REVERSE_LEARNING_RATE", "1e-7")
            )
            execution_values["verify_frozen_base_checksum"] = True
            execution = TrainingExecutionProfile.model_validate(execution_values)
            probe = _AlwaysBenignTestProbe(profile.profile_hash())
            runtime = ClientRuntime(
                data_dir=root / "client",
                client_id="client-a",
                model_profile=profile,
                private_data_path=private_path,
                training_execution_profile=execution,
                safety_probe=probe,
                force_reverse_validation_failure=force_rejection,
            )
            parent_record = runtime.local_train_round(manifest)
            parent, _ = runtime.adapter_store.current()
            self.assertEqual(
                parent.version,
                parent_record["result_adapter_version"],
            )
            self.assertEqual(
                parent.checkpoint_hash,
                parent_record["result_checkpoint_hash"],
            )

            reference_path = root / "reference.jsonl"
            write_reference_jsonl(reference_path, samples)
            runtime.cache_reference_dataset(
                manifest=manifest,
                content=reference_path.read_bytes(),
            )
            alignment = resolve_alignment_profile(
                f"dtw:{POC_DTW_PROFILE_VERSION}"
            )
            tokenizer = load_pinned_tokenizer(
                alignment.client,
                cache_dir=os.getenv("CLIENT_TOKENIZER_CACHE_DIR")
                or os.getenv("HF_HOME"),
                token=os.getenv("HF_TOKEN") or None,
                local_files_only=False,
            ).tokenizer
            encoded = encode_reference_samples(
                samples,
                tokenizer=tokenizer,
                model_profile=profile,
                maximum_sequence_length=manifest.maximum_sequence_length,
                expected_sample_ids=manifest.sample_ids,
            )
            vocabulary_size = int(profile.vocabulary_size or 0)
            client_samples = _knowledge_samples(
                encoded,
                ce_loss=1.0,
                vocabulary_size=vocabulary_size,
            )
            host_samples = _knowledge_samples(
                encoded,
                ce_loss=0.5,
                vocabulary_size=vocabulary_size,
            )
            _, client_descriptor = serialize_knowledge_artifact(client_samples)
            _, host_descriptor = serialize_knowledge_artifact(host_samples)
            alignment_id = f"dtw:{POC_DTW_PROFILE_VERSION}"
            client_package = KnowledgePackage.create_signed(
                identity=runtime.identity,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id="client-a",
                sender_role="client",
                model_profile=profile,
                adapter_version=parent.version,
                alignment_profile_id=alignment_id,
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                top_k=manifest.top_k,
                sample_ids=manifest.sample_ids,
                artifact=client_descriptor,
            )
            host_identity = Ed25519Identity.load_or_create(root / "host.pem")
            host_package = KnowledgePackage.create_signed(
                identity=host_identity,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id="host",
                sender_role="host",
                model_profile=manifest.host_model_profile,
                adapter_version=1,
                alignment_profile_id=alignment_id,
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                top_k=manifest.top_k,
                sample_ids=manifest.sample_ids,
                artifact=host_descriptor,
            )
            partition = client_public_data_partition(
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                sample_ids=manifest.sample_ids,
            )
            encoded_by_id = {sample.sample_id: sample for sample in encoded}
            # Step 6.2 consumes an already verified Step 6.1 artifact. This
            # fixture constructs that artifact directly in the Client token
            # space so the acceptance test loads only the Qwen model.
            batch = integrate_reverse_distillation(
                client_id="client-a",
                parent_adapter_version=parent.version,
                parent_adapter_hash=parent.checkpoint_hash,
                host_adapter_promoted=True,
                partition_hash=partition.partition_hash,
                transfer_sample_ids=partition.transfer_sample_ids,
                host_package=host_package,
                host_samples=host_samples,
                client_package=client_package,
                client_samples=client_samples,
                labels_by_sample={
                    sample_id: encoded_by_id[sample_id].labels
                    for sample_id in partition.transfer_sample_ids
                },
            )
            artifact_path = runtime.store.path(
                runtime._reverse_artifact_path(manifest.round_id)
            )
            descriptor = write_client_reverse_training_artifact(
                artifact_path,
                batch,
                pad_token_id=0,
                maximum_bytes=manifest.maximum_client_reverse_training_job_bytes,
            )
            job = ClientReverseTrainingJob.create(
                manifest=manifest,
                client_id="client-a",
                client_model_profile_hash=profile.profile_hash(),
                parent_adapter_version=parent.version,
                parent_adapter_hash=parent.checkpoint_hash,
                accepted_host_adapter_version=host_package.adapter_version,
                host_package_hash=host_package.package_hash,
                host_adapter_promoted=True,
                public_data_partition=partition,
                host_teacher_sample_ids=batch.audit.host_teacher_sample_ids,
                client_public_data_epochs=1,
                distillation=manifest.distillation,
                integration_audit_hash=batch.audit.audit_hash,
                artifact=descriptor,
                created_at=utc_text(),
            )
            runtime.store.write_json_if_absent(
                runtime._reverse_job_path(manifest.round_id),
                job.model_dump(mode="json"),
            )
            runtime.store.write_json_if_absent(
                runtime._reverse_audit_path(manifest.round_id),
                batch.audit.model_dump(mode="json"),
            )
            runtime.store.write_json_if_absent(
                runtime._reverse_partition_path(manifest.round_id),
                partition.model_dump(mode="json"),
            )

            decision = runtime._complete_reverse_distillation(job)
            result = ClientReverseCandidateResult.model_validate(
                runtime.store.read_json(
                    runtime._reverse_candidate_result_path(manifest.round_id)
                )
            )
            self.assertGreaterEqual(result.optimizer_step_count, 1)
            self.assertTrue(result.lora_tensors_changed)
            self.assertTrue(result.frozen_base_unchanged)
            self.assertTrue(result.reload_verified)
            self.assertTrue(decision.quality_gate_passed)
            self.assertTrue(decision.safety_gate_passed)
            if force_rejection:
                self.assertFalse(decision.adapter_promoted)
                self.assertEqual(
                    decision.decision_reason,
                    "forced_validation_rejection",
                )
                self.assertTrue(decision.rejected_candidate_discarded)
                current, _ = runtime.adapter_store.current()
                self.assertEqual(current.checkpoint_hash, parent.checkpoint_hash)
            else:
                self.assertTrue(decision.adapter_promoted)
                current, _ = runtime.adapter_store.current()
                self.assertEqual(
                    current.version,
                    decision.candidate_adapter_version,
                )
                decision_path = runtime.store.path(
                    runtime._reverse_decision_path(job.manifest.round_id)
                )
                original_bytes = decision_path.read_bytes()
                restarted = ClientRuntime(
                    data_dir=runtime.store.root,
                    client_id="client-a",
                    model_profile=runtime.model_profile,
                    private_data_path=runtime.private_data_path,
                    training_execution_profile=(
                        runtime.training_execution_profile
                    ),
                    safety_probe=probe,
                )
                retry = restarted._complete_reverse_distillation(job)
                self.assertEqual(retry, decision)
                self.assertEqual(decision_path.read_bytes(), original_bytes)
            return decision

    def test_real_qwen_reverse_candidate_promotes_and_restarts(self) -> None:
        decision = self._exercise(
            force_rejection=False,
            round_id="real-qwen-reverse-promotion",
        )
        self.assertTrue(decision.adapter_promoted)

    def test_real_qwen_reverse_candidate_can_be_forced_rejected(self) -> None:
        decision = self._exercise(
            force_rejection=True,
            round_id="real-qwen-reverse-forced-rejection",
        )
        self.assertFalse(decision.adapter_promoted)
        self.assertEqual(decision.decision_reason, "forced_validation_rejection")


if __name__ == "__main__":
    unittest.main()
