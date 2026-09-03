from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import httpx
from pydantic import ValidationError

from coordinator.service import ConflictError, CoordinatorService, HostGateway
from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    MISTRAL_NEMO_HOST_PROFILE_ID,
    pinned_host_profile,
)
from host.main import create_app as create_host_app
from host.peft_backend import (
    HOST_INITIALIZATION_RECORD,
    HostBaselineResult,
    InitializedHostAdapter,
    TransformersPeftHostBackend,
    collate_host_training_rows,
    selective_host_loss,
)
from host.runtime import HostRuntime, HostRuntimeError, default_host_profile
from host.training import (
    HostAdapterInitializationRecord,
    HostValidationRecord,
    HostValidationSampleMetric,
    GraniteHostTrainingContract,
    MISTRAL_NEMO_HOST_TRAINING_CONTRACT_ID,
    PinnedHostTrainingContract,
    HostTrainingExecutionProfile,
    host_execution_profile_from_environment,
)
from shared.adapter_checkpoint import (
    AdapterCheckpointMetadata,
    AdapterCheckpointStore,
)
from shared.alignment_profiles import MISTRAL_NEMO_DTW_PROFILE_VERSION
from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import load_package_samples, write_knowledge_artifact
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import (
    AlignmentConfig,
    HostCandidateTrainingResult,
    HostCandidateValidationResult,
    HostTrainingArtifactDescriptor,
    HostTrainingJob,
    HostTrainingJobReceipt,
    KnowledgePackage,
    ModelProfile,
    HostReferenceDatasetBundle,
    RoundCreateRequest,
    RoundManifest,
    RoundState,
    ServiceIdentity,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
)


FROZEN_REFERENCE_SAMPLE_COUNT = 565
FROZEN_REFERENCE_DATASET_HASH = (
    "5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231"
)
FROZEN_VALIDATION_SAMPLE_COUNT = 173
FROZEN_VALIDATION_DATASET_HASH = (
    "1e40a74799b9900ff8b9a9e05dd379fd0c00226370625f7da1fdca13142b83b5"
)


def fake_initialized_adapter() -> InitializedHostAdapter:
    profile = pinned_host_profile()
    contract = GraniteHostTrainingContract.create(profile)
    execution = HostTrainingExecutionProfile(
        device="cpu",
        precision="float32",
    )
    record = HostAdapterInitializationRecord.create(
        contract=contract,
        contract_hash=contract.contract_hash,
        model_profile_hash=profile.profile_hash(),
        execution_profile=execution,
        execution_profile_hash=execution.profile_hash(),
        adapter_version=0,
        initialization_policy=contract.initial_adapter_policy,
        zero_effect_verified=True,
        reload_verified=True,
        trainable_parameter_count=1024,
        total_parameter_count=2_000_000_000,
        dependency_versions={"torch": "test"},
        created_at=utc_text(),
    )
    metadata = AdapterCheckpointMetadata(
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash(),
        model_profile=profile,
        version=0,
        parent_version=None,
        parent_checkpoint_hash=None,
        round_id=None,
        manifest_hash=None,
        execution_profile_hash=execution.profile_hash(),
        checkpoint_hash="a" * 64,
        file_count=3,
        created_at=utc_text(),
    )
    return InitializedHostAdapter(metadata, Path("/test/v000000"), record)


class FakeHostPeftBackend:
    def __init__(
        self,
        candidate_validation_losses: list[float] | None = None,
    ) -> None:
        self.calls = 0
        self.generate_calls = 0
        self.train_calls = 0
        self.validation_calls = 0
        self.promotion_calls = 0
        self.discard_calls = 0
        self.discarded_candidates: set[tuple[str, int]] = set()
        self.candidate_validation_losses = (
            candidate_validation_losses or [0.1, 0.2]
        )
        self.initialized = fake_initialized_adapter()
        self.contract = self.initialized.initialization.contract
        self.execution_profile = (
            self.initialized.initialization.execution_profile
        )

    def initialize_adapter(self) -> InitializedHostAdapter:
        self.calls += 1
        return self.initialized

    def generate_baseline(
        self,
        reference_samples,
        validation_samples,
        validation_identity,
        manifest,
        *,
        expected_adapter_version,
        expected_checkpoint_hash,
    ) -> HostBaselineResult:
        self.generate_calls += 1
        if expected_adapter_version != self.initialized.metadata.version:
            raise RuntimeError("unexpected adapter version")
        if expected_checkpoint_hash != self.initialized.metadata.checkpoint_hash:
            raise RuntimeError("unexpected checkpoint hash")
        if [sample.sample_id for sample in reference_samples] != manifest.sample_ids:
            raise RuntimeError("unexpected D^P order")
        metrics = [
            HostValidationSampleMetric(
                sample_id=sample.sample_id,
                answer_token_count=index + 1,
                answer_token_ce=0.25 + index * 0.5,
            )
            for index, sample in enumerate(validation_samples)
        ]
        validation = HostValidationRecord.create(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            validation_dataset=validation_identity,
            host_model_profile_hash=pinned_host_profile().profile_hash(),
            adapter_version=expected_adapter_version,
            checkpoint_hash=expected_checkpoint_hash,
            contract_hash=self.contract.contract_hash,
            execution_profile_hash=self.execution_profile.profile_hash(),
            samples=metrics,
        )
        return HostBaselineResult(
            knowledge_samples=deterministic_knowledge_samples(
                manifest=manifest,
                participant_id="legalfedllm-host",
                role="host",
                adapter_version=expected_adapter_version,
            ),
            validation=validation,
        )

    def train_candidate(self, job, artifact_path) -> HostCandidateTrainingResult:
        self.train_calls += 1
        self.asserted_artifact_path = Path(artifact_path)
        return HostCandidateTrainingResult.create(
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            parent_adapter_version=job.host_adapter_version,
            parent_adapter_hash=self.initialized.metadata.checkpoint_hash,
            candidate_adapter_version=job.host_adapter_version + 1,
            candidate_adapter_hash="c" * 64,
            host_model_profile_hash=job.host_model_profile_hash,
            execution_profile_hash=self.execution_profile.profile_hash(),
            host_public_data_epochs=job.host_public_data_epochs,
            supervised_loss_weight=0.9,
            distillation_loss_weight=0.1,
            loss_type="ce",
            temperature=1.0,
            optimizer_step_count=1,
            optimizer_loss=0.75,
            supervised_answer_loss=0.8,
            distillation_answer_loss=0.3,
            trainable_parameter_count=1024,
            total_parameter_count=2_000_000_000,
            lora_tensors_changed=True,
            frozen_base_unchanged=True,
            reload_verified=True,
            dependency_versions={"torch": "test"},
            created_at=utc_text(),
        )

    def validate_candidate(self, result) -> Path:
        self.validated_candidate = result
        return Path("/test/candidates") / result.round_id

    def validate_trained_candidate(
        self,
        validation_samples,
        validation_identity,
        manifest,
        result,
    ) -> HostValidationRecord:
        self.validation_calls += 1
        losses = self.candidate_validation_losses
        if len(losses) != len(validation_samples):
            raise RuntimeError("fake validation loss count differs")
        return HostValidationRecord.create(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            validation_dataset=validation_identity,
            host_model_profile_hash=pinned_host_profile().profile_hash(),
            adapter_version=result.candidate_adapter_version,
            checkpoint_hash=result.candidate_adapter_hash,
            contract_hash=self.contract.contract_hash,
            execution_profile_hash=self.execution_profile.profile_hash(),
            samples=[
                HostValidationSampleMetric(
                    sample_id=sample.sample_id,
                    answer_token_count=index + 1,
                    answer_token_ce=losses[index],
                )
                for index, sample in enumerate(validation_samples)
            ],
        )

    def promote_candidate(self, result) -> InitializedHostAdapter:
        if (
            self.initialized.metadata.version == result.candidate_adapter_version
            and self.initialized.metadata.checkpoint_hash
            == result.candidate_adapter_hash
        ):
            return self.initialized
        self.promotion_calls += 1
        parent = self.initialized.metadata
        metadata = parent.model_copy(
            update={
                "version": result.candidate_adapter_version,
                "parent_version": result.parent_adapter_version,
                "parent_checkpoint_hash": result.parent_adapter_hash,
                "round_id": result.round_id,
                "manifest_hash": result.manifest_hash,
                "checkpoint_hash": result.candidate_adapter_hash,
            }
        )
        self.initialized = InitializedHostAdapter(
            metadata,
            Path("/test/versions") / f"v{metadata.version:06d}",
            self.initialized.initialization,
        )
        return self.initialized

    def discard_candidate(self, result) -> None:
        key = (result.round_id, result.candidate_adapter_version)
        if key in self.discarded_candidates:
            return
        self.discard_calls += 1
        self.discarded_candidates.add(key)

    def generate_post_decision_knowledge(
        self,
        reference_samples,
        manifest,
        *,
        expected_adapter_version,
        expected_checkpoint_hash,
    ):
        if expected_adapter_version != self.initialized.metadata.version:
            raise RuntimeError("unexpected accepted adapter version")
        if expected_checkpoint_hash != self.initialized.metadata.checkpoint_hash:
            raise RuntimeError("unexpected accepted checkpoint hash")
        return deterministic_knowledge_samples(
            manifest=manifest,
            participant_id="legalfedllm-host",
            role="host",
            adapter_version=expected_adapter_version,
        )


def reference_sample(sample_id: str) -> ReferenceSample:
    return ReferenceSample(
        dataset_id="host-reference-v1",
        dataset_version="2026-08-18",
        sample_id=sample_id,
        chapter="Chapter A",
        section="Section B",
        question=f"Question for {sample_id}?",
        gold_answer=f"Answer for {sample_id}.",
    )


def host_round_bundle(
    *,
    reference: list[ReferenceSample] | None = None,
    validation: list[ReferenceSample] | None = None,
    round_id: str = "host-baseline-round",
    maximum_sequence_length: int = 128,
    top_k: int = 2,
    maximum_knowledge_package_bytes: int = 1024 * 1024,
    host_public_data_epochs: int = 5,
    alignment: AlignmentConfig | None = None,
    host_model_profile: ModelProfile | None = None,
) -> tuple[RoundManifest, HostReferenceDatasetBundle]:
    reference = reference or [
        reference_sample("dp-1"),
        reference_sample("dp-2"),
    ]
    validation = validation or [
        reference_sample("dv-1"),
        reference_sample("dv-2"),
    ]
    reference_identity = reference_dataset_identity(reference)
    validation_identity = reference_dataset_identity(validation)
    request = RoundCreateRequest(
        selected_client_ids=["client-a"],
        trusted_client_quorum=1,
        reference_dataset_id=reference_identity.dataset_id,
        reference_dataset_hash=reference_identity.dataset_hash,
        sample_ids=[sample.sample_id for sample in reference],
        prompt_template=PROMPT_TEMPLATE,
        label_format="causal_lm",
        maximum_sequence_length=maximum_sequence_length,
        truncation_policy="reject",
        top_k=top_k,
        maximum_knowledge_package_bytes=maximum_knowledge_package_bytes,
        host_public_data_epochs=host_public_data_epochs,
        alignment=alignment or AlignmentConfig(),
    )
    manifest = RoundManifest.create_signed(
        identity=Ed25519Identity(Ed25519PrivateKey.generate()),
        round_id=round_id,
        coordinator_id="coordinator",
        current_host_adapter_version=0,
        host_model_profile=host_model_profile or pinned_host_profile(),
        selected_client_profile_hashes={"client-a": "b" * 64},
        request=request,
        submission_deadline=(
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat().replace("+00:00", "Z"),
    )
    return manifest, HostReferenceDatasetBundle(
        manifest=manifest,
        reference_samples=reference,
        validation_samples=validation,
        validation_identity=validation_identity,
    )


def write_test_host_training_job(
    manifest: RoundManifest,
    path: Path,
    *,
    integration_audit_hash: str = "e" * 64,
) -> HostTrainingJob:
    import numpy as np
    from safetensors.numpy import save as save_safetensors

    sample_ids = list(manifest.sample_ids)
    if manifest.top_k != 2:
        raise ValueError("test artifact requires top-k 2")
    answer_token_ids = [2 + index % 64 for index in range(len(sample_ids))]
    alternate_token_ids = [
        128 + index % 64 for index in range(len(sample_ids))
    ]
    input_rows = [
        [1, token_id, 0] for token_id in answer_token_ids
    ]
    attention_rows = [[1, 1, 0] for _ in sample_ids]
    label_rows = [
        [-100, token_id, -100] for token_id in answer_token_ids
    ]
    sparse_token_rows = [
        [
            [1, token_id],
            [token_id, alternate_token_id],
            [0, 1],
        ]
        for token_id, alternate_token_id in zip(
            answer_token_ids,
            alternate_token_ids,
        )
    ]
    tensors = {
        "input_ids": np.asarray(input_rows, dtype=np.int32),
        "attention_mask": np.asarray(attention_rows, dtype=np.uint8),
        "labels": np.asarray(label_rows, dtype=np.int32),
        "sparse_target_token_ids": np.asarray(
            sparse_token_rows,
            dtype=np.int32,
        ),
        "sparse_target_probabilities": np.tile(
            np.asarray([0.75, 0.25], dtype=np.float32),
            (len(sample_ids), 3, 1),
        ),
        "sparse_target_valid_mask": np.ones(
            (len(sample_ids), 3, 2), dtype=np.uint8
        ),
    }
    artifact = save_safetensors(tensors)
    path.write_bytes(artifact)
    trainer_inputs_hash = sha256_hex(
        {
            "sample_ids": sample_ids,
            "input_ids": [row[:2] for row in input_rows],
            "attention_lengths": [2 for _ in sample_ids],
            "labels": [row[:2] for row in label_rows],
            "pad_token_id": 0,
            "target_token_ids": tensors[
                "sparse_target_token_ids"
            ].tolist(),
            "target_probabilities": tensors[
                "sparse_target_probabilities"
            ].tolist(),
            "target_valid_mask": tensors[
                "sparse_target_valid_mask"
            ].astype(bool).tolist(),
        }
    )
    descriptor = HostTrainingArtifactDescriptor(
        byte_size=len(artifact),
        sha256=sha256_hex(artifact),
        sample_count=len(sample_ids),
        sample_ids_sha256=sha256_hex(sample_ids),
        padded_sequence_length=3,
        top_k=manifest.top_k,
        pad_token_id=0,
        trainer_inputs_sha256=trainer_inputs_hash,
    )
    return HostTrainingJob.create(
        manifest=manifest,
        dataset_hash="d" * 64,
        integration_audit_hash=integration_audit_hash,
        accepted_client_ids=["client-a"],
        sample_ids=sample_ids,
        host_adapter_version=manifest.current_host_adapter_version,
        host_model_profile_hash=manifest.host_model_profile.profile_hash(),
        host_public_data_epochs=manifest.host_public_data_epochs,
        distillation=manifest.distillation,
        artifact=descriptor,
        created_at=utc_text(),
    )


def write_signed_host_package(
    manifest: RoundManifest,
    artifact_path: str | Path,
    identity: Ed25519Identity,
    adapter_version: int,
) -> KnowledgePackage:
    samples = deterministic_knowledge_samples(
        manifest=manifest,
        participant_id="legalfedllm-host",
        role="host",
        adapter_version=adapter_version,
    )
    descriptor = write_knowledge_artifact(
        artifact_path,
        samples,
        maximum_bytes=manifest.maximum_knowledge_package_bytes,
    )
    return KnowledgePackage.create_signed(
        identity=identity,
        round_id=manifest.round_id,
        manifest_hash=manifest.manifest_hash,
        sender_id="legalfedllm-host",
        sender_role="host",
        model_profile=manifest.host_model_profile,
        adapter_version=adapter_version,
        alignment_profile_id=(
            f"{manifest.alignment.strategy}:"
            f"{manifest.alignment.profile_version}"
        ),
        reference_dataset_id=manifest.reference_dataset_id,
        reference_dataset_hash=manifest.reference_dataset_hash,
        top_k=manifest.top_k,
        sample_ids=manifest.sample_ids,
        artifact=descriptor,
    )


class GraniteHostTrainingContractTests(unittest.TestCase):
    def test_candidate_checkpoint_promotion_and_discard_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AdapterCheckpointStore(directory, pinned_host_profile())

            initial_path = store.staging_path("initial")
            (initial_path / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (initial_path / "adapter_model.safetensors").write_bytes(b"initial")
            initial = store.seal(
                initial_path,
                version=0,
                parent=None,
                round_id=None,
                manifest_hash=None,
                execution_profile_hash="a" * 64,
            )
            store.promote(initial_path, initial)

            candidate_path = store.staging_path("candidate-one")
            (candidate_path / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (candidate_path / "adapter_model.safetensors").write_bytes(
                b"candidate-one"
            )
            candidate = store.seal(
                candidate_path,
                version=1,
                parent=initial,
                round_id="round-1",
                manifest_hash="b" * 64,
                execution_profile_hash="a" * 64,
            )
            store.store_candidate(candidate_path, candidate)

            promoted, promoted_path = store.promote_candidate(
                "round-1",
                1,
                candidate.checkpoint_hash,
            )
            retry, retry_path = store.promote_candidate(
                "round-1",
                1,
                candidate.checkpoint_hash,
            )
            self.assertEqual(retry, promoted)
            self.assertEqual(retry_path, promoted_path)
            self.assertEqual(store.current()[0], promoted)

            rejected_path = store.staging_path("candidate-two")
            (rejected_path / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            (rejected_path / "adapter_model.safetensors").write_bytes(
                b"candidate-two"
            )
            rejected = store.seal(
                rejected_path,
                version=2,
                parent=promoted,
                round_id="round-2",
                manifest_hash="c" * 64,
                execution_profile_hash="a" * 64,
            )
            store.store_candidate(rejected_path, rejected)
            store.discard_candidate(
                "round-2",
                2,
                rejected.checkpoint_hash,
            )
            store.discard_candidate(
                "round-2",
                2,
                rejected.checkpoint_hash,
            )
            self.assertFalse(store.candidate_path("round-2", 2).exists())
            self.assertEqual(store.current()[0], promoted)

    def test_selective_loss_is_answer_only_and_uses_agreed_weights(self) -> None:
        import torch

        logits = torch.tensor(
            [[[0.2, 1.2, -0.4], [8.0, -8.0, 3.0], [1.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )
        inputs = {
            "labels": torch.tensor([[-100, 1, -100]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "sparse_target_token_ids": torch.tensor(
                [[[1, 0], [0, 2], [1, 2]]], dtype=torch.long
            ),
            "sparse_target_probabilities": torch.tensor(
                [[[0.8, 0.2], [0.5, 0.5], [0.5, 0.5]]],
                dtype=torch.float32,
            ),
            "sparse_target_valid_mask": torch.ones(
                (1, 3, 2), dtype=torch.bool
            ),
        }
        combined, supervised, distillation = selective_host_loss(
            torch, logits, inputs
        )
        self.assertTrue(
            torch.allclose(combined, 0.9 * supervised + 0.1 * distillation)
        )

        changed = logits.clone()
        changed[:, 1:, :] = torch.tensor(
            [[[100.0, -100.0, 50.0], [-50.0, 100.0, -100.0]]]
        )
        changed_combined, changed_supervised, changed_distillation = (
            selective_host_loss(torch, changed, inputs)
        )
        self.assertTrue(torch.equal(supervised, changed_supervised))
        self.assertTrue(torch.equal(distillation, changed_distillation))
        self.assertTrue(torch.equal(combined, changed_combined))

    def test_host_training_collator_trims_global_sequence_padding(self) -> None:
        import torch

        def feature(length: int) -> dict[str, torch.Tensor]:
            sequence_length = 7
            top_k = 2
            attention_mask = torch.zeros(sequence_length, dtype=torch.long)
            attention_mask[:length] = 1
            return {
                "input_ids": torch.arange(sequence_length, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.arange(sequence_length, dtype=torch.long),
                "sparse_target_token_ids": torch.zeros(
                    (sequence_length, top_k), dtype=torch.long
                ),
                "sparse_target_probabilities": torch.full(
                    (sequence_length, top_k), 0.5, dtype=torch.float32
                ),
                "sparse_target_valid_mask": torch.ones(
                    (sequence_length, top_k), dtype=torch.bool
                ),
            }

        batch = collate_host_training_rows(
            torch,
            [feature(3), feature(5)],
        )

        for name in ("input_ids", "attention_mask", "labels"):
            self.assertEqual(batch[name].shape, (2, 5))
        for name in (
            "sparse_target_token_ids",
            "sparse_target_probabilities",
            "sparse_target_valid_mask",
        ):
            self.assertEqual(batch[name].shape, (2, 5, 2))
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 1, 1])

    def test_chunked_host_loss_matches_existing_loss_and_gradient(self) -> None:
        import torch

        from shared.fedmkt_core.ml.sparse_targets import (
            SparseTargetBatch,
            answer_only_sparse_distillation_loss,
        )

        torch.manual_seed(7)
        logits = torch.randn((2, 7, 11), dtype=torch.float32)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3, -100, 5, -100],
                [-100, 2, 4, 6, 8, -100, -100],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        )
        token_ids = torch.empty((2, 7, 2), dtype=torch.long)
        for batch_index in range(2):
            for position in range(7):
                first = (batch_index + position) % 11
                token_ids[batch_index, position] = torch.tensor(
                    [first, (first + 3) % 11],
                    dtype=torch.long,
                )
        probabilities = torch.tensor([0.75, 0.25], dtype=torch.float32).repeat(
            2, 7, 1
        )
        valid_mask = torch.ones((2, 7, 2), dtype=torch.bool)
        inputs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "sparse_target_token_ids": token_ids,
            "sparse_target_probabilities": probabilities,
            "sparse_target_valid_mask": valid_mask,
        }

        oracle_logits = logits.clone().requires_grad_(True)
        oracle_supervised = torch.nn.functional.cross_entropy(
            oracle_logits[..., :-1, :].contiguous().view(-1, 11),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        oracle_distillation = answer_only_sparse_distillation_loss(
            oracle_logits,
            SparseTargetBatch(
                token_ids=token_ids,
                probabilities=probabilities,
                valid_mask=valid_mask,
            ),
            labels=labels,
            attention_mask=attention_mask,
            loss_type="ce",
        )
        oracle_combined = 0.9 * oracle_supervised + 0.1 * oracle_distillation
        oracle_combined.backward()

        chunked_logits = logits.clone().requires_grad_(True)
        combined, supervised, distillation = selective_host_loss(
            torch,
            chunked_logits,
            inputs,
            sequence_chunk_size=2,
        )
        combined.backward()

        self.assertTrue(torch.allclose(supervised, oracle_supervised, atol=1e-6))
        self.assertTrue(
            torch.allclose(distillation, oracle_distillation, atol=1e-6)
        )
        self.assertTrue(torch.allclose(combined, oracle_combined, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                chunked_logits.grad,
                oracle_logits.grad,
                atol=1e-6,
            )
        )

    def test_chunked_host_loss_bounds_each_vocabulary_reduction(self) -> None:
        import torch

        logits = torch.tensor(
            [[[0.2, 1.2, -0.4]] * 8],
            dtype=torch.float32,
        )
        inputs = {
            "labels": torch.tensor(
                [[-100, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
            "sparse_target_token_ids": torch.tensor(
                [[[[1, 0]] * 8][0]], dtype=torch.long
            ),
            "sparse_target_probabilities": torch.tensor(
                [[[[0.8, 0.2]] * 8][0]], dtype=torch.float32
            ),
            "sparse_target_valid_mask": torch.ones((1, 8, 2), dtype=torch.bool),
        }
        original_logsumexp = torch.logsumexp
        sequence_widths: list[int] = []

        def bounded_logsumexp(value, *args, **kwargs):
            sequence_widths.append(int(value.shape[1]))
            return original_logsumexp(value, *args, **kwargs)

        with mock.patch.object(torch, "logsumexp", side_effect=bounded_logsumexp):
            selective_host_loss(
                torch,
                logits,
                inputs,
                sequence_chunk_size=3,
            )

        self.assertEqual(sequence_widths, [3, 3, 1])
        self.assertLessEqual(max(sequence_widths), 3)

    def test_contract_freezes_the_agreed_host_decisions(self) -> None:
        profile = pinned_host_profile()
        contract = GraniteHostTrainingContract.create(profile)

        self.assertEqual(
            profile.profile_hash(),
            "842b754defd7751a8b9a415757bbd760a870009045eb2cc4b5c2f045a7dbde44",
        )
        self.assertEqual(
            contract.contract_hash,
            "b9ed41565df46babcc3dbe4593a96ba5cdd60ccfe453fd1ea874ee193b3486dd",
        )
        self.assertEqual(contract.host_model_profile_hash, profile.profile_hash())
        self.assertEqual(contract.initial_adapter_policy, "fresh_zero_effect_lora_v0")
        self.assertEqual(contract.authoritative_public_data_epochs, 5)
        self.assertEqual(contract.development_smoke_epochs, 1)
        self.assertEqual(
            contract.primary_validation_metric,
            "macro_mean_answer_token_ce",
        )
        self.assertEqual(
            contract.secondary_validation_metric,
            "token_weighted_answer_token_ce",
        )
        self.assertEqual(contract.minimum_validation_improvement, 0.001)
        self.assertEqual(
            contract.rejected_candidate_policy,
            "discard_weights_retain_audit",
        )

    def test_contract_rejects_an_unpinned_host(self) -> None:
        changed = pinned_host_profile().model_copy(
            update={"model_revision": "main"}
        )
        with self.assertRaisesRegex(ValueError, "exact pinned Host"):
            GraniteHostTrainingContract.create(changed)

    def test_mistral_nemo_uses_its_own_pinned_contract_and_tokenizer(self) -> None:
        profile = pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID)
        contract = PinnedHostTrainingContract.create(profile)
        execution = HostTrainingExecutionProfile(
            device="cpu",
            precision="float32",
            gradient_checkpointing=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )

        self.assertEqual(
            contract.contract_id,
            MISTRAL_NEMO_HOST_TRAINING_CONTRACT_ID,
        )
        self.assertEqual(
            profile.profile_hash(),
            "8bbca6cdac9166d516e1061fc7df65450b1bf7a1d94aed7fb47350efafb7f652",
        )
        self.assertEqual(
            contract.contract_hash,
            "e6fe93ba0b6f8f1b3eea0cad42a7939d5ddf6a4c1d73709146d815fbcf0cc44f",
        )
        self.assertEqual(contract.host_model_profile_hash, profile.profile_hash())
        self.assertEqual(backend.tokenizer_endpoint.profile_id, profile.profile_id)
        self.assertTrue(backend.tokenizer_endpoint.fix_mistral_regex)
        self.assertTrue(backend.tokenizer_endpoint.bind_existing_pad_token)

    def test_execution_profile_uses_the_upstream_host_defaults(self) -> None:
        profile = HostTrainingExecutionProfile()
        self.assertEqual(profile.public_data_epochs, 5)
        self.assertEqual(profile.micro_batch_size, 1)
        self.assertEqual(profile.gradient_accumulation_steps, 4)
        self.assertEqual(profile.learning_rate, 3e-5)
        self.assertEqual(profile.learning_rate_scheduler, "cosine")
        self.assertEqual(profile.warmup_ratio, 0.008)
        self.assertEqual(profile.adam_beta1, 0.9)
        self.assertEqual(profile.adam_beta2, 0.95)
        self.assertEqual(profile.weight_decay, 0.1)
        self.assertEqual(profile.maximum_gradient_norm, 1.0)

        with self.assertRaises(ValidationError):
            HostTrainingExecutionProfile(device="cpu", precision="bfloat16")
        with self.assertRaises(ValidationError):
            HostTrainingExecutionProfile(device="cuda", precision="float32")
        with self.assertRaises(ValidationError):
            HostTrainingExecutionProfile(public_data_epochs=2)

        initialized = fake_initialized_adapter()
        changed_execution = initialized.initialization.execution_profile.model_copy(
            update={"seed": 43}
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                execution_profile=changed_execution,
            )
            with self.assertRaisesRegex(ValueError, "execution profile"):
                backend._validate_initialization(  # noqa: SLF001
                    initialized.metadata,
                    initialized.initialization,
                )

    def test_initialization_record_hash_detects_tampering(self) -> None:
        record = fake_initialized_adapter().initialization
        payload = record.model_dump(mode="json")
        payload["total_parameter_count"] += 1
        with self.assertRaises(ValidationError):
            HostAdapterInitializationRecord.model_validate(payload)

    def test_real_profile_selection_is_explicit_and_exact(self) -> None:
        environment = {
            "HOST_MODEL_PROFILE": GRANITE_3_3_2B_HOST_PROFILE_ID,
            "HOST_TRAINING_BACKEND": "transformers",
            "HOST_SERVING_BACKEND": "mock",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            self.assertEqual(
                default_host_profile().profile_hash(),
                pinned_host_profile().profile_hash(),
            )

        nemo_environment = {
            "HOST_MODEL_PROFILE": MISTRAL_NEMO_HOST_PROFILE_ID,
            "HOST_TRAINING_BACKEND": "transformers",
            "HOST_SERVING_BACKEND": "mock",
        }
        with mock.patch.dict(os.environ, nemo_environment, clear=False):
            self.assertEqual(
                default_host_profile().profile_hash(),
                pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID).profile_hash(),
            )

        with mock.patch.dict(
            os.environ,
            {"HOST_TRAINING_BACKEND": "transformers"},
            clear=False,
        ):
            os.environ.pop("HOST_MODEL_PROFILE", None)
            with self.assertRaisesRegex(HostRuntimeError, "HOST_MODEL_PROFILE"):
                default_host_profile()

    def test_environment_profile_supports_the_one_step_smoke_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "HOST_TRAINING_DEVICE": "cpu",
                "HOST_TRAINING_PRECISION": "float32",
                "HOST_PUBLIC_DATA_EPOCHS": "1",
                "HOST_GRADIENT_ACCUMULATION_STEPS": "1",
                "HOST_WARMUP_RATIO": "0",
                "HOST_GRADIENT_CHECKPOINTING": "true",
            },
            clear=False,
        ):
            profile = host_execution_profile_from_environment()
        self.assertEqual(profile.public_data_epochs, 1)
        self.assertEqual(profile.gradient_accumulation_steps, 1)
        self.assertEqual(profile.warmup_ratio, 0.0)
        self.assertTrue(profile.gradient_checkpointing)

    def test_validation_record_rejects_metric_and_binding_tampering(self) -> None:
        manifest, bundle = host_round_bundle()
        record = HostValidationRecord.create(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            validation_dataset=bundle.validation_identity,
            host_model_profile_hash=pinned_host_profile().profile_hash(),
            adapter_version=0,
            checkpoint_hash="a" * 64,
            contract_hash=GraniteHostTrainingContract.create(
                pinned_host_profile()
            ).contract_hash,
            execution_profile_hash=HostTrainingExecutionProfile(
                device="cpu",
                precision="float32",
            ).profile_hash(),
            samples=[
                HostValidationSampleMetric(
                    sample_id="dv-1",
                    answer_token_count=1,
                    answer_token_ce=0.25,
                ),
                HostValidationSampleMetric(
                    sample_id="dv-2",
                    answer_token_count=3,
                    answer_token_ce=0.75,
                ),
            ],
        )
        self.assertEqual(record.macro_mean_answer_token_ce, 0.5)
        self.assertEqual(record.token_weighted_answer_token_ce, 0.625)

        payload = record.model_dump(mode="json")
        payload["macro_mean_answer_token_ce"] = 0.4
        with self.assertRaises(ValidationError):
            HostValidationRecord.model_validate(payload)

        payload = record.model_dump(mode="json")
        payload["checkpoint_hash"] = "c" * 64
        with self.assertRaises(ValidationError):
            HostValidationRecord.model_validate(payload)

    def test_runtime_generates_signed_baseline_and_persists_validation(self) -> None:
        backend = FakeHostPeftBackend()
        manifest, bundle = host_round_bundle()
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=(
                    backend.initialized.initialization.execution_profile
                ),
                peft_backend=backend,
            )

            self.assertEqual(backend.calls, 1)
            self.assertEqual(runtime.adapter_version, 0)
            self.assertEqual(
                runtime.active_adapter()["checkpoint_hash"],
                "a" * 64,
            )
            self.assertEqual(
                runtime.service_identity()["model_profile"]["profile_id"],
                GRANITE_3_3_2B_HOST_PROFILE_ID,
            )
            receipt = runtime.load_reference_data(bundle)
            self.assertEqual(receipt.validation_identity, bundle.validation_identity)

            package = runtime.generate_reference_knowledge(manifest)
            self.assertEqual(backend.generate_calls, 1)
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertEqual(package.adapter_version, 0)
            self.assertTrue(package.verify_signature(runtime.identity.public_key_b64))
            samples = load_package_samples(
                runtime.knowledge_artifact_path(
                    manifest,
                    enforce_manifest_parent=True,
                ),
                package,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            self.assertEqual(
                [sample.sample_id for sample in samples],
                manifest.sample_ids,
            )

            validation = runtime.validation_baseline(manifest)
            self.assertEqual(validation.sample_count, 2)
            self.assertEqual(validation.supervised_answer_token_count, 3)
            self.assertEqual(validation.macro_mean_answer_token_ce, 0.5)
            self.assertAlmostEqual(
                validation.token_weighted_answer_token_ce,
                7 / 12,
            )

            cached = runtime.generate_reference_knowledge(manifest)
            self.assertEqual(cached.package_hash, package.package_hash)
            self.assertEqual(backend.generate_calls, 1)

            restarted_backend = FakeHostPeftBackend()
            restarted = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=restarted_backend.execution_profile,
                peft_backend=restarted_backend,
            )
            restarted_package = restarted.generate_reference_knowledge(manifest)
            self.assertEqual(restarted_package.package_hash, package.package_hash)
            self.assertEqual(restarted_backend.generate_calls, 0)

            with self.assertRaisesRegex(HostRuntimeError, "not implemented"):
                runtime.distill(None)  # type: ignore[arg-type]

    def test_runtime_rejects_a_tampered_cached_validation_record(self) -> None:
        backend = FakeHostPeftBackend()
        manifest, bundle = host_round_bundle()
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=backend.execution_profile,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            path = runtime.store.path(
                runtime._baseline_validation_path(manifest.round_id)  # noqa: SLF001
            )
            payload = runtime.store.read_json(
                runtime._baseline_validation_path(manifest.round_id)  # noqa: SLF001
            )
            payload["macro_mean_answer_token_ce"] = 99.0
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                HostRuntimeError,
                "validation baseline is invalid",
            ):
                runtime.generate_reference_knowledge(manifest)

    def test_runtime_promotes_validated_candidate_and_publishes_knowledge(
        self,
    ) -> None:
        backend = FakeHostPeftBackend(candidate_validation_losses=[0.1, 0.2])
        manifest, bundle = host_round_bundle(
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1")
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=backend.execution_profile,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            artifact = Path(directory) / "trainer-inputs.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            runtime.load_training_job(job, artifact)
            candidate = runtime.train_candidate(job)

            decision = runtime.validate_candidate_and_decide(job)

            self.assertTrue(decision.adapter_promoted)
            self.assertEqual(decision.decision_reason, "candidate_improved")
            self.assertEqual(decision.accepted_adapter_version, 1)
            self.assertEqual(decision.accepted_adapter_hash, "c" * 64)
            self.assertGreaterEqual(
                decision.observed_improvement,
                decision.required_improvement,
            )
            self.assertEqual(runtime.adapter_version, 1)
            self.assertEqual(backend.validation_calls, 1)
            self.assertEqual(backend.promotion_calls, 1)
            self.assertEqual(backend.discard_calls, 0)
            self.assertEqual(
                decision.candidate_result_hash,
                candidate.result_hash,
            )

            package = runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(package.adapter_version, 1)
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )
            retry = runtime.validate_candidate_and_decide(job)
            self.assertEqual(retry, decision)
            self.assertEqual(backend.validation_calls, 1)
            self.assertEqual(backend.promotion_calls, 1)
            self.assertEqual(runtime.train_candidate(job), candidate)
            self.assertEqual(backend.train_calls, 1)

    def test_runtime_forced_rejection_discards_candidate_and_retains_parent(
        self,
    ) -> None:
        backend = FakeHostPeftBackend(candidate_validation_losses=[0.1, 0.2])
        manifest, bundle = host_round_bundle(
            round_id="forced-validation-rejection",
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=backend.execution_profile,
                peft_backend=backend,
                force_validation_failure=True,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            artifact = Path(directory) / "trainer-inputs.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            runtime.load_training_job(job, artifact)
            candidate = runtime.train_candidate(job)

            decision = runtime.validate_candidate_and_decide(job)

            self.assertFalse(decision.adapter_promoted)
            self.assertEqual(
                decision.decision_reason,
                "forced_validation_rejection",
            )
            self.assertTrue(decision.rejected_candidate_discarded)
            self.assertEqual(decision.accepted_adapter_version, 0)
            self.assertEqual(decision.accepted_adapter_hash, "a" * 64)
            self.assertEqual(runtime.adapter_version, 0)
            self.assertEqual(backend.promotion_calls, 0)
            self.assertEqual(backend.discard_calls, 1)

            package = runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(package.adapter_version, 0)
            retry = runtime.validate_candidate_and_decide(job)
            self.assertEqual(retry, decision)
            self.assertEqual(backend.validation_calls, 1)
            self.assertEqual(backend.discard_calls, 1)
            self.assertEqual(
                runtime.train_candidate(job).result_hash,
                candidate.result_hash,
            )
            self.assertEqual(backend.train_calls, 1)


class HostMlEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_trains_sealed_dtw_round_without_blocking_status(self) -> None:
        manifest, _ = host_round_bundle(
            round_id="async-candidate-round",
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1"),
        )
        backend = FakeHostPeftBackend()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "prepared.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            receipt = HostTrainingJobReceipt(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                job_hash=job.job_hash,
                artifact_sha256=job.artifact.sha256,
                artifact_byte_size=job.artifact.byte_size,
                accepted_client_ids=job.accepted_client_ids,
            )
            result = backend.train_candidate(job, artifact)
            decision = HostCandidateValidationResult.create(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                job_hash=job.job_hash,
                candidate_result_hash=result.result_hash,
                previous_adapter_version=0,
                previous_adapter_hash="a" * 64,
                candidate_adapter_version=1,
                candidate_adapter_hash="c" * 64,
                accepted_adapter_version=1,
                accepted_adapter_hash="c" * 64,
                baseline_validation_record_hash="b" * 64,
                candidate_validation_record_hash="d" * 64,
                previous_macro_mean_answer_token_ce=0.5,
                candidate_macro_mean_answer_token_ce=0.25,
                previous_token_weighted_answer_token_ce=0.55,
                candidate_token_weighted_answer_token_ce=0.3,
                required_improvement=0.001,
                observed_improvement=0.25,
                adapter_promoted=True,
                decision_reason="candidate_improved",
                rejected_candidate_discarded=False,
                created_at=utc_text(),
            )
            host_identity = Ed25519Identity(Ed25519PrivateKey.generate())
            training_started = asyncio.Event()
            release_training = asyncio.Event()

            async def blocking_train_candidate(value):
                self.assertEqual(value, job)
                training_started.set()
                await release_training.wait()
                return result

            gateway = mock.Mock()
            gateway.load_training_job = mock.AsyncMock(return_value=receipt)
            gateway.train_candidate = mock.AsyncMock(
                side_effect=blocking_train_candidate
            )
            gateway.validate_candidate = mock.AsyncMock(
                return_value=decision
            )

            async def post_decision_knowledge(value, artifact_path):
                return write_signed_host_package(
                    value,
                    artifact_path,
                    host_identity,
                    1,
                )

            gateway.post_decision_knowledge = mock.AsyncMock(
                side_effect=post_decision_knowledge
            )
            gateway.identity = mock.AsyncMock(
                return_value=ServiceIdentity(
                    service_id="legalfedllm-host",
                    public_key=host_identity.public_key_b64,
                    model_profile=manifest.host_model_profile,
                    adapter_version=1,
                )
            )
            service = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=gateway,
            )
            service.store.write_json(
                f"rounds/{manifest.round_id}/manifest.json",
                manifest.model_dump(mode="json"),
            )
            service.store.write_json(
                f"rounds/{manifest.round_id}/state.json",
                RoundState(
                    round_id=manifest.round_id,
                    state="SEALED",
                    accepted_client_ids=["client-a"],
                    sealed_client_ids=["client-a"],
                    host_adapter_before=0,
                    message="trusted quorum reached; submission set sealed",
                    updated_at=utc_text(),
                ).model_dump(mode="json"),
            )
            service.store.write_json(
                "rounds/current.json", {"round_id": manifest.round_id}
            )
            target = service.store.path(
                f"rounds/{manifest.round_id}/host_training_job/"
                "trainer_inputs.safetensors"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifact.read_bytes())

            before = await service.round_status(manifest.round_id)
            self.assertEqual(before.state, "SEALED")
            gateway.load_training_job.assert_not_awaited()
            gateway.train_candidate.assert_not_awaited()

            with mock.patch.object(
                service, "_load_host_training_job", return_value=job
            ):
                pending = asyncio.create_task(service.monitor_once())
                await asyncio.wait_for(training_started.wait(), timeout=1)
                during = await asyncio.wait_for(
                    service.round_status(manifest.round_id), timeout=1
                )
                self.assertEqual(during.state, "DISTILLING")
                self.assertIn("optimization is running", during.message)
                release_training.set()
                await pending

            after = service.get_state(manifest.round_id)
            self.assertEqual(after.state, "COMPLETED")
            self.assertEqual(after.host_adapter_after, 1)
            self.assertTrue(after.adapter_promoted)
            gateway.load_training_job.assert_awaited_once()
            gateway.train_candidate.assert_awaited_once_with(job)
            gateway.validate_candidate.assert_awaited_once_with(job)
            gateway.post_decision_knowledge.assert_awaited_once()
            persisted = HostCandidateTrainingResult.model_validate(
                service.store.read_json(
                    f"rounds/{manifest.round_id}/host_training_job/"
                    "candidate_result.json"
                )
            )
            self.assertEqual(persisted, result)
            persisted_decision = HostCandidateValidationResult.model_validate(
                service.store.read_json(
                    f"rounds/{manifest.round_id}/host_training_job/"
                    "validation_decision.json"
                )
            )
            self.assertEqual(persisted_decision, decision)
            package, package_path = service.get_host_knowledge(
                manifest.round_id
            )
            self.assertEqual(package.adapter_version, 1)
            self.assertTrue(package_path.is_file())

    async def test_coordinator_finalizes_forced_candidate_rejection(self) -> None:
        manifest, _ = host_round_bundle(
            round_id="coordinator-forced-rejection",
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1"),
        )
        backend = FakeHostPeftBackend()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "prepared.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            candidate = backend.train_candidate(job, artifact)
            decision = HostCandidateValidationResult.create(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                job_hash=job.job_hash,
                candidate_result_hash=candidate.result_hash,
                previous_adapter_version=0,
                previous_adapter_hash="a" * 64,
                candidate_adapter_version=1,
                candidate_adapter_hash="c" * 64,
                accepted_adapter_version=0,
                accepted_adapter_hash="a" * 64,
                baseline_validation_record_hash="b" * 64,
                candidate_validation_record_hash="d" * 64,
                previous_macro_mean_answer_token_ce=0.5,
                candidate_macro_mean_answer_token_ce=0.25,
                previous_token_weighted_answer_token_ce=0.55,
                candidate_token_weighted_answer_token_ce=0.3,
                required_improvement=0.001,
                observed_improvement=0.25,
                adapter_promoted=False,
                decision_reason="forced_validation_rejection",
                rejected_candidate_discarded=True,
                created_at=utc_text(),
            )
            host_identity = Ed25519Identity(Ed25519PrivateKey.generate())
            gateway = mock.Mock()
            gateway.validate_candidate = mock.AsyncMock(
                return_value=decision
            )

            async def post_decision_knowledge(value, artifact_path):
                return write_signed_host_package(
                    value,
                    artifact_path,
                    host_identity,
                    0,
                )

            gateway.post_decision_knowledge = mock.AsyncMock(
                side_effect=post_decision_knowledge
            )
            gateway.identity = mock.AsyncMock(
                return_value=ServiceIdentity(
                    service_id="legalfedllm-host",
                    public_key=host_identity.public_key_b64,
                    model_profile=manifest.host_model_profile,
                    adapter_version=0,
                )
            )
            service = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=gateway,
            )
            state = RoundState(
                round_id=manifest.round_id,
                state="DISTILLING",
                accepted_client_ids=["client-a"],
                sealed_client_ids=["client-a"],
                host_adapter_before=0,
                updated_at=utc_text(),
            )
            service.store.write_json(
                f"rounds/{manifest.round_id}/manifest.json",
                manifest.model_dump(mode="json"),
            )

            await service._complete_real_round(  # noqa: SLF001
                manifest,
                state,
                job,
                candidate,
            )

            self.assertEqual(state.state, "COMPLETED")
            self.assertFalse(state.adapter_promoted)
            self.assertEqual(state.host_adapter_after, 0)
            package, path = service.get_host_knowledge(manifest.round_id)
            self.assertEqual(package.adapter_version, 0)
            self.assertTrue(path.is_file())
            events = [
                json.loads(line)
                for line in service.store.path("audit/events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            validation_event = next(
                event
                for event in events
                if event["event"] == "host_candidate_validated"
            )
            self.assertEqual(
                validation_event["decision_reason"],
                "forced_validation_rejection",
            )
            self.assertTrue(
                validation_event["rejected_candidate_discarded"]
            )

    async def test_training_job_and_candidate_are_immutable_without_promotion(
        self,
    ) -> None:
        backend = FakeHostPeftBackend()
        manifest, bundle = host_round_bundle(
            alignment=AlignmentConfig(
                strategy="dtw",
                profile_version="test-v1",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=backend.execution_profile,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            artifact_path = Path(directory) / "trainer-inputs.safetensors"
            job = write_test_host_training_job(manifest, artifact_path)
            adapter_before = runtime.active_adapter()
            app = create_host_app(
                runtime,
                internal_token_override="internal-test-token",
            )
            transport = httpx.ASGITransport(app=app)
            gateway = HostGateway(
                "http://host",
                "internal-test-token",
                transport=transport,
            )

            receipt = await gateway.load_training_job(job, artifact_path)
            self.assertEqual(receipt.job_hash, job.job_hash)

            retry = await gateway.load_training_job(job, artifact_path)
            self.assertEqual(retry, receipt)

            candidate = await gateway.train_candidate(job)
            self.assertEqual(candidate.parent_adapter_version, 0)
            self.assertEqual(candidate.candidate_adapter_version, 1)
            self.assertEqual(candidate.optimizer_step_count, 1)
            self.assertTrue(candidate.lora_tensors_changed)
            self.assertTrue(candidate.frozen_base_unchanged)
            self.assertTrue(candidate.reload_verified)
            self.assertEqual(runtime.active_adapter(), adapter_before)
            self.assertEqual(backend.train_calls, 1)

            candidate_retry = await gateway.train_candidate(job)
            self.assertEqual(candidate_retry, candidate)
            self.assertEqual(backend.train_calls, 1)

            changed = write_test_host_training_job(
                manifest,
                artifact_path,
                integration_audit_hash="f" * 64,
            )
            with self.assertRaisesRegex(ConflictError, "returned 409"):
                await gateway.load_training_job(changed, artifact_path)

            self.assertEqual(runtime.active_adapter(), adapter_before)
            root = runtime.store.path(
                f"rounds/{manifest.round_id}/training_job"
            )
            self.assertTrue((root / "job.json").is_file())
            self.assertTrue((root / "trainer_inputs.safetensors").is_file())
            self.assertTrue(
                runtime.store.path(
                    f"rounds/{manifest.round_id}/candidate/result.json"
                ).is_file()
            )

            decision = await gateway.validate_candidate(job)
            self.assertTrue(decision.adapter_promoted)
            self.assertEqual(runtime.adapter_version, 1)
            incoming = Path(directory) / "post-decision.safetensors"
            package = await gateway.post_decision_knowledge(
                manifest,
                incoming,
            )
            self.assertEqual(package.adapter_version, 1)
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )
            self.assertTrue(incoming.is_file())

    async def test_inference_runs_off_loop_and_rejects_a_concurrent_job(self) -> None:
        backend = FakeHostPeftBackend()
        manifest, bundle = host_round_bundle()
        started = threading.Event()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=pinned_host_profile(),
                training_execution_profile=backend.execution_profile,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            original = runtime.generate_reference_knowledge

            def blocking_inference(value):
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test did not release inference")
                return original(value)

            app = create_host_app(
                runtime,
                internal_token_override="internal-test-token",
            )
            transport = httpx.ASGITransport(app=app)
            headers = {"X-Internal-Token": "internal-test-token"}
            with mock.patch.object(
                runtime,
                "generate_reference_knowledge",
                blocking_inference,
            ):
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://host",
                ) as client:
                    pending = asyncio.create_task(
                        client.post(
                            "/internal/v1/reference-knowledge",
                            headers=headers,
                            json=manifest.model_dump(mode="json"),
                        )
                    )
                    self.assertTrue(
                        await asyncio.to_thread(started.wait, 2)
                    )
                    health = await client.get("/health")
                    self.assertEqual(health.status_code, 200, health.text)
                    conflict = await client.post(
                        "/internal/v1/reference-knowledge",
                        headers=headers,
                        json=manifest.model_dump(mode="json"),
                    )
                    self.assertEqual(conflict.status_code, 409, conflict.text)
                    release.set()
                    response = await pending
                    self.assertEqual(response.status_code, 200, response.text)


@unittest.skipUnless(
    os.getenv("LEGALFEDLLM_RUN_REAL_HOST_TESTS", "").lower()
    in {"1", "true", "yes"},
    "set LEGALFEDLLM_RUN_REAL_HOST_TESTS=true for the pinned Granite Host",
)
class RealHostAdapterAcceptanceTests(unittest.TestCase):
    def test_train_validate_decide_publish_and_restart(self) -> None:
        profile = pinned_host_profile()
        execution = host_execution_profile_from_environment()
        smoke_sample_count = (
            2
            * execution.micro_batch_size
            * execution.gradient_accumulation_steps
        )
        smoke_reference = [
            reference_sample(f"dp-decision-{index:04d}")
            for index in range(smoke_sample_count)
        ]
        manifest, bundle = host_round_bundle(
            reference=smoke_reference,
            round_id="real-host-candidate-decision",
            host_public_data_epochs=execution.public_data_epochs,
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            artifact = Path(directory) / "candidate-decision.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            runtime.load_training_job(job, artifact)
            runtime.train_candidate(job)

            decision = runtime.validate_candidate_and_decide(job)

            expected_promotion = (
                decision.observed_improvement
                >= decision.required_improvement
            )
            self.assertEqual(decision.adapter_promoted, expected_promotion)
            self.assertEqual(
                decision.accepted_adapter_version,
                1 if expected_promotion else 0,
            )
            package = runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(
                package.adapter_version,
                decision.accepted_adapter_version,
            )
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )

            restarted_backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            restarted_runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=restarted_backend,
            )
            self.assertEqual(
                restarted_runtime.adapter_version,
                decision.accepted_adapter_version,
            )
            cached = restarted_runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(cached.package_hash, package.package_hash)

    def test_train_validate_force_rejection_publish_and_restart(self) -> None:
        profile = pinned_host_profile()
        execution = host_execution_profile_from_environment()
        smoke_sample_count = (
            2
            * execution.micro_batch_size
            * execution.gradient_accumulation_steps
        )
        smoke_reference = [
            reference_sample(f"dp-smoke-{index:04d}")
            for index in range(smoke_sample_count)
        ]
        manifest, bundle = host_round_bundle(
            reference=smoke_reference,
            round_id="real-host-candidate-smoke",
            host_public_data_epochs=execution.public_data_epochs,
            alignment=AlignmentConfig(strategy="dtw", profile_version="test-v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=backend,
                force_validation_failure=True,
            )
            runtime.load_reference_data(bundle)
            runtime.generate_reference_knowledge(manifest)
            artifact = Path(directory) / "candidate-smoke.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            runtime.load_training_job(job, artifact)
            active_before = runtime.active_adapter()

            result = runtime.train_candidate(job)

            self.assertGreaterEqual(result.optimizer_step_count, 2)
            self.assertTrue(result.lora_tensors_changed)
            self.assertTrue(result.frozen_base_unchanged)
            self.assertTrue(result.reload_verified)
            self.assertEqual(runtime.active_adapter(), active_before)
            metadata, candidate_path = backend.checkpoints.candidate(
                manifest.round_id,
                result.candidate_adapter_version,
            )
            self.assertEqual(metadata.checkpoint_hash, result.candidate_adapter_hash)
            self.assertTrue(
                (candidate_path / "adapter_model.safetensors").is_file()
            )

            decision = runtime.validate_candidate_and_decide(job)
            self.assertFalse(decision.adapter_promoted)
            self.assertEqual(
                decision.decision_reason,
                "forced_validation_rejection",
            )
            self.assertTrue(decision.rejected_candidate_discarded)
            self.assertEqual(runtime.active_adapter(), active_before)
            self.assertFalse(candidate_path.exists())
            package = runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(package.adapter_version, 0)
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )

            restarted_backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            restarted_runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=restarted_backend,
                force_validation_failure=True,
            )
            self.assertEqual(restarted_runtime.adapter_version, 0)
            self.assertEqual(
                restarted_runtime.candidate_validation_decision(
                    manifest
                ),
                decision,
            )
            cached = restarted_runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(cached.package_hash, package.package_hash)

    def test_initialize_infer_validate_package_and_restart(self) -> None:
        profile = pinned_host_profile()
        execution = host_execution_profile_from_environment()
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            initialized = backend.initialize_adapter()

            self.assertEqual(initialized.metadata.version, 0)
            self.assertIsNone(initialized.metadata.parent_version)
            self.assertIsNone(initialized.metadata.round_id)
            self.assertGreaterEqual(initialized.metadata.file_count, 3)
            self.assertTrue(
                (initialized.checkpoint_path / "adapter_config.json").is_file()
            )
            self.assertTrue(
                (
                    initialized.checkpoint_path
                    / "adapter_model.safetensors"
                ).is_file()
            )
            self.assertTrue(
                (initialized.checkpoint_path / HOST_INITIALIZATION_RECORD).is_file()
            )
            self.assertTrue(initialized.initialization.zero_effect_verified)
            self.assertTrue(initialized.initialization.reload_verified)

            manifest, bundle = host_round_bundle()
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            package = runtime.generate_reference_knowledge(manifest)
            validation = runtime.validation_baseline(manifest)
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertEqual(package.top_k, manifest.top_k)
            self.assertTrue(package.verify_signature(runtime.identity.public_key_b64))
            self.assertEqual(validation.sample_count, 2)
            self.assertGreater(validation.supervised_answer_token_count, 0)
            self.assertGreaterEqual(validation.macro_mean_answer_token_ce, 0)
            self.assertGreaterEqual(
                validation.token_weighted_answer_token_ce,
                0,
            )

            restarted_backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            restarted = restarted_backend.initialize_adapter()
            self.assertEqual(
                restarted.metadata.checkpoint_hash,
                initialized.metadata.checkpoint_hash,
            )
            self.assertEqual(
                restarted.initialization.record_hash,
                initialized.initialization.record_hash,
            )
            restarted_runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=restarted_backend,
            )
            cached = restarted_runtime.generate_reference_knowledge(manifest)
            self.assertEqual(cached.package_hash, package.package_hash)
            self.assertEqual(
                restarted_runtime.validation_baseline(manifest).record_hash,
                validation.record_hash,
            )

    @unittest.skipUnless(
        os.getenv("LEGALFEDLLM_REAL_REFERENCE_DATASET_PATH")
        and os.getenv("LEGALFEDLLM_REAL_VALIDATION_DATASET_PATH"),
        "set both real Host D^P and D^V dataset paths for full-corpus acceptance",
    )
    def test_full_frozen_reference_and_validation_baseline(self) -> None:
        reference = load_reference_jsonl(
            os.environ["LEGALFEDLLM_REAL_REFERENCE_DATASET_PATH"]
        )
        validation_samples = load_reference_jsonl(
            os.environ["LEGALFEDLLM_REAL_VALIDATION_DATASET_PATH"]
        )
        reference_identity = reference_dataset_identity(reference)
        validation_identity = reference_dataset_identity(validation_samples)
        self.assertEqual(
            reference_identity.sample_count,
            FROZEN_REFERENCE_SAMPLE_COUNT,
        )
        self.assertEqual(
            reference_identity.dataset_hash,
            FROZEN_REFERENCE_DATASET_HASH,
        )
        self.assertEqual(
            validation_identity.sample_count,
            FROZEN_VALIDATION_SAMPLE_COUNT,
        )
        self.assertEqual(
            validation_identity.dataset_hash,
            FROZEN_VALIDATION_DATASET_HASH,
        )

        maximum_sequence_length = int(
            os.getenv("LEGALFEDLLM_REAL_HOST_MAX_SEQUENCE_LENGTH", "4096")
        )
        manifest, bundle = host_round_bundle(
            reference=reference,
            validation=validation_samples,
            round_id="host-full-corpus-baseline",
            maximum_sequence_length=maximum_sequence_length,
            top_k=4,
            maximum_knowledge_package_bytes=25 * 1024 * 1024,
        )
        profile = pinned_host_profile()
        execution = host_execution_profile_from_environment()
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=backend,
            )
            runtime.load_reference_data(bundle)
            package = runtime.generate_reference_knowledge(manifest)
            validation = runtime.validation_baseline(manifest)

            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertEqual(
                package.artifact.sample_count,
                FROZEN_REFERENCE_SAMPLE_COUNT,
            )
            self.assertEqual(package.top_k, 4)
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )
            self.assertEqual(
                validation.sample_count,
                FROZEN_VALIDATION_SAMPLE_COUNT,
            )
            self.assertEqual(
                validation.validation_dataset.dataset_hash,
                FROZEN_VALIDATION_DATASET_HASH,
            )
            self.assertGreater(validation.supervised_answer_token_count, 0)

            cached = runtime.generate_reference_knowledge(manifest)
            self.assertEqual(cached.package_hash, package.package_hash)
            self.assertEqual(
                runtime.validation_baseline(manifest).record_hash,
                validation.record_hash,
            )


@unittest.skipUnless(
    os.getenv("LEGALFEDLLM_RUN_REAL_NEMO_HOST_TESTS", "").lower()
    in {"1", "true", "yes"},
    "set LEGALFEDLLM_RUN_REAL_NEMO_HOST_TESTS=true on the Host",
)
class RealMistralNemoHostAcceptanceTests(unittest.TestCase):
    def test_initialize_infer_train_reject_and_restart(self) -> None:
        profile = pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID)
        execution = host_execution_profile_from_environment()
        self.assertEqual(execution.device, "cuda")
        self.assertEqual(execution.precision, "bfloat16")
        self.assertEqual(execution.public_data_epochs, 1)
        self.assertEqual(execution.micro_batch_size, 1)
        self.assertEqual(execution.gradient_accumulation_steps, 1)
        self.assertEqual(execution.learning_rate, 3e-5)
        self.assertEqual(execution.warmup_ratio, 0.0)
        self.assertTrue(execution.gradient_checkpointing)

        manifest, bundle = host_round_bundle(
            reference=[reference_sample("nemo-dp-smoke-0001")],
            round_id="mistral-nemo-real-host-smoke",
            maximum_sequence_length=128,
            host_public_data_epochs=1,
            alignment=AlignmentConfig(
                strategy="dtw",
                profile_version=MISTRAL_NEMO_DTW_PROFILE_VERSION,
            ),
            host_model_profile=profile,
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=backend,
                force_validation_failure=True,
            )
            self.assertTrue(
                runtime.real_adapter.initialization.zero_effect_verified
            )
            self.assertTrue(runtime.real_adapter.initialization.reload_verified)

            runtime.load_reference_data(bundle)
            package = runtime.generate_reference_knowledge(manifest)
            self.assertEqual(package.sample_ids, manifest.sample_ids)
            self.assertTrue(
                package.verify_signature(runtime.identity.public_key_b64)
            )

            artifact = Path(directory) / "nemo-smoke.safetensors"
            job = write_test_host_training_job(manifest, artifact)
            runtime.load_training_job(job, artifact)
            result = runtime.train_candidate(job)
            self.assertEqual(result.optimizer_step_count, 1)
            self.assertTrue(result.lora_tensors_changed)
            self.assertTrue(result.frozen_base_unchanged)
            self.assertTrue(result.reload_verified)

            decision = runtime.validate_candidate_and_decide(job)
            self.assertFalse(decision.adapter_promoted)
            self.assertTrue(decision.rejected_candidate_discarded)
            post_decision = runtime.generate_post_decision_reference_knowledge(
                manifest
            )
            self.assertEqual(post_decision.adapter_version, 0)

            restarted_backend = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            )
            restarted_runtime = HostRuntime(
                data_dir=directory,
                model_profile=profile,
                training_execution_profile=execution,
                peft_backend=restarted_backend,
                force_validation_failure=True,
            )
            self.assertEqual(restarted_runtime.adapter_version, 0)
            self.assertEqual(
                restarted_runtime.candidate_validation_decision(manifest),
                decision,
            )
