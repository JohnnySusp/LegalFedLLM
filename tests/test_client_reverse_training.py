from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.reverse_training import (
    ClientReverseCandidateResult,
    ClientValidationRecord,
    ClientValidationSampleMetric,
    selective_client_loss,
)
from client.runtime import ClientRuntime
from client.safety_probe import (
    ClientSafetyProbeReport,
    SafeFedLoraProbe,
    SafeFedProbeManifest,
)
from client.training import TrainingExecutionProfile
from shared.crypto import Ed25519Identity, sha256_hex
from shared.protocol import (
    ClientReverseTrainingArtifactDescriptor,
    ClientReverseTrainingJob,
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceSample,
    client_public_data_partition,
    reference_dataset_identity,
    write_reference_jsonl,
)


def host_profile() -> ModelProfile:
    return ModelProfile(
        profile_id="reverse-test-host",
        role="host",
        model_id="legalfedllm/reverse-test-host",
        model_revision="v1",
        tokenizer_id="legalfedllm/reverse-test-tokenizer",
        tokenizer_revision="v1",
        tokenizer_class="MockTokenizer",
        training_backend="mock",
        serving_backend="mock",
        prompt_template_hash=sha256_hex(b"reverse-test"),
        lora=LoraProfile(rank=8),
    )


def reverse_fixture(
    root: Path,
    *,
    host_teachers: bool = True,
    force_rejection: bool = False,
    maliciousness_probability: float = 0.1,
    advance_on_candidate_validation: bool = False,
    sample_count: int = 2,
):
    profile = pinned_client_profile(QWEN_PROFILE_ID)
    execution = TrainingExecutionProfile(
        backend="transformers",
        device="cuda",
        precision="bfloat16",
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )
    samples = [
        ReferenceSample(
            schema_version=1,
            dataset_id="reverse-reference",
            dataset_version="v1",
            sample_id=f"sample-{index}",
            chapter="Contracts",
            section=str(index),
            question=f"Question {index}?",
            gold_answer=f"Answer {index}.",
        )
        for index in range(sample_count)
    ]
    identity = Ed25519Identity.load_or_create(root / "coordinator.pem")
    dataset_identity = reference_dataset_identity(samples)
    request = RoundCreateRequest(
        selected_client_ids=["client-a"],
        trusted_client_quorum=1,
        reference_dataset_id=dataset_identity.dataset_id,
        reference_dataset_hash=dataset_identity.dataset_hash,
        sample_ids=[sample.sample_id for sample in samples],
        prompt_template="{question}\n{answer}",
        top_k=2,
        maximum_sequence_length=64,
    )
    manifest = RoundManifest.create_signed(
        identity=identity,
        round_id="reverse-round",
        coordinator_id="coordinator",
        current_host_adapter_version=0,
        host_model_profile=host_profile(),
        selected_client_profile_hashes={"client-a": profile.profile_hash()},
        request=request,
        submission_deadline="2026-08-26T00:00:00Z",
    )
    partition = client_public_data_partition(
        reference_dataset_id=manifest.reference_dataset_id,
        reference_dataset_hash=manifest.reference_dataset_hash,
        sample_ids=manifest.sample_ids,
    )

    backend = FakeReverseBackend(
        profile=profile,
        execution=execution,
        advance_on_candidate_validation=advance_on_candidate_validation,
    )
    probe = FakeSafetyProbe(profile, maliciousness_probability)
    runtime = ClientRuntime(
        data_dir=root / "client",
        client_id="client-a",
        model_profile=profile,
        training_execution_profile=execution,
        reverse_backend=backend,
        safety_probe=probe,
        force_reverse_validation_failure=force_rejection,
    )
    backend.runtime = runtime
    staging = runtime.adapter_store.staging_path("parent")
    FakeReverseBackend.write_adapter(staging, b"parent")
    parent = runtime.adapter_store.seal(
        staging,
        version=1,
        parent=None,
        round_id=manifest.round_id,
        manifest_hash=manifest.manifest_hash,
        execution_profile_hash=execution.profile_hash(),
    )
    runtime.adapter_store.promote(staging, parent)
    state = runtime.state()
    state.update(
        {
            "candidate_adapter_version": parent.version,
            "training_adapter_version": parent.version,
            "training_checkpoint_hash": parent.checkpoint_hash,
        }
    )
    runtime.store.write_json("state.json", state)
    reference_path = root / "reference.jsonl"
    write_reference_jsonl(reference_path, samples)
    runtime.cache_reference_dataset(
        manifest=manifest,
        content=reference_path.read_bytes(),
    )
    artifact = ClientReverseTrainingArtifactDescriptor(
        byte_size=16,
        sha256="a" * 64,
        sample_count=len(partition.transfer_sample_ids),
        sample_ids_sha256=partition.transfer_sample_ids_sha256,
        padded_sequence_length=8,
        top_k=manifest.top_k,
        pad_token_id=0,
        trainer_inputs_sha256="b" * 64,
    )
    job = ClientReverseTrainingJob.create(
        manifest=manifest,
        client_id="client-a",
        client_model_profile_hash=profile.profile_hash(),
        parent_adapter_version=parent.version,
        parent_adapter_hash=parent.checkpoint_hash,
        accepted_host_adapter_version=1,
        host_package_hash="c" * 64,
        host_adapter_promoted=False,
        public_data_partition=partition,
        host_teacher_sample_ids=(
            [partition.transfer_sample_ids[0]] if host_teachers else []
        ),
        client_public_data_epochs=1,
        distillation=manifest.distillation,
        integration_audit_hash="d" * 64,
        artifact=artifact,
        created_at=utc_text(),
    )
    return runtime, backend, probe, job


class FakeSafetyProbe:
    def __init__(self, profile: ModelProfile, probability: float):
        self.probability = probability
        self.manifest = SimpleNamespace(
            artifact_hash=sha256_hex(
                {"probe": "test", "profile": profile.profile_hash()}
            )
        )

    def evaluate(self, **values):
        return ClientSafetyProbeReport.create(
            round_id=values["round_id"],
            manifest_hash=values["manifest_hash"],
            job_hash=values["job_hash"],
            probe_id="test-qwen-probe",
            probe_version="v1",
            probe_artifact_hash=self.manifest.artifact_hash,
            model_profile_hash=pinned_client_profile(
                QWEN_PROFILE_ID
            ).profile_hash(),
            parent_checkpoint_hash=values["parent_checkpoint_hash"],
            candidate_checkpoint_hash=values["candidate_checkpoint_hash"],
            delta_sha256="e" * 64,
            maliciousness_probability=self.probability,
            harmful_threshold=0.8,
            safety_gate_passed=self.probability < 0.8,
            created_at=values["created_at"],
        )


class FakeReverseBackend:
    def __init__(
        self,
        *,
        profile: ModelProfile,
        execution: TrainingExecutionProfile,
        advance_on_candidate_validation: bool,
    ):
        self.profile = profile
        self.execution = execution
        self.advance_on_candidate_validation = advance_on_candidate_validation
        self.runtime = None
        self.train_calls = 0
        self.validation_calls = 0
        self.promote_calls = 0
        self.discard_calls = 0
        self.advanced = False

    @staticmethod
    def write_adapter(path: Path, marker: bytes) -> None:
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA"}),
            encoding="utf-8",
        )
        (path / "adapter_model.safetensors").write_bytes(marker)

    def train_candidate(self, job, artifact_path):
        del artifact_path
        self.train_calls += 1
        staging = self.runtime.adapter_store.staging_path("reverse-candidate")
        self.write_adapter(staging, b"candidate")
        parent, _ = self.runtime.adapter_store.version(job.parent_adapter_version)
        candidate = self.runtime.adapter_store.seal(
            staging,
            version=parent.version + 1,
            parent=parent,
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            execution_profile_hash=self.execution.profile_hash(),
        )
        self.runtime.adapter_store.store_candidate(staging, candidate)
        return ClientReverseCandidateResult.create(
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            parent_adapter_version=parent.version,
            parent_adapter_hash=parent.checkpoint_hash,
            candidate_adapter_version=candidate.version,
            candidate_adapter_hash=candidate.checkpoint_hash,
            client_model_profile_hash=self.profile.profile_hash(),
            execution_profile_hash=self.execution.profile_hash(),
            client_public_data_epochs=1,
            optimizer_step_count=1,
            optimizer_loss=0.5,
            supervised_answer_loss=0.5,
            distillation_answer_loss=0.5,
            trainable_parameter_count=8,
            total_parameter_count=100,
            dependency_versions={"backend": "fake"},
            created_at=utc_text(),
        )

    def validate_candidate(self, result):
        return self.runtime.adapter_store.candidate(
            result.round_id,
            result.candidate_adapter_version,
        )[1]

    def validation_record(
        self,
        *,
        job,
        result,
        validation_samples,
        adapter_role,
    ):
        self.validation_calls += 1
        if adapter_role == "parent":
            version = result.parent_adapter_version
            checkpoint_hash = result.parent_adapter_hash
            ce = 1.0
        else:
            version = result.candidate_adapter_version
            checkpoint_hash = result.candidate_adapter_hash
            ce = 1.0005
        record = ClientValidationRecord.create(
            backend="transformers",
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            partition_hash=job.public_data_partition.partition_hash,
            client_model_profile_hash=self.profile.profile_hash(),
            adapter_role=adapter_role,
            adapter_version=version,
            checkpoint_hash=checkpoint_hash,
            samples=[
                ClientValidationSampleMetric(
                    sample_id=sample.sample_id,
                    answer_token_count=2,
                    answer_token_ce=ce,
                    teacher_forced_exact_match=0.0,
                    teacher_forced_rouge_l=0.5,
                )
                for sample in validation_samples
            ],
            created_at=utc_text(),
        )
        if (
            adapter_role == "candidate"
            and self.advance_on_candidate_validation
            and not self.advanced
        ):
            self.advanced = True
            staging = self.runtime.adapter_store.staging_path("newer-parent")
            self.write_adapter(staging, b"newer-unrelated")
            parent, _ = self.runtime.adapter_store.current()
            newer = self.runtime.adapter_store.seal(
                staging,
                version=parent.version + 1,
                parent=parent,
                round_id="newer-round",
                manifest_hash="f" * 64,
                execution_profile_hash=self.execution.profile_hash(),
            )
            self.runtime.adapter_store.promote(staging, newer)
        return record

    def promote_candidate(self, result):
        self.promote_calls += 1
        return self.runtime.adapter_store.promote_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )

    def discard_candidate(self, result):
        self.discard_calls += 1
        self.runtime.adapter_store.discard_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )


class ClientReverseLossTests(unittest.TestCase):
    def test_selective_loss_is_answer_only_point_nine_point_one(self) -> None:
        import torch

        logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, -1.0]]]
        )
        inputs = {
            "labels": torch.tensor([[-100, 1, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "sparse_target_token_ids": torch.tensor(
                [[[[0, 1]], [[1, 0]], [[0, 1]]]]
            ).squeeze(2),
            "sparse_target_probabilities": torch.tensor(
                [[[[0.75, 0.25]], [[0.8, 0.2]], [[0.6, 0.4]]]]
            ).squeeze(2),
            "sparse_target_valid_mask": torch.tensor(
                [[[[1, 1]], [[1, 1]], [[1, 1]]]],
                dtype=torch.bool,
            ).squeeze(2),
        }
        combined, supervised, distillation = selective_client_loss(
            torch,
            logits,
            inputs,
        )
        self.assertTrue(torch.isfinite(combined))
        self.assertTrue(
            torch.allclose(combined, 0.9 * supervised + 0.1 * distillation)
        )


class SafeFedProbeTests(unittest.TestCase):
    def test_qwen_probe_is_hash_bound_and_uses_strict_point_eight_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = pinned_client_profile(QWEN_PROFILE_ID)
            keys = [
                "base_model.model.model.layers.0.self_attn."
                f"{target}.lora_B.weight"
                for target in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]
            parent = root / "parent"
            candidate = root / "candidate"
            parent.mkdir()
            candidate.mkdir()
            save_file(
                {
                    key: np.zeros((2, 2), dtype=np.float32)
                    for key in keys
                },
                parent / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
            save_file(
                {
                    key: np.ones((2, 2), dtype=np.float32)
                    for key in keys
                },
                candidate / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
            weights_path = root / "linear_probe.safetensors"
            save_file(
                {
                    "linear_weight": np.zeros(16, dtype=np.float32),
                    "linear_bias": np.array([math.log(0.8 / 0.2)], dtype=np.float32),
                },
                weights_path,
            )
            content = weights_path.read_bytes()
            manifest = SafeFedProbeManifest.create(
                probe_id="qwen-safe-probe",
                probe_version="v1",
                model_profile_hash=profile.profile_hash(),
                model_revision=profile.model_revision,
                lora_profile_hash=sha256_hex(profile.lora.model_dump(mode="json")),
                ordered_parameter_keys=keys,
                parameter_sizes={key: 4 for key in keys},
                input_dimension=16,
                weights_file=weights_path.name,
                weights_byte_size=len(content),
                weights_sha256=sha256_hex(content),
                training_corpus_hash="1" * 64,
                validation_corpus_hash="2" * 64,
                created_at=utc_text(),
            )
            manifest_path = root / "probe.json"
            manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
            probe = SafeFedLoraProbe(
                manifest_path=manifest_path,
                model_profile=profile,
            )
            report = probe.evaluate(
                round_id="round-1",
                manifest_hash="3" * 64,
                job_hash="4" * 64,
                parent_checkpoint_hash="5" * 64,
                candidate_checkpoint_hash="6" * 64,
                parent_path=parent,
                candidate_path=candidate,
            )
            self.assertAlmostEqual(report.maliciousness_probability, 0.8, places=6)
            self.assertFalse(report.safety_gate_passed)
            tampered = bytearray(content)
            tampered[-1] ^= 1
            weights_path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                SafeFedLoraProbe(
                    manifest_path=manifest_path,
                    model_profile=profile,
                )


class ClientReverseRuntimeTests(unittest.TestCase):
    def test_natural_promotion_is_restart_and_byte_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, backend, _, job = reverse_fixture(root)
            decision = runtime._complete_reverse_distillation(job)
            self.assertTrue(decision.adapter_promoted)
            self.assertTrue(decision.quality_gate_passed)
            self.assertTrue(decision.safety_gate_passed)
            self.assertEqual(runtime.adapter_store.current()[0].version, 2)
            self.assertEqual(backend.train_calls, 1)
            self.assertEqual(backend.promote_calls, 1)
            decision_path = runtime.store.path(
                runtime._reverse_decision_path(job.manifest.round_id)
            )
            before = decision_path.read_bytes()

            restarted = ClientRuntime(
                data_dir=root / "client",
                client_id="client-a",
                model_profile=runtime.model_profile,
                training_execution_profile=runtime.training_execution_profile,
            )
            retry = restarted._complete_reverse_distillation(job)
            self.assertEqual(retry, decision)
            self.assertEqual(decision_path.read_bytes(), before)
            self.assertEqual(restarted.adapter_store.current()[0].version, 2)

    def test_restart_recovers_candidate_without_a_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, backend, _, job = reverse_fixture(Path(directory))
            parent, _ = runtime.adapter_store.current()
            staging = runtime.adapter_store.staging_path("interrupted-candidate")
            FakeReverseBackend.write_adapter(staging, b"interrupted")
            incomplete = runtime.adapter_store.seal(
                staging,
                version=parent.version + 1,
                parent=parent,
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                execution_profile_hash=(
                    runtime.training_execution_profile.profile_hash()
                ),
            )
            runtime.adapter_store.store_candidate(staging, incomplete)

            decision = runtime._complete_reverse_distillation(job)

            self.assertTrue(decision.adapter_promoted)
            self.assertEqual(backend.train_calls, 1)
            current, _ = runtime.adapter_store.current()
            self.assertEqual(current.version, parent.version + 1)

    def test_forced_rejection_discards_candidate_and_retains_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, backend, _, job = reverse_fixture(
                Path(directory),
                force_rejection=True,
            )
            decision = runtime._complete_reverse_distillation(job)
            self.assertFalse(decision.adapter_promoted)
            self.assertEqual(
                decision.decision_reason,
                "forced_validation_rejection",
            )
            self.assertTrue(decision.rejected_candidate_discarded)
            self.assertEqual(runtime.adapter_store.current()[0].version, 1)
            self.assertEqual(backend.discard_calls, 1)
            decision_path = runtime.store.path(
                runtime._reverse_decision_path(job.manifest.round_id)
            )
            before = decision_path.read_bytes()
            restarted = ClientRuntime(
                data_dir=Path(directory) / "client",
                client_id="client-a",
                model_profile=runtime.model_profile,
                training_execution_profile=runtime.training_execution_profile,
            )
            retry = restarted._complete_reverse_distillation(job)
            self.assertEqual(retry, decision)
            self.assertEqual(decision_path.read_bytes(), before)
            self.assertEqual(restarted.adapter_store.current()[0].version, 1)

    def test_safety_rejection_is_independent_of_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _, _, job = reverse_fixture(
                Path(directory),
                maliciousness_probability=0.9,
            )
            decision = runtime._complete_reverse_distillation(job)
            self.assertTrue(decision.quality_gate_passed)
            self.assertFalse(decision.safety_gate_passed)
            self.assertEqual(decision.decision_reason, "safety_gate_failed")
            self.assertFalse(decision.adapter_promoted)

    def test_no_host_teacher_is_a_successful_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, backend, _, job = reverse_fixture(
                Path(directory),
                host_teachers=False,
            )
            decision = runtime._complete_reverse_distillation(job)
            self.assertEqual(decision.decision_reason, "no_host_teacher_samples")
            self.assertFalse(decision.adapter_promoted)
            self.assertIsNone(decision.candidate_result_hash)
            self.assertEqual(backend.train_calls, 0)
            self.assertEqual(runtime.adapter_store.current()[0].version, 1)

    def test_candidate_adoption_requires_a_held_out_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, backend, _, job = reverse_fixture(
                Path(directory),
                sample_count=1,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "non-empty held-out public validation split",
            ):
                runtime._complete_reverse_distillation(job)
            self.assertEqual(backend.train_calls, 0)
            self.assertEqual(runtime.adapter_store.current()[0].version, 1)

    def test_stale_parent_discards_candidate_without_overwriting_newer_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, backend, _, job = reverse_fixture(
                Path(directory),
                advance_on_candidate_validation=True,
            )
            decision = runtime._complete_reverse_distillation(job)
            current, _ = runtime.adapter_store.current()
            self.assertTrue(decision.stale_parent)
            self.assertEqual(decision.decision_reason, "stale_parent")
            self.assertFalse(decision.adapter_promoted)
            self.assertEqual(current.round_id, "newer-round")
            self.assertEqual(
                decision.accepted_adapter_hash,
                current.checkpoint_hash,
            )
            self.assertEqual(backend.discard_calls, 1)


if __name__ == "__main__":
    unittest.main()
