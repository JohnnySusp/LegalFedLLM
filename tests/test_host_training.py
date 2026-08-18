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

from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    pinned_host_profile,
)
from host.main import create_app as create_host_app
from host.peft_backend import (
    HOST_INITIALIZATION_RECORD,
    HostBaselineResult,
    InitializedHostAdapter,
    TransformersPeftHostBackend,
)
from host.runtime import HostRuntime, HostRuntimeError, default_host_profile
from host.training import (
    HostAdapterInitializationRecord,
    HostValidationRecord,
    HostValidationSampleMetric,
    GraniteHostTrainingContract,
    HostTrainingExecutionProfile,
    host_execution_profile_from_environment,
)
from shared.adapter_checkpoint import AdapterCheckpointMetadata
from shared.crypto import Ed25519Identity
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import load_package_samples
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import (
    HostReferenceDatasetBundle,
    RoundCreateRequest,
    RoundManifest,
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
    def __init__(self) -> None:
        self.calls = 0
        self.generate_calls = 0
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
    )
    manifest = RoundManifest.create_signed(
        identity=Ed25519Identity(Ed25519PrivateKey.generate()),
        round_id=round_id,
        coordinator_id="coordinator",
        current_host_adapter_version=0,
        host_model_profile=pinned_host_profile(),
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


class GraniteHostTrainingContractTests(unittest.TestCase):
    def test_contract_freezes_the_agreed_host_decisions(self) -> None:
        profile = pinned_host_profile()
        contract = GraniteHostTrainingContract.create(profile)

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
        with self.assertRaisesRegex(ValueError, "exact pinned Granite"):
            GraniteHostTrainingContract.create(changed)

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

        with mock.patch.dict(
            os.environ,
            {"HOST_TRAINING_BACKEND": "transformers"},
            clear=False,
        ):
            os.environ.pop("HOST_MODEL_PROFILE", None)
            with self.assertRaisesRegex(HostRuntimeError, "HOST_MODEL_PROFILE"):
                default_host_profile()

    def test_environment_profile_supports_the_one_epoch_smoke_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "HOST_TRAINING_DEVICE": "cpu",
                "HOST_TRAINING_PRECISION": "float32",
                "HOST_PUBLIC_DATA_EPOCHS": "1",
                "HOST_GRADIENT_CHECKPOINTING": "true",
            },
            clear=False,
        ):
            profile = host_execution_profile_from_environment()
        self.assertEqual(profile.public_data_epochs, 1)
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


class HostMlEndpointTests(unittest.IsolatedAsyncioTestCase):
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
