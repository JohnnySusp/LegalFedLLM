from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    pinned_host_profile,
)
from host.peft_backend import (
    HOST_INITIALIZATION_RECORD,
    InitializedHostAdapter,
    TransformersPeftHostBackend,
)
from host.runtime import HostRuntime, HostRuntimeError, default_host_profile
from host.training import (
    HostAdapterInitializationRecord,
    HostStageFiveContract,
    HostTrainingExecutionProfile,
    host_execution_profile_from_environment,
)
from shared.adapter_checkpoint import AdapterCheckpointMetadata
from shared.protocol import utc_text


def fake_initialized_adapter() -> InitializedHostAdapter:
    profile = pinned_host_profile()
    contract = HostStageFiveContract.create(profile)
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
        self.initialized = fake_initialized_adapter()

    def initialize_adapter(self) -> InitializedHostAdapter:
        self.calls += 1
        return self.initialized


class HostStageFiveContractTests(unittest.TestCase):
    def test_contract_freezes_the_agreed_host_decisions(self) -> None:
        profile = pinned_host_profile()
        contract = HostStageFiveContract.create(profile)

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
            HostStageFiveContract.create(changed)

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

    def test_runtime_uses_real_checkpoint_identity_but_blocks_later_steps(self) -> None:
        backend = FakeHostPeftBackend()
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
            with self.assertRaisesRegex(HostRuntimeError, "Step 5.3"):
                runtime.generate_reference_knowledge(None)  # type: ignore[arg-type]
            with self.assertRaisesRegex(HostRuntimeError, "Step 5.5"):
                runtime.distill(None)  # type: ignore[arg-type]


@unittest.skipUnless(
    os.getenv("LEGALFEDLLM_RUN_REAL_HOST_TESTS", "").lower()
    in {"1", "true", "yes"},
    "set LEGALFEDLLM_RUN_REAL_HOST_TESTS=true for the pinned Granite Host",
)
class RealHostAdapterAcceptanceTests(unittest.TestCase):
    def test_initialize_zero_effect_adapter_and_restart(self) -> None:
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

            restarted = TransformersPeftHostBackend(
                data_dir=directory,
                model_profile=profile,
                execution_profile=execution,
            ).initialize_adapter()
            self.assertEqual(
                restarted.metadata.checkpoint_hash,
                initialized.metadata.checkpoint_hash,
            )
            self.assertEqual(
                restarted.initialization.record_hash,
                initialized.initialization.record_hash,
            )
