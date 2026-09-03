from __future__ import annotations

import math
import os
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    MISTRAL_NEMO_HOST_PROFILE_ID,
    pinned_host_profile,
)
from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile, parse_utc, utc_text
from shared.reference_dataset import ReferenceDatasetIdentity


GRANITE_HOST_TRAINING_CONTRACT_ID = "granite-3.3-2b-real-host-v1"
MISTRAL_NEMO_HOST_TRAINING_CONTRACT_ID = (
    "mistral-nemo-instruct-2407-real-host-v1"
)
# Backward-compatible public name for the original Granite contract.
HOST_TRAINING_CONTRACT_ID = GRANITE_HOST_TRAINING_CONTRACT_ID
INITIAL_HOST_ADAPTER_POLICY = "fresh_zero_effect_lora_v0"
PRIMARY_HOST_VALIDATION_METRIC = "macro_mean_answer_token_ce"
SECONDARY_HOST_VALIDATION_METRIC = "token_weighted_answer_token_ce"
REJECTED_CANDIDATE_POLICY = "discard_weights_retain_audit"


class HostTrainingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PinnedHostTrainingContract(HostTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal[
        "granite-3.3-2b-real-host-v1",
        "mistral-nemo-instruct-2407-real-host-v1",
    ] = GRANITE_HOST_TRAINING_CONTRACT_ID
    host_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    initial_adapter_policy: Literal["fresh_zero_effect_lora_v0"] = (
        INITIAL_HOST_ADAPTER_POLICY
    )
    authoritative_public_data_epochs: Literal[5] = 5
    development_smoke_epochs: Literal[1] = 1
    primary_validation_metric: Literal["macro_mean_answer_token_ce"] = (
        PRIMARY_HOST_VALIDATION_METRIC
    )
    secondary_validation_metric: Literal["token_weighted_answer_token_ce"] = (
        SECONDARY_HOST_VALIDATION_METRIC
    )
    minimum_validation_improvement: Literal[0.001] = 0.001
    rejected_candidate_policy: Literal["discard_weights_retain_audit"] = (
        REJECTED_CANDIDATE_POLICY
    )
    contract_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_contract_hash(self) -> "PinnedHostTrainingContract":
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"contract_hash"})
        )
        if self.contract_hash != expected:
            raise ValueError("contract_hash does not match the Host contract")
        return self

    @classmethod
    def create(
        cls,
        model_profile: ModelProfile,
    ) -> "PinnedHostTrainingContract":
        expected = pinned_host_profile(
            model_profile.profile_id,
            serving_backend=model_profile.serving_backend,
        )
        if model_profile.profile_hash() != expected.profile_hash():
            raise ValueError(
                "real Host training requires an exact pinned Host profile"
            )
        contract_ids = {
            GRANITE_3_3_2B_HOST_PROFILE_ID: (
                GRANITE_HOST_TRAINING_CONTRACT_ID
            ),
            MISTRAL_NEMO_HOST_PROFILE_ID: (
                MISTRAL_NEMO_HOST_TRAINING_CONTRACT_ID
            ),
        }
        payload = {
            "schema_version": "1.0",
            "contract_id": contract_ids[model_profile.profile_id],
            "host_model_profile_hash": model_profile.profile_hash(),
            "initial_adapter_policy": INITIAL_HOST_ADAPTER_POLICY,
            "authoritative_public_data_epochs": 5,
            "development_smoke_epochs": 1,
            "primary_validation_metric": PRIMARY_HOST_VALIDATION_METRIC,
            "secondary_validation_metric": SECONDARY_HOST_VALIDATION_METRIC,
            "minimum_validation_improvement": 0.001,
            "rejected_candidate_policy": REJECTED_CANDIDATE_POLICY,
        }
        return cls(**payload, contract_hash=sha256_hex(payload))


# Keep imports and serialized Granite records from earlier milestones valid.
GraniteHostTrainingContract = PinnedHostTrainingContract


class HostTrainingExecutionProfile(HostTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["transformers"] = "transformers"
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantization: Literal["none"] = "none"
    public_data_epochs: Literal[1, 5] = 5
    micro_batch_size: int = Field(default=1, ge=1, le=1024)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=65536)
    optimizer: Literal["adamw_torch"] = "adamw_torch"
    learning_rate_scheduler: Literal["cosine"] = "cosine"
    learning_rate: float = Field(default=3e-5, gt=0)
    warmup_ratio: float = Field(default=0.008, ge=0, lt=1)
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.95, gt=0, lt=1)
    weight_decay: float = Field(default=0.1, ge=0)
    maximum_gradient_norm: float = Field(default=1.0, gt=0)
    dataloader_num_workers: int = Field(default=4, ge=0, le=256)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    gradient_checkpointing: bool = False
    verify_frozen_base_checksum: bool = True

    @model_validator(mode="after")
    def validate_device_precision(self) -> "HostTrainingExecutionProfile":
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU Host training requires float32 precision")
        if self.device == "cuda" and self.precision == "float32":
            raise ValueError(
                "CUDA Host training must explicitly use bfloat16 or float16"
            )
        return self

    def profile_hash(self) -> str:
        return sha256_hex(self.model_dump(mode="json"))


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def host_execution_profile_from_environment() -> HostTrainingExecutionProfile:
    return HostTrainingExecutionProfile(
        device=os.getenv("HOST_TRAINING_DEVICE", "cuda").strip().lower(),
        precision=os.getenv(
            "HOST_TRAINING_PRECISION", "bfloat16"
        ).strip().lower(),
        public_data_epochs=int(os.getenv("HOST_PUBLIC_DATA_EPOCHS", "5")),
        micro_batch_size=int(os.getenv("HOST_MICRO_BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(
            os.getenv("HOST_GRADIENT_ACCUMULATION_STEPS", "4")
        ),
        learning_rate=float(os.getenv("HOST_LEARNING_RATE", "3e-5")),
        warmup_ratio=float(os.getenv("HOST_WARMUP_RATIO", "0.008")),
        dataloader_num_workers=int(
            os.getenv("HOST_DATALOADER_NUM_WORKERS", "4")
        ),
        seed=int(os.getenv("HOST_TRAINING_SEED", "42")),
        gradient_checkpointing=_environment_flag(
            "HOST_GRADIENT_CHECKPOINTING", False
        ),
        verify_frozen_base_checksum=_environment_flag(
            "HOST_VERIFY_FROZEN_BASE_CHECKSUM", True
        ),
    )


class HostAdapterInitializationRecord(HostTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    contract: PinnedHostTrainingContract
    contract_hash: str = Field(pattern=HASH_PATTERN)
    model_profile_hash: str = Field(pattern=HASH_PATTERN)
    execution_profile: HostTrainingExecutionProfile
    execution_profile_hash: str = Field(pattern=HASH_PATTERN)
    adapter_version: Literal[0] = 0
    initialization_policy: Literal["fresh_zero_effect_lora_v0"] = (
        INITIAL_HOST_ADAPTER_POLICY
    )
    zero_effect_verified: Literal[True] = True
    reload_verified: Literal[True] = True
    trainable_parameter_count: int = Field(gt=0)
    total_parameter_count: int = Field(gt=0)
    dependency_versions: dict[str, str]
    created_at: str
    record_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_record(self) -> "HostAdapterInitializationRecord":
        if self.contract_hash != self.contract.contract_hash:
            raise ValueError("Host initialization contract hash differs")
        if self.execution_profile_hash != self.execution_profile.profile_hash():
            raise ValueError("Host execution profile hash differs")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("record_hash does not match Host initialization")
        return self

    @classmethod
    def create(cls, **values: Any) -> "HostAdapterInitializationRecord":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"record_hash"}
        )
        return cls(**payload, record_hash=sha256_hex(payload))


class HostValidationSampleMetric(HostTrainingContract):
    sample_id: str = Field(min_length=1, max_length=256)
    answer_token_count: int = Field(ge=1)
    answer_token_ce: float = Field(ge=0, allow_inf_nan=False)


class HostValidationRecord(HostTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    validation_dataset: ReferenceDatasetIdentity
    validation_sample_ids_hash: str = Field(pattern=HASH_PATTERN)
    host_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    adapter_version: int = Field(ge=0)
    checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    contract_hash: str = Field(pattern=HASH_PATTERN)
    execution_profile_hash: str = Field(pattern=HASH_PATTERN)
    primary_metric: Literal["macro_mean_answer_token_ce"] = (
        PRIMARY_HOST_VALIDATION_METRIC
    )
    secondary_metric: Literal["token_weighted_answer_token_ce"] = (
        SECONDARY_HOST_VALIDATION_METRIC
    )
    sample_count: int = Field(ge=1)
    supervised_answer_token_count: int = Field(ge=1)
    macro_mean_answer_token_ce: float = Field(ge=0, allow_inf_nan=False)
    token_weighted_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    samples: list[HostValidationSampleMetric] = Field(min_length=1)
    created_at: str
    record_hash: str = Field(pattern=HASH_PATTERN)

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        parse_utc(value)
        return value

    @model_validator(mode="after")
    def validate_record(self) -> "HostValidationRecord":
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Host validation sample IDs must be unique")
        if self.validation_sample_ids_hash != sha256_hex(sample_ids):
            raise ValueError("Host validation sample order hash differs")
        if self.sample_count != len(self.samples):
            raise ValueError("Host validation sample count differs")
        if self.validation_dataset.sample_count != self.sample_count:
            raise ValueError("Host validation dataset count differs")

        token_count = sum(
            sample.answer_token_count for sample in self.samples
        )
        if self.supervised_answer_token_count != token_count:
            raise ValueError("Host validation answer-token count differs")

        macro = math.fsum(
            sample.answer_token_ce for sample in self.samples
        ) / self.sample_count
        weighted = math.fsum(
            sample.answer_token_ce * sample.answer_token_count
            for sample in self.samples
        ) / token_count
        if not math.isclose(
            self.macro_mean_answer_token_ce,
            macro,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Host primary validation metric differs")
        if not math.isclose(
            self.token_weighted_answer_token_ce,
            weighted,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Host secondary validation metric differs")

        expected_hash = sha256_hex(
            self.model_dump(mode="json", exclude={"record_hash"})
        )
        if self.record_hash != expected_hash:
            raise ValueError("Host validation record hash differs")
        return self

    @classmethod
    def create(
        cls,
        *,
        round_id: str,
        manifest_hash: str,
        validation_dataset: ReferenceDatasetIdentity,
        host_model_profile_hash: str,
        adapter_version: int,
        checkpoint_hash: str,
        contract_hash: str,
        execution_profile_hash: str,
        samples: list[HostValidationSampleMetric],
        created_at: str | None = None,
    ) -> "HostValidationRecord":
        sample_ids = [sample.sample_id for sample in samples]
        sample_count = len(samples)
        token_count = sum(sample.answer_token_count for sample in samples)
        if sample_count < 1 or token_count < 1:
            raise ValueError("Host validation metrics must not be empty")
        macro = math.fsum(
            sample.answer_token_ce for sample in samples
        ) / sample_count
        weighted = math.fsum(
            sample.answer_token_ce * sample.answer_token_count
            for sample in samples
        ) / token_count
        payload = {
            "schema_version": "1.0",
            "round_id": round_id,
            "manifest_hash": manifest_hash,
            "validation_dataset": validation_dataset.model_dump(mode="json"),
            "validation_sample_ids_hash": sha256_hex(sample_ids),
            "host_model_profile_hash": host_model_profile_hash,
            "adapter_version": adapter_version,
            "checkpoint_hash": checkpoint_hash,
            "contract_hash": contract_hash,
            "execution_profile_hash": execution_profile_hash,
            "primary_metric": PRIMARY_HOST_VALIDATION_METRIC,
            "secondary_metric": SECONDARY_HOST_VALIDATION_METRIC,
            "sample_count": sample_count,
            "supervised_answer_token_count": token_count,
            "macro_mean_answer_token_ce": macro,
            "token_weighted_answer_token_ce": weighted,
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "created_at": created_at or utc_text(),
        }
        return cls(**payload, record_hash=sha256_hex(payload))
