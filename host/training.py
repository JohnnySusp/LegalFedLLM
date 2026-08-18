from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from host.model_profiles import pinned_host_profile
from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile


HOST_STAGE_FIVE_CONTRACT_ID = "granite-3.3-2b-real-host-v1"
INITIAL_HOST_ADAPTER_POLICY = "fresh_zero_effect_lora_v0"
PRIMARY_HOST_VALIDATION_METRIC = "macro_mean_answer_token_ce"
SECONDARY_HOST_VALIDATION_METRIC = "token_weighted_answer_token_ce"
REJECTED_CANDIDATE_POLICY = "discard_weights_retain_audit"


class HostTrainingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HostStageFiveContract(HostTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["granite-3.3-2b-real-host-v1"] = (
        HOST_STAGE_FIVE_CONTRACT_ID
    )
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
    def validate_contract_hash(self) -> "HostStageFiveContract":
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"contract_hash"})
        )
        if self.contract_hash != expected:
            raise ValueError("contract_hash does not match the Host contract")
        return self

    @classmethod
    def create(cls, model_profile: ModelProfile) -> "HostStageFiveContract":
        expected = pinned_host_profile(
            serving_backend=model_profile.serving_backend
        )
        if model_profile.profile_hash() != expected.profile_hash():
            raise ValueError(
                "Step 5 requires the exact pinned Granite Host profile"
            )
        payload = {
            "schema_version": "1.0",
            "contract_id": HOST_STAGE_FIVE_CONTRACT_ID,
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
    contract: HostStageFiveContract
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
