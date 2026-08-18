from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.adapter_checkpoint import (
    AdapterCheckpointMetadata,
    AdapterCheckpointStore,
)
from shared.answer_only import (
    EncodedTrainingExample,
    encode_answer_only_example,
)
from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile


class TrainingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PrivateTrainingExample(TrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    example_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=100_000)
    answer: str = Field(min_length=1, max_length=100_000)

    @field_validator("example_id", "prompt", "answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("private-training text must not be blank")
        return value


def load_private_examples(path: str | Path) -> list[PrivateTrainingExample]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"private training JSONL does not exist: {source}")

    examples: list[PrivateTrainingExample] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"private training JSONL line {line_number} is blank"
                )
            try:
                value = json.loads(line)
                example = PrivateTrainingExample.model_validate(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid private training JSONL line {line_number}: {exc}"
                ) from exc
            if example.example_id in seen_ids:
                raise ValueError(
                    f"duplicate private example_id: {example.example_id!r}"
                )
            seen_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise ValueError("private training JSONL contains no examples")
    if len(examples) > 100_000:
        raise ValueError("private training JSONL exceeds 100000 examples")
    return examples


def private_dataset_semantic_hash(
    examples: list[PrivateTrainingExample],
) -> str:
    return sha256_hex(
        [example.model_dump(mode="json") for example in examples]
    )


class TrainingExecutionProfile(TrainingContract):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    backend: Literal["mock", "transformers"]
    device: Literal["cuda", "cpu"]
    precision: Literal["bfloat16", "float16", "float32"]
    quantization: Literal["none"] = "none"
    micro_batch_size: int = Field(default=1, ge=1, le=1024)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=65536)
    optimizer: Literal["adamw"] = "adamw"
    learning_rate_scheduler: Literal["linear"] = "linear"
    learning_rate: float = Field(default=2e-4, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    gradient_checkpointing: bool = False
    verify_frozen_base_checksum: bool = False

    @model_validator(mode="after")
    def validate_device_precision(self) -> "TrainingExecutionProfile":
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU training requires float32 precision")
        if self.device == "cuda" and self.precision == "float32":
            raise ValueError(
                "CUDA training must explicitly use bfloat16 or float16"
            )
        if self.backend == "mock" and (
            self.device != "cpu" or self.precision != "float32"
        ):
            raise ValueError("mock training uses the CPU/float32 test profile")
        return self

    def profile_hash(self) -> str:
        payload = self.model_dump(mode="json")
        if self.schema_version == "1.0":
            payload.pop("learning_rate_scheduler")
        return sha256_hex(payload)


def execution_profile_from_environment(
    backend: Literal["mock", "transformers"],
) -> TrainingExecutionProfile:
    if backend == "mock":
        return TrainingExecutionProfile(
            backend="mock",
            device="cpu",
            precision="float32",
        )
    return TrainingExecutionProfile(
        backend="transformers",
        device=os.getenv("CLIENT_TRAINING_DEVICE", "cuda").strip().lower(),
        precision=os.getenv(
            "CLIENT_TRAINING_PRECISION", "bfloat16"
        ).strip().lower(),
        quantization=os.getenv(
            "CLIENT_TRAINING_QUANTIZATION", "none"
        ).strip().lower(),
        micro_batch_size=int(os.getenv("CLIENT_MICRO_BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(
            os.getenv("CLIENT_GRADIENT_ACCUMULATION_STEPS", "8")
        ),
        optimizer=os.getenv("CLIENT_OPTIMIZER", "adamw").strip().lower(),
        learning_rate=float(os.getenv("CLIENT_LEARNING_RATE", "2e-4")),
        seed=int(os.getenv("CLIENT_TRAINING_SEED", "42")),
        gradient_checkpointing=os.getenv(
            "CLIENT_GRADIENT_CHECKPOINTING", "false"
        ).strip().lower()
        in {"1", "true", "yes"},
        verify_frozen_base_checksum=os.getenv(
            "CLIENT_VERIFY_FROZEN_BASE_CHECKSUM", "false"
        ).strip().lower()
        in {"1", "true", "yes"},
    )


def encode_private_examples(
    examples: list[PrivateTrainingExample],
    *,
    tokenizer: Any,
    model_profile: ModelProfile,
    maximum_sequence_length: int,
) -> list[EncodedTrainingExample]:
    return [
        encode_answer_only_example(
            item_id=example.example_id,
            item_kind="private example",
            prompt=example.prompt,
            answer=example.answer,
            tokenizer=tokenizer,
            model_profile=model_profile,
            maximum_sequence_length=maximum_sequence_length,
        )
        for example in examples
    ]


class LocalTrainingRecord(TrainingContract):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    round_id: str
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    client_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    training_execution_profile: TrainingExecutionProfile
    training_execution_profile_hash: str = Field(pattern=HASH_PATTERN)
    private_dataset_id: str
    private_dataset_semantic_hash: str = Field(pattern=HASH_PATTERN)
    private_example_count: int = Field(ge=1)
    parent_adapter_version: int | None = Field(default=None, ge=0)
    parent_checkpoint_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    result_adapter_version: int = Field(ge=0)
    result_checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    checkpoint_format: Literal["mock-json", "peft-safetensors"]
    label_format: Literal["chat_sft_answer_only_v1"]
    maximum_sequence_length: int = Field(ge=2)
    truncation_policy: Literal["reject"]
    started_at: str
    completed_at: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    trainable_parameter_count: int = Field(ge=0)
    total_parameter_count: int = Field(ge=0)
    optimizer_step_count: int | None = Field(default=None, ge=0)
    training_loss: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    record_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hashes(self) -> "LocalTrainingRecord":
        if (
            self.training_execution_profile_hash
            != self.training_execution_profile.profile_hash()
        ):
            raise ValueError("training execution profile hash does not match")
        payload = self.model_dump(mode="json", exclude={"record_hash"})
        if self.schema_version == "1.0":
            payload.pop("optimizer_step_count")
            payload.pop("training_loss")
            if self.training_execution_profile.schema_version == "1.0":
                payload["training_execution_profile"].pop(
                    "learning_rate_scheduler"
                )
        elif self.checkpoint_format == "peft-safetensors":
            if not self.optimizer_step_count:
                raise ValueError(
                    "real PEFT training must record an optimizer step"
                )
            if self.training_loss is None:
                raise ValueError("real PEFT training must record its loss")
        expected = sha256_hex(payload)
        if self.record_hash != expected:
            raise ValueError("local training record hash does not match")
        return self

    @classmethod
    def create(cls, **values: Any) -> "LocalTrainingRecord":
        payload = {**values, "schema_version": "1.1"}
        execution_profile = payload.get("training_execution_profile")
        if isinstance(execution_profile, TrainingExecutionProfile):
            payload["training_execution_profile"] = (
                execution_profile.model_dump(mode="json")
            )
        payload["record_hash"] = sha256_hex(payload)
        return cls(**payload)


@dataclass(frozen=True)
class BackendTrainingResult:
    parent_version: int | None
    parent_checkpoint_hash: str | None
    result_version: int
    result_checkpoint_hash: str
    checkpoint_format: Literal["mock-json", "peft-safetensors"]
    dependency_versions: dict[str, str]
    trainable_parameter_count: int
    total_parameter_count: int
    optimizer_step_count: int
    training_loss: float | None
