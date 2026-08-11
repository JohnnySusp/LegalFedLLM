from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile, RoundManifest, utc_text


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
    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["mock", "transformers"]
    device: Literal["cuda", "cpu"]
    precision: Literal["bfloat16", "float16", "float32"]
    quantization: Literal["none"] = "none"
    micro_batch_size: int = Field(default=1, ge=1, le=1024)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=65536)
    optimizer: Literal["adamw"] = "adamw"
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
        return sha256_hex(self.model_dump(mode="json"))


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


@dataclass(frozen=True)
class EncodedTrainingExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(token_id, int) for token_id in value
    ):
        raise ValueError("chat template did not return one token-ID sequence")
    return value


def encode_private_examples(
    examples: list[PrivateTrainingExample],
    *,
    tokenizer: Any,
    model_profile: ModelProfile,
    maximum_sequence_length: int,
) -> list[EncodedTrainingExample]:
    encoded: list[EncodedTrainingExample] = []
    template_options: dict[str, Any] = {}
    if model_profile.chat_template_mode == "qwen_non_thinking":
        template_options["enable_thinking"] = False

    for example in examples:
        user_messages = [{"role": "user", "content": example.prompt}]
        full_messages = [
            *user_messages,
            {"role": "assistant", "content": example.answer},
        ]
        prompt_ids = _input_ids(
            tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                **template_options,
            )
        )
        full_ids = _input_ids(
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                **template_options,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"chat template is not prefix-stable for {example.example_id!r}"
            )
        if len(full_ids) > maximum_sequence_length:
            raise ValueError(
                f"private example {example.example_id!r} has {len(full_ids)} "
                f"tokens, exceeding maximum_sequence_length="
                f"{maximum_sequence_length}"
            )
        answer_ids = full_ids[len(prompt_ids) :]
        if not answer_ids:
            raise ValueError(
                f"private example {example.example_id!r} has no answer tokens"
            )
        labels = [-100] * len(prompt_ids) + answer_ids
        encoded.append(
            EncodedTrainingExample(
                input_ids=full_ids,
                attention_mask=[1] * len(full_ids),
                labels=labels,
            )
        )
    return encoded


class AdapterCheckpointMetadata(TrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str
    profile_hash: str = Field(pattern=HASH_PATTERN)
    model_profile: ModelProfile
    version: int = Field(ge=0)
    parent_version: int | None = Field(default=None, ge=0)
    parent_checkpoint_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    round_id: str | None = None
    manifest_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    execution_profile_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    file_count: int = Field(ge=2)
    created_at: str


def _write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


class AdapterCheckpointStore:
    def __init__(self, root: str | Path, model_profile: ModelProfile):
        if (
            Path(model_profile.profile_id).name != model_profile.profile_id
            or model_profile.profile_id in {".", ".."}
        ):
            raise ValueError("model profile ID is not safe for adapter storage")
        self.root = Path(root).resolve() / model_profile.profile_id
        self.model_profile = model_profile
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)

    def staging_path(self, job_id: str) -> Path:
        if Path(job_id).name != job_id or job_id in {".", ".."}:
            raise ValueError("adapter job ID is not safe")
        path = self.root / "staging" / job_id
        path.mkdir(parents=False, exist_ok=False)
        return path

    def version_path(self, version: int) -> Path:
        return self.root / "versions" / f"v{version:06d}"

    def next_version(self, minimum: int) -> int:
        version = minimum
        while self.version_path(version).exists():
            version += 1
        return version

    @staticmethod
    def _tree_hash(path: Path) -> tuple[str, int]:
        required = {"adapter_config.json", "adapter_model.safetensors"}
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.name != "checkpoint.json"
        )
        relative_names = {item.relative_to(path).as_posix() for item in files}
        if not required.issubset(relative_names):
            raise ValueError("PEFT checkpoint is missing its required files")
        entries: list[dict[str, Any]] = []
        for item in files:
            if item.is_symlink():
                raise ValueError("adapter checkpoints must not contain symlinks")
            content = item.read_bytes()
            entries.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        return sha256_hex(entries), len(files)

    def seal(
        self,
        staging_path: Path,
        *,
        version: int,
        parent: AdapterCheckpointMetadata | None,
        round_id: str | None,
        manifest_hash: str | None,
        execution_profile_hash: str | None,
    ) -> AdapterCheckpointMetadata:
        expected_parent = self.root / "staging"
        if staging_path.parent.resolve() != expected_parent.resolve():
            raise ValueError("adapter staging directory is outside its store")
        checkpoint_hash, file_count = self._tree_hash(staging_path)
        metadata = AdapterCheckpointMetadata(
            profile_id=self.model_profile.profile_id,
            profile_hash=self.model_profile.profile_hash(),
            model_profile=self.model_profile,
            version=version,
            parent_version=parent.version if parent else None,
            parent_checkpoint_hash=parent.checkpoint_hash if parent else None,
            round_id=round_id,
            manifest_hash=manifest_hash,
            execution_profile_hash=execution_profile_hash,
            checkpoint_hash=checkpoint_hash,
            file_count=file_count,
            created_at=utc_text(),
        )
        _write_atomic_json(
            staging_path / "checkpoint.json",
            metadata.model_dump(mode="json"),
        )
        return metadata

    def promote(
        self,
        staging_path: Path,
        metadata: AdapterCheckpointMetadata,
    ) -> Path:
        self._validate_directory(staging_path, metadata)
        target = self.version_path(metadata.version)
        if target.exists():
            raise FileExistsError(str(target))
        staging_path.replace(target)
        _write_atomic_json(
            self.root / "current.json",
            {
                "version": metadata.version,
                "checkpoint_hash": metadata.checkpoint_hash,
                "profile_hash": metadata.profile_hash,
            },
        )
        return target

    def current(self) -> tuple[AdapterCheckpointMetadata, Path] | None:
        pointer_path = self.root / "current.json"
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if set(pointer) != {"version", "checkpoint_hash", "profile_hash"}:
            raise ValueError("adapter current pointer has an invalid schema")
        path = self.version_path(int(pointer["version"]))
        metadata_path = path / "checkpoint.json"
        if not metadata_path.is_file():
            raise ValueError("current adapter checkpoint metadata is missing")
        metadata = AdapterCheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if pointer != {
            "version": metadata.version,
            "checkpoint_hash": metadata.checkpoint_hash,
            "profile_hash": metadata.profile_hash,
        }:
            raise ValueError("adapter current pointer does not match its metadata")
        self._validate_directory(path, metadata)
        return metadata, path

    def _validate_directory(
        self,
        path: Path,
        metadata: AdapterCheckpointMetadata,
    ) -> None:
        if metadata.profile_id != self.model_profile.profile_id:
            raise ValueError("adapter checkpoint has another profile ID")
        if metadata.profile_hash != self.model_profile.profile_hash():
            raise ValueError("adapter checkpoint has an incompatible model profile")
        if metadata.model_profile.profile_hash() != (
            self.model_profile.profile_hash()
        ):
            raise ValueError("adapter checkpoint model metadata is incompatible")
        checkpoint_hash, file_count = self._tree_hash(path)
        if checkpoint_hash != metadata.checkpoint_hash:
            raise ValueError("adapter checkpoint file hash does not match")
        if file_count != metadata.file_count:
            raise ValueError("adapter checkpoint file count does not match")


class LocalTrainingRecord(TrainingContract):
    schema_version: Literal["1.0"] = "1.0"
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
    record_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hashes(self) -> "LocalTrainingRecord":
        if (
            self.training_execution_profile_hash
            != self.training_execution_profile.profile_hash()
        ):
            raise ValueError("training execution profile hash does not match")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("local training record hash does not match")
        return self

    @classmethod
    def create(cls, **values: Any) -> "LocalTrainingRecord":
        payload = {**values, "schema_version": "1.0"}
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
