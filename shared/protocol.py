from __future__ import annotations

import math
import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.crypto import Ed25519Identity, sha256_hex, verify_json
from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    ReferenceSample,
)

PROTOCOL_VERSION = "1.1"
PACKAGE_SCHEMA_VERSION = "2.0"
KNOWLEDGE_ARTIFACT_FORMAT = "safetensors"
KNOWLEDGE_ARTIFACT_SCHEMA_VERSION = "2.0"
HOST_TRAINING_ARTIFACT_SCHEMA_VERSION = "1.0"
HOST_TRAINING_JOB_SCHEMA_VERSION = "1.0"
HOST_CANDIDATE_RESULT_SCHEMA_VERSION = "1.0"
HOST_CANDIDATE_VALIDATION_SCHEMA_VERSION = "1.0"
CLIENT_PUBLIC_DATA_PARTITION_SCHEMA_VERSION = "1.0"
CLIENT_REVERSE_TRAINING_ARTIFACT_SCHEMA_VERSION = "1.0"
CLIENT_REVERSE_TRAINING_JOB_SCHEMA_VERSION = "1.0"
HASH_PATTERN = r"^[0-9a-f]{64}$"
BASE64_PATTERN = r"^[A-Za-z0-9+/]+={0,2}$"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LoraProfile(ContractModel):
    rank: int = Field(ge=1, le=4096)
    alpha: float = Field(default=16.0, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    bias: Literal["none"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    modules_to_save: tuple[str, ...] = ()

    @field_validator("target_modules")
    @classmethod
    def clean_targets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in values)
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("target_modules must contain non-blank names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("target_modules must be unique")
        return cleaned

    @field_validator("modules_to_save")
    @classmethod
    def clean_modules_to_save(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in values)
        if any(not item for item in cleaned):
            raise ValueError("modules_to_save must contain non-blank names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("modules_to_save must be unique")
        return cleaned


class OllamaProfile(ContractModel):
    model: str = Field(min_length=1, max_length=256)
    digest: str | None = Field(default=None, max_length=256)


class ModelProfile(ContractModel):
    profile_schema_version: Literal["1.0"] = "1.0"
    profile_id: str = Field(min_length=1, max_length=128)
    role: Literal["client", "host"]
    model_id: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    model_class: str = Field(
        default="MockForCausalLM",
        min_length=1,
        max_length=256,
    )
    model_type: str = Field(default="mock", min_length=1, max_length=128)
    tokenizer_id: str = Field(min_length=1, max_length=512)
    tokenizer_revision: str = Field(min_length=1, max_length=256)
    tokenizer_class: str = Field(min_length=1, max_length=256)
    vocabulary_size: int | None = Field(default=None, ge=1)
    training_backend: Literal["mock", "transformers"] = "mock"
    serving_backend: Literal["mock", "ollama", "transformers"] = "mock"
    prompt_template_id: str = Field(
        default="chapter-section-question-v1",
        min_length=1,
        max_length=256,
    )
    prompt_template_hash: str = Field(pattern=HASH_PATTERN)
    tokenizer_chat_template_hash: str | None = Field(
        default=None,
        pattern=HASH_PATTERN,
    )
    chat_template_mode: Literal["mock", "standard", "qwen_non_thinking"] = (
        "mock"
    )
    lora: LoraProfile
    ollama: OllamaProfile | None = None

    @model_validator(mode="after")
    def validate_ollama(self) -> "ModelProfile":
        if self.serving_backend == "ollama" and self.ollama is None:
            raise ValueError("ollama profile is required for Ollama serving")
        if self.training_backend == "transformers":
            for name, revision in (
                ("model_revision", self.model_revision),
                ("tokenizer_revision", self.tokenizer_revision),
            ):
                if len(revision) != 40 or any(
                    character not in "0123456789abcdef" for character in revision
                ):
                    raise ValueError(
                        f"{name} must be a full lowercase commit hash"
                    )
            if self.model_revision != self.tokenizer_revision:
                raise ValueError(
                    "model and tokenizer revisions must identify one snapshot"
                )
            if self.tokenizer_chat_template_hash is None:
                raise ValueError(
                    "Transformers profiles require a chat-template hash"
                )
            if self.vocabulary_size is None:
                raise ValueError(
                    "Transformers profiles require an expected vocabulary size"
                )
            if self.chat_template_mode == "mock":
                raise ValueError(
                    "Transformers profiles require a real chat-template mode"
                )
        return self

    def profile_hash(self) -> str:
        return sha256_hex(
            self.model_dump(
                mode="json",
                exclude={"serving_backend", "ollama"},
            )
        )


class DifferentialPrivacyPolicy(ContractModel):
    required: bool = False
    mechanism: str = "dp_sgd"
    max_epsilon: float | None = Field(default=None, gt=0)
    delta: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_policy(self) -> "DifferentialPrivacyPolicy":
        mechanism = self.mechanism.strip().lower()

        if self.required:
            if mechanism == "none":
                raise ValueError("a required DP policy must name a mechanism")

            if self.max_epsilon is None or self.delta is None:
                raise ValueError(
                    "a required DP policy must define max_epsilon and delta"
                )

        return self


class DifferentialPrivacyReport(ContractModel):
    enabled: bool = False
    mechanism: str = "none"
    epsilon_spent: float | None = Field(default=None, ge=0)
    delta: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_report(self) -> "DifferentialPrivacyReport":
        mechanism = self.mechanism.strip().lower()

        if self.enabled:
            if mechanism == "none":
                raise ValueError("an enabled DP report must name a mechanism")

            if self.epsilon_spent is None or self.delta is None:
                raise ValueError(
                    "an enabled DP report must include epsilon_spent and delta"
                )

        elif (
            mechanism != "none"
            or self.epsilon_spent is not None
            or self.delta is not None
        ):
            raise ValueError(
                "a disabled DP report must use mechanism=none "
                "without privacy values"
            )

        return self


class DistillationConfig(ContractModel):
    strategy: Literal["dual_min_ce"] = "dual_min_ce"
    loss_type: Literal["ce", "kl"] = "ce"
    lm_loss_weight: float = Field(default=0.9, ge=0, le=1)
    temperature: float = Field(default=1.0, gt=0)
    minimum_validation_improvement: float = Field(default=0.001, ge=0)


class AlignmentConfig(ContractModel):
    strategy: Literal["mock_identity", "dtw"] = "mock_identity"
    profile_version: str = Field(default="1", min_length=1, max_length=128)

    @property
    def profile_id(self) -> str:
        return f"{self.strategy}:{self.profile_version}"


class RoundCreateRequest(ContractModel):
    selected_client_ids: list[str] = Field(min_length=1, max_length=256)
    trusted_client_quorum: int = Field(ge=1)
    reference_dataset_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    reference_dataset_hash: str | None = Field(
        default=None,
        pattern=HASH_PATTERN,
    )
    sample_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=100_000,
    )
    prompt_template: str = Field(min_length=1, max_length=20_000)
    label_format: str = Field(default="causal_lm", min_length=1, max_length=128)
    maximum_sequence_length: int = Field(default=512, ge=2, le=131_072)
    truncation_policy: Literal["right", "left", "reject"] = "right"
    top_k: int = Field(default=20, ge=1, le=4096)
    training_epochs: int = Field(default=1, ge=1, le=100)
    host_public_data_epochs: Literal[1, 5] = 5
    client_public_data_epochs: Literal[1] = 1
    client_public_validation_fraction: Literal[0.1] = 0.1
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    dp_policy: DifferentialPrivacyPolicy = Field(default_factory=DifferentialPrivacyPolicy)
    maximum_knowledge_package_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, le=2 * 1024 * 1024 * 1024
    )
    maximum_host_training_job_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1024,
        le=2 * 1024 * 1024 * 1024,
    )
    maximum_client_reverse_training_job_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1024,
        le=2 * 1024 * 1024 * 1024,
    )
    submission_window_seconds: int = Field(default=3600, ge=1, le=31_536_000)

    @model_validator(mode="after")
    def validate_round_request(self) -> "RoundCreateRequest":
        if len(self.selected_client_ids) != len(set(self.selected_client_ids)):
            raise ValueError("selected_client_ids must be unique")
        if (
            self.sample_ids is not None
            and len(self.sample_ids) != len(set(self.sample_ids))
        ):
            raise ValueError("sample_ids must be unique")

        metadata_fields = (
            self.reference_dataset_id is not None,
            self.reference_dataset_hash is not None,
            self.sample_ids is not None,
        )

        if any(metadata_fields) and not all(metadata_fields):
            raise ValueError(
                "reference dataset ID, hash and sample IDs "
                "must be provided together"
            )
        if self.trusted_client_quorum > len(self.selected_client_ids):
            raise ValueError("quorum cannot exceed selected Client count")
        return self


class RoundManifest(ContractModel):
    protocol_version: Literal["1.1"] = PROTOCOL_VERSION
    round_id: str = Field(min_length=1, max_length=128)
    selected_client_ids: list[str]
    selected_client_profile_hashes: dict[str, str]
    trusted_client_quorum: int = Field(ge=1)
    current_host_adapter_version: int = Field(ge=0)
    host_model_profile: ModelProfile
    reference_dataset_id: str
    reference_dataset_hash: str = Field(pattern=HASH_PATTERN)
    sample_ids: list[str] = Field(min_length=1, max_length=100_000)
    prompt_template: str
    prompt_template_hash: str = Field(pattern=HASH_PATTERN)
    label_format: str
    maximum_sequence_length: int = Field(ge=2)
    truncation_policy: Literal["right", "left", "reject"]
    top_k: int = Field(ge=1)
    training_epochs: int = Field(ge=1)
    host_public_data_epochs: Literal[1, 5]
    client_public_data_epochs: Literal[1]
    client_public_validation_fraction: Literal[0.1]
    selected_client_alignment_profiles: dict[str, str]
    distillation: DistillationConfig
    dp_policy: DifferentialPrivacyPolicy
    maximum_knowledge_package_bytes: int = Field(ge=1024)
    maximum_host_training_job_bytes: int = Field(ge=1024)
    maximum_client_reverse_training_job_bytes: int = Field(ge=1024)
    submission_deadline: str
    round_nonce: str = Field(min_length=16, max_length=256)
    coordinator_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    coordinator_signature: str = Field(pattern=BASE64_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> "RoundManifest":
        parse_utc(self.submission_deadline)
        if self.host_model_profile.role != "host":
            raise ValueError("host_model_profile must have role=host")
        if self.trusted_client_quorum > len(self.selected_client_ids):
            raise ValueError("quorum cannot exceed selected Client count")
        if len(self.selected_client_ids) != len(set(self.selected_client_ids)):
            raise ValueError("selected_client_ids must be unique")
        if set(self.selected_client_profile_hashes) != set(
            self.selected_client_ids
        ):
            raise ValueError(
                "selected Client profile hashes must match selected Client IDs"
            )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.selected_client_profile_hashes.values()
        ):
            raise ValueError("selected Client profile hashes must be SHA-256 hex")
        if set(self.selected_client_alignment_profiles) != set(
            self.selected_client_ids
        ):
            raise ValueError(
                "selected Client alignment profiles must match selected Client IDs"
            )
        strategies: set[str] = set()
        for profile_id in self.selected_client_alignment_profiles.values():
            if not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError(
                    "selected Client alignment profile IDs must be non-blank strings"
                )
            strategy, separator, version = profile_id.partition(":")
            if separator != ":" or not version or strategy not in {
                "mock_identity",
                "dtw",
            }:
                raise ValueError(
                    "selected Client alignment profile IDs must use "
                    "mock_identity:<version> or dtw:<version>"
                )
            if len(profile_id) > 256:
                raise ValueError(
                    "selected Client alignment profile IDs must not exceed 256 characters"
                )
            strategies.add(strategy)
        if len(strategies) != 1:
            raise ValueError(
                "selected Client alignment profiles must use one alignment strategy"
            )
        if (
            self.sample_ids is not None
            and len(self.sample_ids) != len(set(self.sample_ids))
        ):
            raise ValueError("sample_ids must be unique")

        metadata_fields = (
            self.reference_dataset_id is not None,
            self.reference_dataset_hash is not None,
            self.sample_ids is not None,
        )

        if any(metadata_fields) and not all(metadata_fields):
            raise ValueError(
                "reference dataset ID, hash and sample IDs "
                "must be provided together"
            )
        expected = sha256_hex(self.hash_payload())
        if self.manifest_hash != expected:
            raise ValueError("manifest_hash does not match the manifest payload")
        return self

    def alignment_profile_id_for(self, client_id: str) -> str:
        try:
            return self.selected_client_alignment_profiles[client_id]
        except KeyError:
            raise ValueError(
                f"Client {client_id!r} has no signed alignment assignment"
            ) from None

    @property
    def alignment_strategy(self) -> str:
        first_client_id = self.selected_client_ids[0]
        return self.alignment_profile_id_for(first_client_id).split(":", 1)[0]

    @property
    def host_package_alignment_profile_id(self) -> str:
        """Return the deterministic Host-package alignment anchor for the round."""

        return self.alignment_profile_id_for(self.selected_client_ids[0])

    @property
    def homogeneous_alignment_profile_id(self) -> str:
        profile_ids = set(self.selected_client_alignment_profiles.values())
        if len(profile_ids) != 1:
            raise ValueError(
                "round has multiple Client alignment profiles"
            )
        return next(iter(profile_ids))

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json", exclude={"manifest_hash", "coordinator_signature"}
        )

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"coordinator_signature"})

    def verify_signature(self, public_key_b64: str) -> bool:
        return verify_json(
            public_key_b64, self.signed_payload(), self.coordinator_signature
        )

    @classmethod
    def create_signed(
        cls,
        *,
        identity: Ed25519Identity,
        round_id: str,
        coordinator_id: str,
        current_host_adapter_version: int,
        host_model_profile: ModelProfile,
        selected_client_profile_hashes: dict[str, str],
        selected_client_alignment_profiles: dict[str, str],
        request: RoundCreateRequest,
        submission_deadline: str,
    ) -> "RoundManifest":
        if (
            request.reference_dataset_id is None
            or request.reference_dataset_hash is None
            or request.sample_ids is None
        ):
            raise ValueError(
                "resolved reference dataset metadata is required"
            )

        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "round_id": round_id,
            "selected_client_ids": request.selected_client_ids,
            "selected_client_profile_hashes": selected_client_profile_hashes,
            "selected_client_alignment_profiles": (
                selected_client_alignment_profiles
            ),
            "trusted_client_quorum": request.trusted_client_quorum,
            "current_host_adapter_version": current_host_adapter_version,
            "host_model_profile": host_model_profile.model_dump(mode="json"),
            "reference_dataset_id": request.reference_dataset_id,
            "reference_dataset_hash": request.reference_dataset_hash,
            "sample_ids": request.sample_ids,
            "prompt_template": request.prompt_template,
            "prompt_template_hash": sha256_hex(request.prompt_template.encode("utf-8")),
            "label_format": request.label_format,
            "maximum_sequence_length": request.maximum_sequence_length,
            "truncation_policy": request.truncation_policy,
            "top_k": request.top_k,
            "training_epochs": request.training_epochs,
            "host_public_data_epochs": request.host_public_data_epochs,
            "client_public_data_epochs": request.client_public_data_epochs,
            "client_public_validation_fraction": (
                request.client_public_validation_fraction
            ),
            "distillation": request.distillation.model_dump(mode="json"),
            "dp_policy": request.dp_policy.model_dump(mode="json"),
            "maximum_knowledge_package_bytes": request.maximum_knowledge_package_bytes,
            "maximum_host_training_job_bytes": (
                request.maximum_host_training_job_bytes
            ),
            "maximum_client_reverse_training_job_bytes": (
                request.maximum_client_reverse_training_job_bytes
            ),
            "submission_deadline": submission_deadline,
            "round_nonce": secrets.token_urlsafe(24),
            "coordinator_id": coordinator_id,
        }
        manifest_hash = sha256_hex(payload)
        signed_payload = {**payload, "manifest_hash": manifest_hash}
        return cls(
            **signed_payload,
            coordinator_signature=identity.sign_json(signed_payload),
        )


class HostReferenceDatasetBundle(ContractModel):
    manifest: RoundManifest
    reference_samples: list[ReferenceSample] = Field(
        min_length=1,
        max_length=100_000,
    )
    validation_samples: list[ReferenceSample] = Field(
        min_length=1,
        max_length=100_000,
    )
    validation_identity: ReferenceDatasetIdentity


class HostReferenceDatasetReceipt(ContractModel):
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    reference_identity: ReferenceDatasetIdentity
    validation_identity: ReferenceDatasetIdentity


class KnowledgeSample(ContractModel):
    sample_id: str = Field(min_length=1, max_length=256)
    source_input_ids: list[int] = Field(min_length=1)
    attention_length: int = Field(ge=1)
    top_k_token_ids: list[list[int]] = Field(min_length=1)
    top_k_logits: list[list[float]] = Field(min_length=1)
    full_logsumexp: list[float] = Field(min_length=1)
    gold_token_ids: list[int] = Field(min_length=1)
    gold_token_logits: list[float] = Field(min_length=1)
    gold_token_nll: list[float] = Field(min_length=1)
    ce_loss: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_shapes(self) -> "KnowledgeSample":
        length = len(self.source_input_ids)
        if self.attention_length > length:
            raise ValueError("attention_length exceeds source_input_ids length")
        if len(self.top_k_token_ids) != len(self.top_k_logits):
            raise ValueError("top-k token and logit sequence lengths differ")
        if len(self.top_k_token_ids) != length:
            raise ValueError("top-k sequence length must match source_input_ids")
        if len(self.full_logsumexp) != length:
            raise ValueError("full_logsumexp length must match source_input_ids")
        if len(self.gold_token_ids) != length:
            raise ValueError("gold_token_ids length must match source_input_ids")
        if len(self.gold_token_logits) != length:
            raise ValueError("gold_token_logits length must match source_input_ids")
        if len(self.gold_token_nll) != length:
            raise ValueError("gold_token_nll length must match source_input_ids")
        widths = {len(row) for row in self.top_k_token_ids}
        logit_widths = {len(row) for row in self.top_k_logits}
        if not widths or 0 in widths or widths != logit_widths or len(widths) != 1:
            raise ValueError("top-k rows must have one consistent non-zero width")
        if not math.isfinite(self.ce_loss):
            raise ValueError("ce_loss must be finite")
        if any(token < 0 for row in self.top_k_token_ids for token in row):
            raise ValueError("token IDs must be non-negative")
        if any(
            token != -100 and token < 0
            for token in self.gold_token_ids
        ):
            raise ValueError("gold token IDs must be -100 or non-negative")
        if any(not math.isfinite(value) for row in self.top_k_logits for value in row):
            raise ValueError("logits must be finite")
        if any(not math.isfinite(value) for value in self.full_logsumexp):
            raise ValueError("full_logsumexp values must be finite")
        if any(not math.isfinite(value) for value in self.gold_token_logits):
            raise ValueError("gold_token_logits values must be finite")
        if any(
            not math.isfinite(value) or value < 0
            for value in self.gold_token_nll
        ):
            raise ValueError("gold_token_nll values must be finite and non-negative")
        supervised = 0
        for token_id, gold_logit, nll in zip(
            self.gold_token_ids,
            self.gold_token_logits,
            self.gold_token_nll,
            strict=True,
        ):
            if token_id == -100:
                if gold_logit != 0.0 or nll != 0.0:
                    raise ValueError(
                        "unsupervised gold-token positions must use logit/NLL 0.0"
                    )
            else:
                supervised += 1
        if supervised == 0:
            raise ValueError("knowledge sample requires a supervised gold token")
        return self

    @property
    def top_k(self) -> int:
        return len(self.top_k_token_ids[0])


class KnowledgeArtifactDescriptor(ContractModel):
    format: Literal["safetensors"] = KNOWLEDGE_ARTIFACT_FORMAT
    schema_version: Literal["2.0"] = KNOWLEDGE_ARTIFACT_SCHEMA_VERSION
    byte_size: int = Field(ge=1, le=2 * 1024 * 1024 * 1024)
    sha256: str = Field(pattern=HASH_PATTERN)
    sample_count: int = Field(ge=1, le=100_000)
    sample_ids_sha256: str = Field(pattern=HASH_PATTERN)
    total_token_count: int = Field(ge=1)
    top_k: int = Field(ge=1, le=4096)


class KnowledgePackage(ContractModel):
    protocol_version: Literal["1.1"] = PROTOCOL_VERSION
    package_schema_version: Literal["2.0"] = PACKAGE_SCHEMA_VERSION
    round_id: str
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    sender_id: str = Field(min_length=1, max_length=128)
    sender_role: Literal["client", "host"]
    model_profile: ModelProfile
    adapter_version: int = Field(ge=0)
    alignment_profile_id: str = Field(min_length=1, max_length=256)
    reference_dataset_id: str
    reference_dataset_hash: str = Field(pattern=HASH_PATTERN)
    sample_ids: list[str] = Field(min_length=1, max_length=100_000)
    top_k: int = Field(ge=1)
    artifact: KnowledgeArtifactDescriptor
    dp_report: DifferentialPrivacyReport = Field(
        default_factory=DifferentialPrivacyReport
    )
    nonce: str = Field(min_length=16, max_length=256)
    created_at: str
    package_hash: str = Field(pattern=HASH_PATTERN)
    signature: str = Field(pattern=BASE64_PATTERN)

    @model_validator(mode="after")
    def validate_package(self) -> "KnowledgePackage":
        parse_utc(self.created_at)
        if self.model_profile.role != self.sender_role:
            raise ValueError("model profile role does not match sender_role")
        if not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("sample_ids must contain non-blank values")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("sample_ids must be unique")
        if self.artifact.sample_count != len(self.sample_ids):
            raise ValueError("artifact sample count does not match sample_ids")
        if self.artifact.sample_ids_sha256 != sha256_hex(self.sample_ids):
            raise ValueError("artifact is bound to another sample order")
        if self.artifact.top_k != self.top_k:
            raise ValueError("artifact top-k width does not match package top_k")
        expected = sha256_hex(self.hash_payload())
        if self.package_hash != expected:
            raise ValueError("package_hash does not match the package payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json", exclude={"package_hash", "signature"}
        )

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"signature"})

    def verify_signature(self, public_key_b64: str) -> bool:
        return verify_json(public_key_b64, self.signed_payload(), self.signature)

    @classmethod
    def create_signed(
        cls,
        *,
        identity: Ed25519Identity,
        round_id: str,
        manifest_hash: str,
        sender_id: str,
        sender_role: Literal["client", "host"],
        model_profile: ModelProfile,
        adapter_version: int,
        alignment_profile_id: str,
        reference_dataset_id: str,
        reference_dataset_hash: str,
        top_k: int,
        sample_ids: list[str],
        artifact: KnowledgeArtifactDescriptor,
        dp_report: DifferentialPrivacyReport | None = None,
        nonce: str | None = None,
        created_at: str | None = None,
    ) -> "KnowledgePackage":
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "package_schema_version": PACKAGE_SCHEMA_VERSION,
            "round_id": round_id,
            "manifest_hash": manifest_hash,
            "sender_id": sender_id,
            "sender_role": sender_role,
            "model_profile": model_profile.model_dump(mode="json"),
            "adapter_version": adapter_version,
            "alignment_profile_id": alignment_profile_id,
            "reference_dataset_id": reference_dataset_id,
            "reference_dataset_hash": reference_dataset_hash,
            "sample_ids": sample_ids,
            "top_k": top_k,
            "artifact": artifact.model_dump(mode="json"),
            "dp_report": (dp_report or DifferentialPrivacyReport()).model_dump(
                mode="json"
            ),
            "nonce": nonce or secrets.token_urlsafe(24),
            "created_at": created_at or utc_text(),
        }
        package_hash = sha256_hex(payload)
        signed_payload = {**payload, "package_hash": package_hash}
        return cls(**signed_payload, signature=identity.sign_json(signed_payload))


class EnrollmentTokenIssue(ContractModel):
    token: str = Field(min_length=32, max_length=512)
    issued_at: str

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at(cls, value: str) -> str:
        parse_utc(value)
        return value


class ClientRequestAuthentication(ContractModel):
    client_id: str = Field(min_length=1, max_length=128)
    method: str = Field(min_length=1, max_length=16)
    path: str = Field(min_length=1, max_length=2048)
    timestamp: str
    nonce: str = Field(min_length=16, max_length=256)
    signature: str = Field(pattern=BASE64_PATTERN)

    @model_validator(mode="after")
    def validate_request(self) -> "ClientRequestAuthentication":
        if self.method != self.method.upper():
            raise ValueError("authenticated request method must be uppercase")
        if not self.path.startswith("/"):
            raise ValueError("authenticated request path must be absolute")
        parse_utc(self.timestamp)
        return self

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"signature"})

    def verify_signature(self, public_key_b64: str) -> bool:
        return verify_json(public_key_b64, self.signed_payload(), self.signature)

    @classmethod
    def create_signed(
        cls,
        *,
        identity: Ed25519Identity,
        client_id: str,
        method: str,
        path: str,
        timestamp: str | None = None,
        nonce: str | None = None,
    ) -> "ClientRequestAuthentication":
        payload = {
            "client_id": client_id,
            "method": method.upper(),
            "path": path,
            "timestamp": timestamp or utc_text(),
            "nonce": nonce or secrets.token_urlsafe(24),
        }
        return cls(**payload, signature=identity.sign_json(payload))


class ClientRegistrationRequest(ContractModel):
    client_id: str = Field(min_length=1, max_length=128)
    public_key: str = Field(pattern=BASE64_PATTERN)
    model_profile: ModelProfile

    @model_validator(mode="after")
    def validate_role(self) -> "ClientRegistrationRequest":
        if self.model_profile.role != "client":
            raise ValueError("Client registration requires role=client")
        return self


class RegistrationRecord(ClientRegistrationRequest):
    registered_at: str


class SafetyReport(ContractModel):
    accepted: bool
    trust_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    probe_stage: Literal["pre_alignment", "post_alignment"] = "post_alignment"
    score_components: dict[str, float] = Field(default_factory=dict)
    sample_risks: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_safety_score(self) -> "SafetyReport":
        if self.trust_score != round(self.trust_score, 2):
            raise ValueError("trust_score must use at most two decimal places")
        for values, label in (
            (self.score_components, "safety score components"),
            (self.sample_risks, "sample risks"),
        ):
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in values.values()
            ):
                raise ValueError(f"{label} must be finite values in [0, 1]")
        return self


class ClientTrustHistory(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    client_id: str = Field(min_length=1, max_length=128)
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    completed_rounds: int = Field(ge=0)
    last_round_id: str | None = Field(default=None, max_length=128)

    @property
    def reliability(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ValidatedDistillationSample(ContractModel):
    sample_id: str
    teacher_id: str
    teacher_ce_loss: float = Field(ge=0)
    host_ce_loss: float = Field(ge=0)
    source_input_ids: list[int]
    attention_length: int = Field(ge=1)
    aligned_top_k_token_ids: list[list[int]]
    aligned_top_k_logits: list[list[float]]
    trust_score: float = Field(gt=0, le=1)


class ValidatedDistillationDataset(ContractModel):
    round_id: str
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    host_adapter_version: int = Field(ge=0)
    accepted_client_ids: list[str]
    samples: list[ValidatedDistillationSample]
    dataset_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> "ValidatedDistillationDataset":
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"dataset_hash"})
        )
        if self.dataset_hash != expected:
            raise ValueError("dataset_hash does not match distillation dataset")
        return self

    @classmethod
    def create(
        cls,
        *,
        round_id: str,
        manifest_hash: str,
        host_adapter_version: int,
        accepted_client_ids: list[str],
        samples: list[ValidatedDistillationSample],
    ) -> "ValidatedDistillationDataset":
        payload = {
            "round_id": round_id,
            "manifest_hash": manifest_hash,
            "host_adapter_version": host_adapter_version,
            "accepted_client_ids": accepted_client_ids,
            "samples": [sample.model_dump(mode="json") for sample in samples],
        }
        return cls(**payload, dataset_hash=sha256_hex(payload))


class DistillationJob(ContractModel):
    manifest: RoundManifest
    dataset: ValidatedDistillationDataset


class HostTrainingArtifactDescriptor(ContractModel):
    format: Literal["safetensors"] = KNOWLEDGE_ARTIFACT_FORMAT
    schema_version: Literal["1.0"] = HOST_TRAINING_ARTIFACT_SCHEMA_VERSION
    byte_size: int = Field(ge=1, le=2 * 1024 * 1024 * 1024)
    sha256: str = Field(pattern=HASH_PATTERN)
    sample_count: int = Field(ge=1, le=100_000)
    sample_ids_sha256: str = Field(pattern=HASH_PATTERN)
    padded_sequence_length: int = Field(ge=2, le=131_072)
    top_k: int = Field(ge=1, le=4096)
    pad_token_id: int = Field(ge=0)
    trainer_inputs_sha256: str = Field(pattern=HASH_PATTERN)


class ClientPublicDataPartition(ContractModel):
    schema_version: Literal["1.0"] = CLIENT_PUBLIC_DATA_PARTITION_SCHEMA_VERSION
    reference_dataset_id: str = Field(min_length=1, max_length=256)
    reference_dataset_hash: str = Field(pattern=HASH_PATTERN)
    transfer_sample_ids: list[str] = Field(min_length=1, max_length=100_000)
    validation_sample_ids: list[str] = Field(max_length=100_000)
    transfer_sample_ids_sha256: str = Field(pattern=HASH_PATTERN)
    validation_sample_ids_sha256: str = Field(pattern=HASH_PATTERN)
    validation_fraction: Literal[0.1] = 0.1
    partition_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_partition(self) -> ClientPublicDataPartition:
        transfer = self.transfer_sample_ids
        validation = self.validation_sample_ids
        if len(transfer) != len(set(transfer)):
            raise ValueError("Client transfer sample IDs must be unique")
        if len(validation) != len(set(validation)):
            raise ValueError("Client validation sample IDs must be unique")
        if set(transfer).intersection(validation):
            raise ValueError("Client public-data partition overlaps")
        if self.transfer_sample_ids_sha256 != sha256_hex(transfer):
            raise ValueError("Client transfer sample hash differs")
        if self.validation_sample_ids_sha256 != sha256_hex(validation):
            raise ValueError("Client validation sample hash differs")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"partition_hash"})
        )
        if self.partition_hash != expected:
            raise ValueError("Client public-data partition hash differs")
        return self

    @classmethod
    def create(cls, **values: Any) -> ClientPublicDataPartition:
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"partition_hash"}
        )
        return cls(**payload, partition_hash=sha256_hex(payload))


class ClientReverseTrainingArtifactDescriptor(ContractModel):
    format: Literal["safetensors"] = KNOWLEDGE_ARTIFACT_FORMAT
    schema_version: Literal["1.0"] = CLIENT_REVERSE_TRAINING_ARTIFACT_SCHEMA_VERSION
    byte_size: int = Field(ge=1, le=2 * 1024 * 1024 * 1024)
    sha256: str = Field(pattern=HASH_PATTERN)
    sample_count: int = Field(ge=1, le=100_000)
    sample_ids_sha256: str = Field(pattern=HASH_PATTERN)
    padded_sequence_length: int = Field(ge=2, le=131_072)
    top_k: int = Field(ge=1, le=4096)
    pad_token_id: int = Field(ge=0)
    trainer_inputs_sha256: str = Field(pattern=HASH_PATTERN)


class ClientReverseTrainingJob(ContractModel):
    schema_version: Literal["1.0"] = CLIENT_REVERSE_TRAINING_JOB_SCHEMA_VERSION
    manifest: RoundManifest
    client_id: str = Field(min_length=1, max_length=128)
    client_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    parent_adapter_version: int = Field(ge=0)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    accepted_host_adapter_version: int = Field(ge=0)
    host_package_hash: str = Field(pattern=HASH_PATTERN)
    host_adapter_promoted: bool
    public_data_partition: ClientPublicDataPartition
    host_teacher_sample_ids: list[str] = Field(max_length=100_000)
    client_public_data_epochs: Literal[1]
    distillation: DistillationConfig
    integration_audit_hash: str = Field(pattern=HASH_PATTERN)
    artifact: ClientReverseTrainingArtifactDescriptor
    created_at: str
    job_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_job(self) -> ClientReverseTrainingJob:
        parse_utc(self.created_at)
        manifest = self.manifest
        partition = self.public_data_partition
        if partition.reference_dataset_id != manifest.reference_dataset_id:
            raise ValueError("Client partition uses another reference dataset")
        if partition.reference_dataset_hash != manifest.reference_dataset_hash:
            raise ValueError("Client partition uses another dataset hash")
        combined = set(partition.transfer_sample_ids) | set(
            partition.validation_sample_ids
        )
        if combined != set(manifest.sample_ids):
            raise ValueError("Client partition does not cover the manifest")
        validation_ids = set(partition.validation_sample_ids)
        expected_transfer = [
            value
            for value in manifest.sample_ids
            if value in combined - validation_ids
        ]
        expected_validation = [
            value for value in manifest.sample_ids if value in validation_ids
        ]
        if partition.transfer_sample_ids != expected_transfer or (
            partition.validation_sample_ids != expected_validation
        ):
            raise ValueError("Client partition changed signed sample order")
        if not set(self.host_teacher_sample_ids).issubset(
            partition.transfer_sample_ids
        ):
            raise ValueError("Host teacher sample IDs leave the transfer split")
        expected_host_order = [
            value
            for value in partition.transfer_sample_ids
            if value in set(self.host_teacher_sample_ids)
        ]
        if self.host_teacher_sample_ids != expected_host_order:
            raise ValueError("Host teacher sample IDs changed transfer order")
        if self.artifact.sample_count != len(partition.transfer_sample_ids):
            raise ValueError("Client reverse artifact sample count differs")
        if self.artifact.sample_ids_sha256 != partition.transfer_sample_ids_sha256:
            raise ValueError("Client reverse artifact sample order differs")
        if self.artifact.top_k != manifest.top_k:
            raise ValueError("Client reverse artifact top-k differs")
        if self.artifact.padded_sequence_length > manifest.maximum_sequence_length:
            raise ValueError("Client reverse artifact exceeds sequence limit")
        if self.client_public_data_epochs != manifest.client_public_data_epochs:
            raise ValueError("Client public-data epochs differ from manifest")
        if self.distillation != manifest.distillation:
            raise ValueError("Client reverse objective differs from manifest")
        if (
            self.distillation.loss_type != "ce"
            or self.distillation.temperature != 1.0
            or self.distillation.lm_loss_weight != 0.9
        ):
            raise ValueError(
                "Client reverse training requires answer-only 0.9 supervised "
                "+ 0.1 sparse CE at temperature 1.0"
            )
        if self.artifact.byte_size > manifest.maximum_client_reverse_training_job_bytes:
            raise ValueError("Client reverse artifact exceeds signed size limit")
        expected = sha256_hex(self.model_dump(mode="json", exclude={"job_hash"}))
        if self.job_hash != expected:
            raise ValueError("job_hash does not match Client reverse training job")
        return self

    @classmethod
    def create(cls, **values: Any) -> ClientReverseTrainingJob:
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"job_hash"}
        )
        return cls(**payload, job_hash=sha256_hex(payload))


class HostTrainingJob(ContractModel):
    schema_version: Literal["1.0"] = HOST_TRAINING_JOB_SCHEMA_VERSION
    manifest: RoundManifest
    dataset_hash: str = Field(pattern=HASH_PATTERN)
    integration_audit_hash: str = Field(pattern=HASH_PATTERN)
    accepted_client_ids: list[str] = Field(min_length=1, max_length=256)
    sample_ids: list[str] = Field(min_length=1, max_length=100_000)
    host_adapter_version: int = Field(ge=0)
    host_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    host_public_data_epochs: Literal[1, 5]
    distillation: DistillationConfig
    artifact: HostTrainingArtifactDescriptor
    created_at: str
    job_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_job(self) -> "HostTrainingJob":
        parse_utc(self.created_at)
        manifest = self.manifest
        if self.sample_ids != manifest.sample_ids:
            raise ValueError("Host training sample order differs from the manifest")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Host training sample IDs must be unique")
        if self.artifact.sample_count != len(self.sample_ids):
            raise ValueError("Host training artifact sample count differs")
        if self.artifact.sample_ids_sha256 != sha256_hex(self.sample_ids):
            raise ValueError("Host training artifact has another sample order")
        if self.artifact.top_k != manifest.top_k:
            raise ValueError("Host training artifact has another top-k width")
        if (
            self.artifact.padded_sequence_length
            > manifest.maximum_sequence_length
        ):
            raise ValueError("Host training artifact exceeds the sequence limit")
        if self.host_adapter_version != manifest.current_host_adapter_version:
            raise ValueError("Host training job has another parent adapter")
        if self.host_model_profile_hash != (
            manifest.host_model_profile.profile_hash()
        ):
            raise ValueError("Host training job has another model profile")
        if self.host_public_data_epochs != manifest.host_public_data_epochs:
            raise ValueError("Host training epochs differ from the manifest")
        if self.distillation != manifest.distillation:
            raise ValueError("Host training distillation config differs")
        if self.artifact.byte_size > manifest.maximum_host_training_job_bytes:
            raise ValueError("Host training artifact exceeds the signed size limit")
        if len(self.accepted_client_ids) < manifest.trusted_client_quorum:
            raise ValueError("Host training job no longer satisfies quorum")
        if len(self.accepted_client_ids) != len(set(self.accepted_client_ids)):
            raise ValueError("accepted Host training Client IDs must be unique")
        accepted = set(self.accepted_client_ids)
        if not accepted.issubset(manifest.selected_client_ids):
            raise ValueError("Host training job contains an unselected Client")
        expected_order = [
            client_id
            for client_id in manifest.selected_client_ids
            if client_id in accepted
        ]
        if self.accepted_client_ids != expected_order:
            raise ValueError("accepted Host training Clients changed signed order")
        expected_hash = sha256_hex(
            self.model_dump(mode="json", exclude={"job_hash"})
        )
        if self.job_hash != expected_hash:
            raise ValueError("job_hash does not match the Host training job")
        return self

    @classmethod
    def create(cls, **values: Any) -> "HostTrainingJob":
        payload = cls.model_construct(**values).model_dump(
            mode="json",
            exclude={"job_hash"},
        )
        return cls(**payload, job_hash=sha256_hex(payload))


class HostTrainingJobReceipt(ContractModel):
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    artifact_sha256: str = Field(pattern=HASH_PATTERN)
    artifact_byte_size: int = Field(ge=1)
    accepted_client_ids: list[str] = Field(min_length=1, max_length=256)


class HostCandidateTrainingResult(ContractModel):
    schema_version: Literal["1.0"] = HOST_CANDIDATE_RESULT_SCHEMA_VERSION
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    parent_adapter_version: int = Field(ge=0)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    candidate_adapter_version: int = Field(ge=1)
    candidate_adapter_hash: str = Field(pattern=HASH_PATTERN)
    host_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    execution_profile_hash: str = Field(pattern=HASH_PATTERN)
    host_public_data_epochs: Literal[1, 5]
    supervised_loss_weight: Literal[0.9] = 0.9
    distillation_loss_weight: Literal[0.1] = 0.1
    loss_type: Literal["ce"] = "ce"
    temperature: Literal[1.0] = 1.0
    optimizer_step_count: int = Field(ge=1)
    optimizer_loss: float = Field(ge=0, allow_inf_nan=False)
    supervised_answer_loss: float = Field(ge=0, allow_inf_nan=False)
    distillation_answer_loss: float = Field(ge=0, allow_inf_nan=False)
    trainable_parameter_count: int = Field(gt=0)
    total_parameter_count: int = Field(gt=0)
    lora_tensors_changed: Literal[True] = True
    frozen_base_unchanged: Literal[True] = True
    reload_verified: Literal[True] = True
    dependency_versions: dict[str, str]
    created_at: str
    result_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> "HostCandidateTrainingResult":
        parse_utc(self.created_at)
        if self.candidate_adapter_version != self.parent_adapter_version + 1:
            raise ValueError("Host candidate version must follow its parent")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"result_hash"})
        )
        if self.result_hash != expected:
            raise ValueError("result_hash does not match Host candidate training")
        return self

    @classmethod
    def create(cls, **values: Any) -> "HostCandidateTrainingResult":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"result_hash"}
        )
        return cls(**payload, result_hash=sha256_hex(payload))


class HostCandidateValidationResult(ContractModel):
    schema_version: Literal["1.0"] = (
        HOST_CANDIDATE_VALIDATION_SCHEMA_VERSION
    )
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    candidate_result_hash: str = Field(pattern=HASH_PATTERN)
    previous_adapter_version: int = Field(ge=0)
    previous_adapter_hash: str = Field(pattern=HASH_PATTERN)
    candidate_adapter_version: int = Field(ge=1)
    candidate_adapter_hash: str = Field(pattern=HASH_PATTERN)
    accepted_adapter_version: int = Field(ge=0)
    accepted_adapter_hash: str = Field(pattern=HASH_PATTERN)
    baseline_validation_record_hash: str = Field(pattern=HASH_PATTERN)
    candidate_validation_record_hash: str = Field(pattern=HASH_PATTERN)
    primary_metric: Literal["macro_mean_answer_token_ce"] = (
        "macro_mean_answer_token_ce"
    )
    secondary_metric: Literal["token_weighted_answer_token_ce"] = (
        "token_weighted_answer_token_ce"
    )
    previous_macro_mean_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    candidate_macro_mean_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    previous_token_weighted_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    candidate_token_weighted_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    required_improvement: float = Field(ge=0, allow_inf_nan=False)
    observed_improvement: float = Field(allow_inf_nan=False)
    adapter_promoted: bool
    decision_reason: Literal[
        "candidate_improved",
        "insufficient_improvement",
        "forced_validation_rejection",
    ]
    rejected_candidate_discarded: bool
    created_at: str
    decision_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_decision(self) -> "HostCandidateValidationResult":
        parse_utc(self.created_at)
        if self.candidate_adapter_version != self.previous_adapter_version + 1:
            raise ValueError("Host validation candidate must follow its parent")
        improvement = (
            self.previous_macro_mean_answer_token_ce
            - self.candidate_macro_mean_answer_token_ce
        )
        if not math.isclose(
            self.observed_improvement,
            improvement,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Host validation improvement differs")
        if self.adapter_promoted:
            if self.decision_reason != "candidate_improved":
                raise ValueError("promoted Host candidate has another decision reason")
            if self.observed_improvement < self.required_improvement:
                raise ValueError("promoted Host candidate did not meet the threshold")
            if self.rejected_candidate_discarded:
                raise ValueError("promoted Host candidate cannot be discarded")
            if (
                self.accepted_adapter_version != self.candidate_adapter_version
                or self.accepted_adapter_hash != self.candidate_adapter_hash
            ):
                raise ValueError("promoted Host adapter identity differs")
        else:
            if self.decision_reason == "candidate_improved":
                raise ValueError("rejected Host candidate has a promotion reason")
            if (
                self.decision_reason == "insufficient_improvement"
                and self.observed_improvement >= self.required_improvement
            ):
                raise ValueError("rejected Host candidate met the threshold")
            if not self.rejected_candidate_discarded:
                raise ValueError("rejected Host candidate weights must be discarded")
            if (
                self.accepted_adapter_version != self.previous_adapter_version
                or self.accepted_adapter_hash != self.previous_adapter_hash
            ):
                raise ValueError("rejected Host candidate changed the active adapter")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"decision_hash"})
        )
        if self.decision_hash != expected:
            raise ValueError("decision_hash does not match Host validation")
        return self

    @classmethod
    def create(cls, **values: Any) -> "HostCandidateValidationResult":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"decision_hash"}
        )
        return cls(**payload, decision_hash=sha256_hex(payload))


class DistillationResult(ContractModel):
    round_id: str
    previous_adapter_version: int = Field(ge=0)
    candidate_adapter_version: int = Field(ge=0)
    accepted_adapter_version: int = Field(ge=0)
    previous_validation_loss: float = Field(ge=0)
    candidate_validation_loss: float = Field(ge=0)
    required_improvement: float = Field(ge=0)
    adapter_promoted: bool
    candidate_artifact_hash: str = Field(pattern=HASH_PATTERN)
    host_knowledge_package: KnowledgePackage


class RoundState(ContractModel):
    round_id: str
    state: Literal[
        "COLLECTING",
        "SEALED",
        "DISTILLING",
        "COMPLETED",
        "SKIPPED",
        "ABORTED",
    ]
    accepted_client_ids: list[str] = Field(default_factory=list)
    rejected_client_ids: list[str] = Field(default_factory=list)

    seen_nonces: list[str] = Field(default_factory=list)
    seen_package_hashes: list[str] = Field(default_factory=list)

    used_nonces: list[str] = Field(default_factory=list)
    submission_hashes: list[str] = Field(default_factory=list)

    host_nonces: list[str] = Field(default_factory=list)
    host_package_hashes: list[str] = Field(default_factory=list)

    sealed_client_ids: list[str] = Field(default_factory=list)
    host_adapter_before: int = Field(ge=0)
    host_adapter_after: int | None = Field(default=None, ge=0)
    adapter_promoted: bool | None = None
    message: str = ""
    updated_at: str


class SubmissionReceipt(ContractModel):
    round_id: str
    client_id: str
    package_hash: str = Field(pattern=HASH_PATTERN)
    state: str
    accepted_count: int = Field(ge=0)
    quorum: int = Field(ge=1)


class ServiceIdentity(ContractModel):
    service_id: str
    public_key: str = Field(pattern=BASE64_PATTERN)
    model_profile: ModelProfile | None = None
    adapter_version: int | None = Field(default=None, ge=0)
    host_public_key: str | None = Field(default=None, pattern=BASE64_PATTERN)
    host_service_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
