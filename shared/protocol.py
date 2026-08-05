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

PROTOCOL_VERSION = "1.0"
PACKAGE_SCHEMA_VERSION = "1.0"
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
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    @field_validator("target_modules")
    @classmethod
    def clean_targets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in values)
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("target_modules must contain non-blank names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("target_modules must be unique")
        return cleaned


class OllamaProfile(ContractModel):
    model: str = Field(min_length=1, max_length=256)
    digest: str | None = Field(default=None, max_length=256)


class ModelProfile(ContractModel):
    profile_id: str = Field(min_length=1, max_length=128)
    role: Literal["client", "host"]
    model_id: str = Field(min_length=1, max_length=512)
    model_revision: str = Field(min_length=1, max_length=256)
    tokenizer_id: str = Field(min_length=1, max_length=512)
    tokenizer_revision: str = Field(min_length=1, max_length=256)
    tokenizer_class: str = Field(min_length=1, max_length=256)
    training_backend: Literal["mock", "transformers"] = "mock"
    serving_backend: Literal["mock", "ollama"] = "mock"
    prompt_template_hash: str = Field(pattern=HASH_PATTERN)
    lora: LoraProfile
    ollama: OllamaProfile | None = None

    @model_validator(mode="after")
    def validate_ollama(self) -> "ModelProfile":
        if self.serving_backend == "ollama" and self.ollama is None:
            raise ValueError("ollama profile is required for Ollama serving")
        return self


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
    strategy: Literal["mock_identity", "dtw", "greedy_dp"] = "mock_identity"
    profile_version: str = "1"


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
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    dp_policy: DifferentialPrivacyPolicy = Field(default_factory=DifferentialPrivacyPolicy)
    maximum_knowledge_package_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, le=2 * 1024 * 1024 * 1024
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
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    round_id: str = Field(min_length=1, max_length=128)
    selected_client_ids: list[str]
    trusted_client_quorum: int = Field(ge=1)
    current_host_adapter_version: int = Field(ge=0)
    host_model_profile: ModelProfile
    reference_dataset_id: str
    reference_dataset_hash: str = Field(pattern=HASH_PATTERN)
    sample_ids: list[str]
    prompt_template: str
    prompt_template_hash: str = Field(pattern=HASH_PATTERN)
    label_format: str
    maximum_sequence_length: int = Field(ge=2)
    truncation_policy: Literal["right", "left", "reject"]
    top_k: int = Field(ge=1)
    training_epochs: int = Field(ge=1)
    alignment: AlignmentConfig
    distillation: DistillationConfig
    dp_policy: DifferentialPrivacyPolicy
    maximum_knowledge_package_bytes: int = Field(ge=1024)
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
            "alignment": request.alignment.model_dump(mode="json"),
            "distillation": request.distillation.model_dump(mode="json"),
            "dp_policy": request.dp_policy.model_dump(mode="json"),
            "maximum_knowledge_package_bytes": request.maximum_knowledge_package_bytes,
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
    ce_loss: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_shapes(self) -> "KnowledgeSample":
        if self.attention_length > len(self.source_input_ids):
            raise ValueError("attention_length exceeds source_input_ids length")
        if len(self.top_k_token_ids) != len(self.top_k_logits):
            raise ValueError("top-k token and logit sequence lengths differ")
        if len(self.top_k_token_ids) != len(self.source_input_ids):
            raise ValueError("top-k sequence length must match source_input_ids")
        widths = {len(row) for row in self.top_k_token_ids}
        logit_widths = {len(row) for row in self.top_k_logits}
        if not widths or 0 in widths or widths != logit_widths or len(widths) != 1:
            raise ValueError("top-k rows must have one consistent non-zero width")
        if not math.isfinite(self.ce_loss):
            raise ValueError("ce_loss must be finite")
        if any(token < 0 for row in self.top_k_token_ids for token in row):
            raise ValueError("token IDs must be non-negative")
        if any(not math.isfinite(value) for row in self.top_k_logits for value in row):
            raise ValueError("logits must be finite")
        return self

    @property
    def top_k(self) -> int:
        return len(self.top_k_token_ids[0])


class KnowledgePackage(ContractModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    package_schema_version: Literal["1.0"] = PACKAGE_SCHEMA_VERSION
    round_id: str
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    sender_id: str = Field(min_length=1, max_length=128)
    sender_role: Literal["client", "host"]
    model_profile: ModelProfile
    adapter_version: int = Field(ge=0)
    alignment_profile_id: str = Field(min_length=1, max_length=256)
    reference_dataset_id: str
    reference_dataset_hash: str = Field(pattern=HASH_PATTERN)
    sample_ids: list[str]
    top_k: int = Field(ge=1)
    samples: list[KnowledgeSample] = Field(min_length=1)
    dp_report: DifferentialPrivacyReport = Field(
        default_factory=DifferentialPrivacyReport
    )
    nonce: str = Field(min_length=16, max_length=256)
    created_at: str
    artifact_sha256: str = Field(pattern=HASH_PATTERN)
    signature: str = Field(pattern=BASE64_PATTERN)

    @model_validator(mode="after")
    def validate_package(self) -> "KnowledgePackage":
        parse_utc(self.created_at)
        if self.model_profile.role != self.sender_role:
            raise ValueError("model profile role does not match sender_role")
        if self.sample_ids != [sample.sample_id for sample in self.samples]:
            raise ValueError("sample_ids must match package sample order")
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
        if any(sample.top_k != self.top_k for sample in self.samples):
            raise ValueError("sample top-k width does not match package top_k")
        expected = sha256_hex(self.hash_payload())
        if self.artifact_sha256 != expected:
            raise ValueError("artifact_sha256 does not match the package payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json", exclude={"artifact_sha256", "signature"}
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
        samples: list[KnowledgeSample],
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
            "sample_ids": [sample.sample_id for sample in samples],
            "top_k": top_k,
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "dp_report": (dp_report or DifferentialPrivacyReport()).model_dump(
                mode="json"
            ),
            "nonce": nonce or secrets.token_urlsafe(24),
            "created_at": created_at or utc_text(),
        }
        artifact_sha256 = sha256_hex(payload)
        signed_payload = {**payload, "artifact_sha256": artifact_sha256}
        return cls(**signed_payload, signature=identity.sign_json(signed_payload))


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


class ValidatedDistillationSample(ContractModel):
    sample_id: str
    teacher_id: str
    teacher_ce_loss: float = Field(ge=0)
    host_ce_loss: float = Field(ge=0)
    source_input_ids: list[int]
    attention_length: int = Field(ge=1)
    aligned_top_k_token_ids: list[list[int]]
    aligned_top_k_logits: list[list[float]]
    trust_weight: float = Field(gt=0, le=1)


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
