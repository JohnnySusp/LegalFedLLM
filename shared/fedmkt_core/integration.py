from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch
from pydantic import Field, model_validator

from shared.alignment_profiles import BidirectionalAlignmentProfile
from shared.crypto import sha256_hex
from shared.fedmkt_core.ml.sparse_targets import (
    SparseTargetBatch,
    build_sparse_target_batch,
)
from shared.fedmkt_core.ml.token_alignment import transform_step_logits
from shared.fedmkt_core.safety import (
    MINIMUM_TRUST_SCORE,
    finalize_aligned_safety_reports,
    is_eligible_for_distillation,
)
from shared.protocol import (
    HASH_PATTERN,
    ContractModel,
    KnowledgePackage,
    KnowledgeSample,
    SafetyReport,
    ValidatedDistillationDataset,
    ValidatedDistillationSample,
)
from shared.tokenizer_validation import ValidatedTokenizer
from shared.vocabulary_mapping import (
    UnaddressableTokenId,
    VocabularyMappingCache,
    VocabularyMappingCacheError,
    VocabularyMappingError,
    validate_addressable_token_ids,
)


INTEGRATION_AUDIT_SCHEMA_VERSION = "3.0"
CLIENT_TO_HOST_DIRECTION = "client_to_host"


class DistillationIntegrationError(ValueError):
    pass


class TrustedClientQuorumError(DistillationIntegrationError):
    pass


class RejectedClientAudit(ContractModel):
    client_id: str = Field(min_length=1, max_length=128)
    reasons: list[str] = Field(min_length=1)


class SourcePackageAudit(ContractModel):
    sender_id: str = Field(min_length=1, max_length=128)
    package_hash: str = Field(pattern=HASH_PATTERN)


class AlignmentMappingAudit(ContractModel):
    alignment_profile_id: str = Field(min_length=1, max_length=256)
    alignment_direction: Literal["client_to_host"] = CLIENT_TO_HOST_DIRECTION
    client_ids: list[str] = Field(min_length=1)
    mapping_identity_sha256: str = Field(pattern=HASH_PATTERN)
    mapping_payload_sha256: str = Field(pattern=HASH_PATTERN)


class SampleTeacherAudit(ContractModel):
    sample_id: str = Field(min_length=1, max_length=256)
    teacher_id: str = Field(min_length=1, max_length=128)
    teacher_ce_loss: float = Field(ge=0)
    host_ce_loss: float = Field(ge=0)
    trust_score: float = Field(gt=0, le=1)
    source_package_hash: str = Field(pattern=HASH_PATTERN)
    empty_aligned_row_fallback_count: int = Field(ge=0)


class DistillationIntegrationAudit(ContractModel):
    schema_version: Literal["3.0"] = INTEGRATION_AUDIT_SCHEMA_VERSION
    round_id: str = Field(min_length=1, max_length=128)
    dataset_hash: str = Field(pattern=HASH_PATTERN)
    alignments: list[AlignmentMappingAudit] = Field(min_length=1)
    accepted_client_ids: list[str]
    rejected_clients: list[RejectedClientAudit]
    safety_reports: dict[str, SafetyReport]
    source_packages: list[SourcePackageAudit] = Field(min_length=1)
    samples: list[SampleTeacherAudit] = Field(min_length=1)
    empty_aligned_row_fallback_count: int = Field(ge=0)
    trainer_inputs_sha256: str = Field(pattern=HASH_PATTERN)
    target_temperature: float = Field(gt=0)
    distillation_loss_type: Literal["ce", "kl"]
    audit_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_audit_hash(self) -> "DistillationIntegrationAudit":
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"audit_hash"})
        )
        if self.audit_hash != expected:
            raise ValueError("audit_hash does not match integration audit")
        return self

    @classmethod
    def create(cls, **values: object) -> "DistillationIntegrationAudit":
        payload = cls.model_construct(**values).model_dump(
            mode="json",
            exclude={"audit_hash"},
        )
        return cls(**payload, audit_hash=sha256_hex(payload))


@dataclass(frozen=True, slots=True)
class PreparedDistillationBatch:
    dataset: ValidatedDistillationDataset
    sample_ids: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sparse_targets: SparseTargetBatch
    audit: DistillationIntegrationAudit

    def trainer_inputs(self) -> dict[str, torch.Tensor]:
        """Return CPU tensors with explicit sparse-target field names."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "sparse_target_token_ids": self.sparse_targets.token_ids,
            "sparse_target_probabilities": self.sparse_targets.probabilities,
            "sparse_target_valid_mask": self.sparse_targets.valid_mask,
        }


Aligner = Callable[..., tuple[list[list[float]], list[list[int]]]]


def _alignment_contracts(
    *,
    profile: BidirectionalAlignmentProfile | None,
    client_tokenizer: ValidatedTokenizer | None,
    alignment_profiles: Mapping[str, BidirectionalAlignmentProfile] | None,
    client_tokenizers: Mapping[str, ValidatedTokenizer] | None,
) -> tuple[
    dict[str, BidirectionalAlignmentProfile],
    dict[str, ValidatedTokenizer],
]:
    legacy_values = profile is not None or client_tokenizer is not None
    heterogeneous_values = (
        alignment_profiles is not None or client_tokenizers is not None
    )
    if legacy_values and heterogeneous_values:
        raise DistillationIntegrationError(
            "provide either one legacy alignment profile/tokenizer pair or "
            "the per-profile mappings, not both"
        )
    if legacy_values:
        if profile is None or client_tokenizer is None:
            raise DistillationIntegrationError(
                "the alignment profile and Client tokenizer must be provided "
                "together"
            )
        profiles = {profile.profile_id: profile}
        tokenizers = {profile.profile_id: client_tokenizer}
    else:
        if not alignment_profiles or not client_tokenizers:
            raise DistillationIntegrationError(
                "per-profile alignment contracts and Client tokenizers are "
                "required"
            )
        profiles = dict(alignment_profiles)
        tokenizers = dict(client_tokenizers)
        if set(profiles) != set(tokenizers):
            raise DistillationIntegrationError(
                "alignment-profile and Client-tokenizer IDs differ"
            )

    for profile_id, value in profiles.items():
        if profile_id != value.profile_id:
            raise DistillationIntegrationError(
                "alignment-profile mapping key differs from its signed ID"
            )
        tokenizer = tokenizers[profile_id]
        if tokenizer.endpoint != value.client:
            raise DistillationIntegrationError(
                f"validated Client tokenizer differs from {profile_id!r}"
            )
        if value.strategy != "dtw":
            raise DistillationIntegrationError(
                f"alignment profile {profile_id!r} does not use DTW"
            )
        if value.client_to_host_owner != "coordinator":
            raise DistillationIntegrationError(
                f"alignment profile {profile_id!r} is not assigned to the "
                "Coordinator"
            )
    return profiles, tokenizers


def _package_hash(package: KnowledgePackage, label: str) -> str:
    value = getattr(package, "package_hash", None)
    if not isinstance(value, str) or len(value) != 64:
        raise DistillationIntegrationError(
            f"{label} does not carry a validated package hash"
        )
    return value


def _samples_by_id(
    samples: Sequence[KnowledgeSample],
    *,
    expected_ids: Sequence[str],
    label: str,
) -> dict[str, KnowledgeSample]:
    ids = [sample.sample_id for sample in samples]
    if ids != list(expected_ids):
        raise DistillationIntegrationError(
            f"{label} sample order differs from the signed package"
        )
    return {sample.sample_id: sample for sample in samples}


def _client_protocol_reasons(
    package: KnowledgePackage,
    samples: Sequence[KnowledgeSample] | None,
    *,
    host_package: KnowledgePackage,
    profile: BidirectionalAlignmentProfile,
) -> list[str]:
    reasons: list[str] = []
    expected = {
        "round_id": host_package.round_id,
        "manifest_hash": host_package.manifest_hash,
        "reference_dataset_id": host_package.reference_dataset_id,
        "reference_dataset_hash": host_package.reference_dataset_hash,
        "top_k": host_package.top_k,
    }
    for name, value in expected.items():
        if getattr(package, name, None) != value:
            reasons.append(f"{name} differs from the Host package")
    if package.alignment_profile_id != profile.profile_id:
        reasons.append("alignment_profile_id differs from the approved pair")
    if getattr(package, "sender_role", None) != "client":
        reasons.append("sender_role is not client")
    model_profile = getattr(package, "model_profile", None)
    if model_profile is None:
        reasons.append("model_profile is missing")
    else:
        mismatches = profile.client.mismatches(model_profile)
        if mismatches:
            reasons.append(
                "model_profile differs from the approved Client endpoint: "
                + ", ".join(mismatches)
            )
    if package.sample_ids != host_package.sample_ids:
        reasons.append("sample IDs differ from the Host package")
    if samples is None:
        reasons.append("validated knowledge samples are missing")
    elif [sample.sample_id for sample in samples] != package.sample_ids:
        reasons.append("loaded sample order differs from the signed package")
    elif any(sample.top_k != package.top_k for sample in samples):
        reasons.append("loaded top-k width differs from the signed package")
    elif any(
        sample.attention_length != len(sample.source_input_ids)
        for sample in samples
    ):
        reasons.append("Client alignment inputs contain padded token sequences")
    return reasons


def _demanded_ids(samples: Sequence[KnowledgeSample]) -> tuple[int, ...]:
    values: set[int] = set()
    for sample in samples:
        values.update(sample.source_input_ids)
        for row in sample.top_k_token_ids:
            values.update(row)
    return tuple(sorted(values))


def _rejection(
    rejected: dict[str, list[str]],
    client_id: str,
    reasons: Sequence[str],
) -> None:
    rejected[client_id] = list(dict.fromkeys(str(reason) for reason in reasons))


def _check_quorum(
    accepted_client_ids: Sequence[str],
    trusted_client_quorum: int,
    rejected: Mapping[str, Sequence[str]],
) -> None:
    if len(accepted_client_ids) < trusted_client_quorum:
        details = "; ".join(
            f"{client_id}: {', '.join(reasons)}"
            for client_id, reasons in rejected.items()
        )
        suffix = f" ({details})" if details else ""
        raise TrustedClientQuorumError(
            "trusted Client quorum is no longer satisfied after eligibility "
            f"and alignment checks: {len(accepted_client_ids)}/"
            f"{trusted_client_quorum}{suffix}"
        )


def _validate_host_contract(
    *,
    host_package: KnowledgePackage,
    host_samples: Sequence[KnowledgeSample],
    host_tokenizer: ValidatedTokenizer,
    profile: BidirectionalAlignmentProfile,
) -> dict[str, KnowledgeSample]:
    if profile.strategy != "dtw":
        raise DistillationIntegrationError("the integrated path requires DTW")
    if profile.client_to_host_owner != "coordinator":
        raise DistillationIntegrationError(
            "Client-to-Host alignment is not assigned to the Coordinator"
        )
    if host_tokenizer.endpoint != profile.host:
        raise DistillationIntegrationError(
            "validated Host tokenizer differs from the alignment profile"
        )
    if host_package.sender_role != "host":
        raise DistillationIntegrationError("Host package sender_role is not host")
    if host_package.alignment_profile_id != profile.profile_id:
        raise DistillationIntegrationError(
            "Host package uses another alignment profile"
        )
    mismatches = profile.host.mismatches(host_package.model_profile)
    if mismatches:
        raise DistillationIntegrationError(
            "Host package differs from the approved Host endpoint: "
            + ", ".join(mismatches)
        )
    _package_hash(host_package, "Host package")
    by_id = _samples_by_id(
        host_samples,
        expected_ids=host_package.sample_ids,
        label="Host",
    )
    if any(sample.top_k != host_package.top_k for sample in host_samples):
        raise DistillationIntegrationError(
            "Host sample top-k width differs from the signed package"
        )
    if any(
        sample.attention_length != len(sample.source_input_ids)
        for sample in host_samples
    ):
        raise DistillationIntegrationError(
            "Host alignment inputs must be unpadded token sequences"
        )
    return by_id


def _validate_aligned_rows(
    token_ids: Sequence[Sequence[int]],
    logits: Sequence[Sequence[float]],
    *,
    top_k: int,
    vocabulary_size: int,
) -> int:
    fallback_count = 0
    for ids, values in zip(token_ids, logits):
        if not ids and not values:
            fallback_count += 1
            continue
        if not ids or len(ids) != len(values):
            raise ValueError("aligned token IDs and logits differ in width")
        if len(ids) > top_k:
            raise ValueError("aligned row exceeds the signed top-k width")
        if any(
            type(token_id) is not int
            or token_id < 0
            or token_id >= vocabulary_size
            for token_id in ids
        ):
            raise ValueError("aligned row contains an invalid Host token ID")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("aligned row contains a non-finite logit")
    return fallback_count


def _pad_trainer_inputs(
    *,
    host_samples: Sequence[KnowledgeSample],
    labels_by_sample: Mapping[str, Sequence[int]],
    pad_token_id: int,
    vocabulary_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    expected_ids = [sample.sample_id for sample in host_samples]
    if set(labels_by_sample) != set(expected_ids):
        raise DistillationIntegrationError(
            "answer-only labels must match the Host sample IDs exactly"
        )
    maximum_length = max(len(sample.source_input_ids) for sample in host_samples)
    input_ids = torch.full(
        (len(host_samples), maximum_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    serialized_labels: list[list[int]] = []
    for index, sample in enumerate(host_samples):
        length = len(sample.source_input_ids)
        values = list(labels_by_sample[sample.sample_id])
        if len(values) != length:
            raise DistillationIntegrationError(
                f"labels for {sample.sample_id!r} differ from Host sequence length"
            )
        if any(
            type(value) is not int
            or (value != -100 and (value < 0 or value >= vocabulary_size))
            for value in values
        ):
            raise DistillationIntegrationError(
                f"labels for {sample.sample_id!r} contain an invalid token ID"
            )
        if any(
            token_id < 0 or token_id >= vocabulary_size
            for token_id in sample.source_input_ids
        ):
            raise DistillationIntegrationError(
                f"Host input IDs for {sample.sample_id!r} exceed the vocabulary"
            )
        input_ids[index, :length] = torch.tensor(
            sample.source_input_ids,
            dtype=torch.long,
        )
        attention_mask[index, : sample.attention_length] = 1
        labels[index, :length] = torch.tensor(values, dtype=torch.long)
        if not (
            labels[index, 1:length].ne(-100)
            & attention_mask[index, 1:length].bool()
        ).any():
            raise DistillationIntegrationError(
                f"labels for {sample.sample_id!r} have no supervised causal "
                "answer position"
            )
        serialized_labels.append(values)
    hash_payload: dict[str, object] = {
        "sample_ids": expected_ids,
        "input_ids": [sample.source_input_ids for sample in host_samples],
        "attention_lengths": [sample.attention_length for sample in host_samples],
        "labels": serialized_labels,
        "pad_token_id": pad_token_id,
    }
    return input_ids, attention_mask, labels, hash_payload


def integrate_distillation_round(
    *,
    profile: BidirectionalAlignmentProfile | None = None,
    client_tokenizer: ValidatedTokenizer | None = None,
    alignment_profiles: Mapping[
        str, BidirectionalAlignmentProfile
    ] | None = None,
    client_tokenizers: Mapping[str, ValidatedTokenizer] | None = None,
    host_tokenizer: ValidatedTokenizer,
    mapping_cache: VocabularyMappingCache,
    host_package: KnowledgePackage,
    host_samples: Sequence[KnowledgeSample],
    client_packages: Sequence[KnowledgePackage],
    client_samples: Mapping[str, Sequence[KnowledgeSample]],
    safety_reports: Mapping[str, SafetyReport],
    historical_reliability: Mapping[str, float] | None = None,
    selected_client_ids: Sequence[str],
    trusted_client_quorum: int,
    labels_by_sample: Mapping[str, Sequence[int]],
    temperature: float = 1.0,
    loss_type: Literal["ce", "kl"] = "ce",
    aligner: Aligner = transform_step_logits,
) -> PreparedDistillationBatch:
    """Build one deterministic Client-to-Host sparse distillation batch.

    Inputs are already authenticated protocol models and validated tokenizer
    artifacts. Client-originated validation or alignment failures reject that
    entire package; Host, tokenizer, profile, and cache failures abort.
    """

    if type(trusted_client_quorum) is not int or trusted_client_quorum < 1:
        raise DistillationIntegrationError("trusted Client quorum must be positive")
    if len(selected_client_ids) != len(set(selected_client_ids)):
        raise DistillationIntegrationError(
            "selected_client_ids must be unique and in signed order"
        )
    if trusted_client_quorum > len(selected_client_ids):
        raise DistillationIntegrationError(
            "trusted Client quorum exceeds the signed Client set"
        )
    profiles, tokenizers = _alignment_contracts(
        profile=profile,
        client_tokenizer=client_tokenizer,
        alignment_profiles=alignment_profiles,
        client_tokenizers=client_tokenizers,
    )
    host_endpoints = {value.host for value in profiles.values()}
    if len(host_endpoints) != 1:
        raise DistillationIntegrationError(
            "all Client alignment profiles must share one Host endpoint"
        )
    host_endpoint = next(iter(host_endpoints))
    if host_tokenizer.endpoint != host_endpoint:
        raise DistillationIntegrationError(
            "validated Host tokenizer differs from the alignment profiles"
        )
    try:
        host_profile = profiles[host_package.alignment_profile_id]
    except KeyError:
        raise DistillationIntegrationError(
            "Host package alignment profile is not approved for this round"
        ) from None
    host_by_id = _validate_host_contract(
        host_package=host_package,
        host_samples=host_samples,
        host_tokenizer=host_tokenizer,
        profile=host_profile,
    )
    package_by_id: dict[str, KnowledgePackage] = {}
    for package in client_packages:
        if package.sender_id in package_by_id:
            raise DistillationIntegrationError(
                f"duplicate Client package for {package.sender_id!r}"
            )
        package_by_id[package.sender_id] = package

    rejected: dict[str, list[str]] = {}
    eligible: list[str] = []
    client_profile_ids: dict[str, str] = {}
    for client_id in selected_client_ids:
        package = package_by_id.get(client_id)
        if package is None:
            _rejection(rejected, client_id, ["signed Client package is missing"])
            continue
        client_profile = profiles.get(package.alignment_profile_id)
        if client_profile is None:
            _rejection(
                rejected,
                client_id,
                ["alignment profile is not approved for this round"],
            )
            continue
        reasons = _client_protocol_reasons(
            package,
            client_samples.get(client_id),
            host_package=host_package,
            profile=client_profile,
        )
        report = safety_reports.get(client_id)
        if report is None:
            reasons.append("safety report is missing")
        elif not report.accepted:
            reasons.extend(report.reasons or ["hard protocol checks failed"])
        elif (
            report.probe_stage == "post_alignment"
            and report.trust_score < MINIMUM_TRUST_SCORE
        ):
            reasons.append(
                f"trust_score {report.trust_score} is below "
                f"{MINIMUM_TRUST_SCORE}"
            )
        if reasons:
            _rejection(rejected, client_id, reasons)
            continue
        if (
            report.probe_stage == "post_alignment"
            and not is_eligible_for_distillation(report)
        ):
            raise AssertionError("eligible report gate and reasons diverged")
        try:
            validate_addressable_token_ids(
                tokenizers[client_profile.profile_id],
                _demanded_ids(client_samples[client_id]),
            )
        except UnaddressableTokenId as exc:
            _rejection(rejected, client_id, [str(exc)])
            continue
        eligible.append(client_id)
        client_profile_ids[client_id] = client_profile.profile_id

    _check_quorum(eligible, trusted_client_quorum, rejected)

    aligned_by_client: dict[
        str,
        dict[str, tuple[list[list[int]], list[list[float]], int]],
    ] = {}
    mapping_resolutions = {}
    while True:
        ordered_profile_ids = list(
            dict.fromkeys(client_profile_ids[client_id] for client_id in eligible)
        )
        mapping_resolutions = {}
        for profile_id in ordered_profile_ids:
            requested_ids = sorted(
                {
                    token_id
                    for client_id in eligible
                    if client_profile_ids[client_id] == profile_id
                    for token_id in _demanded_ids(client_samples[client_id])
                }
            )
            try:
                mapping_resolutions[profile_id] = mapping_cache.resolve(
                    profile=profiles[profile_id],
                    direction=CLIENT_TO_HOST_DIRECTION,
                    source=tokenizers[profile_id],
                    target=host_tokenizer,
                    requested_token_ids=requested_ids,
                )
            except VocabularyMappingCacheError as exc:
                raise DistillationIntegrationError(
                    "persistent vocabulary-mapping cache failed validation"
                ) from exc
            except VocabularyMappingError as exc:
                raise DistillationIntegrationError(
                    "Client-to-Host vocabulary mapping failed"
                ) from exc

        aligned_by_client = {}
        failed: dict[str, str] = {}
        for client_id in eligible:
            profile_id = client_profile_ids[client_id]
            client_profile = profiles[profile_id]
            selected_tokenizer = tokenizers[profile_id]
            upstream_mapping = mapping_resolutions[
                profile_id
            ].mapping.as_upstream_token_mapping()
            per_sample: dict[
                str,
                tuple[list[list[int]], list[list[float]], int],
            ] = {}
            client_by_id = {
                sample.sample_id: sample
                for sample in client_samples[client_id]
            }
            try:
                for sample_id in host_package.sample_ids:
                    host_sample = host_by_id[sample_id]
                    client_sample = client_by_id[sample_id]
                    aligned_logits, aligned_ids = aligner(
                        base_model_tokenizer=host_tokenizer.tokenizer,
                        blending_model_tokenizer=selected_tokenizer.tokenizer,
                        base_model_vocab=host_tokenizer.tokenizer.get_vocab(),
                        base_model_input_ids=host_sample.source_input_ids,
                        blending_model_input_ids=client_sample.source_input_ids,
                        blending_model_per_step_logits=client_sample.top_k_logits,
                        blending_model_per_step_indices=client_sample.top_k_token_ids,
                        blending_to_base_mapping=upstream_mapping,
                        align_strategy=client_profile.strategy,
                        base_model_special_token=(
                            client_profile.host.word_boundary_marker
                        ),
                        blending_model_special_token=(
                            client_profile.client.word_boundary_marker
                        ),
                    )
                    if (
                        len(aligned_ids) != len(host_sample.source_input_ids)
                        or len(aligned_logits) != len(host_sample.source_input_ids)
                        or any(
                            len(ids) != len(logits)
                            for ids, logits in zip(aligned_ids, aligned_logits)
                        )
                    ):
                        raise ValueError(
                            "aligned rows differ from the Host sequence shape"
                        )
                    fallback_count = _validate_aligned_rows(
                        aligned_ids,
                        aligned_logits,
                        top_k=host_package.top_k,
                        vocabulary_size=host_endpoint.vocabulary_size,
                    )
                    per_sample[sample_id] = (
                        aligned_ids,
                        aligned_logits,
                        fallback_count,
                    )
            except Exception as exc:
                failed[client_id] = (
                    f"alignment failed ({type(exc).__name__})"
                )
                continue
            aligned_by_client[client_id] = per_sample
        if not failed:
            break
        for client_id in eligible:
            if client_id in failed:
                _rejection(rejected, client_id, [failed[client_id]])
        eligible = [client_id for client_id in eligible if client_id not in failed]
        _check_quorum(eligible, trusted_client_quorum, rejected)

    final_safety_reports = dict(safety_reports)
    if any(
        final_safety_reports[client_id].probe_stage == "pre_alignment"
        for client_id in eligible
    ):
        finalized = finalize_aligned_safety_reports(
            host_samples=host_samples,
            client_samples=client_samples,
            aligned_by_client=aligned_by_client,
            pre_alignment_reports={
                client_id: final_safety_reports[client_id]
                for client_id in eligible
            },
            selected_client_ids=eligible,
            historical_reliability=historical_reliability,
        )
        final_safety_reports.update(finalized)

    trust_rejected: list[str] = []
    for client_id in eligible:
        report = final_safety_reports[client_id]
        if not is_eligible_for_distillation(report):
            _rejection(
                rejected,
                client_id,
                report.reasons
                or [
                    f"trust_score {report.trust_score} is below "
                    f"{MINIMUM_TRUST_SCORE}"
                ],
            )
            trust_rejected.append(client_id)
    eligible = [
        client_id for client_id in eligible if client_id not in trust_rejected
    ]
    _check_quorum(eligible, trusted_client_quorum, rejected)

    selected_samples: list[ValidatedDistillationSample] = []
    sample_audits: list[SampleTeacherAudit] = []
    selected_ids: list[list[list[int]]] = []
    selected_logits: list[list[list[float]]] = []
    host_ids: list[list[list[int]]] = []
    host_logits: list[list[list[float]]] = []
    package_hashes = {
        host_package.sender_id: _package_hash(host_package, "Host package"),
        **{
            client_id: _package_hash(
                package_by_id[client_id],
                f"Client package {client_id!r}",
            )
            for client_id in eligible
        },
    }
    for sample_id in host_package.sample_ids:
        host_sample = host_by_id[sample_id]
        candidates: list[
            tuple[
                float,
                str,
                float,
                list[list[int]],
                list[list[float]],
                int,
            ]
        ] = [
            (
                host_sample.ce_loss,
                host_package.sender_id,
                1.0,
                host_sample.top_k_token_ids,
                host_sample.top_k_logits,
                0,
            )
        ]
        for client_id in eligible:
            sample = next(
                value
                for value in client_samples[client_id]
                if value.sample_id == sample_id
            )
            ids, logits, fallback_count = aligned_by_client[client_id][sample_id]
            candidates.append(
                (
                    sample.ce_loss,
                    client_id,
                    final_safety_reports[client_id].trust_score,
                    ids,
                    logits,
                    fallback_count,
                )
            )
        teacher_loss, teacher_id, trust_score, ids, logits, fallback_count = min(
            candidates,
            key=lambda value: value[0],
        )
        selected_samples.append(
            ValidatedDistillationSample(
                sample_id=sample_id,
                teacher_id=teacher_id,
                teacher_ce_loss=teacher_loss,
                host_ce_loss=host_sample.ce_loss,
                source_input_ids=host_sample.source_input_ids,
                attention_length=host_sample.attention_length,
                aligned_top_k_token_ids=ids,
                aligned_top_k_logits=logits,
                trust_score=trust_score,
            )
        )
        sample_audits.append(
            SampleTeacherAudit(
                sample_id=sample_id,
                teacher_id=teacher_id,
                teacher_ce_loss=teacher_loss,
                host_ce_loss=host_sample.ce_loss,
                trust_score=trust_score,
                source_package_hash=package_hashes[teacher_id],
                empty_aligned_row_fallback_count=fallback_count,
            )
        )
        selected_ids.append(ids)
        selected_logits.append(logits)
        host_ids.append(host_sample.top_k_token_ids)
        host_logits.append(host_sample.top_k_logits)

    dataset = ValidatedDistillationDataset.create(
        round_id=host_package.round_id,
        manifest_hash=host_package.manifest_hash,
        host_adapter_version=host_package.adapter_version,
        accepted_client_ids=list(eligible),
        samples=selected_samples,
    )
    pad_token_id = host_endpoint.pad_token_id
    if pad_token_id is None:
        raise DistillationIntegrationError(
            "approved Host tokenizer does not define a padding token ID"
        )
    input_ids, attention_mask, labels, trainer_hash_payload = _pad_trainer_inputs(
        host_samples=host_samples,
        labels_by_sample=labels_by_sample,
        pad_token_id=pad_token_id,
        vocabulary_size=host_endpoint.vocabulary_size,
    )
    sparse_targets = build_sparse_target_batch(
        selected_ids,
        selected_logits,
        max_length=input_ids.shape[1],
        top_k=host_package.top_k,
        vocab_size=host_endpoint.vocabulary_size,
        pad_token_id=pad_token_id,
        temperature=temperature,
        dtype=torch.float32,
        fallback_token_id_rows=host_ids,
        fallback_logit_rows=host_logits,
    )
    source_packages = [
        SourcePackageAudit(
            sender_id=host_package.sender_id,
            package_hash=package_hashes[host_package.sender_id],
        ),
        *[
            SourcePackageAudit(
                sender_id=client_id,
                package_hash=package_hashes[client_id],
            )
            for client_id in eligible
        ],
    ]
    rejected_audits = [
        RejectedClientAudit(client_id=client_id, reasons=rejected[client_id])
        for client_id in selected_client_ids
        if client_id in rejected
    ]
    trainer_inputs_sha256 = sha256_hex(
        {
            **trainer_hash_payload,
            "target_token_ids": sparse_targets.token_ids.tolist(),
            "target_probabilities": sparse_targets.probabilities.tolist(),
            "target_valid_mask": sparse_targets.valid_mask.tolist(),
        }
    )
    audit = DistillationIntegrationAudit.create(
        schema_version=INTEGRATION_AUDIT_SCHEMA_VERSION,
        round_id=host_package.round_id,
        dataset_hash=dataset.dataset_hash,
        alignments=[
            AlignmentMappingAudit(
                alignment_profile_id=profile_id,
                alignment_direction=CLIENT_TO_HOST_DIRECTION,
                client_ids=[
                    client_id
                    for client_id in eligible
                    if client_profile_ids[client_id] == profile_id
                ],
                mapping_identity_sha256=(
                    mapping_resolutions[profile_id].mapping.identity_sha256
                ),
                mapping_payload_sha256=(
                    mapping_resolutions[profile_id].mapping.payload_sha256
                ),
            )
            for profile_id in mapping_resolutions
            if any(
                client_profile_ids[client_id] == profile_id
                for client_id in eligible
            )
        ],
        accepted_client_ids=list(eligible),
        rejected_clients=rejected_audits,
        safety_reports={
            client_id: final_safety_reports[client_id]
            for client_id in selected_client_ids
            if client_id in final_safety_reports
        },
        source_packages=source_packages,
        samples=sample_audits,
        empty_aligned_row_fallback_count=sum(
            sample.empty_aligned_row_fallback_count for sample in sample_audits
        ),
        trainer_inputs_sha256=trainer_inputs_sha256,
        target_temperature=temperature,
        distillation_loss_type=loss_type,
    )
    return PreparedDistillationBatch(
        dataset=dataset,
        sample_ids=tuple(host_package.sample_ids),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sparse_targets=sparse_targets,
        audit=audit,
    )
