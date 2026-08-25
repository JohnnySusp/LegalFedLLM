from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field, model_validator

from shared.alignment_profiles import BidirectionalAlignmentProfile
from shared.crypto import sha256_hex
from shared.fedmkt_core.ml.sparse_targets import (
    SparseTargetBatch,
    build_sparse_target_batch,
)
from shared.protocol import (
    HASH_PATTERN,
    ContractModel,
    KnowledgePackage,
    KnowledgeSample,
)
from shared.tokenizer_validation import ValidatedTokenizer
from shared.vocabulary_mapping import VocabularyMappingCache

REVERSE_INTEGRATION_AUDIT_SCHEMA_VERSION = "1.0"
HOST_TO_CLIENT_DIRECTION = "host_to_client"


class ReverseDistillationIntegrationError(ValueError):
    pass


class ReverseAlignmentAudit(ContractModel):
    alignment_profile_id: str = Field(min_length=1)
    alignment_direction: Literal["host_to_client"] = HOST_TO_CLIENT_DIRECTION
    mapping_identity_sha256: str = Field(pattern=HASH_PATTERN)
    mapping_payload_sha256: str = Field(pattern=HASH_PATTERN)


class ReverseSampleAudit(ContractModel):
    sample_id: str = Field(min_length=1)
    teacher_role: Literal["host", "client"]
    host_ce_loss: float = Field(ge=0)
    client_ce_loss: float = Field(ge=0)
    empty_aligned_row_fallback_count: int = Field(ge=0)


class ReverseDistillationIntegrationAudit(ContractModel):
    schema_version: Literal["1.0"] = REVERSE_INTEGRATION_AUDIT_SCHEMA_VERSION
    round_id: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    client_id: str = Field(min_length=1)
    parent_adapter_version: int = Field(ge=0)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    accepted_host_adapter_version: int = Field(ge=0)
    host_package_hash: str = Field(pattern=HASH_PATTERN)
    client_package_hash: str = Field(pattern=HASH_PATTERN)
    host_adapter_promoted: bool
    partition_hash: str = Field(pattern=HASH_PATTERN)
    alignment: ReverseAlignmentAudit | None = None
    samples: list[ReverseSampleAudit] = Field(min_length=1)
    host_teacher_sample_ids: list[str]
    trainer_inputs_sha256: str = Field(pattern=HASH_PATTERN)
    target_temperature: Literal[1.0] = 1.0
    distillation_loss_type: Literal["ce"] = "ce"
    audit_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> ReverseDistillationIntegrationAudit:
        expected = sha256_hex(self.model_dump(mode="json", exclude={"audit_hash"}))
        if self.audit_hash != expected:
            raise ValueError("reverse integration audit hash differs")
        return self

    @classmethod
    def create(cls, **values: object) -> ReverseDistillationIntegrationAudit:
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"audit_hash"}
        )
        return cls(**payload, audit_hash=sha256_hex(payload))


@dataclass(frozen=True, slots=True)
class PreparedReverseDistillationBatch:
    sample_ids: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sparse_targets: SparseTargetBatch
    audit: ReverseDistillationIntegrationAudit

    def trainer_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "sparse_target_token_ids": self.sparse_targets.token_ids,
            "sparse_target_probabilities": self.sparse_targets.probabilities,
            "sparse_target_valid_mask": self.sparse_targets.valid_mask,
        }


def _by_id(
    samples: Sequence[KnowledgeSample], expected_ids: Sequence[str], label: str
) -> dict[str, KnowledgeSample]:
    if [sample.sample_id for sample in samples] != list(expected_ids):
        raise ReverseDistillationIntegrationError(f"{label} sample order differs")
    return {sample.sample_id: sample for sample in samples}


def _pad_client_inputs(
    samples: Sequence[KnowledgeSample],
    labels_by_sample: Mapping[str, Sequence[int]],
    *,
    pad_token_id: int,
    vocabulary_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    maximum = max(len(sample.source_input_ids) for sample in samples)
    input_ids = torch.full((len(samples), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    serialized_labels: list[list[int]] = []
    for index, sample in enumerate(samples):
        length = len(sample.source_input_ids)
        values = list(labels_by_sample.get(sample.sample_id, ()))
        if len(values) != length:
            raise ReverseDistillationIntegrationError(
                f"labels for {sample.sample_id!r} differ from Client sequence length"
            )
        if any(value != -100 and not 0 <= value < vocabulary_size for value in values):
            raise ReverseDistillationIntegrationError("Client labels exceed vocabulary")
        input_ids[index, :length] = torch.tensor(sample.source_input_ids)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(values)
        if not (labels[index, 1:length].ne(-100)).any():
            raise ReverseDistillationIntegrationError(
                f"labels for {sample.sample_id!r} have no answer-only causal position"
            )
        serialized_labels.append(values)
    return (
        input_ids,
        attention_mask,
        labels,
        {
            "sample_ids": [sample.sample_id for sample in samples],
            "input_ids": [sample.source_input_ids for sample in samples],
            "attention_lengths": [sample.attention_length for sample in samples],
            "labels": serialized_labels,
            "pad_token_id": pad_token_id,
        },
    )


def integrate_reverse_distillation(
    *,
    client_id: str,
    parent_adapter_version: int,
    parent_adapter_hash: str,
    host_adapter_promoted: bool,
    partition_hash: str,
    transfer_sample_ids: Sequence[str],
    host_package: KnowledgePackage,
    host_samples: Sequence[KnowledgeSample],
    client_package: KnowledgePackage,
    client_samples: Sequence[KnowledgeSample],
    labels_by_sample: Mapping[str, Sequence[int]],
    profile: BidirectionalAlignmentProfile | None = None,
    host_tokenizer: ValidatedTokenizer | None = None,
    client_tokenizer: ValidatedTokenizer | None = None,
    mapping_cache: VocabularyMappingCache | None = None,
) -> PreparedReverseDistillationBatch:
    """Create the immutable Step 6.1 Client reverse-distillation tensors."""

    expected = host_package.sample_ids
    if client_package.sample_ids != expected:
        raise ReverseDistillationIntegrationError(
            "Host and Client package order differs"
        )
    if host_package.round_id != client_package.round_id or (
        host_package.manifest_hash != client_package.manifest_hash
    ):
        raise ReverseDistillationIntegrationError(
            "Host and Client package binding differs"
        )
    host_by_id = _by_id(host_samples, expected, "Host")
    client_by_id = _by_id(client_samples, expected, "Client")
    transfer = list(transfer_sample_ids)
    if not transfer or not set(transfer).issubset(expected):
        raise ReverseDistillationIntegrationError("invalid Client transfer split")
    if transfer != [sample_id for sample_id in expected if sample_id in set(transfer)]:
        raise ReverseDistillationIntegrationError("Client transfer split changed order")

    alignment_audit = None
    aligned_host: dict[str, tuple[list[list[int]], list[list[float]], int]] = {}
    if profile is None:
        # The deterministic mock profile shares its synthetic token IDs.
        for sample_id in transfer:
            host_sample = host_by_id[sample_id]
            client_sample = client_by_id[sample_id]
            if len(host_sample.top_k_token_ids) != len(client_sample.source_input_ids):
                raise ReverseDistillationIntegrationError(
                    "mock token sequence shapes differ"
                )
            aligned_host[sample_id] = (
                host_sample.top_k_token_ids,
                host_sample.top_k_logits,
                0,
            )
        vocabulary_size = (
            max(
                token_id
                for sample_id in transfer
                for sample in (client_by_id[sample_id], host_by_id[sample_id])
                for row in ([sample.source_input_ids] + sample.top_k_token_ids)
                for token_id in row
            )
            + 1
        )
        pad_token_id = 0
    else:
        from shared.fedmkt_core.ml.token_alignment import transform_step_logits

        if profile.host_to_client_owner != "client" or profile.strategy != "dtw":
            raise ReverseDistillationIntegrationError(
                "Host-to-Client DTW is not Client-owned"
            )
        if not host_tokenizer or not client_tokenizer or not mapping_cache:
            raise ReverseDistillationIntegrationError(
                "validated tokenizers and mapping cache are required"
            )
        if (
            host_tokenizer.endpoint != profile.host
            or client_tokenizer.endpoint != profile.client
        ):
            raise ReverseDistillationIntegrationError(
                "validated tokenizer differs from alignment profile"
            )
        host_mismatches = profile.host.mismatches(host_package.model_profile)
        client_mismatches = profile.client.mismatches(client_package.model_profile)
        if host_mismatches or client_mismatches:
            raise ReverseDistillationIntegrationError(
                "Knowledge Package model profile differs from alignment profile"
            )
        requested = sorted(
            {
                token_id
                for sample_id in transfer
                for sample in (host_by_id[sample_id],)
                for row in ([sample.source_input_ids] + sample.top_k_token_ids)
                for token_id in row
            }
        )
        resolution = mapping_cache.resolve(
            profile=profile,
            direction=HOST_TO_CLIENT_DIRECTION,
            source=host_tokenizer,
            target=client_tokenizer,
            requested_token_ids=requested,
        )
        mapping = resolution.mapping.as_upstream_token_mapping()
        for sample_id in transfer:
            host_sample = host_by_id[sample_id]
            client_sample = client_by_id[sample_id]
            logits, token_ids = transform_step_logits(
                base_model_tokenizer=client_tokenizer.tokenizer,
                blending_model_tokenizer=host_tokenizer.tokenizer,
                base_model_vocab=client_tokenizer.tokenizer.get_vocab(),
                base_model_input_ids=client_sample.source_input_ids,
                blending_model_input_ids=host_sample.source_input_ids,
                blending_model_per_step_logits=host_sample.top_k_logits,
                blending_model_per_step_indices=host_sample.top_k_token_ids,
                blending_to_base_mapping=mapping,
                align_strategy=profile.strategy,
                base_model_special_token=profile.client.word_boundary_marker,
                blending_model_special_token=profile.host.word_boundary_marker,
            )
            if len(token_ids) != len(client_sample.source_input_ids):
                raise ReverseDistillationIntegrationError(
                    "aligned Host rows differ from Client shape"
                )
            fallback_count = sum(1 for row in token_ids if not row)
            aligned_host[sample_id] = (token_ids, logits, fallback_count)
        alignment_audit = ReverseAlignmentAudit(
            alignment_profile_id=profile.profile_id,
            mapping_identity_sha256=resolution.mapping.identity_sha256,
            mapping_payload_sha256=resolution.mapping.payload_sha256,
        )
        vocabulary_size = profile.client.vocabulary_size
        if profile.client.pad_token_id is None:
            raise ReverseDistillationIntegrationError("Client profile has no pad token")
        pad_token_id = profile.client.pad_token_id

    selected_ids: list[list[list[int]]] = []
    selected_logits: list[list[list[float]]] = []
    fallback_ids: list[list[list[int]]] = []
    fallback_logits: list[list[list[float]]] = []
    audits: list[ReverseSampleAudit] = []
    host_teacher_ids: list[str] = []
    selected_client_samples = [client_by_id[sample_id] for sample_id in transfer]
    for sample_id in transfer:
        host = host_by_id[sample_id]
        client = client_by_id[sample_id]
        use_host = host.ce_loss < client.ce_loss
        host_ids, host_logits, fallback_count = aligned_host[sample_id]
        selected_ids.append(host_ids if use_host else client.top_k_token_ids)
        selected_logits.append(host_logits if use_host else client.top_k_logits)
        fallback_ids.append(client.top_k_token_ids)
        fallback_logits.append(client.top_k_logits)
        if use_host:
            host_teacher_ids.append(sample_id)
        audits.append(
            ReverseSampleAudit(
                sample_id=sample_id,
                teacher_role="host" if use_host else "client",
                host_ce_loss=host.ce_loss,
                client_ce_loss=client.ce_loss,
                empty_aligned_row_fallback_count=fallback_count if use_host else 0,
            )
        )
    input_ids, attention_mask, labels, hash_payload = _pad_client_inputs(
        selected_client_samples,
        labels_by_sample,
        pad_token_id=pad_token_id,
        vocabulary_size=vocabulary_size,
    )
    sparse_targets = build_sparse_target_batch(
        selected_ids,
        selected_logits,
        max_length=input_ids.shape[1],
        top_k=host_package.top_k,
        vocab_size=vocabulary_size,
        pad_token_id=pad_token_id,
        temperature=1.0,
        dtype=torch.float32,
        fallback_token_id_rows=fallback_ids,
        fallback_logit_rows=fallback_logits,
    )
    trainer_hash = sha256_hex(
        {
            **hash_payload,
            "target_token_ids": sparse_targets.token_ids.tolist(),
            "target_probabilities": sparse_targets.probabilities.tolist(),
            "target_valid_mask": sparse_targets.valid_mask.tolist(),
        }
    )
    audit = ReverseDistillationIntegrationAudit.create(
        round_id=host_package.round_id,
        manifest_hash=host_package.manifest_hash,
        client_id=client_id,
        parent_adapter_version=parent_adapter_version,
        parent_adapter_hash=parent_adapter_hash,
        accepted_host_adapter_version=host_package.adapter_version,
        host_package_hash=host_package.package_hash,
        client_package_hash=client_package.package_hash,
        host_adapter_promoted=host_adapter_promoted,
        partition_hash=partition_hash,
        alignment=alignment_audit,
        samples=audits,
        host_teacher_sample_ids=host_teacher_ids,
        trainer_inputs_sha256=trainer_hash,
        target_temperature=1.0,
        distillation_loss_type="ce",
    )
    return PreparedReverseDistillationBatch(
        sample_ids=tuple(transfer),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sparse_targets=sparse_targets,
        audit=audit,
    )
