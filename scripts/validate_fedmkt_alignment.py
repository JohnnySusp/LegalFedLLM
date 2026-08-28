from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.knowledge import EncodedReferenceSample, encode_reference_samples
from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
    pinned_client_profile,
)
from host.model_profiles import pinned_host_profile
from shared.alignment_profiles import (
    GRANITE_IDENTITY_DTW_PROFILE,
    POC_DTW_PROFILE,
)
from shared.crypto import (
    Ed25519Identity,
    canonical_json_bytes,
    sha256_hex,
)
from shared.knowledge_artifact import (
    load_knowledge_artifact,
    serialize_knowledge_artifact,
)
from shared.protocol import KnowledgePackage, KnowledgeSample, SafetyReport, utc_text
from shared.reference_dataset import (
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
)
from shared.storage import JsonFileStore
from shared.tokenizer_validation import (
    ValidatedTokenizer,
    load_pinned_tokenizer,
)
from shared.vocabulary_mapping import VocabularyMappingCache


REPORT_SCHEMA_VERSION = "1.0"
EXPECTED_SAMPLE_COUNT = 565
EXPECTED_DATASET_HASH = (
    "5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231"
)
SELECTED_CLIENT_IDS = ("client-b", "client-a")
TRUST_SCORES = {"client-b": 0.5, "client-a": 1.0}
TRUSTED_CLIENT_QUORUM = 2
FIXED_PACKAGE_TIME = "2026-08-15T00:00:00Z"
DEFAULT_MAXIMUM_SEQUENCE_LENGTH = 4096
DEFAULT_TOP_K = 4
MAXIMUM_KNOWLEDGE_PACKAGE_BYTES = 25 * 1024 * 1024


TokenizerLoader = Callable[..., ValidatedTokenizer]


def _encoded_statistics(
    samples: Sequence[EncodedReferenceSample],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("encoded reference sample collection is empty")
    lengths = [len(sample.input_ids) for sample in samples]
    maximum = max(lengths)
    return {
        "sample_count": len(samples),
        "total_token_count": sum(lengths),
        "maximum_observed_tokens": maximum,
        "maximum_observed_sample_ids": [
            sample.sample_id
            for sample, length in zip(samples, lengths, strict=True)
            if length == maximum
        ],
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in ("torch", "transformers", "rapidfuzz", "safetensors", "numpy"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def _safe_token_pool(
    tokenizer: ValidatedTokenizer,
    *,
    top_k: int,
) -> list[int]:
    vocabulary = tokenizer.tokenizer.get_vocab()
    values = sorted(
        {
            token_id
            for token_id in vocabulary.values()
            if type(token_id) is int
            and 0 <= token_id < tokenizer.endpoint.vocabulary_size
        }
    )
    if len(values) < top_k:
        raise ValueError(
            f"tokenizer {tokenizer.endpoint.profile_id!r} has fewer than "
            f"{top_k} addressable token IDs"
        )
    return values[: top_k + 1]


def _knowledge_rows(
    encoded: EncodedReferenceSample,
    *,
    token_pool: Sequence[int],
    top_k: int,
    ce_loss: float,
) -> KnowledgeSample:
    token_rows: list[list[int]] = []
    logit_rows: list[list[float]] = []
    full_logsumexp: list[float] = []
    gold_token_ids = [-100] * len(encoded.input_ids)
    gold_token_logits = [0.0] * len(encoded.input_ids)
    gold_token_nll = [0.0] * len(encoded.input_ids)
    for position in range(len(encoded.input_ids) - 1):
        gold_token_ids[position] = encoded.labels[position + 1]

    for position, source_token_id in enumerate(encoded.input_ids):
        gold_token_id = gold_token_ids[position]
        candidates = [
            *([gold_token_id] if gold_token_id != -100 else []),
            source_token_id,
            *token_pool,
        ]
        selected: list[int] = []
        for token_id in candidates:
            if token_id not in selected:
                selected.append(token_id)
            if len(selected) == top_k:
                break
        if len(selected) != top_k:
            raise ValueError("could not construct a unique top-k row")
        token_rows.append(selected)
        offset = (position % 11) / 1000
        logits = [
            4.0 + offset,
            *[
                float(-rank) + offset
                for rank in range(top_k - 1)
            ],
        ]
        logit_rows.append(logits)
        if gold_token_id != -100:
            gold_logit = logits[0]
            gold_token_logits[position] = gold_logit
            gold_token_nll[position] = ce_loss
            full_logsumexp.append(gold_logit + ce_loss)
        else:
            maximum = max(logits)
            top_k_logsumexp = maximum + math.log(
                sum(math.exp(value - maximum) for value in logits)
            )
            full_logsumexp.append(top_k_logsumexp + 0.25)

    return KnowledgeSample(
        sample_id=encoded.sample_id,
        source_input_ids=encoded.input_ids,
        attention_length=len(encoded.input_ids),
        top_k_token_ids=token_rows,
        top_k_logits=logit_rows,
        full_logsumexp=full_logsumexp,
        gold_token_ids=gold_token_ids,
        gold_token_logits=gold_token_logits,
        gold_token_nll=gold_token_nll,
        ce_loss=ce_loss,
    )


def _losses(index: int) -> tuple[float, float, float]:
    mode = index % 4
    if mode == 0:
        return 0.10, 0.20, 0.30
    if mode == 1:
        return 0.20, 0.20, 0.20
    if mode == 2:
        return 0.30, 0.10, 0.10
    return 0.30, 0.20, 0.10


def _knowledge_samples(
    encoded: Sequence[EncodedReferenceSample],
    *,
    tokenizer: ValidatedTokenizer,
    top_k: int,
    loss_index: int,
) -> list[KnowledgeSample]:
    pool = _safe_token_pool(tokenizer, top_k=top_k)
    values: list[KnowledgeSample] = []
    for index, item in enumerate(encoded):
        values.append(
            _knowledge_rows(
                item,
                token_pool=pool,
                top_k=top_k,
                ce_loss=_losses(index)[loss_index],
            )
        )
    return values


def _signed_package(
    *,
    identity: Ed25519Identity,
    sender_id: str,
    sender_role: str,
    model_profile: Any,
    alignment_profile_id: str,
    manifest_hash: str,
    dataset_id: str,
    dataset_hash: str,
    sample_ids: list[str],
    artifact_descriptor: Any,
    top_k: int,
) -> KnowledgePackage:
    return KnowledgePackage.create_signed(
        identity=identity,
        round_id="fedmkt-alignment-validation",
        manifest_hash=manifest_hash,
        sender_id=sender_id,
        sender_role=sender_role,
        model_profile=model_profile,
        adapter_version=0,
        alignment_profile_id=alignment_profile_id,
        reference_dataset_id=dataset_id,
        reference_dataset_hash=dataset_hash,
        top_k=top_k,
        sample_ids=sample_ids,
        artifact=artifact_descriptor,
        nonce=f"fedmkt-alignment-validation-{sender_id}",
        created_at=FIXED_PACKAGE_TIME,
    )


def _serialize_and_reload(
    samples: Sequence[KnowledgeSample],
    *,
    artifact_path: Path,
) -> tuple[bytes, Any, list[KnowledgeSample]]:
    artifact, descriptor = serialize_knowledge_artifact(samples)
    if len(artifact) > MAXIMUM_KNOWLEDGE_PACKAGE_BYTES:
        raise ValueError(
            f"{artifact_path.stem} validation artifact is {len(artifact)} "
            "bytes and exceeds the 25 MiB bound"
        )
    artifact_path.write_bytes(artifact)
    loaded = load_knowledge_artifact(
        artifact_path,
        descriptor,
        [sample.sample_id for sample in samples],
        maximum_bytes=MAXIMUM_KNOWLEDGE_PACKAGE_BYTES,
    )
    return artifact, descriptor, loaded


def _tensor_bytes(batch: Any) -> int:
    tensors = batch.trainer_inputs().values()
    return sum(value.numel() * value.element_size() for value in tensors)


def _same_batch(first: Any, second: Any) -> bool:
    import torch

    return (
        first.dataset.dataset_hash == second.dataset.dataset_hash
        and first.audit.audit_hash == second.audit.audit_hash
        and first.audit.trainer_inputs_sha256
        == second.audit.trainer_inputs_sha256
        and all(
            torch.equal(first.trainer_inputs()[name], second.trainer_inputs()[name])
            for name in first.trainer_inputs()
        )
    )


def _load_validation_tokenizers(
    *,
    cache_dir: Path | None,
    local_files_only: bool,
    loader: TokenizerLoader,
) -> tuple[ValidatedTokenizer, ValidatedTokenizer, ValidatedTokenizer]:
    qwen = loader(
        POC_DTW_PROFILE.client,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    host = loader(
        POC_DTW_PROFILE.host,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if replace(
        GRANITE_IDENTITY_DTW_PROFILE.client,
        role=host.endpoint.role,
        profile_id=host.endpoint.profile_id,
    ) != host.endpoint:
        raise ValueError(
            "Granite Client and validation Host tokenizer contracts differ"
        )
    granite_client = ValidatedTokenizer(
        endpoint=GRANITE_IDENTITY_DTW_PROFILE.client,
        tokenizer=host.tokenizer,
        artifact_sha256=host.artifact_sha256,
        artifact_path=host.artifact_path,
    )
    return qwen, granite_client, host


def run_validation(
    *,
    reference_path: str | Path,
    output_path: str | Path,
    mapping_cache_dir: str | Path,
    identity_dir: str | Path,
    hf_cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    expected_dataset_hash: str = EXPECTED_DATASET_HASH,
    maximum_sequence_length: int = DEFAULT_MAXIMUM_SEQUENCE_LENGTH,
    top_k: int = DEFAULT_TOP_K,
    tokenizer_loader: TokenizerLoader = load_pinned_tokenizer,
) -> dict[str, Any]:
    from shared.fedmkt_core.integration import integrate_distillation_round

    if maximum_sequence_length < 1:
        raise ValueError("maximum_sequence_length must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    started = time.perf_counter()
    phase_times: dict[str, float] = {}
    reference = load_reference_jsonl(reference_path)
    identity = reference_dataset_identity(reference)
    if identity.sample_count != expected_sample_count:
        raise ValueError(
            f"expected {expected_sample_count} D^P samples, "
            f"found {identity.sample_count}"
        )
    if identity.dataset_hash != expected_dataset_hash:
        raise ValueError(
            "D^P semantic hash does not match the accepted frozen dataset"
        )
    sample_ids = [sample.sample_id for sample in reference]
    phase_times["dataset_load_seconds"] = time.perf_counter() - started

    phase_started = time.perf_counter()
    qwen_tokenizer, granite_client_tokenizer, host_tokenizer = (
        _load_validation_tokenizers(
            cache_dir=Path(hf_cache_dir) if hf_cache_dir is not None else None,
            local_files_only=local_files_only,
            loader=tokenizer_loader,
        )
    )
    phase_times["tokenizer_validation_seconds"] = (
        time.perf_counter() - phase_started
    )

    qwen_profile = pinned_client_profile(QWEN_PROFILE_ID)
    granite_client_profile = pinned_client_profile(
        GRANITE_3_3_2B_CLIENT_PROFILE_ID
    )
    host_profile = pinned_host_profile()
    phase_started = time.perf_counter()
    qwen_encoded = encode_reference_samples(
        reference,
        tokenizer=qwen_tokenizer.tokenizer,
        model_profile=qwen_profile,
        maximum_sequence_length=maximum_sequence_length,
        expected_sample_ids=sample_ids,
    )
    granite_encoded = encode_reference_samples(
        reference,
        tokenizer=host_tokenizer.tokenizer,
        model_profile=host_profile,
        maximum_sequence_length=maximum_sequence_length,
        expected_sample_ids=sample_ids,
    )
    host_samples = _knowledge_samples(
        granite_encoded,
        tokenizer=host_tokenizer,
        top_k=top_k,
        loss_index=0,
    )
    qwen_samples = _knowledge_samples(
        qwen_encoded,
        tokenizer=qwen_tokenizer,
        top_k=top_k,
        loss_index=1,
    )
    granite_samples = _knowledge_samples(
        granite_encoded,
        tokenizer=granite_client_tokenizer,
        top_k=top_k,
        loss_index=2,
    )
    phase_times["deterministic_package_generation_seconds"] = (
        time.perf_counter() - phase_started
    )

    identity_root = Path(identity_dir)
    identities = {
        sender_id: Ed25519Identity.load_or_create(
            identity_root / f"{sender_id}.pem"
        )
        for sender_id in ("host", *SELECTED_CLIENT_IDS)
    }
    manifest_hash = sha256_hex(
        {
            "round_id": "fedmkt-alignment-validation",
            "selected_client_ids": list(SELECTED_CLIENT_IDS),
            "trusted_client_quorum": TRUSTED_CLIENT_QUORUM,
            "trust_scores": TRUST_SCORES,
            "dataset": identity.model_dump(mode="json"),
            "host_profile_hash": host_profile.profile_hash(),
            "client_profile_hashes": {
                "client-b": qwen_profile.profile_hash(),
                "client-a": granite_client_profile.profile_hash(),
            },
            "client_alignment_profiles": {
                "client-b": POC_DTW_PROFILE.profile_id,
                "client-a": GRANITE_IDENTITY_DTW_PROFILE.profile_id,
            },
            "top_k": top_k,
            "maximum_sequence_length": maximum_sequence_length,
        }
    )

    phase_started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        package_inputs = {
            "host": (
                host_samples,
                host_profile,
                POC_DTW_PROFILE.profile_id,
                "host",
            ),
            "client-b": (
                qwen_samples,
                qwen_profile,
                POC_DTW_PROFILE.profile_id,
                "client",
            ),
            "client-a": (
                granite_samples,
                granite_client_profile,
                GRANITE_IDENTITY_DTW_PROFILE.profile_id,
                "client",
            ),
        }
        packages: dict[str, KnowledgePackage] = {}
        loaded_samples: dict[str, list[KnowledgeSample]] = {}
        artifact_measurements: dict[str, dict[str, Any]] = {}
        for sender_id, (
            samples,
            model_profile,
            alignment_profile_id,
            role,
        ) in package_inputs.items():
            artifact, descriptor, loaded = _serialize_and_reload(
                samples,
                artifact_path=temporary / f"{sender_id}.safetensors",
            )
            package = _signed_package(
                identity=identities[sender_id],
                sender_id=sender_id,
                sender_role=role,
                model_profile=model_profile,
                alignment_profile_id=alignment_profile_id,
                manifest_hash=manifest_hash,
                dataset_id=identity.dataset_id,
                dataset_hash=identity.dataset_hash,
                sample_ids=sample_ids,
                artifact_descriptor=descriptor,
                top_k=top_k,
            )
            if not package.verify_signature(identities[sender_id].public_key_b64):
                raise AssertionError(f"{sender_id} signature did not verify")
            package_json = canonical_json_bytes(package.model_dump(mode="json"))
            logical_package_bytes = len(package_json) + len(artifact)
            if logical_package_bytes > MAXIMUM_KNOWLEDGE_PACKAGE_BYTES:
                raise ValueError(
                    f"{sender_id} logical package exceeds the 25 MiB bound"
                )
            packages[sender_id] = package
            loaded_samples[sender_id] = loaded
            artifact_measurements[sender_id] = {
                "package_hash": package.package_hash,
                "artifact_sha256": descriptor.sha256,
                "sample_count": descriptor.sample_count,
                "total_token_count": descriptor.total_token_count,
                "top_k": descriptor.top_k,
                "package_json_bytes": len(package_json),
                "safetensors_bytes": len(artifact),
                "logical_package_bytes": logical_package_bytes,
                "public_key_sha256": sha256_hex(
                    identities[sender_id].public_key_b64
                ),
            }
        phase_times["artifact_round_trip_seconds"] = (
            time.perf_counter() - phase_started
        )

        integration_values = {
            "alignment_profiles": {
                POC_DTW_PROFILE.profile_id: POC_DTW_PROFILE,
                GRANITE_IDENTITY_DTW_PROFILE.profile_id: (
                    GRANITE_IDENTITY_DTW_PROFILE
                ),
            },
            "client_tokenizers": {
                POC_DTW_PROFILE.profile_id: qwen_tokenizer,
                GRANITE_IDENTITY_DTW_PROFILE.profile_id: (
                    granite_client_tokenizer
                ),
            },
            "host_tokenizer": host_tokenizer,
            "mapping_cache": VocabularyMappingCache(mapping_cache_dir),
            "host_package": packages["host"],
            "host_samples": loaded_samples["host"],
            "client_packages": [
                packages[client_id] for client_id in SELECTED_CLIENT_IDS
            ],
            "client_samples": {
                client_id: loaded_samples[client_id]
                for client_id in SELECTED_CLIENT_IDS
            },
            "safety_reports": {
                client_id: SafetyReport(
                    accepted=True,
                    trust_score=TRUST_SCORES[client_id],
                )
                for client_id in SELECTED_CLIENT_IDS
            },
            "selected_client_ids": list(SELECTED_CLIENT_IDS),
            "trusted_client_quorum": TRUSTED_CLIENT_QUORUM,
            "labels_by_sample": {
                item.sample_id: item.labels for item in granite_encoded
            },
            "temperature": 1.0,
            "loss_type": "ce",
        }
        phase_started = time.perf_counter()
        first = integrate_distillation_round(**integration_values)
        phase_times["first_integration_seconds"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        second = integrate_distillation_round(**integration_values)
        phase_times["second_integration_seconds"] = (
            time.perf_counter() - phase_started
        )

    if not _same_batch(first, second):
        raise AssertionError("the two validation passes are not deterministic")
    if first.dataset.accepted_client_ids != list(SELECTED_CLIENT_IDS):
        raise AssertionError("inclusive trust threshold did not preserve quorum")
    selected_counts = Counter(
        sample.teacher_id for sample in first.dataset.samples
    )
    expected_counts = {
        "host": (identity.sample_count + 1) // 2,
        "client-b": identity.sample_count // 4,
        "client-a": identity.sample_count // 4,
    }
    if dict(selected_counts) != expected_counts:
        raise AssertionError(
            f"unexpected deterministic teacher counts: {dict(selected_counts)}"
        )
    if first.dataset.samples[2].teacher_id != "client-b":
        raise AssertionError("signed Client order did not resolve the tie")
    if first.dataset.samples[2].trust_score != 0.5:
        raise AssertionError("trust score was used as a teacher-selection weight")

    mapping_records = []
    for alignment in first.audit.alignments:
        profile = {
            POC_DTW_PROFILE.profile_id: POC_DTW_PROFILE,
            GRANITE_IDENTITY_DTW_PROFILE.profile_id: (
                GRANITE_IDENTITY_DTW_PROFILE
            ),
        }[alignment.alignment_profile_id]
        cache_path = (
            Path(mapping_cache_dir)
            / profile.profile_version
            / "client_to_host"
            / f"{alignment.mapping_identity_sha256}.json"
        )
        mapping_records.append(
            {
                **alignment.model_dump(mode="json"),
                "cache_path": str(cache_path),
                "cache_file_bytes": cache_path.stat().st_size,
                "cache_file_sha256": sha256_hex(cache_path.read_bytes()),
            }
        )

    total_seconds = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "generated_at": utc_text(),
        "repository": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "dependencies": _dependency_versions(),
        },
        "dataset": identity.model_dump(mode="json"),
        "round": {
            "manifest_hash": manifest_hash,
            "host": {
                "sender_id": "host",
                "status": "temporary_validation_endpoint",
                "model_profile": host_profile.model_dump(mode="json"),
            },
            "selected_client_ids": list(SELECTED_CLIENT_IDS),
            "trusted_client_quorum": TRUSTED_CLIENT_QUORUM,
            "clients": [
                {
                    "client_id": "client-b",
                    "trust_score": TRUST_SCORES["client-b"],
                    "hard_protocol_checks_passed": True,
                    "model_profile": qwen_profile.model_dump(mode="json"),
                    "alignment_profile_id": POC_DTW_PROFILE.profile_id,
                },
                {
                    "client_id": "client-a",
                    "trust_score": TRUST_SCORES["client-a"],
                    "hard_protocol_checks_passed": True,
                    "model_profile": granite_client_profile.model_dump(
                        mode="json"
                    ),
                    "alignment_profile_id": (
                        GRANITE_IDENTITY_DTW_PROFILE.profile_id
                    ),
                },
            ],
            "trust_policy": {
                "minimum_score": 0.5,
                "threshold_is_inclusive": True,
                "trust_is_teacher_selection_weight": False,
            },
        },
        "tokenization": {
            "maximum_sequence_length": maximum_sequence_length,
            "overlength_policy": "reject_without_truncation",
            "qwen_client": _encoded_statistics(qwen_encoded),
            "granite_client_and_validation_host": _encoded_statistics(
                granite_encoded
            ),
        },
        "knowledge_artifact_policy": {
            "top_k": top_k,
            "maximum_logical_package_bytes": (
                MAXIMUM_KNOWLEDGE_PACKAGE_BYTES
            ),
            "position_retention": "all_non_padding_source_positions",
            "logit_source": "deterministic_validation_fixture",
        },
        "packages": artifact_measurements,
        "alignment_mappings": mapping_records,
        "result": {
            "accepted_client_ids": first.dataset.accepted_client_ids,
            "rejected_clients": [
                value.model_dump(mode="json")
                for value in first.audit.rejected_clients
            ],
            "selected_source_counts": dict(selected_counts),
            "empty_aligned_row_fallback_count": (
                first.audit.empty_aligned_row_fallback_count
            ),
            "dataset_hash": first.dataset.dataset_hash,
            "trainer_inputs_sha256": first.audit.trainer_inputs_sha256,
            "integration_audit_hash": first.audit.audit_hash,
            "sparse_and_trainer_tensor_bytes": _tensor_bytes(first),
        },
        "determinism": {
            "passes": 2,
            "identical": True,
            "dataset_hash": first.dataset.dataset_hash,
            "trainer_inputs_sha256": first.audit.trainer_inputs_sha256,
            "integration_audit_hash": first.audit.audit_hash,
        },
        "resources": {
            "execution_device": "cpu",
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "wall_time_seconds": total_seconds,
            "phase_wall_times": phase_times,
            "vram": {
                "status": "not_applicable",
                "peak_bytes": None,
                "model_forward_passes": 0,
                "cuda_tensors_created": False,
                "reason": (
                    "The validation performs tokenizer encoding, vocabulary "
                    "mapping, DTW, selection and sparse-target construction "
                    "without loading model weights or allocating CUDA tensors."
                ),
            },
        },
    }
    output = Path(output_path).resolve()
    JsonFileStore(output.parent).write_json(output.name, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate heterogeneous FedMKT alignment over the frozen D^P and "
            "write a machine-readable local report."
        )
    )
    parser.add_argument(
        "--reference",
        default="data/derived/gld2012/reference.jsonl",
        help="canonical frozen D^P JSONL path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="explicit JSON report path, normally under ignored artifacts/",
    )
    parser.add_argument(
        "--mapping-cache",
        default="artifacts/fedmkt-alignment-cache",
        help="persistent vocabulary-mapping cache directory",
    )
    parser.add_argument(
        "--identity-dir",
        default="artifacts/fedmkt-validation-identities",
        help="local validation-only Ed25519 identity directory",
    )
    parser.add_argument(
        "--hf-cache",
        default=None,
        help="optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="refuse tokenizer downloads and use the supplied cache only",
    )
    parser.add_argument(
        "--maximum-sequence-length",
        type=int,
        default=DEFAULT_MAXIMUM_SEQUENCE_LENGTH,
        help=(
            "reject a reference sample above this token count without "
            f"truncation (default: {DEFAULT_MAXIMUM_SEQUENCE_LENGTH})"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "top-k width for each deterministic validation package "
            f"(default: {DEFAULT_TOP_K})"
        ),
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_validation(
        reference_path=arguments.reference,
        output_path=arguments.output,
        mapping_cache_dir=arguments.mapping_cache,
        identity_dir=arguments.identity_dir,
        hf_cache_dir=arguments.hf_cache,
        local_files_only=arguments.local_files_only,
        maximum_sequence_length=arguments.maximum_sequence_length,
        top_k=arguments.top_k,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
