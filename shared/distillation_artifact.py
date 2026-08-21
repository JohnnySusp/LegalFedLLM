from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save as save_safetensors

from shared.crypto import sha256_hex
from shared.protocol import (
    HOST_TRAINING_ARTIFACT_SCHEMA_VERSION,
    KNOWLEDGE_ARTIFACT_FORMAT,
    HostTrainingArtifactDescriptor,
)

if TYPE_CHECKING:
    from shared.fedmkt_core.integration import PreparedDistillationBatch


TENSOR_NAMES = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "sparse_target_token_ids",
        "sparse_target_probabilities",
        "sparse_target_valid_mask",
    }
)
TENSOR_DTYPES = {
    "input_ids": np.dtype("int32"),
    "attention_mask": np.dtype("uint8"),
    "labels": np.dtype("int32"),
    "sparse_target_token_ids": np.dtype("int32"),
    "sparse_target_probabilities": np.dtype("float32"),
    "sparse_target_valid_mask": np.dtype("uint8"),
}


def _cpu_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(result)


def _artifact_tensors(batch: PreparedDistillationBatch) -> dict[str, np.ndarray]:
    values = batch.trainer_inputs()
    tensors = {
        name: _cpu_numpy(values[name], dtype)
        for name, dtype in TENSOR_DTYPES.items()
    }
    if not np.isfinite(tensors["sparse_target_probabilities"]).all():
        raise ValueError("Host training probabilities must be finite")
    if not np.isin(tensors["attention_mask"], (0, 1)).all():
        raise ValueError("Host training attention mask must be binary")
    if not np.isin(tensors["sparse_target_valid_mask"], (0, 1)).all():
        raise ValueError("Host training sparse mask must be binary")
    return tensors


def serialize_host_training_artifact(
    batch: PreparedDistillationBatch,
    *,
    pad_token_id: int,
) -> tuple[bytes, HostTrainingArtifactDescriptor]:
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Host padding token ID must be non-negative")
    return _serialize_host_training_artifact(
        _artifact_tensors(batch),
        batch=batch,
        pad_token_id=pad_token_id,
    )


def _serialize_host_training_artifact(
    tensors: dict[str, np.ndarray],
    *,
    batch: PreparedDistillationBatch,
    pad_token_id: int,
) -> tuple[bytes, HostTrainingArtifactDescriptor]:
    sample_count, sequence_length = tensors["input_ids"].shape
    top_k = tensors["sparse_target_token_ids"].shape[-1]
    artifact = save_safetensors(tensors)
    descriptor = HostTrainingArtifactDescriptor(
        format=KNOWLEDGE_ARTIFACT_FORMAT,
        schema_version=HOST_TRAINING_ARTIFACT_SCHEMA_VERSION,
        byte_size=len(artifact),
        sha256=sha256_hex(artifact),
        sample_count=sample_count,
        sample_ids_sha256=sha256_hex(list(batch.sample_ids)),
        padded_sequence_length=sequence_length,
        top_k=top_k,
        pad_token_id=pad_token_id,
        trainer_inputs_sha256=batch.audit.trainer_inputs_sha256,
    )
    return artifact, descriptor


def write_host_training_artifact(
    path: str | Path,
    batch: PreparedDistillationBatch,
    *,
    pad_token_id: int,
    maximum_bytes: int,
) -> HostTrainingArtifactDescriptor:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Host padding token ID must be non-negative")
    tensors = _artifact_tensors(batch)
    if sum(tensor.nbytes for tensor in tensors.values()) > maximum_bytes:
        raise ValueError("Host training tensors exceed maximum_bytes")
    artifact, descriptor = _serialize_host_training_artifact(
        tensors,
        batch=batch,
        pad_token_id=pad_token_id,
    )
    if descriptor.byte_size > maximum_bytes:
        raise ValueError("Host training artifact exceeds maximum_bytes")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        temporary.write_bytes(artifact)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return descriptor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tensor_schema(
    tensors: dict[str, np.ndarray],
    descriptor: HostTrainingArtifactDescriptor,
    vocabulary_size: int,
) -> None:
    if type(vocabulary_size) is not int or vocabulary_size < 1:
        raise ValueError("Host vocabulary size must be positive")
    if descriptor.pad_token_id >= vocabulary_size:
        raise ValueError("Host training padding token is outside the vocabulary")
    names = set(tensors)
    if names != TENSOR_NAMES:
        raise ValueError(
            "Host training artifact tensor names are invalid; "
            f"missing={sorted(TENSOR_NAMES - names)}, "
            f"unknown={sorted(names - TENSOR_NAMES)}"
        )
    for name, dtype in TENSOR_DTYPES.items():
        if tensors[name].dtype != dtype:
            raise ValueError(f"Host training tensor {name} must use {dtype.name}")
    sample_count = descriptor.sample_count
    sequence_length = descriptor.padded_sequence_length
    top_k = descriptor.top_k
    matrix_shape = (sample_count, sequence_length)
    sparse_shape = (sample_count, sequence_length, top_k)
    for name in ("input_ids", "attention_mask", "labels"):
        if tensors[name].shape != matrix_shape:
            raise ValueError(f"Host training tensor {name} has an invalid shape")
    for name in (
        "sparse_target_token_ids",
        "sparse_target_probabilities",
        "sparse_target_valid_mask",
    ):
        if tensors[name].shape != sparse_shape:
            raise ValueError(f"Host training tensor {name} has an invalid shape")
    if not np.isin(tensors["attention_mask"], (0, 1)).all():
        raise ValueError("Host training attention mask is not binary")
    if not np.isin(tensors["sparse_target_valid_mask"], (0, 1)).all():
        raise ValueError("Host training sparse mask is not binary")
    if np.any(tensors["input_ids"] < 0) or np.any(
        tensors["input_ids"] >= vocabulary_size
    ):
        raise ValueError("Host training input IDs are outside the vocabulary")
    labels = tensors["labels"]
    if np.any((labels != -100) & ((labels < 0) | (labels >= vocabulary_size))):
        raise ValueError("Host training labels are outside the vocabulary")
    if np.any(tensors["sparse_target_token_ids"] < 0) or np.any(
        tensors["sparse_target_token_ids"] >= vocabulary_size
    ):
        raise ValueError("Host training target IDs are outside the vocabulary")
    probabilities = tensors["sparse_target_probabilities"]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Host training probabilities are invalid")
    mask = tensors["sparse_target_valid_mask"].astype(bool)
    if np.any(probabilities[~mask] != 0):
        raise ValueError("masked Host training probabilities must be zero")
    if np.any(mask.sum(axis=-1) == 0):
        raise ValueError("every Host training target row needs one valid entry")
    row_sums = (probabilities * mask).sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-6):
        raise ValueError("Host training target rows must sum to one")


def _trainer_inputs_hash(
    tensors: dict[str, np.ndarray],
    sample_ids: Sequence[str],
    pad_token_id: int,
) -> str:
    attention_lengths = tensors["attention_mask"].sum(axis=1).tolist()
    input_ids = [
        tensors["input_ids"][index, :length].tolist()
        for index, length in enumerate(attention_lengths)
    ]
    labels = [
        tensors["labels"][index, :length].tolist()
        for index, length in enumerate(attention_lengths)
    ]
    return sha256_hex(
        {
            "sample_ids": list(sample_ids),
            "input_ids": input_ids,
            "attention_lengths": attention_lengths,
            "labels": labels,
            "pad_token_id": pad_token_id,
            "target_token_ids": tensors["sparse_target_token_ids"].tolist(),
            "target_probabilities": tensors[
                "sparse_target_probabilities"
            ].tolist(),
            "target_valid_mask": tensors[
                "sparse_target_valid_mask"
            ].astype(bool).tolist(),
        }
    )


def load_host_training_artifact(
    path: str | Path,
    descriptor: HostTrainingArtifactDescriptor,
    sample_ids: Sequence[str],
    *,
    maximum_bytes: int,
    vocabulary_size: int,
) -> dict[str, np.ndarray]:
    artifact_path = Path(path)
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    actual_size = artifact_path.stat().st_size
    if actual_size > maximum_bytes or descriptor.byte_size > maximum_bytes:
        raise ValueError("Host training artifact exceeds maximum_bytes")
    if actual_size != descriptor.byte_size:
        raise ValueError("Host training artifact byte size differs")
    if _sha256_file(artifact_path) != descriptor.sha256:
        raise ValueError("Host training artifact SHA-256 differs")
    values = list(sample_ids)
    if len(values) != descriptor.sample_count:
        raise ValueError("Host training artifact sample count differs")
    if sha256_hex(values) != descriptor.sample_ids_sha256:
        raise ValueError("Host training artifact sample order differs")
    try:
        with safe_open(artifact_path, framework="np") as artifact:
            if artifact.metadata():
                raise ValueError("Host training artifact metadata is not allowed")
            tensors = artifact.get_tensors()
    except (OSError, SafetensorError) as exc:
        raise ValueError("invalid safetensors Host training artifact") from exc
    _validate_tensor_schema(tensors, descriptor, vocabulary_size)
    actual_hash = _trainer_inputs_hash(
        tensors,
        values,
        descriptor.pad_token_id,
    )
    if actual_hash != descriptor.trainer_inputs_sha256:
        raise ValueError("Host training semantic tensor hash differs")
    return tensors
