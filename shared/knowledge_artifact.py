from __future__ import annotations

import hashlib
import operator
import os
import secrets
from pathlib import Path
from typing import Sequence

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save as save_safetensors

from shared.crypto import sha256_hex
from shared.protocol import (
    KNOWLEDGE_ARTIFACT_FORMAT,
    KNOWLEDGE_ARTIFACT_SCHEMA_VERSION,
    KnowledgeArtifactDescriptor,
    KnowledgeSample,
)

TENSOR_NAMES = frozenset(
    {
        "sample_offsets",
        "source_input_ids",
        "attention_lengths",
        "top_k_token_ids",
        "top_k_logits",
        "ce_losses",
    }
)

TENSOR_DTYPES = {
    "sample_offsets": np.dtype("int64"),
    "source_input_ids": np.dtype("int32"),
    "attention_lengths": np.dtype("int32"),
    "top_k_token_ids": np.dtype("int32"),
    "top_k_logits": np.dtype("float32"),
    "ce_losses": np.dtype("float32"),
}

def ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    values = list(sample_ids)
    if not values:
        raise ValueError("sample_ids must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("sample_ids must contain non-blank strings")
    if len(values) != len(set(values)):
        raise ValueError("sample_ids must be unique")
    return sha256_hex(values)


def _validated_samples(
    samples: Sequence[KnowledgeSample],
) -> tuple[list[KnowledgeSample], int]:
    values = list(samples)
    if not values:
        raise ValueError("knowledge artifact requires at least one sample")
    if any(not isinstance(sample, KnowledgeSample) for sample in values):
        raise TypeError("samples must contain KnowledgeSample values")

    ordered_sample_ids_sha256([sample.sample_id for sample in values])
    top_k_values = {sample.top_k for sample in values}
    if len(top_k_values) != 1:
        raise ValueError("all samples must use one top-k width")
    return values, top_k_values.pop()


def _as_integer(values, name: str, dtype: np.dtype) -> np.ndarray:
    limits = np.iinfo(dtype)
    try:
        object_values = np.asarray(values, dtype=object)
        for value in object_values.flat:
            integer = operator.index(value)
            if integer < limits.min or integer > limits.max:
                raise ValueError
        result = np.asarray(values, dtype=dtype)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} cannot be represented as {dtype.name}"
        ) from exc
    return np.ascontiguousarray(result)


def _as_int64(values, name: str) -> np.ndarray:
    return _as_integer(values, name, np.dtype("int64"))


def _as_int32(values, name: str) -> np.ndarray:
    return _as_integer(values, name, np.dtype("int32"))


def _as_float32(values, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} cannot be represented as finite float32")
    return np.ascontiguousarray(result)


def _artifact_tensors(
    samples: Sequence[KnowledgeSample],
) -> tuple[dict[str, np.ndarray], list[str], int]:
    values, top_k = _validated_samples(samples)
    sample_ids = [sample.sample_id for sample in values]
    lengths = [len(sample.source_input_ids) for sample in values]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    tensors = {
        "sample_offsets": _as_int64(offsets, "sample_offsets"),
        "source_input_ids": _as_int32(
            [token for sample in values for token in sample.source_input_ids],
            "source_input_ids",
        ),
        "attention_lengths": _as_int32(
            [sample.attention_length for sample in values],
            "attention_lengths",
        ),
        "top_k_token_ids": _as_int32(
            [row for sample in values for row in sample.top_k_token_ids],
            "top_k_token_ids",
        ).reshape(offsets[-1], top_k),
        "top_k_logits": _as_float32(
            [row for sample in values for row in sample.top_k_logits],
            "top_k_logits",
        ).reshape(offsets[-1], top_k),
        "ce_losses": _as_float32(
            [sample.ce_loss for sample in values],
            "ce_losses",
        ),
    }
    if np.any(tensors["source_input_ids"] < 0):
        raise ValueError("source token IDs must be non-negative")
    return tensors, sample_ids, top_k


def serialize_knowledge_artifact(
    samples: Sequence[KnowledgeSample],
) -> tuple[bytes, KnowledgeArtifactDescriptor]:
    tensors, sample_ids, top_k = _artifact_tensors(samples)
    artifact = save_safetensors(tensors)
    descriptor = KnowledgeArtifactDescriptor(
        format=KNOWLEDGE_ARTIFACT_FORMAT,
        schema_version=KNOWLEDGE_ARTIFACT_SCHEMA_VERSION,
        byte_size=len(artifact),
        sha256=sha256_hex(artifact),
        sample_count=len(sample_ids),
        sample_ids_sha256=ordered_sample_ids_sha256(sample_ids),
        total_token_count=int(tensors["source_input_ids"].shape[0]),
        top_k=top_k,
    )
    return artifact, descriptor


def write_knowledge_artifact(
    path: str | Path,
    samples: Sequence[KnowledgeSample],
    *,
    maximum_bytes: int,
) -> KnowledgeArtifactDescriptor:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")

    artifact, descriptor = serialize_knowledge_artifact(samples)
    if descriptor.byte_size > maximum_bytes:
        raise ValueError("knowledge artifact exceeds maximum_bytes")

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


def _validate_descriptor_and_file(
    path: Path,
    descriptor: KnowledgeArtifactDescriptor,
    sample_ids: Sequence[str],
    maximum_bytes: int,
) -> None:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")

    actual_size = path.stat().st_size
    if actual_size > maximum_bytes or descriptor.byte_size > maximum_bytes:
        raise ValueError("knowledge artifact exceeds maximum_bytes")
    if actual_size != descriptor.byte_size:
        raise ValueError("knowledge artifact byte size does not match descriptor")
    if _sha256_file(path) != descriptor.sha256:
        raise ValueError("knowledge artifact SHA-256 does not match descriptor")

    values = list(sample_ids)
    if len(values) != descriptor.sample_count:
        raise ValueError("sample count does not match descriptor")
    if ordered_sample_ids_sha256(values) != descriptor.sample_ids_sha256:
        raise ValueError("ordered sample IDs do not match descriptor")


def _validate_tensor_schema(
    tensors: dict[str, np.ndarray],
    descriptor: KnowledgeArtifactDescriptor,
) -> None:
    names = set(tensors)
    if names != TENSOR_NAMES:
        missing = sorted(TENSOR_NAMES - names)
        unknown = sorted(names - TENSOR_NAMES)
        raise ValueError(
            f"knowledge artifact tensor names are invalid; "
            f"missing={missing}, unknown={unknown}"
        )

    for name, expected_dtype in TENSOR_DTYPES.items():
        if tensors[name].dtype != expected_dtype:
            raise ValueError(
                f"tensor {name} must use dtype {expected_dtype.name}"
            )

    sample_count = descriptor.sample_count
    total_tokens = descriptor.total_token_count
    top_k = descriptor.top_k
    expected_shapes = {
        "sample_offsets": (sample_count + 1,),
        "source_input_ids": (total_tokens,),
        "attention_lengths": (sample_count,),
        "top_k_token_ids": (total_tokens, top_k),
        "top_k_logits": (total_tokens, top_k),
        "ce_losses": (sample_count,),
    }
    for name, expected_shape in expected_shapes.items():
        if tensors[name].shape != expected_shape:
            raise ValueError(
                f"tensor {name} has shape {tensors[name].shape}; "
                f"expected {expected_shape}"
            )

    offsets = tensors["sample_offsets"]
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("sample offsets do not span the token tensors")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("sample offsets must be strictly increasing")

    attention_lengths = tensors["attention_lengths"]
    sample_lengths = np.diff(offsets)
    if np.any(attention_lengths < 1) or np.any(attention_lengths > sample_lengths):
        raise ValueError("attention lengths are outside their sample bounds")

    if np.any(tensors["source_input_ids"] < 0):
        raise ValueError("source token IDs must be non-negative")
    if np.any(tensors["top_k_token_ids"] < 0):
        raise ValueError("top-k token IDs must be non-negative")
    if not np.isfinite(tensors["top_k_logits"]).all():
        raise ValueError("top-k logits must be finite")
    if not np.isfinite(tensors["ce_losses"]).all():
        raise ValueError("CE losses must be finite")
    if np.any(tensors["ce_losses"] < 0):
        raise ValueError("CE losses must be non-negative")


def load_knowledge_artifact(
    path: str | Path,
    descriptor: KnowledgeArtifactDescriptor,
    sample_ids: Sequence[str],
    *,
    maximum_bytes: int,
) -> list[KnowledgeSample]:
    artifact_path = Path(path)
    _validate_descriptor_and_file(
        artifact_path,
        descriptor,
        sample_ids,
        maximum_bytes,
    )

    try:
        with safe_open(artifact_path, framework="np") as artifact:
            if artifact.metadata():
                raise ValueError("knowledge artifact metadata is not allowed")
            tensors = artifact.get_tensors()
    except (OSError, SafetensorError) as exc:
        raise ValueError("invalid safetensors knowledge artifact") from exc
    _validate_tensor_schema(tensors, descriptor)

    offsets = tensors["sample_offsets"]
    samples: list[KnowledgeSample] = []
    for index, sample_id in enumerate(sample_ids):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        samples.append(
            KnowledgeSample(
                sample_id=sample_id,
                source_input_ids=tensors["source_input_ids"][start:end].tolist(),
                attention_length=int(tensors["attention_lengths"][index]),
                top_k_token_ids=tensors["top_k_token_ids"][start:end].tolist(),
                top_k_logits=tensors["top_k_logits"][start:end].tolist(),
                ce_loss=float(tensors["ce_losses"][index]),
            )
        )
    return samples
