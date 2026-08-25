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
from shared.distillation_artifact import (
    TENSOR_DTYPES,
    TENSOR_NAMES,
    _trainer_inputs_hash,
    _validate_tensor_schema,
)
from shared.protocol import ClientReverseTrainingArtifactDescriptor

if TYPE_CHECKING:
    from shared.fedmkt_core.reverse_integration import PreparedReverseDistillationBatch


def _tensors(batch: PreparedReverseDistillationBatch) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, dtype in TENSOR_DTYPES.items():
        value: Any = batch.trainer_inputs()[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[name] = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return result


def write_client_reverse_training_artifact(
    path: str | Path,
    batch: PreparedReverseDistillationBatch,
    *,
    pad_token_id: int,
    maximum_bytes: int,
) -> ClientReverseTrainingArtifactDescriptor:
    tensors = _tensors(batch)
    if sum(value.nbytes for value in tensors.values()) > maximum_bytes:
        raise ValueError("Client reverse training tensors exceed maximum_bytes")
    artifact = save_safetensors(tensors)
    if len(artifact) > maximum_bytes:
        raise ValueError("Client reverse training artifact exceeds maximum_bytes")
    descriptor = ClientReverseTrainingArtifactDescriptor(
        byte_size=len(artifact),
        sha256=sha256_hex(artifact),
        sample_count=len(batch.sample_ids),
        sample_ids_sha256=sha256_hex(list(batch.sample_ids)),
        padded_sequence_length=tensors["input_ids"].shape[1],
        top_k=tensors["sparse_target_token_ids"].shape[-1],
        pad_token_id=pad_token_id,
        trainer_inputs_sha256=batch.audit.trainer_inputs_sha256,
    )
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


def load_client_reverse_training_artifact(
    path: str | Path,
    descriptor: ClientReverseTrainingArtifactDescriptor,
    sample_ids: Sequence[str],
    *,
    maximum_bytes: int,
    vocabulary_size: int,
) -> dict[str, np.ndarray]:
    artifact_path = Path(path)
    actual_size = artifact_path.stat().st_size
    if (
        actual_size != descriptor.byte_size
        or actual_size > maximum_bytes
        or descriptor.byte_size > maximum_bytes
    ):
        raise ValueError("Client reverse training artifact byte size differs")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != descriptor.sha256:
        raise ValueError("Client reverse training artifact SHA-256 differs")
    values = list(sample_ids)
    if sha256_hex(values) != descriptor.sample_ids_sha256:
        raise ValueError("Client reverse training artifact sample order differs")
    try:
        with safe_open(artifact_path, framework="np") as artifact:
            if artifact.metadata():
                raise ValueError(
                    "Client reverse training artifact metadata is not allowed"
                )
            tensors = artifact.get_tensors()
    except (OSError, SafetensorError) as exc:
        raise ValueError(
            "invalid safetensors Client reverse training artifact"
        ) from exc
    if set(tensors) != TENSOR_NAMES:
        raise ValueError("Client reverse training tensor names are invalid")
    if any(tensors[name].dtype != dtype for name, dtype in TENSOR_DTYPES.items()):
        raise ValueError("Client reverse training tensor dtype is invalid")
    _validate_tensor_schema(tensors, descriptor, vocabulary_size)
    if (
        _trainer_inputs_hash(tensors, values, descriptor.pad_token_id)
        != descriptor.trainer_inputs_sha256
    ):
        raise ValueError("Client reverse training semantic tensor hash differs")
    return tensors
