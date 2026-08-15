from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence

import torch


SparseRows = Sequence[Sequence[Sequence[int]]]
SparseLogits = Sequence[Sequence[Sequence[float]]]


class SparseTargetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SparseTargetBatch:
    token_ids: torch.Tensor
    probabilities: torch.Tensor
    valid_mask: torch.Tensor


def _validate_dtype(dtype: torch.dtype) -> None:
    try:
        value = torch.empty((), dtype=dtype)
    except (TypeError, RuntimeError) as exc:
        raise SparseTargetError("sparse-target dtype is invalid") from exc
    if not value.is_floating_point():
        raise SparseTargetError("sparse-target dtype must be floating point")


def _validate_row(
    token_ids: Sequence[int],
    logits: Sequence[float],
    *,
    vocab_size: int,
    top_k: int,
    label: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if len(token_ids) != len(logits):
        raise SparseTargetError(f"{label} token IDs and logits differ in length")
    if not token_ids:
        raise SparseTargetError(f"{label} is empty")
    if len(token_ids) > top_k:
        raise SparseTargetError(f"{label} exceeds configured top-k width")

    unique_ids: list[int] = []
    unique_logits: list[float] = []
    seen: set[int] = set()
    for token_id, logit in zip(token_ids, logits):
        if type(token_id) is not int or token_id < 0 or token_id >= vocab_size:
            raise SparseTargetError(f"{label} contains an out-of-range token ID")
        if isinstance(logit, bool) or not isinstance(logit, Real):
            raise SparseTargetError(f"{label} contains a non-numeric logit")
        logit_value = float(logit)
        if not math.isfinite(logit_value):
            raise SparseTargetError(f"{label} contains a non-finite logit")
        if token_id in seen:
            continue
        seen.add(token_id)
        unique_ids.append(token_id)
        unique_logits.append(logit_value)

    return tuple(unique_ids), tuple(unique_logits)


def _validate_nested_rows(
    token_id_rows: SparseRows,
    logit_rows: SparseLogits,
    *,
    max_length: int,
    label: str,
) -> tuple[int, ...]:
    if len(token_id_rows) != len(logit_rows):
        raise SparseTargetError(f"{label} batch dimensions differ")
    if not token_id_rows:
        raise SparseTargetError(f"{label} batch is empty")

    lengths: list[int] = []
    for sample_index, (sample_ids, sample_logits) in enumerate(
        zip(token_id_rows, logit_rows)
    ):
        if len(sample_ids) != len(sample_logits):
            raise SparseTargetError(
                f"{label} sample {sample_index} sequence dimensions differ"
            )
        if not sample_ids:
            raise SparseTargetError(f"{label} sample {sample_index} is empty")
        if len(sample_ids) > max_length:
            raise SparseTargetError(
                f"{label} sample {sample_index} exceeds maximum length"
            )
        lengths.append(len(sample_ids))
    return tuple(lengths)


def build_sparse_target_batch(
    token_id_rows: SparseRows,
    logit_rows: SparseLogits,
    *,
    max_length: int,
    top_k: int,
    vocab_size: int,
    pad_token_id: int,
    temperature: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    fallback_token_id_rows: SparseRows | None = None,
    fallback_logit_rows: SparseLogits | None = None,
) -> SparseTargetBatch:
    if type(max_length) is not int or max_length < 1:
        raise SparseTargetError("maximum length must be a positive integer")
    if type(top_k) is not int or top_k < 1:
        raise SparseTargetError("top-k width must be a positive integer")
    if type(vocab_size) is not int or vocab_size < 1:
        raise SparseTargetError("vocabulary size must be a positive integer")
    if (
        type(pad_token_id) is not int
        or pad_token_id < 0
        or pad_token_id >= vocab_size
    ):
        raise SparseTargetError("padding token ID is outside the vocabulary")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise SparseTargetError("distillation temperature must be finite and positive")
    _validate_dtype(dtype)

    sequence_lengths = _validate_nested_rows(
        token_id_rows,
        logit_rows,
        max_length=max_length,
        label="target",
    )
    if (fallback_token_id_rows is None) != (fallback_logit_rows is None):
        raise SparseTargetError(
            "fallback token IDs and logits must either both be provided or omitted"
        )
    if fallback_token_id_rows is not None and fallback_logit_rows is not None:
        fallback_lengths = _validate_nested_rows(
            fallback_token_id_rows,
            fallback_logit_rows,
            max_length=max_length,
            label="fallback",
        )
        if fallback_lengths != sequence_lengths:
            raise SparseTargetError(
                "fallback sequence lengths differ from target sequence lengths"
            )
        for sample_index, (sample_ids, sample_logits) in enumerate(
            zip(fallback_token_id_rows, fallback_logit_rows)
        ):
            for position, (row_ids, row_logits) in enumerate(
                zip(sample_ids, sample_logits)
            ):
                _validate_row(
                    row_ids,
                    row_logits,
                    vocab_size=vocab_size,
                    top_k=top_k,
                    label=f"fallback sample {sample_index} position {position}",
                )

    batch_size = len(token_id_rows)
    token_ids = torch.full(
        (batch_size, max_length, top_k),
        pad_token_id,
        dtype=torch.long,
    )
    probabilities = torch.zeros(
        (batch_size, max_length, top_k),
        dtype=dtype,
    )
    valid_mask = torch.zeros(
        (batch_size, max_length, top_k),
        dtype=torch.bool,
    )

    for sample_index, sequence_length in enumerate(sequence_lengths):
        for position in range(sequence_length):
            row_ids = token_id_rows[sample_index][position]
            row_logits = logit_rows[sample_index][position]
            if not row_ids and not row_logits:
                if fallback_token_id_rows is None or fallback_logit_rows is None:
                    raise SparseTargetError(
                        f"target sample {sample_index} position {position} is empty"
                    )
                row_ids = fallback_token_id_rows[sample_index][position]
                row_logits = fallback_logit_rows[sample_index][position]

            unique_ids, unique_logits = _validate_row(
                row_ids,
                row_logits,
                vocab_size=vocab_size,
                top_k=top_k,
                label=f"target sample {sample_index} position {position}",
            )
            row_tensor = torch.tensor(unique_logits, dtype=dtype)
            row_probabilities = torch.softmax(
                row_tensor / float(temperature),
                dim=-1,
            )
            if not torch.isfinite(row_probabilities).all():
                raise SparseTargetError(
                    f"target sample {sample_index} position {position} "
                    "produced non-finite probabilities"
                )

            width = len(unique_ids)
            token_ids[sample_index, position, :width] = torch.tensor(
                unique_ids,
                dtype=torch.long,
            )
            probabilities[sample_index, position, :width] = row_probabilities
            valid_mask[sample_index, position, :width] = True

        for position in range(sequence_length, max_length):
            token_ids[sample_index, position, 0] = pad_token_id
            probabilities[sample_index, position, 0] = 1.0
            valid_mask[sample_index, position, 0] = True

    batch = SparseTargetBatch(
        token_ids=token_ids,
        probabilities=probabilities,
        valid_mask=valid_mask,
    )
    validate_sparse_target_batch(batch, vocab_size=vocab_size)
    return batch


def validate_sparse_target_batch(
    targets: SparseTargetBatch,
    *,
    vocab_size: int,
) -> None:
    if type(vocab_size) is not int or vocab_size < 1:
        raise SparseTargetError("vocabulary size must be a positive integer")
    if (
        targets.token_ids.ndim != 3
        or targets.probabilities.ndim != 3
        or targets.valid_mask.ndim != 3
        or targets.token_ids.shape != targets.probabilities.shape
        or targets.token_ids.shape != targets.valid_mask.shape
        or any(size < 1 for size in targets.token_ids.shape)
    ):
        raise SparseTargetError(
            "sparse-target tensors must share a non-empty [batch, sequence, top-k] shape"
        )
    if targets.token_ids.dtype != torch.long:
        raise SparseTargetError("sparse-target token IDs must use torch.long")
    if not targets.probabilities.is_floating_point():
        raise SparseTargetError("sparse-target probabilities must be floating point")
    if targets.valid_mask.dtype != torch.bool:
        raise SparseTargetError("sparse-target valid mask must use torch.bool")
    if targets.token_ids.device != targets.probabilities.device or (
        targets.token_ids.device != targets.valid_mask.device
    ):
        raise SparseTargetError("sparse-target tensors must share one device")
    if torch.any(targets.token_ids < 0) or torch.any(
        targets.token_ids >= vocab_size
    ):
        raise SparseTargetError("sparse-target token ID is outside the vocabulary")
    if not torch.isfinite(targets.probabilities).all():
        raise SparseTargetError("sparse-target probabilities must be finite")
    if torch.any(targets.probabilities < 0):
        raise SparseTargetError("sparse-target probabilities must be non-negative")
    if torch.any(targets.probabilities[~targets.valid_mask] != 0):
        raise SparseTargetError("masked sparse-target probabilities must be zero")
    if torch.any(targets.valid_mask.sum(dim=-1) == 0):
        raise SparseTargetError("every sparse-target row must contain a valid entry")

    row_sums = (
        targets.probabilities
        * targets.valid_mask.to(targets.probabilities.dtype)
    ).sum(dim=-1)
    tolerance = max(1e-6, 4 * torch.finfo(targets.probabilities.dtype).eps)
    if not torch.allclose(
        row_sums.float(),
        torch.ones_like(row_sums, dtype=torch.float32),
        rtol=0,
        atol=tolerance,
    ):
        raise SparseTargetError("sparse-target rows must sum to one")

    masked_ids = torch.where(
        targets.valid_mask,
        targets.token_ids,
        torch.full_like(targets.token_ids, vocab_size),
    )
    sorted_ids = torch.sort(masked_ids, dim=-1).values
    duplicates = (
        sorted_ids[..., 1:] == sorted_ids[..., :-1]
    ) & sorted_ids[..., 1:].ne(vocab_size)
    if torch.any(duplicates):
        raise SparseTargetError(
            "valid sparse-target token IDs must be unique within each row"
        )


def densify_sparse_targets_for_test(
    targets: SparseTargetBatch,
    *,
    vocab_size: int,
    maximum_elements: int = 1_000_000,
) -> torch.Tensor:
    validate_sparse_target_batch(targets, vocab_size=vocab_size)
    if type(maximum_elements) is not int or maximum_elements < 1:
        raise SparseTargetError("dense-oracle element limit must be positive")
    batch_size, sequence_length, _ = targets.token_ids.shape
    element_count = batch_size * sequence_length * vocab_size
    if element_count > maximum_elements:
        raise SparseTargetError("dense parity oracle exceeds its fixture-only limit")

    dense = torch.zeros(
        (batch_size, sequence_length, vocab_size),
        dtype=targets.probabilities.dtype,
        device=targets.probabilities.device,
    )
    dense.scatter_add_(
        -1,
        targets.token_ids,
        targets.probabilities * targets.valid_mask.to(targets.probabilities.dtype),
    )
    return dense


def sparse_distillation_loss_per_position(
    model_logits: torch.Tensor,
    targets: SparseTargetBatch,
    *,
    loss_type: Literal["ce", "kl"] = "ce",
) -> torch.Tensor:
    if model_logits.ndim != 3 or any(size < 1 for size in model_logits.shape):
        raise SparseTargetError(
            "model logits must have a non-empty [batch, sequence, vocabulary] shape"
        )
    if not model_logits.is_floating_point():
        raise SparseTargetError("model logits must be floating point")
    validate_sparse_target_batch(targets, vocab_size=model_logits.shape[-1])
    if model_logits.shape[:2] != targets.token_ids.shape[:2]:
        raise SparseTargetError(
            "model logits and sparse targets differ in batch or sequence shape"
        )
    if model_logits.device != targets.token_ids.device:
        raise SparseTargetError("model logits and sparse targets must share one device")

    selected_logits = torch.gather(
        model_logits,
        -1,
        targets.token_ids,
    )
    selected_log_probabilities = selected_logits - torch.logsumexp(
        model_logits,
        dim=-1,
        keepdim=True,
    )
    probabilities = targets.probabilities.to(model_logits.dtype)
    mask = targets.valid_mask

    if loss_type == "ce":
        values = -probabilities * selected_log_probabilities
    elif loss_type == "kl":
        positive = mask & probabilities.gt(0)
        safe_probabilities = torch.where(
            positive,
            probabilities,
            torch.ones_like(probabilities),
        )
        values = torch.where(
            positive,
            probabilities
            * (torch.log(safe_probabilities) - selected_log_probabilities),
            torch.zeros_like(probabilities),
        )
    else:
        raise SparseTargetError(f"unsupported distillation loss type: {loss_type!r}")

    return (values * mask.to(values.dtype)).sum(dim=-1)


def answer_only_sparse_distillation_loss(
    model_logits: torch.Tensor,
    targets: SparseTargetBatch,
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_type: Literal["ce", "kl"] = "ce",
) -> torch.Tensor:
    if labels.shape != model_logits.shape[:2] or attention_mask.shape != labels.shape:
        raise SparseTargetError(
            "labels and attention mask must match model batch and sequence shape"
        )
    if labels.device != model_logits.device or attention_mask.device != model_logits.device:
        raise SparseTargetError(
            "model logits, labels and attention mask must share one device"
        )
    per_position = sparse_distillation_loss_per_position(
        model_logits,
        targets,
        loss_type=loss_type,
    )
    distillation_mask = labels[..., 1:].ne(-100) & attention_mask[..., 1:].bool()
    supervised_positions = distillation_mask.sum()
    if supervised_positions == 0:
        raise SparseTargetError(
            "answer-only distillation requires at least one supervised target token"
        )
    return (
        per_position[..., :-1] * distillation_mask
    ).sum() / supervised_positions
