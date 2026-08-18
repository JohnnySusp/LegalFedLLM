from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from shared.answer_only import encode_answer_only_example
from shared.prompt import render_reference_prompt
from shared.protocol import KnowledgeSample, ModelProfile
from shared.reference_dataset import ReferenceSample


@dataclass(frozen=True)
class EncodedReferenceSample:
    sample_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


@dataclass(frozen=True)
class FedMKTGenerationArguments:
    top_k_logits_keep: int
    metric_type: str = "ce"
    top_k_strategy: str = "highest"


def encode_reference_samples(
    samples: Sequence[ReferenceSample],
    *,
    tokenizer: Any,
    model_profile: ModelProfile,
    maximum_sequence_length: int,
    expected_sample_ids: Sequence[str],
    dataset_label: str = "signed D^P",
) -> list[EncodedReferenceSample]:
    values = list(samples)
    expected_ids = list(expected_sample_ids)
    actual_ids = [sample.sample_id for sample in values]
    if actual_ids != expected_ids:
        raise ValueError(
            f"reference samples differ from the {dataset_label} order"
        )

    encoded: list[EncodedReferenceSample] = []
    overlength: list[tuple[str, int]] = []
    for sample in values:
        item = encode_answer_only_example(
            item_id=sample.sample_id,
            item_kind="reference sample",
            prompt=render_reference_prompt(sample),
            answer=sample.gold_answer,
            tokenizer=tokenizer,
            model_profile=model_profile,
            maximum_sequence_length=None,
        )
        length = len(item.input_ids)
        if length > maximum_sequence_length:
            overlength.append((sample.sample_id, length))
        encoded.append(
            EncodedReferenceSample(
                sample_id=sample.sample_id,
                input_ids=item.input_ids,
                attention_mask=item.attention_mask,
                labels=item.labels,
            )
        )

    if overlength:
        maximum = max(length for _, length in overlength)
        examples = ", ".join(
            f"{sample_id}={length}" for sample_id, length in overlength[:10]
        )
        if len(overlength) > 10:
            examples += f", and {len(overlength) - 10} more"
        raise ValueError(
            f"{len(overlength)} reference sample(s) exceed "
            f"maximum_sequence_length={maximum_sequence_length}; "
            f"maximum observed length={maximum}; {examples}"
        )
    return encoded


def knowledge_sample_from_rows(
    encoded: EncodedReferenceSample,
    *,
    top_k_token_ids: Sequence[Sequence[int]],
    top_k_logits: Sequence[Sequence[float]],
    ce_loss: float,
) -> KnowledgeSample:
    length = len(encoded.input_ids)
    return KnowledgeSample(
        sample_id=encoded.sample_id,
        source_input_ids=encoded.input_ids,
        attention_length=length,
        top_k_token_ids=[list(row) for row in top_k_token_ids[:length]],
        top_k_logits=[list(row) for row in top_k_logits[:length]],
        ce_loss=ce_loss,
    )
