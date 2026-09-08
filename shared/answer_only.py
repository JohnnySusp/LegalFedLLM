from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.protocol import ModelProfile


@dataclass(frozen=True)
class EncodedTrainingExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(token_id, int) for token_id in value
    ):
        raise ValueError("chat template did not return one token-ID sequence")
    return value


def encode_answer_only_example(
    *,
    item_id: str,
    item_kind: str,
    prompt: str,
    answer: str,
    tokenizer: Any,
    model_profile: ModelProfile,
    maximum_sequence_length: int | None,
) -> EncodedTrainingExample:
    template_options: dict[str, Any] = {}
    if model_profile.chat_template_mode == "qwen_non_thinking":
        template_options["enable_thinking"] = False

    user_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        *user_messages,
        {"role": "assistant", "content": answer},
    ]
    prompt_ids = _input_ids(
        tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_options,
        )
    )
    full_ids = _input_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            **template_options,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"chat template is not prefix-stable for {item_id!r}"
        )
    if (
        maximum_sequence_length is not None
        and len(full_ids) > maximum_sequence_length
    ):
        raise ValueError(
            f"{item_kind} {item_id!r} has {len(full_ids)} tokens, "
            f"exceeding maximum_sequence_length="
            f"{maximum_sequence_length}"
        )
    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        raise ValueError(f"{item_kind} {item_id!r} has no answer tokens")
    return EncodedTrainingExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=[-100] * len(prompt_ids) + answer_ids,
    )


def encode_chat_prompt(
    *,
    tokenizer: Any,
    model_profile: ModelProfile,
    messages: list[dict[str, str]],
) -> list[int]:
    if not messages:
        raise ValueError("chat generation requires at least one message")
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat message content must not be blank")
        normalized.append({"role": role, "content": content})

    template_options: dict[str, Any] = {}
    if model_profile.chat_template_mode == "qwen_non_thinking":
        template_options["enable_thinking"] = False
    return _input_ids(
        tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            **template_options,
        )
    )


class AnswerOnlyCollator:
    def __init__(self, torch: Any, pad_token_id: int):
        self.torch = torch
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {
            name: self.torch.tensor(values, dtype=self.torch.long)
            for name, values in batch.items()
        }
