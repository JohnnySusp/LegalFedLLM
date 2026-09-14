from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
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


ANSWER_ONLY_LOSS_SEQUENCE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class BoundedCausalLmState:
    hidden_states: Any
    output_embeddings: Any
    logits_scaling: float
    vocabulary_size: int


def client_saved_activation_context(torch: Any, device: Any) -> Any:
    if device.type == "cuda" and torch.is_grad_enabled():
        return torch.autograd.graph.save_on_cpu(pin_memory=True)
    return nullcontext()


def bounded_causal_lm_state(
    torch: Any,
    model: Any,
    inputs: dict[str, Any],
) -> BoundedCausalLmState:
    causal_lm = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    config = getattr(causal_lm, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in {"qwen3", "granite"}:
        raise RuntimeError(
            "bounded Client loss supports only pinned Qwen3 and Granite models"
        )
    decoder = getattr(causal_lm, "model", None)
    output_embeddings = (
        causal_lm.get_output_embeddings()
        if hasattr(causal_lm, "get_output_embeddings")
        else None
    )
    if decoder is None or output_embeddings is None:
        raise RuntimeError(
            "Client model does not expose its decoder and output embeddings"
        )

    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    if labels.shape != input_ids.shape or attention_mask.shape != labels.shape:
        raise ValueError(
            "labels and attention mask must match Client training input shape"
        )
    if input_ids.device != labels.device or attention_mask.device != labels.device:
        raise ValueError(
            "Client training inputs, labels and attention mask must share one device"
        )

    # Only saved backward tensors move to RAM. Model execution stays on GPU.
    with client_saved_activation_context(torch, input_ids.device):
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    hidden_states = outputs.last_hidden_state
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            "Client training hidden states differ in batch or sequence shape"
        )
    if hidden_states.device != labels.device:
        raise ValueError(
            "Client training hidden states and labels must share one device"
        )

    vocabulary_size = int(getattr(config, "vocab_size", 0))
    if vocabulary_size < 1:
        raise RuntimeError("Client model has no valid vocabulary size")
    logits_scaling = (
        float(config.logits_scaling) if model_type == "granite" else 1.0
    )
    if not math.isfinite(logits_scaling) or logits_scaling <= 0:
        raise RuntimeError("Client model has invalid logits scaling")
    return BoundedCausalLmState(
        hidden_states=hidden_states,
        output_embeddings=output_embeddings,
        logits_scaling=logits_scaling,
        vocabulary_size=vocabulary_size,
    )


def checkpointed_causal_lm_chunk(
    torch: Any,
    state: BoundedCausalLmState,
    hidden_chunk: Any,
    loss_function: Any,
    *loss_inputs: Any,
) -> Any:
    from torch.utils.checkpoint import checkpoint

    def project_and_loss(hidden_states: Any, *values: Any) -> Any:
        logits = state.output_embeddings(hidden_states)
        if state.logits_scaling != 1.0:
            logits = logits / state.logits_scaling
        return loss_function(logits, *values)

    return checkpoint(
        project_and_loss,
        hidden_chunk,
        *loss_inputs,
        use_reentrant=False,
    )


def bounded_answer_only_loss(
    torch: Any,
    model: Any,
    inputs: dict[str, Any],
    *,
    sequence_chunk_size: int = ANSWER_ONLY_LOSS_SEQUENCE_CHUNK_SIZE,
    num_items_in_batch: Any | None = None,
) -> Any:
    if type(sequence_chunk_size) is not int or sequence_chunk_size < 1:
        raise ValueError("answer-only loss sequence chunk size must be positive")
    state = bounded_causal_lm_state(torch, model, inputs)
    shifted_labels = inputs["labels"][..., 1:]
    supervised_mask = shifted_labels.ne(-100)
    supervised_positions = supervised_mask.sum()
    if int(supervised_positions.item()) < 1:
        raise ValueError(
            "answer-only supervision requires at least one supervised target token"
        )

    def chunk_loss(logits: Any, labels: Any) -> Any:
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, state.vocabulary_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    loss_sum = None
    active_positions = torch.nonzero(
        supervised_mask.any(dim=0),
        as_tuple=False,
    ).flatten()
    first_position = int(active_positions[0].item())
    last_position = int(active_positions[-1].item()) + 1
    first_chunk = (first_position // sequence_chunk_size) * sequence_chunk_size
    for start in range(first_chunk, last_position, sequence_chunk_size):
        end = min(start + sequence_chunk_size, last_position)
        chunk_labels = shifted_labels[:, start:end]
        if not bool(chunk_labels.ne(-100).any().item()):
            continue
        chunk_sum = checkpointed_causal_lm_chunk(
            torch,
            state,
            state.hidden_states[:, start:end, :],
            chunk_loss,
            chunk_labels,
        )
        loss_sum = chunk_sum if loss_sum is None else loss_sum + chunk_sum

    if loss_sum is None:
        raise ValueError("answer-only loss requires at least two sequence tokens")
    if num_items_in_batch is None:
        denominator = supervised_positions.to(loss_sum.dtype)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(
            device=loss_sum.device,
            dtype=loss_sum.dtype,
        )
    else:
        denominator = torch.tensor(
            float(num_items_in_batch),
            device=loss_sum.device,
            dtype=loss_sum.dtype,
        )
    if float(denominator.detach().item()) <= 0:
        raise ValueError("answer-only loss denominator must be positive")
    return loss_sum / denominator
