from __future__ import annotations

from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile, OllamaProfile


LLAMA_PROFILE_ID = "llama-3.2-1b-instruct-lora-v1"
QWEN_PROFILE_ID = "qwen3-1.7b-lora-v1"

LLAMA_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

LLAMA_CHAT_TEMPLATE_HASH = (
    "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"
)
QWEN_CHAT_TEMPLATE_HASH = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)


def _lora_profile() -> LoraProfile:
    return LoraProfile(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=(),
    )


def pinned_client_profile(
    profile_id: str,
    *,
    serving_backend: str = "mock",
) -> ModelProfile:
    if profile_id == LLAMA_PROFILE_ID:
        values = {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "revision": LLAMA_REVISION,
            "model_class": "LlamaForCausalLM",
            "model_type": "llama",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "vocabulary_size": 128256,
            "chat_template_hash": LLAMA_CHAT_TEMPLATE_HASH,
            "chat_template_mode": "standard",
            "ollama_model": "llama3.2:1b",
        }
    elif profile_id == QWEN_PROFILE_ID:
        values = {
            "model_id": "Qwen/Qwen3-1.7B",
            "revision": QWEN_REVISION,
            "model_class": "Qwen3ForCausalLM",
            "model_type": "qwen3",
            "tokenizer_class": "Qwen2TokenizerFast",
            "vocabulary_size": 151936,
            "chat_template_hash": QWEN_CHAT_TEMPLATE_HASH,
            "chat_template_mode": "qwen_non_thinking",
            "ollama_model": "qwen3:1.7b",
        }
    else:
        raise ValueError(f"unknown pinned Client profile: {profile_id!r}")

    if serving_backend not in {"mock", "ollama"}:
        raise ValueError("serving_backend must be mock or ollama")

    return ModelProfile(
        profile_id=profile_id,
        role="client",
        model_id=values["model_id"],
        model_revision=values["revision"],
        model_class=values["model_class"],
        model_type=values["model_type"],
        tokenizer_id=values["model_id"],
        tokenizer_revision=values["revision"],
        tokenizer_class=values["tokenizer_class"],
        vocabulary_size=values["vocabulary_size"],
        training_backend="transformers",
        serving_backend=serving_backend,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        tokenizer_chat_template_hash=values["chat_template_hash"],
        chat_template_mode=values["chat_template_mode"],
        lora=_lora_profile(),
        ollama=(
            OllamaProfile(model=values["ollama_model"])
            if serving_backend == "ollama"
            else None
        ),
    )


def supported_profile_ids() -> tuple[str, ...]:
    return LLAMA_PROFILE_ID, QWEN_PROFILE_ID
