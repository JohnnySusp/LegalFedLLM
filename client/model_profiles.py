from __future__ import annotations

from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile, OllamaProfile


QWEN_PROFILE_ID = "qwen3-1.7b-lora-v1"
GRANITE_3_3_2B_CLIENT_PROFILE_ID = (
    "granite-3.3-2b-instruct-client-lora-v1"
)

QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

QWEN_CHAT_TEMPLATE_HASH = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
GRANITE_3_3_2B_REVISION = "652c333dc5066f2a1764854a1bcd0ce67163d74f"
GRANITE_3_3_2B_CHAT_TEMPLATE_HASH = (
    "6bc46d1fc4c69468e21e79809662cc0a5c4a1e3e979ecb3de0dd51d4788191a0"
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
    if profile_id == QWEN_PROFILE_ID:
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
    elif profile_id == GRANITE_3_3_2B_CLIENT_PROFILE_ID:
        values = {
            "model_id": "ibm-granite/granite-3.3-2b-instruct",
            "revision": GRANITE_3_3_2B_REVISION,
            "model_class": "GraniteForCausalLM",
            "model_type": "granite",
            "tokenizer_class": "GPT2TokenizerFast",
            "vocabulary_size": 49159,
            "chat_template_hash": GRANITE_3_3_2B_CHAT_TEMPLATE_HASH,
            "chat_template_mode": "standard",
            "ollama_model": "granite3.3:2b",
        }
    else:
        raise ValueError(f"unknown pinned Client profile: {profile_id!r}")

    if serving_backend not in {"mock", "ollama", "transformers"}:
        raise ValueError("serving_backend must be mock, ollama, or transformers")

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


def ollama_model_for_profile(profile_id: str) -> str:
    if profile_id == QWEN_PROFILE_ID:
        return "qwen3:1.7b"
    if profile_id == GRANITE_3_3_2B_CLIENT_PROFILE_ID:
        return "granite3.3:2b"
    raise ValueError(f"unknown pinned Client profile: {profile_id!r}")


def supported_ollama_models() -> tuple[str, ...]:
    return tuple(ollama_model_for_profile(profile_id) for profile_id in supported_profile_ids())


def supported_profile_ids() -> tuple[str, ...]:
    return (QWEN_PROFILE_ID, GRANITE_3_3_2B_CLIENT_PROFILE_ID)
