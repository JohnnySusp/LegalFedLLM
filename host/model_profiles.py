from __future__ import annotations

from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile, OllamaProfile


LLAMA_3_2_3B_HOST_PROFILE_ID = "llama-3.2-3b-instruct-host-lora-v1"
LLAMA_3_2_3B_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
LLAMA_3_2_CHAT_TEMPLATE_HASH = (
    "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"
)


def pinned_host_profile(*, serving_backend: str = "mock") -> ModelProfile:
    if serving_backend not in {"mock", "ollama"}:
        raise ValueError("serving_backend must be mock or ollama")

    return ModelProfile(
        profile_id=LLAMA_3_2_3B_HOST_PROFILE_ID,
        role="host",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_revision=LLAMA_3_2_3B_REVISION,
        model_class="LlamaForCausalLM",
        model_type="llama",
        tokenizer_id="meta-llama/Llama-3.2-3B-Instruct",
        tokenizer_revision=LLAMA_3_2_3B_REVISION,
        tokenizer_class="PreTrainedTokenizerFast",
        vocabulary_size=128256,
        training_backend="transformers",
        serving_backend=serving_backend,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        tokenizer_chat_template_hash=LLAMA_3_2_CHAT_TEMPLATE_HASH,
        chat_template_mode="standard",
        lora=LoraProfile(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=(),
        ),
        ollama=(
            OllamaProfile(model="llama3.2:3b")
            if serving_backend == "ollama"
            else None
        ),
    )
