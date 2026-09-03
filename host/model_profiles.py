from __future__ import annotations

from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile, OllamaProfile


GRANITE_3_3_2B_HOST_PROFILE_ID = "granite-3.3-2b-instruct-host-lora-v1"
GRANITE_3_3_2B_REVISION = "652c333dc5066f2a1764854a1bcd0ce67163d74f"
GRANITE_3_3_2B_CHAT_TEMPLATE_HASH = (
    "6bc46d1fc4c69468e21e79809662cc0a5c4a1e3e979ecb3de0dd51d4788191a0"
)
MISTRAL_NEMO_HOST_PROFILE_ID = (
    "mistral-nemo-instruct-2407-host-lora-v1"
)
MISTRAL_NEMO_REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
MISTRAL_NEMO_CHAT_TEMPLATE_HASH = (
    "e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272"
)


def supported_host_profile_ids() -> tuple[str, ...]:
    return (
        GRANITE_3_3_2B_HOST_PROFILE_ID,
        MISTRAL_NEMO_HOST_PROFILE_ID,
    )


def pinned_host_profile(
    profile_id: str = GRANITE_3_3_2B_HOST_PROFILE_ID,
    *,
    serving_backend: str = "mock",
) -> ModelProfile:
    if serving_backend not in {"mock", "ollama"}:
        raise ValueError("serving_backend must be mock or ollama")

    if profile_id == MISTRAL_NEMO_HOST_PROFILE_ID:
        if serving_backend != "mock":
            raise ValueError(
                "the pinned Mistral Nemo Host supports mock serving only"
            )
        return ModelProfile(
            profile_id=MISTRAL_NEMO_HOST_PROFILE_ID,
            role="host",
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            model_revision=MISTRAL_NEMO_REVISION,
            model_class="MistralForCausalLM",
            model_type="mistral",
            tokenizer_id="mistralai/Mistral-Nemo-Instruct-2407",
            tokenizer_revision=MISTRAL_NEMO_REVISION,
            tokenizer_class="PreTrainedTokenizerFast",
            vocabulary_size=131072,
            training_backend="transformers",
            serving_backend="mock",
            prompt_template_id=PROMPT_TEMPLATE_ID,
            prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
            tokenizer_chat_template_hash=MISTRAL_NEMO_CHAT_TEMPLATE_HASH,
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
        )

    if profile_id != GRANITE_3_3_2B_HOST_PROFILE_ID:
        supported = ", ".join(supported_host_profile_ids())
        raise ValueError(
            f"unsupported pinned Host profile {profile_id!r}; "
            f"supported: {supported}"
        )

    return ModelProfile(
        profile_id=GRANITE_3_3_2B_HOST_PROFILE_ID,
        role="host",
        model_id="ibm-granite/granite-3.3-2b-instruct",
        model_revision=GRANITE_3_3_2B_REVISION,
        model_class="GraniteForCausalLM",
        model_type="granite",
        tokenizer_id="ibm-granite/granite-3.3-2b-instruct",
        tokenizer_revision=GRANITE_3_3_2B_REVISION,
        tokenizer_class="GPT2TokenizerFast",
        vocabulary_size=49159,
        training_backend="transformers",
        serving_backend=serving_backend,
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        tokenizer_chat_template_hash=GRANITE_3_3_2B_CHAT_TEMPLATE_HASH,
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
            OllamaProfile(model="granite3.3:2b")
            if serving_backend == "ollama"
            else None
        ),
    )
