from __future__ import annotations

from dataclasses import dataclass, fields

from shared.protocol import ModelProfile


POC_DTW_PROFILE_VERSION = "qwen3-1.7b--llama3.2-3b-v1"
POC_DTW_PROFILE_ID = f"dtw:{POC_DTW_PROFILE_VERSION}"


class UnsupportedAlignmentProfile(ValueError):
    """Raised when a requested alignment profile is not an approved PoC pair."""


@dataclass(frozen=True, slots=True)
class TokenizerEndpoint:
    role: str
    profile_id: str
    model_id: str
    model_revision: str
    model_class: str
    model_type: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_class: str
    vocabulary_size: int
    tokenizer_chat_template_hash: str
    chat_template_mode: str

    def mismatches(self, profile: ModelProfile) -> tuple[str, ...]:
        actual = {
            "role": profile.role,
            "profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "model_class": profile.model_class,
            "model_type": profile.model_type,
            "tokenizer_id": profile.tokenizer_id,
            "tokenizer_revision": profile.tokenizer_revision,
            "tokenizer_class": profile.tokenizer_class,
            "vocabulary_size": profile.vocabulary_size,
            "tokenizer_chat_template_hash": (
                profile.tokenizer_chat_template_hash
            ),
            "chat_template_mode": profile.chat_template_mode,
        }
        return tuple(
            item.name
            for item in fields(self)
            if actual[item.name] != getattr(self, item.name)
        )


@dataclass(frozen=True, slots=True)
class BidirectionalAlignmentProfile:
    profile_id: str
    strategy: str
    profile_version: str
    client: TokenizerEndpoint
    host: TokenizerEndpoint
    client_to_host_owner: str
    host_to_client_owner: str


POC_DTW_PROFILE = BidirectionalAlignmentProfile(
    profile_id=POC_DTW_PROFILE_ID,
    strategy="dtw",
    profile_version=POC_DTW_PROFILE_VERSION,
    client=TokenizerEndpoint(
        role="client",
        profile_id="qwen3-1.7b-lora-v1",
        model_id="Qwen/Qwen3-1.7B",
        model_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        model_class="Qwen3ForCausalLM",
        model_type="qwen3",
        tokenizer_id="Qwen/Qwen3-1.7B",
        tokenizer_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        tokenizer_class="Qwen2TokenizerFast",
        vocabulary_size=151936,
        tokenizer_chat_template_hash=(
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        ),
        chat_template_mode="qwen_non_thinking",
    ),
    host=TokenizerEndpoint(
        role="host",
        profile_id="llama-3.2-3b-instruct-host-lora-v1",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
        model_class="LlamaForCausalLM",
        model_type="llama",
        tokenizer_id="meta-llama/Llama-3.2-3B-Instruct",
        tokenizer_revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
        tokenizer_class="PreTrainedTokenizerFast",
        vocabulary_size=128256,
        tokenizer_chat_template_hash=(
            "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"
        ),
        chat_template_mode="standard",
    ),
    client_to_host_owner="coordinator",
    host_to_client_owner="client",
)


def supported_alignment_profile_ids() -> tuple[str, ...]:
    return (POC_DTW_PROFILE_ID,)


def resolve_alignment_profile(profile_id: str) -> BidirectionalAlignmentProfile:
    if profile_id != POC_DTW_PROFILE_ID:
        supported = ", ".join(supported_alignment_profile_ids())
        raise UnsupportedAlignmentProfile(
            f"unsupported alignment profile {profile_id!r}; supported: {supported}"
        )
    return POC_DTW_PROFILE


def validate_alignment_pair(
    profile_id: str,
    *,
    client_profile: ModelProfile,
    host_profile: ModelProfile,
) -> BidirectionalAlignmentProfile:
    profile = resolve_alignment_profile(profile_id)
    client_mismatches = profile.client.mismatches(client_profile)
    host_mismatches = profile.host.mismatches(host_profile)
    if client_mismatches or host_mismatches:
        details: list[str] = []
        if client_mismatches:
            details.append("Client " + ", ".join(client_mismatches))
        if host_mismatches:
            details.append("Host " + ", ".join(host_mismatches))
        raise UnsupportedAlignmentProfile(
            f"alignment profile {profile_id!r} does not match the signed "
            f"model/tokenizer profiles ({'; '.join(details)})"
        )
    return profile
