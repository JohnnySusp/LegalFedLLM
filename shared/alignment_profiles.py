from __future__ import annotations

from dataclasses import dataclass, replace

from shared.protocol import ModelProfile


POC_DTW_PROFILE_VERSION = "qwen3-1.7b--granite3.3-2b-v1"
POC_DTW_PROFILE_ID = f"dtw:{POC_DTW_PROFILE_VERSION}"
MISTRAL_NEMO_DTW_PROFILE_VERSION = (
    "qwen3-1.7b--mistral-nemo-instruct-2407-v1"
)
MISTRAL_NEMO_DTW_PROFILE_ID = f"dtw:{MISTRAL_NEMO_DTW_PROFILE_VERSION}"
GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION = (
    "granite3.3-2b-client--mistral-nemo-instruct-2407-v1"
)
GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID = (
    f"dtw:{GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION}"
)
GRANITE_IDENTITY_DTW_PROFILE_VERSION = (
    "granite3.3-2b-client--granite3.3-2b-host-v1"
)
GRANITE_IDENTITY_DTW_PROFILE_ID = (
    f"dtw:{GRANITE_IDENTITY_DTW_PROFILE_VERSION}"
)
GRANITE_3_3_2B_CLIENT_PROFILE_ID = (
    "granite-3.3-2b-instruct-client-lora-v1"
)
MOCK_IDENTITY_PROFILE_ID = "mock_identity:1"


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
    tokenizer_artifact_sha256: str
    tokenizer_base_vocabulary_size: int
    tokenizer_vocabulary_size: int
    tokenizer_max_token_id: int
    word_boundary_marker: str
    bos_token: str | None
    bos_token_id: int | None
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    unk_token: str | None
    unk_token_id: int | None
    additional_special_token_ids: tuple[int, ...]
    model_max_length: int
    padding_side: str
    fix_mistral_regex: bool = False
    bind_existing_pad_token: bool = False

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
            name for name, value in actual.items() if value != getattr(self, name)
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
        tokenizer_artifact_sha256=(
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        ),
        tokenizer_base_vocabulary_size=151643,
        tokenizer_vocabulary_size=151669,
        tokenizer_max_token_id=151668,
        word_boundary_marker="Ġ",
        bos_token=None,
        bos_token_id=None,
        eos_token="<|im_end|>",
        eos_token_id=151645,
        pad_token="<|endoftext|>",
        pad_token_id=151643,
        unk_token=None,
        unk_token_id=None,
        additional_special_token_ids=tuple(range(151644, 151657)),
        model_max_length=131072,
        padding_side="right",
    ),
    host=TokenizerEndpoint(
        role="host",
        profile_id="granite-3.3-2b-instruct-host-lora-v1",
        model_id="ibm-granite/granite-3.3-2b-instruct",
        model_revision="652c333dc5066f2a1764854a1bcd0ce67163d74f",
        model_class="GraniteForCausalLM",
        model_type="granite",
        tokenizer_id="ibm-granite/granite-3.3-2b-instruct",
        tokenizer_revision="652c333dc5066f2a1764854a1bcd0ce67163d74f",
        tokenizer_class="GPT2TokenizerFast",
        vocabulary_size=49159,
        tokenizer_chat_template_hash=(
            "6bc46d1fc4c69468e21e79809662cc0a5c4a1e3e979ecb3de0dd51d4788191a0"
        ),
        chat_template_mode="standard",
        tokenizer_artifact_sha256=(
            "91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e"
        ),
        tokenizer_base_vocabulary_size=49152,
        tokenizer_vocabulary_size=49159,
        tokenizer_max_token_id=49158,
        word_boundary_marker="Ġ",
        bos_token="<|end_of_text|>",
        bos_token_id=0,
        eos_token="<|end_of_text|>",
        eos_token_id=0,
        pad_token="<|end_of_text|>",
        pad_token_id=0,
        unk_token="<|end_of_text|>",
        unk_token_id=0,
        additional_special_token_ids=tuple(range(49152, 49159)),
        model_max_length=9223372036854775807,
        padding_side="left",
    ),
    client_to_host_owner="coordinator",
    host_to_client_owner="client",
)


GRANITE_IDENTITY_DTW_PROFILE = BidirectionalAlignmentProfile(
    profile_id=GRANITE_IDENTITY_DTW_PROFILE_ID,
    strategy="dtw",
    profile_version=GRANITE_IDENTITY_DTW_PROFILE_VERSION,
    client=replace(
        POC_DTW_PROFILE.host,
        role="client",
        profile_id=GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    ),
    host=POC_DTW_PROFILE.host,
    client_to_host_owner="coordinator",
    host_to_client_owner="client",
)


MISTRAL_NEMO_HOST_ENDPOINT = TokenizerEndpoint(
    role="host",
    profile_id="mistral-nemo-instruct-2407-host-lora-v1",
    model_id="mistralai/Mistral-Nemo-Instruct-2407",
    model_revision="04d8a90549d23fc6bd7f642064003592df51e9b3",
    model_class="MistralForCausalLM",
    model_type="mistral",
    tokenizer_id="mistralai/Mistral-Nemo-Instruct-2407",
    tokenizer_revision="04d8a90549d23fc6bd7f642064003592df51e9b3",
    tokenizer_class="PreTrainedTokenizerFast",
    vocabulary_size=131072,
    tokenizer_chat_template_hash=(
        "e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272"
    ),
    chat_template_mode="standard",
    tokenizer_artifact_sha256=(
        "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db"
    ),
    tokenizer_base_vocabulary_size=131072,
    tokenizer_vocabulary_size=131072,
    tokenizer_max_token_id=131071,
    word_boundary_marker="Ġ",
    bos_token="<s>",
    bos_token_id=1,
    eos_token="</s>",
    eos_token_id=2,
    pad_token="<pad>",
    pad_token_id=10,
    unk_token="<unk>",
    unk_token_id=0,
    additional_special_token_ids=(),
    model_max_length=1000000000000000019884624838656,
    padding_side="right",
    fix_mistral_regex=True,
    bind_existing_pad_token=True,
)


MISTRAL_NEMO_DTW_PROFILE = BidirectionalAlignmentProfile(
    profile_id=MISTRAL_NEMO_DTW_PROFILE_ID,
    strategy="dtw",
    profile_version=MISTRAL_NEMO_DTW_PROFILE_VERSION,
    client=POC_DTW_PROFILE.client,
    host=MISTRAL_NEMO_HOST_ENDPOINT,
    client_to_host_owner="coordinator",
    host_to_client_owner="client",
)


GRANITE_MISTRAL_NEMO_DTW_PROFILE = BidirectionalAlignmentProfile(
    profile_id=GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
    strategy="dtw",
    profile_version=GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION,
    client=GRANITE_IDENTITY_DTW_PROFILE.client,
    host=MISTRAL_NEMO_HOST_ENDPOINT,
    client_to_host_owner="coordinator",
    host_to_client_owner="client",
)


_SUPPORTED_PROFILES = {
    POC_DTW_PROFILE_ID: POC_DTW_PROFILE,
    GRANITE_IDENTITY_DTW_PROFILE_ID: GRANITE_IDENTITY_DTW_PROFILE,
    MISTRAL_NEMO_DTW_PROFILE_ID: MISTRAL_NEMO_DTW_PROFILE,
    GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID: GRANITE_MISTRAL_NEMO_DTW_PROFILE,
}

_HOST_ENDPOINTS = {
    POC_DTW_PROFILE.host.profile_id: POC_DTW_PROFILE.host,
    MISTRAL_NEMO_HOST_ENDPOINT.profile_id: MISTRAL_NEMO_HOST_ENDPOINT,
}


def supported_alignment_profile_ids() -> tuple[str, ...]:
    return tuple(_SUPPORTED_PROFILES)


def resolve_alignment_profile(profile_id: str) -> BidirectionalAlignmentProfile:
    try:
        return _SUPPORTED_PROFILES[profile_id]
    except KeyError:
        supported = ", ".join(supported_alignment_profile_ids())
        raise UnsupportedAlignmentProfile(
            f"unsupported alignment profile {profile_id!r}; supported: {supported}"
        ) from None


def resolve_host_tokenizer_endpoint(profile_id: str) -> TokenizerEndpoint:
    try:
        return _HOST_ENDPOINTS[profile_id]
    except KeyError:
        supported = ", ".join(_HOST_ENDPOINTS)
        raise UnsupportedAlignmentProfile(
            f"unsupported Host tokenizer profile {profile_id!r}; "
            f"supported: {supported}"
        ) from None


def resolve_alignment_profile_id_for_pair(
    *,
    client_profile: ModelProfile,
    host_profile: ModelProfile,
) -> str:
    if (
        client_profile.training_backend == "mock"
        and host_profile.training_backend == "mock"
    ):
        return MOCK_IDENTITY_PROFILE_ID

    matches = [
        profile.profile_id
        for profile in _SUPPORTED_PROFILES.values()
        if not profile.client.mismatches(client_profile)
        and not profile.host.mismatches(host_profile)
    ]
    if not matches:
        raise UnsupportedAlignmentProfile(
            "no approved alignment profile matches the registered Client and "
            "Host model/tokenizer profiles"
        )
    if len(matches) != 1:
        raise UnsupportedAlignmentProfile(
            "multiple approved alignment profiles match the registered Client "
            "and Host model/tokenizer profiles"
        )
    return matches[0]

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
