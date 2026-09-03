from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.alignment_profiles import TokenizerEndpoint
from shared.crypto import sha256_hex


class TokenizerValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedTokenizer:
    endpoint: TokenizerEndpoint
    tokenizer: Any
    artifact_sha256: str
    artifact_path: Path | None = None


def tokenizer_artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_loaded_tokenizer(
    endpoint: TokenizerEndpoint,
    tokenizer: Any,
    *,
    artifact_sha256: str,
    artifact_path: str | Path | None = None,
) -> ValidatedTokenizer:
    failures: list[str] = []

    if tokenizer.__class__.__name__ != endpoint.tokenizer_class:
        failures.append("tokenizer_class")
    if getattr(tokenizer, "is_fast", False) is not True:
        failures.append("is_fast")
    if artifact_sha256 != endpoint.tokenizer_artifact_sha256:
        failures.append("tokenizer_artifact_sha256")

    try:
        vocabulary = tokenizer.get_vocab()
    except Exception as exc:
        raise TokenizerValidationError("tokenizer get_vocab() failed") from exc
    if not isinstance(vocabulary, dict) or not vocabulary:
        failures.append("vocabulary")
        vocabulary = {}

    token_ids = list(vocabulary.values())
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        failures.append("token_ids")
    elif token_ids:
        unique_ids = set(token_ids)
        if len(vocabulary) != endpoint.tokenizer_vocabulary_size:
            failures.append("tokenizer_vocabulary_size")
        if len(unique_ids) != len(token_ids):
            failures.append("duplicate_token_ids")
        if min(unique_ids) != 0 or max(unique_ids) != endpoint.tokenizer_max_token_id:
            failures.append("tokenizer_token_id_range")
        if len(unique_ids) != endpoint.tokenizer_max_token_id + 1:
            failures.append("tokenizer_token_id_gaps")

    if getattr(tokenizer, "vocab_size", None) != (
        endpoint.tokenizer_base_vocabulary_size
    ):
        failures.append("tokenizer_base_vocabulary_size")
    try:
        tokenizer_length = len(tokenizer)
    except Exception:
        tokenizer_length = None
    if tokenizer_length != endpoint.tokenizer_vocabulary_size:
        failures.append("tokenizer_length")

    special_values = {
        "bos_token": endpoint.bos_token,
        "bos_token_id": endpoint.bos_token_id,
        "eos_token": endpoint.eos_token,
        "eos_token_id": endpoint.eos_token_id,
        "pad_token": endpoint.pad_token,
        "pad_token_id": endpoint.pad_token_id,
        "unk_token": endpoint.unk_token,
        "unk_token_id": endpoint.unk_token_id,
    }
    for name, expected in special_values.items():
        if getattr(tokenizer, name, None) != expected:
            failures.append(name)

    additional_ids = tuple(
        getattr(tokenizer, "additional_special_tokens_ids", None) or ()
    )
    if additional_ids != endpoint.additional_special_token_ids:
        failures.append("additional_special_token_ids")
    if getattr(tokenizer, "model_max_length", None) != endpoint.model_max_length:
        failures.append("model_max_length")
    if getattr(tokenizer, "padding_side", None) != endpoint.padding_side:
        failures.append("padding_side")

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        failures.append("chat_template")
    elif sha256_hex(chat_template.encode("utf-8")) != (
        endpoint.tokenizer_chat_template_hash
    ):
        failures.append("tokenizer_chat_template_hash")

    if vocabulary and not any(
        token.startswith(endpoint.word_boundary_marker) for token in vocabulary
    ):
        failures.append("word_boundary_marker")

    if failures:
        raise TokenizerValidationError(
            f"tokenizer {endpoint.profile_id!r} differs from its pinned artifact: "
            + ", ".join(dict.fromkeys(failures))
        )

    return ValidatedTokenizer(
        endpoint=endpoint,
        tokenizer=tokenizer,
        artifact_sha256=artifact_sha256,
        artifact_path=Path(artifact_path) if artifact_path is not None else None,
    )


def _bind_existing_pad_token(
    endpoint: TokenizerEndpoint,
    tokenizer: Any,
) -> None:
    if endpoint.pad_token is None or endpoint.pad_token_id is None:
        raise TokenizerValidationError(
            "existing pad-token binding requires a pinned token and ID"
        )

    try:
        vocabulary_before = tokenizer.get_vocab()
        length_before = len(tokenizer)
    except Exception as exc:
        raise TokenizerValidationError(
            "tokenizer state could not be recorded before pad-token binding"
        ) from exc
    if not isinstance(vocabulary_before, dict):
        raise TokenizerValidationError(
            "tokenizer vocabulary is not a dictionary"
        )
    vocabulary_size_before = getattr(tokenizer, "vocab_size", None)

    actual_id = vocabulary_before.get(endpoint.pad_token)
    if actual_id is None:
        raise TokenizerValidationError(
            f"pinned pad token {endpoint.pad_token!r} is absent from the vocabulary"
        )
    if actual_id != endpoint.pad_token_id:
        raise TokenizerValidationError(
            f"pinned pad token {endpoint.pad_token!r} has ID {actual_id}, "
            f"expected {endpoint.pad_token_id}"
        )

    tokenizer.pad_token = endpoint.pad_token

    try:
        vocabulary_after = tokenizer.get_vocab()
        length_after = len(tokenizer)
    except Exception as exc:
        raise TokenizerValidationError(
            "tokenizer state could not be verified after pad-token binding"
        ) from exc
    vocabulary_size_after = getattr(tokenizer, "vocab_size", None)
    if (
        vocabulary_after != vocabulary_before
        or length_after != length_before
        or vocabulary_size_after != vocabulary_size_before
    ):
        raise TokenizerValidationError(
            "binding the existing pad token changed the tokenizer vocabulary"
        )
    if getattr(tokenizer, "pad_token_id", None) != endpoint.pad_token_id:
        raise TokenizerValidationError(
            "bound pad-token ID differs from the pinned tokenizer endpoint"
        )


def load_pinned_tokenizer(
    endpoint: TokenizerEndpoint,
    *,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> ValidatedTokenizer:
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "pinned tokenizer validation requires the ML dependencies"
        ) from exc

    cache_value = str(cache_dir) if cache_dir is not None else None
    artifact_path = Path(
        hf_hub_download(
            repo_id=endpoint.tokenizer_id,
            filename="tokenizer.json",
            revision=endpoint.tokenizer_revision,
            cache_dir=cache_value,
            token=token,
            local_files_only=local_files_only,
        )
    )
    tokenizer_arguments = {
        "revision": endpoint.tokenizer_revision,
        "trust_remote_code": False,
        "use_fast": True,
        "cache_dir": cache_value,
        "token": token,
        "local_files_only": local_files_only,
    }
    if endpoint.fix_mistral_regex:
        tokenizer_arguments["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        endpoint.tokenizer_id,
        **tokenizer_arguments,
    )
    if endpoint.bind_existing_pad_token:
        _bind_existing_pad_token(endpoint, tokenizer)
    return validate_loaded_tokenizer(
        endpoint,
        tokenizer,
        artifact_sha256=tokenizer_artifact_sha256(artifact_path),
        artifact_path=artifact_path,
    )
