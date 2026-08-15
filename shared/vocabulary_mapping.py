from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field, ValidationError, model_validator
from rapidfuzz.distance import Levenshtein

from shared.alignment_profiles import (
    BidirectionalAlignmentProfile,
    TokenizerEndpoint,
)
from shared.crypto import sha256_hex
from shared.protocol import ContractModel, HASH_PATTERN
from shared.storage import JsonFileStore
from shared.tokenizer_validation import ValidatedTokenizer


MAPPING_SCHEMA_VERSION = "1.0"
MAPPING_RULES_ID = "fedmkt_levenshtein_lowest_target_id_v1"
MappingDirection = Literal["client_to_host", "host_to_client"]


class VocabularyMappingError(ValueError):
    pass


class UnaddressableTokenId(VocabularyMappingError):
    pass


class VocabularyMappingCacheError(VocabularyMappingError):
    pass


class TokenizerMappingIdentity(ContractModel):
    role: Literal["client", "host"]
    profile_id: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    tokenizer_class: str = Field(min_length=1)
    model_vocabulary_size: int = Field(ge=1)
    tokenizer_vocabulary_size: int = Field(ge=1)
    tokenizer_max_token_id: int = Field(ge=0)
    tokenizer_artifact_sha256: str = Field(pattern=HASH_PATTERN)
    word_boundary_marker: str = Field(min_length=1, max_length=8)
    endpoint_sha256: str = Field(pattern=HASH_PATTERN)


class VocabularyMappingIdentity(ContractModel):
    mapping_schema_version: str
    mapping_rules_id: str
    alignment_profile_id: str = Field(min_length=1)
    direction: MappingDirection
    source: TokenizerMappingIdentity
    target: TokenizerMappingIdentity
    requested_token_ids: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_requested_ids(self) -> "VocabularyMappingIdentity":
        if self.mapping_schema_version != MAPPING_SCHEMA_VERSION:
            raise ValueError("unsupported vocabulary-mapping schema version")
        if self.mapping_rules_id != MAPPING_RULES_ID:
            raise ValueError("unsupported vocabulary-mapping rules")
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in self.requested_token_ids
        ):
            raise ValueError("requested token IDs must be non-negative integers")
        if tuple(sorted(set(self.requested_token_ids))) != self.requested_token_ids:
            raise ValueError("requested token IDs must be sorted and unique")
        return self


class VocabularyMappingEntry(ContractModel):
    source_token_id: int = Field(ge=0)
    source_token: str
    normalized_source_token: str
    target_token_id: int = Field(ge=0)
    target_token: str
    levenshtein_distance: int = Field(ge=0)
    exact_match: bool


class DemandVocabularyMapping(ContractModel):
    schema_version: str
    mapping_rules_id: str
    identity: VocabularyMappingIdentity
    identity_sha256: str = Field(pattern=HASH_PATTERN)
    entries: tuple[VocabularyMappingEntry, ...]
    payload_sha256: str = Field(pattern=HASH_PATTERN)

    @classmethod
    def create(
        cls,
        identity: VocabularyMappingIdentity,
        entries: Iterable[VocabularyMappingEntry],
    ) -> "DemandVocabularyMapping":
        values = tuple(entries)
        identity_sha256 = sha256_hex(identity.model_dump(mode="json"))
        payload = {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "mapping_rules_id": MAPPING_RULES_ID,
            "identity": identity.model_dump(mode="json"),
            "identity_sha256": identity_sha256,
            "entries": [entry.model_dump(mode="json") for entry in values],
        }
        return cls(
            **payload,
            payload_sha256=sha256_hex(payload),
        )

    def hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mapping_rules_id": self.mapping_rules_id,
            "identity": self.identity.model_dump(mode="json"),
            "identity_sha256": self.identity_sha256,
            "entries": [entry.model_dump(mode="json") for entry in self.entries],
        }

    @model_validator(mode="after")
    def validate_hashes_and_entries(self) -> "DemandVocabularyMapping":
        if self.schema_version != MAPPING_SCHEMA_VERSION:
            raise ValueError("unsupported vocabulary-mapping schema version")
        if self.mapping_rules_id != MAPPING_RULES_ID:
            raise ValueError("unsupported vocabulary-mapping rules")
        if self.identity_sha256 != sha256_hex(
            self.identity.model_dump(mode="json")
        ):
            raise ValueError("vocabulary-mapping identity hash differs")
        source_ids = tuple(entry.source_token_id for entry in self.entries)
        if source_ids != self.identity.requested_token_ids:
            raise ValueError("mapping entries differ from the requested token set")
        if self.payload_sha256 != sha256_hex(self.hash_payload()):
            raise ValueError("vocabulary-mapping payload hash differs")
        return self

    def as_upstream_token_mapping(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for entry in self.entries:
            values[entry.normalized_source_token] = entry.target_token
        return values


@dataclass(frozen=True, slots=True)
class VocabularyMappingResolution:
    mapping: DemandVocabularyMapping
    cache_path: Path
    cache_hit: bool


def _tokenizer_identity(endpoint: TokenizerEndpoint) -> TokenizerMappingIdentity:
    return TokenizerMappingIdentity(
        role=endpoint.role,
        profile_id=endpoint.profile_id,
        tokenizer_id=endpoint.tokenizer_id,
        tokenizer_revision=endpoint.tokenizer_revision,
        tokenizer_class=endpoint.tokenizer_class,
        model_vocabulary_size=endpoint.vocabulary_size,
        tokenizer_vocabulary_size=endpoint.tokenizer_vocabulary_size,
        tokenizer_max_token_id=endpoint.tokenizer_max_token_id,
        tokenizer_artifact_sha256=endpoint.tokenizer_artifact_sha256,
        word_boundary_marker=endpoint.word_boundary_marker,
        endpoint_sha256=sha256_hex(asdict(endpoint)),
    )


def _direction_endpoints(
    profile: BidirectionalAlignmentProfile,
    direction: MappingDirection,
) -> tuple[TokenizerEndpoint, TokenizerEndpoint]:
    if direction == "client_to_host":
        return profile.client, profile.host
    if direction == "host_to_client":
        return profile.host, profile.client
    raise VocabularyMappingError(f"unsupported mapping direction: {direction!r}")


def _requested_token_ids(values: Iterable[int]) -> tuple[int, ...]:
    requested: set[int] = set()
    for value in values:
        if type(value) is not int or value < 0:
            raise VocabularyMappingError(
                "requested token IDs must be non-negative integers"
            )
        requested.add(value)
    if not requested:
        raise VocabularyMappingError("at least one source token ID is required")
    return tuple(sorted(requested))


def _build_entries(
    source: ValidatedTokenizer,
    target: ValidatedTokenizer,
    requested_token_ids: tuple[int, ...],
) -> tuple[VocabularyMappingEntry, ...]:
    source_vocab = source.tokenizer.get_vocab()
    target_vocab = target.tokenizer.get_vocab()
    target_tokens = [
        token
        for token, _token_id in sorted(
            target_vocab.items(), key=lambda item: (item[1], item[0])
        )
    ]
    if not target_tokens:
        raise VocabularyMappingError("target tokenizer vocabulary is empty")

    entries: list[VocabularyMappingEntry] = []
    for source_token_id in requested_token_ids:
        source_token = _addressable_source_token(
            source,
            source_vocab,
            source_token_id,
        )
        normalized = source_token.replace(
            source.endpoint.word_boundary_marker,
            target.endpoint.word_boundary_marker,
        )
        if normalized in target_vocab:
            target_token = normalized
            distance = 0
            exact_match = True
        else:
            # Python's min() keeps the first equal-distance candidate. Sorting
            # by token ID makes that inherited tie behavior reproducible even
            # when tokenizer vocabulary dictionaries were built differently.
            target_token = min(
                target_tokens,
                key=lambda value: Levenshtein.distance(normalized, value),
            )
            distance = Levenshtein.distance(normalized, target_token)
            exact_match = False
        entries.append(
            VocabularyMappingEntry(
                source_token_id=source_token_id,
                source_token=source_token,
                normalized_source_token=normalized,
                target_token_id=target_vocab[target_token],
                target_token=target_token,
                levenshtein_distance=distance,
                exact_match=exact_match,
            )
        )
    return tuple(entries)


def _addressable_source_token(
    source: ValidatedTokenizer,
    source_vocab: dict[str, int],
    source_token_id: int,
) -> str:
    if source_token_id >= source.endpoint.vocabulary_size:
        raise UnaddressableTokenId(
            f"source token ID {source_token_id} exceeds the model-output "
            f"vocabulary for {source.endpoint.profile_id!r}"
        )
    try:
        converted = source.tokenizer.convert_ids_to_tokens([source_token_id])
    except Exception as exc:
        raise UnaddressableTokenId(
            f"source token ID {source_token_id} cannot be converted by "
            f"{source.endpoint.profile_id!r}"
        ) from exc
    if (
        not isinstance(converted, list)
        or len(converted) != 1
        or not isinstance(converted[0], str)
        or source_vocab.get(converted[0]) != source_token_id
    ):
        raise UnaddressableTokenId(
            f"source token ID {source_token_id} is not addressable by "
            f"{source.endpoint.profile_id!r}"
        )
    return converted[0]


def validate_addressable_token_ids(
    source: ValidatedTokenizer,
    requested_token_ids: Iterable[int],
) -> tuple[int, ...]:
    """Validate a package's demanded IDs without creating a cache entry."""

    requested = _requested_token_ids(requested_token_ids)
    source_vocab = source.tokenizer.get_vocab()
    for source_token_id in requested:
        _addressable_source_token(source, source_vocab, source_token_id)
    return requested


class VocabularyMappingCache:
    def __init__(self, root: str | Path):
        self.store = JsonFileStore(root)

    def resolve(
        self,
        *,
        profile: BidirectionalAlignmentProfile,
        direction: MappingDirection,
        source: ValidatedTokenizer,
        target: ValidatedTokenizer,
        requested_token_ids: Iterable[int],
    ) -> VocabularyMappingResolution:
        expected_source, expected_target = _direction_endpoints(profile, direction)
        if source.endpoint != expected_source or target.endpoint != expected_target:
            raise VocabularyMappingError(
                f"validated tokenizers do not match {direction!r} for "
                f"{profile.profile_id!r}"
            )
        if (
            source.artifact_sha256 != source.endpoint.tokenizer_artifact_sha256
            or target.artifact_sha256 != target.endpoint.tokenizer_artifact_sha256
        ):
            raise VocabularyMappingError(
                "validated tokenizer artifact hash differs from its endpoint"
            )

        requested = _requested_token_ids(requested_token_ids)
        identity = VocabularyMappingIdentity(
            mapping_schema_version=MAPPING_SCHEMA_VERSION,
            mapping_rules_id=MAPPING_RULES_ID,
            alignment_profile_id=profile.profile_id,
            direction=direction,
            source=_tokenizer_identity(source.endpoint),
            target=_tokenizer_identity(target.endpoint),
            requested_token_ids=requested,
        )
        identity_sha256 = sha256_hex(identity.model_dump(mode="json"))
        relative = (
            f"{profile.profile_version}/{direction}/{identity_sha256}.json"
        )
        path = self.store.path(relative)
        if self.store.exists(relative):
            return VocabularyMappingResolution(
                mapping=self._load(relative, identity),
                cache_path=path,
                cache_hit=True,
            )

        mapping = DemandVocabularyMapping.create(
            identity,
            _build_entries(source, target, requested),
        )
        try:
            self.store.write_json_if_absent(
                relative,
                mapping.model_dump(mode="json"),
            )
        except FileExistsError:
            return VocabularyMappingResolution(
                mapping=self._load(relative, identity),
                cache_path=path,
                cache_hit=True,
            )
        return VocabularyMappingResolution(
            mapping=mapping,
            cache_path=path,
            cache_hit=False,
        )

    def _load(
        self,
        relative: str,
        expected_identity: VocabularyMappingIdentity,
    ) -> DemandVocabularyMapping:
        try:
            mapping = DemandVocabularyMapping.model_validate(
                self.store.read_json(relative)
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise VocabularyMappingCacheError(
                f"cached vocabulary mapping {relative!r} is invalid"
            ) from exc
        if mapping.identity != expected_identity:
            raise VocabularyMappingCacheError(
                f"cached vocabulary mapping {relative!r} has stale identity"
            )
        return mapping
