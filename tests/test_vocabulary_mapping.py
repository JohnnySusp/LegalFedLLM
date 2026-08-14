from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


try:
    from shared.alignment_profiles import (
        POC_DTW_PROFILE,
        BidirectionalAlignmentProfile,
        TokenizerEndpoint,
    )
    from shared.fedmkt_core.ml.vocab_mapping import find_best_mapping
    from shared.tokenizer_validation import ValidatedTokenizer
    from shared.vocabulary_mapping import (
        UnaddressableTokenId,
        VocabularyMappingError,
        VocabularyMappingCache,
        VocabularyMappingCacheError,
    )
except ModuleNotFoundError as exc:
    MAPPING_IMPORT_ERROR = exc
else:
    MAPPING_IMPORT_ERROR = None


class FakeTokenizer:
    def __init__(self, vocabulary: dict[str, int]):
        self.vocabulary = vocabulary
        self.tokens = {token_id: token for token, token_id in vocabulary.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str | None]:
        return [self.tokens.get(token_id) for token_id in token_ids]


def endpoint(
    *,
    role: str,
    profile_id: str,
    marker: str,
    artifact_hash: str,
    vocabulary_size: int,
    tokenizer_vocabulary_size: int,
) -> TokenizerEndpoint:
    return TokenizerEndpoint(
        role=role,
        profile_id=profile_id,
        model_id=f"test/{profile_id}",
        model_revision="revision",
        model_class="FakeForCausalLM",
        model_type="fake",
        tokenizer_id=f"test/{profile_id}",
        tokenizer_revision="revision",
        tokenizer_class="FakeTokenizer",
        vocabulary_size=vocabulary_size,
        tokenizer_chat_template_hash="a" * 64,
        chat_template_mode="standard",
        tokenizer_artifact_sha256=artifact_hash,
        tokenizer_base_vocabulary_size=tokenizer_vocabulary_size,
        tokenizer_vocabulary_size=tokenizer_vocabulary_size,
        tokenizer_max_token_id=tokenizer_vocabulary_size - 1,
        word_boundary_marker=marker,
        bos_token=None,
        bos_token_id=None,
        eos_token=None,
        eos_token_id=None,
        pad_token=None,
        pad_token_id=None,
        unk_token=None,
        unk_token_id=None,
        additional_special_token_ids=(),
        model_max_length=128,
        padding_side="right",
    )


@unittest.skipUnless(
    MAPPING_IMPORT_ERROR is None,
    f"optional ML dependencies are unavailable: {MAPPING_IMPORT_ERROR}",
)
class DemandVocabularyMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_endpoint = endpoint(
            role="client",
            profile_id="client-test",
            marker="Ġ",
            artifact_hash="1" * 64,
            vocabulary_size=32,
            tokenizer_vocabulary_size=13,
        )
        self.host_endpoint = endpoint(
            role="host",
            profile_id="host-test",
            marker="▁",
            artifact_hash="2" * 64,
            vocabulary_size=16,
            tokenizer_vocabulary_size=4,
        )
        self.profile = BidirectionalAlignmentProfile(
            profile_id="dtw:test-v1",
            strategy="dtw",
            profile_version="test-v1",
            client=self.client_endpoint,
            host=self.host_endpoint,
            client_to_host_owner="coordinator",
            host_to_client_owner="client",
        )
        self.client = ValidatedTokenizer(
            endpoint=self.client_endpoint,
            tokenizer=FakeTokenizer(
                {
                    "unused-0": 0,
                    "unused-1": 1,
                    "unused-2": 2,
                    "unused-3": 3,
                    "unused-4": 4,
                    "unused-5": 5,
                    "unused-6": 6,
                    "unused-7": 7,
                    "unused-8": 8,
                    "unused-9": 9,
                    "Ġlaw": 10,
                    "Ġlegal": 11,
                    "zz": 12,
                }
            ),
            artifact_sha256=self.client_endpoint.tokenizer_artifact_sha256,
        )
        self.host = ValidatedTokenizer(
            endpoint=self.host_endpoint,
            tokenizer=FakeTokenizer(
                {"▁law": 0, "▁legal": 1, "za": 2, "zb": 3}
            ),
            artifact_sha256=self.host_endpoint.tokenizer_artifact_sha256,
        )

    def test_demanded_results_match_inherited_mapping_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = VocabularyMappingCache(directory).resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[12, 10, 10],
            )

        self.assertFalse(result.cache_hit)
        self.assertEqual(
            result.mapping.identity.requested_token_ids,
            (10, 12),
        )
        self.assertEqual(
            result.mapping.identity.mapping_rules_id,
            "fedmkt_levenshtein_lowest_target_id_v1",
        )
        self.assertEqual(len(result.mapping.entries), 2)
        target_tokens = ["▁law", "▁legal", "za", "zb"]
        expected = dict(
            find_best_mapping(
                token,
                target_tokens,
                "Ġ",
                "▁",
            )
            for token in ("Ġlaw", "zz")
        )
        self.assertEqual(result.mapping.as_upstream_token_mapping(), expected)
        self.assertEqual(result.mapping.entries[1].target_token, "za")
        self.assertEqual(result.mapping.entries[1].levenshtein_distance, 1)

    def test_cache_is_persistent_hashed_and_exact_set_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = VocabularyMappingCache(directory)
            first = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[10, 11],
            )
            second = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[11, 10],
            )
            different = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[10],
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.mapping, second.mapping)
            self.assertNotEqual(first.cache_path, different.cache_path)
            files = sorted(Path(directory).rglob("*.json"))
            self.assertEqual(files, sorted([first.cache_path, different.cache_path]))

    def test_corrupt_cache_is_rejected_without_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = VocabularyMappingCache(directory)
            result = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[10],
            )
            payload = json.loads(result.cache_path.read_text(encoding="utf-8"))
            payload["entries"][0]["target_token"] = "tampered"
            result.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                VocabularyMappingCacheError,
                "is invalid",
            ):
                cache.resolve(
                    profile=self.profile,
                    direction="client_to_host",
                    source=self.client,
                    target=self.host,
                    requested_token_ids=[10],
                )
            self.assertEqual(
                json.loads(result.cache_path.read_text(encoding="utf-8"))[
                    "entries"
                ][0]["target_token"],
                "tampered",
            )

    def test_stale_but_internally_valid_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = VocabularyMappingCache(directory)
            first = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[10],
            )
            second = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[11],
            )
            first.cache_path.write_bytes(second.cache_path.read_bytes())

            with self.assertRaisesRegex(
                VocabularyMappingCacheError,
                "stale identity",
            ):
                cache.resolve(
                    profile=self.profile,
                    direction="client_to_host",
                    source=self.client,
                    target=self.host,
                    requested_token_ids=[10],
                )

    def test_both_approved_directions_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = VocabularyMappingCache(directory)
            client_to_host = cache.resolve(
                profile=self.profile,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[10],
            )
            host_to_client = cache.resolve(
                profile=self.profile,
                direction="host_to_client",
                source=self.host,
                target=self.client,
                requested_token_ids=[0],
            )

        self.assertNotEqual(client_to_host.cache_path, host_to_client.cache_path)
        self.assertEqual(
            host_to_client.mapping.entries[0].target_token_id,
            10,
        )

    def test_role_reversed_tokenizers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                VocabularyMappingError,
                "do not match",
            ):
                VocabularyMappingCache(directory).resolve(
                    profile=self.profile,
                    direction="client_to_host",
                    source=self.host,
                    target=self.client,
                    requested_token_ids=[0],
                )

    def test_unaddressable_model_output_id_is_rejected(self) -> None:
        qwen = ValidatedTokenizer(
            endpoint=POC_DTW_PROFILE.client,
            tokenizer=FakeTokenizer({"</think>": 151668}),
            artifact_sha256=(
                POC_DTW_PROFILE.client.tokenizer_artifact_sha256
            ),
        )
        granite = ValidatedTokenizer(
            endpoint=POC_DTW_PROFILE.host,
            tokenizer=FakeTokenizer({"<|start_of_role|>": 49152}),
            artifact_sha256=POC_DTW_PROFILE.host.tokenizer_artifact_sha256,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                UnaddressableTokenId,
                "151669 is not addressable",
            ):
                VocabularyMappingCache(directory).resolve(
                    profile=POC_DTW_PROFILE,
                    direction="client_to_host",
                    source=qwen,
                    target=granite,
                    requested_token_ids=[151669],
                )


if __name__ == "__main__":
    unittest.main()
