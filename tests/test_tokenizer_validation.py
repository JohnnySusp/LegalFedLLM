from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from shared.alignment_profiles import (
    MISTRAL_NEMO_HOST_ENDPOINT,
    POC_DTW_PROFILE,
)
from shared.crypto import sha256_hex
from shared.tokenizer_validation import (
    TokenizerValidationError,
    load_pinned_tokenizer,
    tokenizer_artifact_sha256,
    validate_loaded_tokenizer,
)


class SyntheticFastTokenizer:
    is_fast = True

    def __init__(self, *, chat_template: str = "synthetic chat") -> None:
        self.vocabulary = {"plain": 0, "Ġlaw": 1, "<eos>": 2}
        self.vocab_size = 2
        self.bos_token = None
        self.bos_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self._pad_token = None
        self.pad_token_id = None
        self.unk_token = None
        self.unk_token_id = None
        self.additional_special_tokens_ids = ()
        self.model_max_length = 128
        self.padding_side = "right"
        self.chat_template = chat_template

    def __len__(self) -> int:
        return len(self.vocabulary)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value) -> None:
        self._pad_token = value
        self.pad_token_id = (
            None if value is None else self.vocabulary.get(value)
        )


class SyntheticPadTokenizer(SyntheticFastTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.vocabulary["<pad>"] = 3
        self.vocab_size = 4


def synthetic_endpoint():
    chat_template = "synthetic chat"
    return replace(
        POC_DTW_PROFILE.client,
        tokenizer_class="SyntheticFastTokenizer",
        tokenizer_artifact_sha256="1" * 64,
        tokenizer_base_vocabulary_size=2,
        tokenizer_vocabulary_size=3,
        tokenizer_max_token_id=2,
        tokenizer_chat_template_hash=sha256_hex(
            chat_template.encode("utf-8")
        ),
        word_boundary_marker="Ġ",
        bos_token=None,
        bos_token_id=None,
        eos_token="<eos>",
        eos_token_id=2,
        pad_token=None,
        pad_token_id=None,
        unk_token=None,
        unk_token_id=None,
        additional_special_token_ids=(),
        model_max_length=128,
        padding_side="right",
    )


def synthetic_pad_endpoint():
    return replace(
        synthetic_endpoint(),
        tokenizer_class="SyntheticPadTokenizer",
        tokenizer_base_vocabulary_size=4,
        tokenizer_vocabulary_size=4,
        tokenizer_max_token_id=3,
        pad_token="<pad>",
        pad_token_id=3,
        bind_existing_pad_token=True,
    )


class TokenizerValidationTests(unittest.TestCase):
    def _load_with_fake_dependencies(self, endpoint, tokenizer):
        arguments = {}
        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub.hf_hub_download = lambda **kwargs: "/tmp/tokenizer.json"
        transformers = ModuleType("transformers")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                arguments["model_id"] = model_id
                arguments.update(kwargs)
                return tokenizer

        transformers.AutoTokenizer = FakeAutoTokenizer
        with patch.dict(
            sys.modules,
            {
                "huggingface_hub": huggingface_hub,
                "transformers": transformers,
            },
        ), patch(
            "shared.tokenizer_validation.tokenizer_artifact_sha256",
            return_value=endpoint.tokenizer_artifact_sha256,
        ):
            validated = load_pinned_tokenizer(endpoint)
        return validated, arguments

    def test_existing_profiles_keep_default_loading_policies(self) -> None:
        for endpoint in (POC_DTW_PROFILE.client, POC_DTW_PROFILE.host):
            with self.subTest(profile_id=endpoint.profile_id):
                self.assertFalse(endpoint.fix_mistral_regex)
                self.assertFalse(endpoint.bind_existing_pad_token)

    def test_loader_passes_corrected_mistral_regex_only_when_requested(
        self,
    ) -> None:
        _, default_arguments = self._load_with_fake_dependencies(
            synthetic_endpoint(),
            SyntheticFastTokenizer(),
        )
        corrected_endpoint = replace(
            synthetic_endpoint(),
            fix_mistral_regex=True,
        )
        _, corrected_arguments = self._load_with_fake_dependencies(
            corrected_endpoint,
            SyntheticFastTokenizer(),
        )

        self.assertNotIn("fix_mistral_regex", default_arguments)
        self.assertIs(corrected_arguments["fix_mistral_regex"], True)

    def test_loader_binds_existing_pad_without_changing_vocabulary(
        self,
    ) -> None:
        endpoint = synthetic_pad_endpoint()
        tokenizer = SyntheticPadTokenizer()
        vocabulary_before = tokenizer.get_vocab()
        length_before = len(tokenizer)
        vocabulary_size_before = tokenizer.vocab_size

        validated, _ = self._load_with_fake_dependencies(endpoint, tokenizer)

        self.assertIs(validated.tokenizer, tokenizer)
        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.pad_token_id, 3)
        self.assertEqual(tokenizer.get_vocab(), vocabulary_before)
        self.assertEqual(len(tokenizer), length_before)
        self.assertEqual(tokenizer.vocab_size, vocabulary_size_before)

    def test_existing_pad_binding_rejects_missing_token_or_wrong_id(self) -> None:
        cases = (
            (
                replace(
                    synthetic_pad_endpoint(),
                    pad_token="<missing>",
                ),
                "absent from the vocabulary",
            ),
            (
                replace(synthetic_pad_endpoint(), pad_token_id=2),
                "has ID 3, expected 2",
            ),
        )
        for endpoint, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TokenizerValidationError, message):
                    self._load_with_fake_dependencies(
                        endpoint,
                        SyntheticPadTokenizer(),
                    )

    def test_poc_tokenizer_fingerprints_are_exact(self) -> None:
        client = POC_DTW_PROFILE.client
        host = POC_DTW_PROFILE.host

        self.assertEqual(
            client.tokenizer_artifact_sha256,
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        )
        self.assertEqual(client.tokenizer_id, "Qwen/Qwen3-1.7B")
        self.assertEqual(client.tokenizer_class, "Qwen2TokenizerFast")
        self.assertEqual(
            (
                client.tokenizer_base_vocabulary_size,
                client.tokenizer_vocabulary_size,
                client.tokenizer_max_token_id,
                client.word_boundary_marker,
            ),
            (151643, 151669, 151668, "Ġ"),
        )
        self.assertEqual(client.pad_token_id, 151643)
        self.assertEqual(client.eos_token_id, 151645)
        self.assertEqual(
            client.additional_special_token_ids,
            tuple(range(151644, 151657)),
        )

        self.assertEqual(
            host.tokenizer_artifact_sha256,
            "91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e",
        )
        self.assertEqual(
            host.tokenizer_id,
            "ibm-granite/granite-3.3-2b-instruct",
        )
        self.assertEqual(host.tokenizer_class, "GPT2TokenizerFast")
        self.assertEqual(
            (
                host.tokenizer_base_vocabulary_size,
                host.tokenizer_vocabulary_size,
                host.tokenizer_max_token_id,
                host.word_boundary_marker,
            ),
            (49152, 49159, 49158, "Ġ"),
        )
        self.assertEqual(host.bos_token, "<|end_of_text|>")
        self.assertEqual(host.bos_token_id, 0)
        self.assertEqual(host.eos_token, "<|end_of_text|>")
        self.assertEqual(host.eos_token_id, 0)
        self.assertEqual(host.pad_token, "<|end_of_text|>")
        self.assertEqual(host.pad_token_id, 0)
        self.assertEqual(host.unk_token, "<|end_of_text|>")
        self.assertEqual(host.unk_token_id, 0)
        self.assertEqual(
            host.additional_special_token_ids,
            tuple(range(49152, 49159)),
        )
        self.assertEqual(host.model_max_length, 9223372036854775807)
        self.assertEqual(host.padding_side, "left")

    def test_matching_loaded_tokenizer_is_accepted(self) -> None:
        endpoint = synthetic_endpoint()
        tokenizer = SyntheticFastTokenizer()

        validated = validate_loaded_tokenizer(
            endpoint,
            tokenizer,
            artifact_sha256=endpoint.tokenizer_artifact_sha256,
        )

        self.assertIs(validated.tokenizer, tokenizer)
        self.assertEqual(validated.endpoint, endpoint)

    def test_changed_artifact_class_vocabulary_or_template_fails(self) -> None:
        endpoint = synthetic_endpoint()
        cases = []

        changed_class = type(
            "SubstitutedTokenizer",
            (SyntheticFastTokenizer,),
            {},
        )()
        cases.append(("tokenizer_class", changed_class, "1" * 64))

        changed_vocabulary = SyntheticFastTokenizer()
        changed_vocabulary.vocabulary["extra"] = 3
        cases.append(
            ("tokenizer_vocabulary_size", changed_vocabulary, "1" * 64)
        )

        cases.append(
            (
                "tokenizer_chat_template_hash",
                SyntheticFastTokenizer(chat_template="changed chat"),
                "1" * 64,
            )
        )
        cases.append(
            (
                "tokenizer_artifact_sha256",
                SyntheticFastTokenizer(),
                "2" * 64,
            )
        )
        changed_special = SyntheticFastTokenizer()
        changed_special.eos_token_id = 1
        cases.append(("eos_token_id", changed_special, "1" * 64))

        for expected, tokenizer, artifact_hash in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    TokenizerValidationError,
                    expected,
                ):
                    validate_loaded_tokenizer(
                        endpoint,
                        tokenizer,
                        artifact_sha256=artifact_hash,
                    )

    def test_tokenizer_file_hash_is_streamed_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tokenizer.json")
            payload = b'{"version":"1.0","unicode":"\xce\xb4\xce\xaf\xce\xba\xce\xb1\xce\xb9\xce\xbf"}'
            path.write_bytes(payload)

            self.assertEqual(
                tokenizer_artifact_sha256(path),
                sha256_hex(payload),
            )


RUN_REAL_TOKENIZER_TESTS = os.getenv(
    "LEGALFEDLLM_RUN_REAL_TOKENIZER_TESTS",
    "",
).lower() in {"1", "true", "yes"}

RUN_REAL_NEMO_TOKENIZER_TESTS = os.getenv(
    "LEGALFEDLLM_RUN_REAL_NEMO_TOKENIZER_TESTS",
    "",
).lower() in {"1", "true", "yes"}


@unittest.skipUnless(
    RUN_REAL_TOKENIZER_TESTS,
    "set LEGALFEDLLM_RUN_REAL_TOKENIZER_TESTS=true for pinned artifacts",
)
class RealPinnedTokenizerAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cache_dir = os.getenv("LEGALFEDLLM_TOKENIZER_CACHE") or None
        token = os.getenv("HF_TOKEN") or None
        local_only = os.getenv(
            "LEGALFEDLLM_TOKENIZER_LOCAL_FILES_ONLY",
            "",
        ).lower() in {"1", "true", "yes"}
        cls.client = load_pinned_tokenizer(
            POC_DTW_PROFILE.client,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_only,
        )
        cls.host = load_pinned_tokenizer(
            POC_DTW_PROFILE.host,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_only,
        )

    def test_actual_tokenizers_map_and_align_in_both_directions(self) -> None:
        from shared.fedmkt_core.ml.token_alignment import transform_step_logits
        from shared.vocabulary_mapping import VocabularyMappingCache

        text = "Ελληνικό δίκαιο. Article 5: contract"
        client_ids = self.client.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        host_ids = self.host.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        self.assertTrue(client_ids)
        self.assertTrue(host_ids)

        with tempfile.TemporaryDirectory() as directory:
            cache = VocabularyMappingCache(directory)
            client_to_host = cache.resolve(
                profile=POC_DTW_PROFILE,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[client_ids[0]],
            )
            host_to_client = cache.resolve(
                profile=POC_DTW_PROFILE,
                direction="host_to_client",
                source=self.host,
                target=self.client,
                requested_token_ids=[host_ids[0]],
            )
            reused = cache.resolve(
                profile=POC_DTW_PROFILE,
                direction="client_to_host",
                source=self.client,
                target=self.host,
                requested_token_ids=[client_ids[0]],
            )

            self.assertFalse(client_to_host.cache_hit)
            self.assertFalse(host_to_client.cache_hit)
            self.assertTrue(reused.cache_hit)
            self.assertEqual(client_to_host.mapping, reused.mapping)
            self._assert_alignment(
                base=self.host,
                blending=self.client,
                base_ids=host_ids,
                blending_ids=client_ids,
                top_k_id=client_ids[0],
                mapping=client_to_host.mapping.as_upstream_token_mapping(),
                transform=transform_step_logits,
            )
            self._assert_alignment(
                base=self.client,
                blending=self.host,
                base_ids=client_ids,
                blending_ids=host_ids,
                top_k_id=host_ids[0],
                mapping=host_to_client.mapping.as_upstream_token_mapping(),
                transform=transform_step_logits,
            )

    def _assert_alignment(
        self,
        *,
        base,
        blending,
        base_ids: list[int],
        blending_ids: list[int],
        top_k_id: int,
        mapping: dict[str, str],
        transform,
    ) -> None:
        arguments = {
            "base_model_tokenizer": base.tokenizer,
            "blending_model_tokenizer": blending.tokenizer,
            "base_model_vocab": base.tokenizer.get_vocab(),
            "base_model_input_ids": base_ids,
            "blending_model_input_ids": blending_ids,
            "blending_model_per_step_logits": [
                [0.75] for _ in blending_ids
            ],
            "blending_model_per_step_indices": [
                [top_k_id] for _ in blending_ids
            ],
            "blending_to_base_mapping": mapping,
            "base_model_special_token": base.endpoint.word_boundary_marker,
            "blending_model_special_token": (
                blending.endpoint.word_boundary_marker
            ),
        }
        logits, indices = transform(**arguments)
        repeated_logits, repeated_indices = transform(**arguments)

        self.assertEqual(logits, repeated_logits)
        self.assertEqual(indices, repeated_indices)
        self.assertEqual(len(logits), len(base_ids))
        self.assertEqual(len(indices), len(base_ids))
        self.assertTrue(all(row for row in logits))
        self.assertTrue(all(row for row in indices))
        self.assertTrue(
            all(
                0 <= token_id <= base.endpoint.tokenizer_max_token_id
                for row in indices
                for token_id in row
            )
        )


@unittest.skipUnless(
    RUN_REAL_NEMO_TOKENIZER_TESTS,
    "set LEGALFEDLLM_RUN_REAL_NEMO_TOKENIZER_TESTS=true for pinned artifacts",
)
class RealMistralNemoTokenizerAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cache_dir = os.getenv("LEGALFEDLLM_TOKENIZER_CACHE") or None
        token = os.getenv("HF_TOKEN") or None
        local_only = os.getenv(
            "LEGALFEDLLM_TOKENIZER_LOCAL_FILES_ONLY",
            "",
        ).lower() in {"1", "true", "yes"}
        cls.validated = load_pinned_tokenizer(
            MISTRAL_NEMO_HOST_ENDPOINT,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_only,
        )

    def test_actual_nemo_uses_pinned_corrected_tokenizer_and_existing_pad(
        self,
    ) -> None:
        endpoint = self.validated.endpoint
        tokenizer = self.validated.tokenizer

        self.assertTrue(endpoint.fix_mistral_regex)
        self.assertTrue(endpoint.bind_existing_pad_token)
        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.pad_token_id, 10)
        self.assertEqual(tokenizer.vocab_size, 131072)
        self.assertEqual(len(tokenizer), 131072)
        self.assertEqual(len(tokenizer.get_vocab()), 131072)


if __name__ == "__main__":
    unittest.main()
