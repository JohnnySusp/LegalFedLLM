from __future__ import annotations

import unittest

from pydantic import ValidationError

from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
    pinned_client_profile,
    supported_profile_ids,
)
from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    GRANITE_3_3_2B_REVISION,
    MISTRAL_NEMO_HOST_PROFILE_ID,
    MISTRAL_NEMO_REVISION,
    pinned_host_profile,
    supported_host_profile_ids,
)
from shared.alignment_profiles import (
    GRANITE_IDENTITY_DTW_PROFILE_ID,
    GRANITE_IDENTITY_DTW_PROFILE_VERSION,
    GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
    GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION,
    MISTRAL_NEMO_DTW_PROFILE_ID,
    MISTRAL_NEMO_DTW_PROFILE_VERSION,
    MISTRAL_NEMO_HOST_ENDPOINT,
    POC_DTW_PROFILE_ID,
    POC_DTW_PROFILE_VERSION,
    UnsupportedAlignmentProfile,
    resolve_alignment_profile,
    supported_alignment_profile_ids,
    validate_alignment_pair,
)
from shared.protocol import AlignmentConfig, ModelProfile


class HostModelProfileTests(unittest.TestCase):
    def test_host_profile_is_exact_and_immutable(self) -> None:
        profile = pinned_host_profile()

        self.assertEqual(profile.profile_id, GRANITE_3_3_2B_HOST_PROFILE_ID)
        self.assertEqual(
            profile.model_id,
            "ibm-granite/granite-3.3-2b-instruct",
        )
        self.assertEqual(profile.model_revision, GRANITE_3_3_2B_REVISION)
        self.assertEqual(profile.model_revision, profile.tokenizer_revision)
        self.assertEqual(profile.model_class, "GraniteForCausalLM")
        self.assertEqual(profile.model_type, "granite")
        self.assertEqual(profile.tokenizer_class, "GPT2TokenizerFast")
        self.assertEqual(profile.vocabulary_size, 49159)
        self.assertEqual(profile.training_backend, "transformers")
        self.assertEqual(profile.lora.rank, 8)
        self.assertEqual(
            pinned_host_profile(serving_backend="ollama").ollama.model,
            "granite3.3:2b",
        )
        self.assertEqual(
            profile.profile_hash(),
            pinned_host_profile(serving_backend="ollama").profile_hash(),
        )

    def test_mistral_nemo_profile_is_exact_and_supports_transformers_serving(self) -> None:
        profile = pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID)

        self.assertEqual(profile.model_id, "mistralai/Mistral-Nemo-Instruct-2407")
        self.assertEqual(profile.model_revision, MISTRAL_NEMO_REVISION)
        self.assertEqual(profile.model_revision, profile.tokenizer_revision)
        self.assertEqual(profile.model_class, "MistralForCausalLM")
        self.assertEqual(profile.model_type, "mistral")
        self.assertEqual(profile.tokenizer_class, "PreTrainedTokenizerFast")
        self.assertEqual(profile.vocabulary_size, 131072)
        self.assertEqual(profile.serving_backend, "mock")
        self.assertEqual(
            profile.lora.target_modules,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        self.assertEqual(
            pinned_host_profile(
                MISTRAL_NEMO_HOST_PROFILE_ID,
                serving_backend="transformers",
            ).serving_backend,
            "transformers",
        )
        with self.assertRaisesRegex(ValueError, "mock or transformers serving"):
            pinned_host_profile(
                MISTRAL_NEMO_HOST_PROFILE_ID,
                serving_backend="ollama",
            )

    def test_supported_host_profiles_preserve_granite_and_add_nemo(self) -> None:
        self.assertEqual(
            supported_host_profile_ids(),
            (GRANITE_3_3_2B_HOST_PROFILE_ID, MISTRAL_NEMO_HOST_PROFILE_ID),
        )


class AlignmentProfileContractTests(unittest.TestCase):
    def test_approved_poc_profiles_are_advertised(self) -> None:
        self.assertEqual(
            supported_alignment_profile_ids(),
            (
                POC_DTW_PROFILE_ID,
                GRANITE_IDENTITY_DTW_PROFILE_ID,
                MISTRAL_NEMO_DTW_PROFILE_ID,
                GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
            ),
        )
        profile = resolve_alignment_profile(POC_DTW_PROFILE_ID)
        self.assertEqual(profile.strategy, "dtw")
        self.assertEqual(profile.profile_version, POC_DTW_PROFILE_VERSION)
        self.assertEqual(profile.client_to_host_owner, "coordinator")
        self.assertEqual(profile.host_to_client_owner, "client")
        self.assertEqual(
            AlignmentConfig(
                strategy="dtw",
                profile_version=POC_DTW_PROFILE_VERSION,
            ).profile_id,
            POC_DTW_PROFILE_ID,
        )
        self.assertEqual(profile.client.word_boundary_marker, "Ġ")
        self.assertEqual(profile.host.word_boundary_marker, "Ġ")
        self.assertEqual(profile.client.tokenizer_vocabulary_size, 151669)
        self.assertEqual(profile.host.tokenizer_vocabulary_size, 49159)

        granite = resolve_alignment_profile(GRANITE_IDENTITY_DTW_PROFILE_ID)
        self.assertEqual(
            granite.profile_version,
            GRANITE_IDENTITY_DTW_PROFILE_VERSION,
        )
        self.assertEqual(granite.client.role, "client")
        self.assertEqual(granite.host.role, "host")
        self.assertEqual(granite.client.model_id, granite.host.model_id)
        self.assertEqual(
            granite.client.tokenizer_artifact_sha256,
            granite.host.tokenizer_artifact_sha256,
        )

        nemo = resolve_alignment_profile(MISTRAL_NEMO_DTW_PROFILE_ID)
        self.assertEqual(
            nemo.profile_version,
            MISTRAL_NEMO_DTW_PROFILE_VERSION,
        )
        self.assertEqual(nemo.host, MISTRAL_NEMO_HOST_ENDPOINT)
        self.assertTrue(nemo.host.fix_mistral_regex)
        self.assertTrue(nemo.host.bind_existing_pad_token)
        self.assertEqual(nemo.host.pad_token_id, 10)

        granite_nemo = resolve_alignment_profile(
            GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID
        )
        self.assertEqual(
            granite_nemo.profile_version,
            GRANITE_MISTRAL_NEMO_DTW_PROFILE_VERSION,
        )
        self.assertEqual(
            granite_nemo.client.profile_id,
            GRANITE_3_3_2B_CLIENT_PROFILE_ID,
        )
        self.assertEqual(granite_nemo.host, MISTRAL_NEMO_HOST_ENDPOINT)

    def test_qwen_and_granite_are_approved_client_profiles(self) -> None:
        self.assertEqual(
            supported_profile_ids(),
            (QWEN_PROFILE_ID, GRANITE_3_3_2B_CLIENT_PROFILE_ID),
        )
        granite = pinned_client_profile(GRANITE_3_3_2B_CLIENT_PROFILE_ID)
        self.assertEqual(granite.role, "client")
        self.assertEqual(granite.model_id, pinned_host_profile().model_id)
        self.assertNotEqual(granite.profile_id, pinned_host_profile().profile_id)

    def test_exact_qwen_to_granite_pair_is_accepted(self) -> None:
        profile = validate_alignment_pair(
            POC_DTW_PROFILE_ID,
            client_profile=pinned_client_profile(QWEN_PROFILE_ID),
            host_profile=pinned_host_profile(),
        )
        self.assertEqual(profile.profile_id, POC_DTW_PROFILE_ID)

    def test_exact_granite_client_to_granite_host_pair_is_accepted(self) -> None:
        profile = validate_alignment_pair(
            GRANITE_IDENTITY_DTW_PROFILE_ID,
            client_profile=pinned_client_profile(
                GRANITE_3_3_2B_CLIENT_PROFILE_ID
            ),
            host_profile=pinned_host_profile(),
        )
        self.assertEqual(profile.profile_id, GRANITE_IDENTITY_DTW_PROFILE_ID)

    def test_exact_qwen_to_mistral_nemo_pair_is_accepted(self) -> None:
        profile = validate_alignment_pair(
            MISTRAL_NEMO_DTW_PROFILE_ID,
            client_profile=pinned_client_profile(QWEN_PROFILE_ID),
            host_profile=pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID),
        )
        self.assertEqual(profile.profile_id, MISTRAL_NEMO_DTW_PROFILE_ID)

    def test_exact_granite_client_to_mistral_nemo_pair_is_accepted(self) -> None:
        profile = validate_alignment_pair(
            GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
            client_profile=pinned_client_profile(
                GRANITE_3_3_2B_CLIENT_PROFILE_ID
            ),
            host_profile=pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID),
        )
        self.assertEqual(
            profile.profile_id,
            GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
        )

    def test_unknown_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedAlignmentProfile,
            "unsupported alignment profile",
        ):
            resolve_alignment_profile("dtw:generic-v1")

    def test_greedy_dp_is_not_a_protocol_strategy(self) -> None:
        with self.assertRaises(ValidationError):
            AlignmentConfig(strategy="greedy_dp")

    def test_unapproved_client_tokenizer_fails_closed(self) -> None:
        values = pinned_client_profile(QWEN_PROFILE_ID).model_dump(mode="json")
        values["profile_id"] = "unapproved-client-v1"
        unapproved = ModelProfile.model_validate(values)

        with self.assertRaisesRegex(
            UnsupportedAlignmentProfile,
            "Client profile_id",
        ):
            validate_alignment_pair(
                POC_DTW_PROFILE_ID,
                client_profile=unapproved,
                host_profile=pinned_host_profile(),
            )

    def test_changed_host_revision_fails_closed(self) -> None:
        values = pinned_host_profile().model_dump(mode="json")
        values["model_revision"] = "0" * 40
        values["tokenizer_revision"] = "0" * 40
        changed = ModelProfile.model_validate(values)

        with self.assertRaisesRegex(
            UnsupportedAlignmentProfile,
            "Host model_revision",
        ):
            validate_alignment_pair(
                POC_DTW_PROFILE_ID,
                client_profile=pinned_client_profile(QWEN_PROFILE_ID),
                host_profile=changed,
            )


if __name__ == "__main__":
    unittest.main()
