from __future__ import annotations

import unittest

from pydantic import ValidationError

from client.model_profiles import (
    QWEN_PROFILE_ID,
    pinned_client_profile,
)
from host.model_profiles import (
    GRANITE_3_3_2B_HOST_PROFILE_ID,
    GRANITE_3_3_2B_REVISION,
    pinned_host_profile,
)
from shared.alignment_profiles import (
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


class AlignmentProfileContractTests(unittest.TestCase):
    def test_only_the_approved_poc_profile_is_advertised(self) -> None:
        self.assertEqual(supported_alignment_profile_ids(), (POC_DTW_PROFILE_ID,))
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

    def test_exact_qwen_to_granite_pair_is_accepted(self) -> None:
        profile = validate_alignment_pair(
            POC_DTW_PROFILE_ID,
            client_profile=pinned_client_profile(QWEN_PROFILE_ID),
            host_profile=pinned_host_profile(),
        )
        self.assertEqual(profile.profile_id, POC_DTW_PROFILE_ID)

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
