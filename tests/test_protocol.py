from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta

from pydantic import ValidationError

from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.protocol import (
    AlignmentConfig,
    LoraProfile,
    ModelProfile,
    RoundCreateRequest,
    RoundManifest,
    KnowledgePackage,
    utc_now,
    utc_text,
)


def profile(role: str) -> ModelProfile:
    return ModelProfile(
        profile_id=f"{role}-profile",
        role=role,
        model_id=f"mock/{role}",
        model_revision="v1",
        tokenizer_id="mock/tokenizer",
        tokenizer_revision="v1",
        tokenizer_class="MockTokenizer",
        prompt_template_hash=sha256_hex(b"prompt"),
        lora=LoraProfile(rank=4),
    )


class ProtocolSecurityTests(unittest.TestCase):
    def test_manifest_and_package_signatures_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Ed25519Identity.load_or_create(
                f"{directory}/coordinator.pem"
            )
            client = Ed25519Identity.load_or_create(f"{directory}/client.pem")
            request = RoundCreateRequest(
                selected_client_ids=["client-a"],
                trusted_client_quorum=1,
                reference_dataset_id="reference-v1",
                reference_dataset_hash=sha256_hex(b"reference-v1"),
                sample_ids=["sample-1"],
                prompt_template="Question: {question}\nAnswer: {answer}",
                top_k=3,
                alignment=AlignmentConfig(strategy="mock_identity"),
            )
            manifest = RoundManifest.create_signed(
                identity=coordinator,
                round_id="round-000001",
                coordinator_id="coordinator",
                current_host_adapter_version=0,
                host_model_profile=profile("host"),
                request=request,
                submission_deadline=utc_text(utc_now() + timedelta(hours=1)),
            )
            self.assertTrue(manifest.verify_signature(coordinator.public_key_b64))
            samples = deterministic_knowledge_samples(
                manifest=manifest,
                participant_id="client-a",
                role="client",
                adapter_version=0,
            )
            package = KnowledgePackage.create_signed(
                identity=client,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id="client-a",
                sender_role="client",
                model_profile=profile("client"),
                adapter_version=0,
                alignment_profile_id="mock_identity:1",
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                top_k=manifest.top_k,
                samples=samples,
            )
            self.assertTrue(package.verify_signature(client.public_key_b64))

            tampered = package.model_dump(mode="json")
            tampered["samples"][0]["top_k_logits"][0][0] += 1.0
            with self.assertRaises(ValidationError):
                KnowledgePackage.model_validate(tampered)

            forged = package.model_copy(update={"signature": coordinator.sign_json({})})
            self.assertFalse(forged.verify_signature(client.public_key_b64))


if __name__ == "__main__":
    unittest.main()
