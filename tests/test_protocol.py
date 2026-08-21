from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta

from pydantic import ValidationError

from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import serialize_knowledge_artifact
from shared.protocol import (
    AlignmentConfig,
    HostCandidateTrainingResult,
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
    def test_host_candidate_result_hash_and_version_are_bound(self) -> None:
        result = HostCandidateTrainingResult.create(
            round_id="round-1",
            manifest_hash="a" * 64,
            job_hash="b" * 64,
            parent_adapter_version=0,
            parent_adapter_hash="c" * 64,
            candidate_adapter_version=1,
            candidate_adapter_hash="d" * 64,
            host_model_profile_hash="e" * 64,
            execution_profile_hash="f" * 64,
            host_public_data_epochs=5,
            optimizer_step_count=2,
            optimizer_loss=0.5,
            supervised_answer_loss=0.55,
            distillation_answer_loss=0.05,
            trainable_parameter_count=10,
            total_parameter_count=100,
            dependency_versions={"torch": "test"},
            created_at=utc_text(),
        )
        payload = result.model_dump(mode="json")
        payload["optimizer_step_count"] = 3
        with self.assertRaises(ValidationError):
            HostCandidateTrainingResult.model_validate(payload)

        payload = result.model_dump(mode="json")
        payload["candidate_adapter_version"] = 2
        payload["result_hash"] = sha256_hex(
            {key: value for key, value in payload.items() if key != "result_hash"}
        )
        with self.assertRaises(ValidationError):
            HostCandidateTrainingResult.model_validate(payload)

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
                selected_client_profile_hashes={
                    "client-a": profile("client").profile_hash()
                },
                request=request,
                submission_deadline=utc_text(utc_now() + timedelta(hours=1)),
            )
            self.assertTrue(manifest.verify_signature(coordinator.public_key_b64))
            self.assertEqual(manifest.host_public_data_epochs, 5)
            self.assertEqual(
                manifest.maximum_host_training_job_bytes,
                256 * 1024 * 1024,
            )
            tampered_manifest = manifest.model_dump(mode="json")
            tampered_manifest["host_public_data_epochs"] = 1
            with self.assertRaises(ValidationError):
                RoundManifest.model_validate(tampered_manifest)
            samples = deterministic_knowledge_samples(
                manifest=manifest,
                participant_id="client-a",
                role="client",
                adapter_version=0,
            )
            _, descriptor = serialize_knowledge_artifact(samples)
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
                sample_ids=[sample.sample_id for sample in samples],
                artifact=descriptor,
            )
            self.assertTrue(package.verify_signature(client.public_key_b64))
            self.assertEqual(package.package_schema_version, "2.0")
            self.assertEqual(package.artifact, descriptor)

            tampered = package.model_dump(mode="json")
            tampered["artifact"]["sha256"] = sha256_hex(b"tampered")
            with self.assertRaises(ValidationError):
                KnowledgePackage.model_validate(tampered)

            forged = package.model_copy(update={"signature": coordinator.sign_json({})})
            self.assertFalse(forged.verify_signature(client.public_key_b64))


if __name__ == "__main__":
    unittest.main()
