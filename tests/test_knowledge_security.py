from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import httpx
from pydantic import ValidationError

from client.runtime import default_client_profile
from host.runtime import default_host_profile
from shared.crypto import Ed25519Identity, canonical_json_bytes, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import (
    load_package_samples,
    serialize_knowledge_artifact,
)
from shared.protocol import (
    KnowledgeArtifactDescriptor,
    KnowledgePackage,
    KnowledgeSample,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)
from tests.test_round import Stack


class KnowledgePackageSecurityTests(unittest.TestCase):
    def _fixture(
        self,
        directory: str,
    ) -> tuple[
        Ed25519Identity,
        Ed25519Identity,
        RoundManifest,
        list[KnowledgeSample],
        bytes,
        KnowledgePackage,
    ]:
        coordinator = Ed25519Identity.load_or_create(
            Path(directory) / "coordinator.pem"
        )
        client = Ed25519Identity.load_or_create(
            Path(directory) / "client.pem"
        )
        request = RoundCreateRequest(
            selected_client_ids=["client-a"],
            trusted_client_quorum=1,
            reference_dataset_id="reference-v1",
            reference_dataset_hash=sha256_hex(b"reference-v1"),
            sample_ids=["sample-1", "sample-2"],
            prompt_template="Question: {question}\nAnswer: {answer}",
            top_k=3,
        )
        manifest = RoundManifest.create_signed(
            identity=coordinator,
            round_id="round-security",
            coordinator_id="coordinator",
            current_host_adapter_version=0,
            host_model_profile=default_host_profile(),
            selected_client_profile_hashes={
                "client-a": default_client_profile().profile_hash()
            },
            selected_client_alignment_profiles={
                "client-a": "mock_identity:1"
            },
            request=request,
            submission_deadline=utc_text(utc_now() + timedelta(hours=1)),
        )
        samples = deterministic_knowledge_samples(
            manifest=manifest,
            participant_id="client-a",
            role="client",
            adapter_version=0,
        )
        artifact, descriptor = serialize_knowledge_artifact(samples)
        package = KnowledgePackage.create_signed(
            identity=client,
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            sender_id="client-a",
            sender_role="client",
            model_profile=default_client_profile(),
            adapter_version=0,
            alignment_profile_id="mock_identity:1",
            reference_dataset_id=manifest.reference_dataset_id,
            reference_dataset_hash=manifest.reference_dataset_hash,
            top_k=manifest.top_k,
            sample_ids=manifest.sample_ids,
            artifact=descriptor,
            nonce="security-fixture-nonce",
            created_at="2026-08-10T00:00:00Z",
        )
        return coordinator, client, manifest, samples, artifact, package

    def test_every_package_field_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, _, package = self._fixture(directory)
            mutations = {
                "protocol_version": lambda value: value.update(
                    protocol_version="9.9"
                ),
                "package_schema_version": lambda value: value.update(
                    package_schema_version="9.9"
                ),
                "round_id": lambda value: value.update(round_id="another-round"),
                "manifest_hash": lambda value: value.update(
                    manifest_hash="0" * 64
                ),
                "sender_id": lambda value: value.update(sender_id="client-b"),
                "sender_role": lambda value: value.update(sender_role="host"),
                "model_profile": lambda value: value["model_profile"].update(
                    model_id="changed/model"
                ),
                "adapter_version": lambda value: value.update(adapter_version=1),
                "alignment_profile_id": lambda value: value.update(
                    alignment_profile_id="mock_identity:2"
                ),
                "reference_dataset_id": lambda value: value.update(
                    reference_dataset_id="another-reference"
                ),
                "reference_dataset_hash": lambda value: value.update(
                    reference_dataset_hash="1" * 64
                ),
                "sample_ids": lambda value: value.update(
                    sample_ids=list(reversed(value["sample_ids"]))
                ),
                "top_k": lambda value: value.update(top_k=value["top_k"] + 1),
                "artifact": lambda value: value["artifact"].update(
                    sha256="2" * 64
                ),
                "dp_report": lambda value: value.update(
                    dp_report={
                        "enabled": True,
                        "mechanism": "dp_sgd",
                        "epsilon_spent": 0.1,
                        "delta": 1e-5,
                    }
                ),
                "nonce": lambda value: value.update(
                    nonce="different-security-nonce"
                ),
                "created_at": lambda value: value.update(
                    created_at="2026-08-10T00:00:01Z"
                ),
                "package_hash": lambda value: value.update(
                    package_hash="3" * 64
                ),
            }

            original = package.model_dump(mode="json")
            for field, mutate in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    with self.assertRaises(ValidationError):
                        KnowledgePackage.model_validate(changed)

    def test_rehashing_modified_metadata_does_not_forge_a_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, client, _, _, _, package = self._fixture(directory)
            mutations = {
                "round_id": lambda value: value.update(round_id="another-round"),
                "manifest_hash": lambda value: value.update(
                    manifest_hash="0" * 64
                ),
                "sender_id": lambda value: value.update(sender_id="client-b"),
                "model_profile": lambda value: value["model_profile"].update(
                    model_id="changed/model"
                ),
                "adapter_version": lambda value: value.update(adapter_version=1),
                "alignment_profile_id": lambda value: value.update(
                    alignment_profile_id="mock_identity:2"
                ),
                "reference_dataset_id": lambda value: value.update(
                    reference_dataset_id="another-reference"
                ),
                "reference_dataset_hash": lambda value: value.update(
                    reference_dataset_hash="1" * 64
                ),
                "dp_report": lambda value: value.update(
                    dp_report={
                        "enabled": True,
                        "mechanism": "dp_sgd",
                        "epsilon_spent": 0.1,
                        "delta": 1e-5,
                    }
                ),
                "nonce": lambda value: value.update(
                    nonce="different-security-nonce"
                ),
                "created_at": lambda value: value.update(
                    created_at="2026-08-10T00:00:01Z"
                ),
            }

            original = package.model_dump(mode="json")
            for field, mutate in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    changed["package_hash"] = sha256_hex(
                        {
                            key: value
                            for key, value in changed.items()
                            if key not in {"package_hash", "signature"}
                        }
                    )
                    rehashed = KnowledgePackage.model_validate(changed)
                    self.assertFalse(
                        rehashed.verify_signature(client.public_key_b64)
                    )

    def test_invalid_signature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator, client, _, _, _, package = self._fixture(directory)
            forged = package.model_copy(
                update={"signature": coordinator.sign_json({})}
            )
            self.assertFalse(forged.verify_signature(client.public_key_b64))

    def test_wrong_artifact_size_and_sha256_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, client, manifest, _, artifact, package = self._fixture(directory)
            path = Path(directory) / "knowledge.safetensors"
            path.write_bytes(artifact)

            descriptors = {
                "byte size": package.artifact.model_copy(
                    update={"byte_size": package.artifact.byte_size + 1}
                ),
                "SHA-256": package.artifact.model_copy(
                    update={"sha256": "0" * 64}
                ),
            }
            for message, descriptor in descriptors.items():
                with self.subTest(message=message):
                    changed = KnowledgePackage.create_signed(
                        identity=client,
                        round_id=package.round_id,
                        manifest_hash=package.manifest_hash,
                        sender_id=package.sender_id,
                        sender_role=package.sender_role,
                        model_profile=package.model_profile,
                        adapter_version=package.adapter_version,
                        alignment_profile_id=package.alignment_profile_id,
                        reference_dataset_id=package.reference_dataset_id,
                        reference_dataset_hash=package.reference_dataset_hash,
                        top_k=package.top_k,
                        sample_ids=package.sample_ids,
                        artifact=KnowledgeArtifactDescriptor.model_validate(
                            descriptor.model_dump(mode="json")
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_package_samples(
                            path,
                            changed,
                            maximum_bytes=(
                                manifest.maximum_knowledge_package_bytes
                            ),
                        )

    def test_substitution_is_rejected_but_identical_artifacts_are_allowed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, client_a, manifest, samples, artifact, package_a = self._fixture(
                directory
            )
            changed_samples = [sample.model_copy(deep=True) for sample in samples]
            changed_logits = [
                row[:]
                for row in changed_samples[0].top_k_logits
            ]
            changed_logits[0][0] += 0.5
            changed_samples[0] = changed_samples[0].model_copy(
                update={"top_k_logits": changed_logits}
            )
            substituted_artifact, _ = serialize_knowledge_artifact(
                changed_samples
            )
            substituted_path = Path(directory) / "substituted.safetensors"
            substituted_path.write_bytes(substituted_artifact)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_package_samples(
                    substituted_path,
                    package_a,
                    maximum_bytes=manifest.maximum_knowledge_package_bytes,
                )

            client_b = Ed25519Identity.load_or_create(
                Path(directory) / "client-b.pem"
            )
            package_b = KnowledgePackage.create_signed(
                identity=client_b,
                round_id=package_a.round_id,
                manifest_hash=package_a.manifest_hash,
                sender_id="client-b",
                sender_role="client",
                model_profile=package_a.model_profile,
                adapter_version=package_a.adapter_version,
                alignment_profile_id=package_a.alignment_profile_id,
                reference_dataset_id=package_a.reference_dataset_id,
                reference_dataset_hash=package_a.reference_dataset_hash,
                top_k=package_a.top_k,
                sample_ids=package_a.sample_ids,
                artifact=package_a.artifact,
                nonce="security-fixture-nonce-b",
                created_at=package_a.created_at,
            )
            identical_path = Path(directory) / "identical.safetensors"
            identical_path.write_bytes(artifact)

            self.assertEqual(
                package_a.artifact.sha256,
                package_b.artifact.sha256,
            )
            self.assertNotEqual(package_a.package_hash, package_b.package_hash)
            self.assertTrue(package_a.verify_signature(client_a.public_key_b64))
            self.assertTrue(package_b.verify_signature(client_b.public_key_b64))
            self.assertEqual(
                load_package_samples(
                    identical_path,
                    package_a,
                    maximum_bytes=manifest.maximum_knowledge_package_bytes,
                ),
                load_package_samples(
                    identical_path,
                    package_b,
                    maximum_bytes=manifest.maximum_knowledge_package_bytes,
                ),
            )


class ClientPackageBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_resigned_mismatched_client_bindings_are_rejected(self) -> None:
        scenarios = {
            "sender ID": "Client is not selected",
            "sender role": "only Client Knowledge Packages",
            "round ID": "round ID path mismatch",
            "manifest hash": "stale manifest hash",
            "model profile": "differs from the signed round manifest",
            "alignment": "alignment profile",
            "dataset ID": "dataset ID",
            "dataset hash": "dataset hash",
            "sample order": "sample order",
            "top-k": "top-k",
            "timestamp": "timestamp",
        }
        for scenario, expected in scenarios.items():
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as directory:
                    stack = Stack(directory)
                    runtime, app = stack.client("client-a")
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://client-a",
                        headers=stack.client_headers,
                    ) as client:
                        registration = await client.post("/v1/register")
                    self.assertEqual(
                        registration.status_code,
                        201,
                        registration.text,
                    )
                    create = await stack.coordinator_request(
                        "POST",
                        "/v1/rounds",
                        headers={"X-Admin-Token": stack.admin_token},
                        json={
                            "selected_client_ids": ["client-a"],
                            "trusted_client_quorum": 1,
                            "reference_dataset_id": "reference",
                            "reference_dataset_hash": sha256_hex(b"reference"),
                            "sample_ids": ["s1", "s2"],
                            "prompt_template": "{question} {answer}",
                            "top_k": 2,
                        },
                    )
                    self.assertEqual(create.status_code, 201, create.text)
                    manifest = RoundManifest.model_validate(create.json())
                    original = runtime.create_knowledge_package(manifest)
                    artifact = runtime.package_artifact_path(
                        original
                    ).read_bytes()
                    values = {
                        "round_id": original.round_id,
                        "manifest_hash": original.manifest_hash,
                        "sender_id": original.sender_id,
                        "sender_role": original.sender_role,
                        "model_profile": original.model_profile,
                        "adapter_version": original.adapter_version,
                        "alignment_profile_id": (
                            original.alignment_profile_id
                        ),
                        "reference_dataset_id": (
                            original.reference_dataset_id
                        ),
                        "reference_dataset_hash": (
                            original.reference_dataset_hash
                        ),
                        "top_k": original.top_k,
                        "sample_ids": original.sample_ids,
                        "artifact": original.artifact,
                        "created_at": original.created_at,
                    }

                    if scenario == "sender ID":
                        values["sender_id"] = "client-b"
                    elif scenario == "sender role":
                        values["sender_role"] = "host"
                        values["model_profile"] = default_host_profile()
                    elif scenario == "round ID":
                        values["round_id"] = "another-round"
                    elif scenario == "manifest hash":
                        values["manifest_hash"] = "0" * 64
                    elif scenario == "model profile":
                        values["model_profile"] = (
                            original.model_profile.model_copy(
                                update={"model_id": "changed/model"}
                            )
                        )
                    elif scenario == "alignment":
                        values["alignment_profile_id"] = "mock_identity:2"
                    elif scenario == "dataset ID":
                        values["reference_dataset_id"] = "another-reference"
                    elif scenario == "dataset hash":
                        values["reference_dataset_hash"] = "1" * 64
                    elif scenario == "sample order":
                        values["sample_ids"] = list(
                            reversed(original.sample_ids)
                        )
                        values["artifact"] = original.artifact.model_copy(
                            update={
                                "sample_ids_sha256": sha256_hex(
                                    values["sample_ids"]
                                )
                            }
                        )
                    elif scenario == "top-k":
                        values["top_k"] = original.top_k + 1
                        values["artifact"] = original.artifact.model_copy(
                            update={"top_k": original.top_k + 1}
                        )
                    elif scenario == "timestamp":
                        values["created_at"] = "2020-01-01T00:00:00Z"

                    changed = KnowledgePackage.create_signed(
                        identity=runtime.identity,
                        **values,
                    )
                    response = await stack.coordinator_request(
                        "POST",
                        f"/v1/rounds/{manifest.round_id}/knowledge",
                        files=[
                            (
                                "package",
                                (
                                    "package.json",
                                    canonical_json_bytes(
                                        changed.model_dump(mode="json")
                                    ),
                                    "application/json",
                                ),
                            ),
                            (
                                "artifact",
                                (
                                    "knowledge.safetensors",
                                    artifact,
                                    "application/octet-stream",
                                ),
                            ),
                        ],
                    )
                    self.assertEqual(
                        response.status_code,
                        409,
                        response.text,
                    )
                    self.assertIn(expected, response.text)
                    state = stack.coordinator_service.get_state(
                        manifest.round_id
                    )
                    self.assertEqual(state.accepted_client_ids, [])
                    incoming = (
                        Path(directory)
                        / "coordinator"
                        / "rounds"
                        / manifest.round_id
                        / "incoming"
                    )
                    self.assertEqual(
                        list(incoming.glob("*.safetensors")),
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
