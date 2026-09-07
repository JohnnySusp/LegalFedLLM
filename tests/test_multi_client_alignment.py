from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
    pinned_client_profile,
)
from client.runtime import ClientRuntime
from coordinator.service import ConflictError
from host.model_profiles import MISTRAL_NEMO_HOST_PROFILE_ID, pinned_host_profile
from shared.alignment_profiles import (
    GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
    MISTRAL_NEMO_DTW_PROFILE_ID,
    UnsupportedAlignmentProfile,
    resolve_alignment_profile_id_for_pair,
)
from shared.crypto import Ed25519Identity, canonical_json_bytes, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import write_knowledge_artifact
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import (
    ClientRegistrationRequest,
    KnowledgePackage,
    RoundCreateRequest,
    RoundManifest,
    utc_now,
    utc_text,
)
from tests.test_round import Stack


class MixedClientAlignmentContractTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _request() -> RoundCreateRequest:
        return RoundCreateRequest(
            selected_client_ids=["qwen-client", "granite-client"],
            trusted_client_quorum=2,
            reference_dataset_id="reference-v1",
            reference_dataset_hash=sha256_hex(b"reference-v1"),
            sample_ids=["sample-1", "sample-2"],
            prompt_template=PROMPT_TEMPLATE,
            label_format="chat_sft_answer_only_v1",
            maximum_sequence_length=128,
            truncation_policy="reject",
            top_k=2,
            host_public_data_epochs=1,
        )

    @staticmethod
    def _register_real_clients(stack: Stack, root: Path):
        identities = {
            "qwen-client": Ed25519Identity.load_or_create(root / "qwen.pem"),
            "granite-client": Ed25519Identity.load_or_create(root / "granite.pem"),
        }
        profiles = {
            "qwen-client": pinned_client_profile(QWEN_PROFILE_ID),
            "granite-client": pinned_client_profile(
                GRANITE_3_3_2B_CLIENT_PROFILE_ID
            ),
        }
        for client_id in ("qwen-client", "granite-client"):
            stack.coordinator_service.register_client(
                ClientRegistrationRequest(
                    client_id=client_id,
                    public_key=identities[client_id].public_key_b64,
                    model_profile=profiles[client_id],
                ),
                stack.issue_enrollment_token(),
            )
        return identities, profiles

    @staticmethod
    def _signed_package(
        *,
        root: Path,
        manifest: RoundManifest,
        client_id: str,
        identity: Ed25519Identity,
        profile,
        alignment_profile_id: str,
    ) -> tuple[KnowledgePackage, Path, int]:
        samples = deterministic_knowledge_samples(
            manifest=manifest,
            participant_id=client_id,
            role="client",
            adapter_version=0,
        )
        artifact_path = root / f"{client_id}-{alignment_profile_id.replace(':', '-')}.safetensors"
        descriptor = write_knowledge_artifact(
            artifact_path,
            samples,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        package = KnowledgePackage.create_signed(
            identity=identity,
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            sender_id=client_id,
            sender_role="client",
            model_profile=profile,
            adapter_version=0,
            alignment_profile_id=alignment_profile_id,
            reference_dataset_id=manifest.reference_dataset_id,
            reference_dataset_hash=manifest.reference_dataset_hash,
            top_k=manifest.top_k,
            sample_ids=manifest.sample_ids,
            artifact=descriptor,
        )
        content_size = descriptor.byte_size + len(
            canonical_json_bytes(package.model_dump(mode="json"))
        )
        return package, artifact_path, content_size

    async def test_mixed_round_assignments_are_generated_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = Stack(directory)
            stack.host_runtime.model_profile = pinned_host_profile(
                MISTRAL_NEMO_HOST_PROFILE_ID
            )
            identities, profiles = self._register_real_clients(stack, root)

            manifest = await stack.coordinator_service.create_round(self._request())

            self.assertEqual(
                manifest.selected_client_alignment_profiles,
                {
                    "qwen-client": MISTRAL_NEMO_DTW_PROFILE_ID,
                    "granite-client": GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
                },
            )
            self.assertEqual(manifest.protocol_version, "1.1")
            self.assertEqual(manifest.alignment_strategy, "dtw")
            self.assertEqual(
                manifest.host_package_alignment_profile_id,
                MISTRAL_NEMO_DTW_PROFILE_ID,
            )

            qwen_runtime = ClientRuntime(
                data_dir=root / "qwen-runtime",
                client_id="qwen-client",
                model_profile=profiles["qwen-client"],
            )
            granite_runtime = ClientRuntime(
                data_dir=root / "granite-runtime",
                client_id="granite-client",
                model_profile=profiles["granite-client"],
            )
            qwen_runtime._validate_training_manifest(manifest)
            granite_runtime._validate_training_manifest(manifest)

            qwen_wrong = self._signed_package(
                root=root,
                manifest=manifest,
                client_id="qwen-client",
                identity=identities["qwen-client"],
                profile=profiles["qwen-client"],
                alignment_profile_id=GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
            )
            with self.assertRaisesRegex(ConflictError, "alignment profile"):
                await stack.coordinator_service.submit_knowledge(*qwen_wrong)

            granite_wrong = self._signed_package(
                root=root,
                manifest=manifest,
                client_id="granite-client",
                identity=identities["granite-client"],
                profile=profiles["granite-client"],
                alignment_profile_id=MISTRAL_NEMO_DTW_PROFILE_ID,
            )
            with self.assertRaisesRegex(ConflictError, "alignment profile"):
                await stack.coordinator_service.submit_knowledge(*granite_wrong)

            qwen = self._signed_package(
                root=root,
                manifest=manifest,
                client_id="qwen-client",
                identity=identities["qwen-client"],
                profile=profiles["qwen-client"],
                alignment_profile_id=MISTRAL_NEMO_DTW_PROFILE_ID,
            )
            qwen_receipt = await stack.coordinator_service.submit_knowledge(*qwen)
            self.assertEqual(qwen_receipt.state, "COLLECTING")
            self.assertEqual(qwen_receipt.accepted_count, 1)
            with self.assertRaisesRegex(ConflictError, "already submitted"):
                await stack.coordinator_service.submit_knowledge(*qwen)

            granite = self._signed_package(
                root=root,
                manifest=manifest,
                client_id="granite-client",
                identity=identities["granite-client"],
                profile=profiles["granite-client"],
                alignment_profile_id=GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
            )
            granite_receipt = await stack.coordinator_service.submit_knowledge(
                *granite
            )
            self.assertEqual(granite_receipt.state, "SEALED")
            self.assertEqual(granite_receipt.accepted_count, 2)
            self.assertEqual(
                stack.coordinator_service.get_state(manifest.round_id).sealed_client_ids,
                ["qwen-client", "granite-client"],
            )
            self.assertEqual(qwen[0].protocol_version, "1.1")
            self.assertEqual(granite[0].protocol_version, "1.1")

    def test_manifest_mapping_is_exact_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = Ed25519Identity.load_or_create(Path(directory) / "coordinator.pem")
            request = self._request()
            host_profile = pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID)
            profile_hashes = {
                "qwen-client": pinned_client_profile(QWEN_PROFILE_ID).profile_hash(),
                "granite-client": pinned_client_profile(
                    GRANITE_3_3_2B_CLIENT_PROFILE_ID
                ).profile_hash(),
            }
            first_mapping = {
                "qwen-client": MISTRAL_NEMO_DTW_PROFILE_ID,
                "granite-client": GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
            }
            second_mapping = {
                "granite-client": GRANITE_MISTRAL_NEMO_DTW_PROFILE_ID,
                "qwen-client": MISTRAL_NEMO_DTW_PROFILE_ID,
            }
            deadline = utc_text(utc_now() + timedelta(hours=1))
            with mock.patch("shared.protocol.secrets.token_urlsafe", return_value="round-nonce-constant"):
                first = RoundManifest.create_signed(
                    identity=identity,
                    round_id="round-canonical",
                    coordinator_id="coordinator",
                    current_host_adapter_version=0,
                    host_model_profile=host_profile,
                    selected_client_profile_hashes=profile_hashes,
                    selected_client_alignment_profiles=first_mapping,
                    request=request,
                    submission_deadline=deadline,
                )
                second = RoundManifest.create_signed(
                    identity=identity,
                    round_id="round-canonical",
                    coordinator_id="coordinator",
                    current_host_adapter_version=0,
                    host_model_profile=host_profile,
                    selected_client_profile_hashes=profile_hashes,
                    selected_client_alignment_profiles=second_mapping,
                    request=request,
                    submission_deadline=deadline,
                )
            self.assertEqual(first.manifest_hash, second.manifest_hash)
            self.assertEqual(
                first.coordinator_signature,
                second.coordinator_signature,
            )
            self.assertEqual(first.protocol_version, "1.1")
            self.assertEqual(
                first.host_package_alignment_profile_id,
                MISTRAL_NEMO_DTW_PROFILE_ID,
            )

            old_protocol = first.model_dump(mode="json")
            old_protocol["protocol_version"] = "1.0"
            with self.assertRaisesRegex(ValidationError, "protocol_version"):
                RoundManifest.model_validate(old_protocol)

            missing = first.model_dump(mode="json")
            missing["selected_client_alignment_profiles"].pop("granite-client")
            with self.assertRaisesRegex(ValidationError, "alignment profiles"):
                RoundManifest.model_validate(missing)

            extra = first.model_dump(mode="json")
            extra["selected_client_alignment_profiles"]["other-client"] = (
                MISTRAL_NEMO_DTW_PROFILE_ID
            )
            with self.assertRaisesRegex(ValidationError, "alignment profiles"):
                RoundManifest.model_validate(extra)

            malformed = first.model_dump(mode="json")
            malformed["selected_client_alignment_profiles"]["qwen-client"] = "dtw"
            with self.assertRaisesRegex(ValidationError, "alignment profile IDs"):
                RoundManifest.model_validate(malformed)

    def test_unknown_real_pair_fails_instead_of_guessing(self) -> None:
        client = pinned_client_profile(QWEN_PROFILE_ID).model_copy(
            update={"model_revision": "unsupported-revision"}
        )
        host = pinned_host_profile(MISTRAL_NEMO_HOST_PROFILE_ID)
        with self.assertRaises(UnsupportedAlignmentProfile):
            resolve_alignment_profile_id_for_pair(
                client_profile=client,
                host_profile=host,
            )


if __name__ == "__main__":
    unittest.main()
