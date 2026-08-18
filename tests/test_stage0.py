from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
from pydantic import ValidationError

from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.runtime import ClientRuntime, default_client_profile
from coordinator.main import create_app as create_coordinator_app
from coordinator.service import CoordinatorService
from host.model_profiles import pinned_host_profile
from host.runtime import (
    HostRuntime,
    HostRuntimeError,
    default_host_profile,
)
from shared.alignment_profiles import POC_DTW_PROFILE_VERSION
from shared.crypto import Ed25519Identity, canonical_json_bytes, sha256_hex
from shared.knowledge_artifact import (
    load_package_samples,
    serialize_knowledge_artifact,
)
from shared.protocol import (
    ClientRegistrationRequest,
    DifferentialPrivacyPolicy,
    DifferentialPrivacyReport,
    KnowledgePackage,
    KnowledgeSample,
    ModelProfile,
    RoundManifest,
    utc_text,
)
from tests.test_round import Stack


class ProtocolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def _register(self, stack: Stack, app) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://client",
            headers=stack.client_headers,
        ) as client:
            response = await client.post("/v1/register")

        self.assertEqual(response.status_code, 201, response.text)

    async def _create_round(
        self,
        stack: Stack,
        *,
        selected_client_ids: list[str],
        quorum: int,
        alignment: dict | None = None,
    ) -> RoundManifest:
        payload = {
            "selected_client_ids": selected_client_ids,
            "trusted_client_quorum": quorum,
            "reference_dataset_id": "reference",
            "reference_dataset_hash": sha256_hex(b"reference"),
            "sample_ids": ["s1", "s2"],
            "prompt_template": "{question} {answer}",
            "top_k": 2,
        }

        if alignment is not None:
            payload["alignment"] = alignment

        response = await stack.coordinator_request(
            "POST",
            "/v1/rounds",
            headers={"X-Admin-Token": stack.admin_token},
            json=payload,
        )

        self.assertEqual(response.status_code, 201, response.text)
        return RoundManifest.model_validate(response.json())

    async def test_client_admin_endpoints_require_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
            ) as client:
                denied = await client.post("/v1/register")

            self.assertEqual(denied.status_code, 401)
            await self._register(stack, app)

    async def test_non_mock_alignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")
            await self._register(stack, app)

            response = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a"],
                    "trusted_client_quorum": 1,
                    "reference_dataset_id": "reference",
                    "reference_dataset_hash": sha256_hex(b"reference"),
                    "sample_ids": ["s1"],
                    "prompt_template": "{question} {answer}",
                    "top_k": 2,
                    "alignment": {
                        "strategy": "dtw",
                        "profile_version": "1",
                    },
                },
            )

            self.assertEqual(response.status_code, 409, response.text)

    async def test_pinned_dtw_pair_remains_blocked_before_integration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            stack.host_runtime.model_profile = pinned_host_profile()
            client_identity = Ed25519Identity.load_or_create(
                Path(directory) / "qwen-client.pem"
            )
            stack.coordinator_service.register_client(
                ClientRegistrationRequest(
                    client_id="client-a",
                    public_key=client_identity.public_key_b64,
                    model_profile=pinned_client_profile(QWEN_PROFILE_ID),
                )
            )

            response = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a"],
                    "trusted_client_quorum": 1,
                    "reference_dataset_id": "reference",
                    "reference_dataset_hash": sha256_hex(b"reference"),
                    "sample_ids": ["s1"],
                    "prompt_template": "{question} {answer}",
                    "top_k": 2,
                    "alignment": {
                        "strategy": "dtw",
                        "profile_version": POC_DTW_PROFILE_VERSION,
                    },
                },
            )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn(
                "real alignment execution is not yet available",
                response.json()["detail"],
            )

    async def test_accepted_client_cache_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime_a, app_a = stack.client("client-a")
            _, app_b = stack.client("client-b")

            await self._register(stack, app_a)
            await self._register(stack, app_b)

            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a", "client-b"],
                quorum=2,
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                first = await client.post("/v1/participate")
                self.assertEqual(first.status_code, 201, first.text)

                accepted_path = runtime_a.store.path(
                    f"knowledge_cache/accepted/{manifest.round_id}/package.json"
                )
                before = accepted_path.read_bytes()
                accepted_artifact_path = runtime_a.store.path(
                    f"knowledge_cache/accepted/{manifest.round_id}/"
                    "knowledge.safetensors"
                )
                artifact_before = accepted_artifact_path.read_bytes()

                trained = await client.post(
                    "/v1/local-train",
                    json={"examples": ["new private example"]},
                )
                self.assertEqual(trained.status_code, 200, trained.text)

                duplicate = await client.post("/v1/participate")

            self.assertEqual(duplicate.status_code, 409, duplicate.text)
            self.assertEqual(before, accepted_path.read_bytes())
            self.assertEqual(
                artifact_before,
                accepted_artifact_path.read_bytes(),
            )

            snapshot = runtime_a.store.read_json(
                f"adapter_snapshots/accepted/{manifest.round_id}.json"
            )
            accepted = KnowledgePackage.model_validate(
                runtime_a.store.read_json(
                    f"knowledge_cache/accepted/{manifest.round_id}/package.json"
                )
            )

            self.assertEqual(snapshot["adapter_version"], accepted.adapter_version)
            self.assertEqual(snapshot["package_hash"], accepted.package_hash)

    async def test_rejected_package_cannot_be_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime_a, app_a = stack.client("client-a")
            _, app_b = stack.client("client-b")

            await self._register(stack, app_a)
            await self._register(stack, app_b)

            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a", "client-b"],
                quorum=2,
            )

            original = runtime_a.create_knowledge_package(manifest)
            original_samples = load_package_samples(
                runtime_a.package_artifact_path(original),
                original,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            bad_samples = []

            for sample in original_samples:
                logits = [row[:] for row in sample.top_k_logits]
                logits[0][0] = 101.0
                bad_samples.append(
                    KnowledgeSample(
                        sample_id=sample.sample_id,
                        source_input_ids=sample.source_input_ids,
                        attention_length=sample.attention_length,
                        top_k_token_ids=sample.top_k_token_ids,
                        top_k_logits=logits,
                        ce_loss=sample.ce_loss,
                    )
                )

            bad_artifact, bad_descriptor = serialize_knowledge_artifact(
                bad_samples
            )
            bad = KnowledgePackage.create_signed(
                identity=runtime_a.identity,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id=runtime_a.client_id,
                sender_role="client",
                model_profile=runtime_a.model_profile,
                adapter_version=original.adapter_version,
                alignment_profile_id=original.alignment_profile_id,
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                top_k=manifest.top_k,
                sample_ids=[sample.sample_id for sample in bad_samples],
                artifact=bad_descriptor,
            )

            payload = bad.model_dump(mode="json")
            endpoint = f"/v1/rounds/{manifest.round_id}/knowledge"
            files = [
                (
                    "package",
                    (
                        "package.json",
                        canonical_json_bytes(payload),
                        "application/json",
                    ),
                ),
                (
                    "artifact",
                    (
                        "knowledge.safetensors",
                        bad_artifact,
                        "application/octet-stream",
                    ),
                ),
            ]

            first = await stack.coordinator_request(
                "POST",
                endpoint,
                files=files,
            )
            restarted = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=stack.host_gateway,
                registration_token=stack.registration_token,
                admin_token=stack.admin_token,
                now_fn=stack.coordinator_service.now_fn,
            )
            restarted_app = create_coordinator_app(restarted)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://coordinator",
            ) as coordinator:
                second = await coordinator.post(endpoint, files=files)

                nonce_replay = KnowledgePackage.create_signed(
                    identity=runtime_a.identity,
                    round_id=bad.round_id,
                    manifest_hash=bad.manifest_hash,
                    sender_id=bad.sender_id,
                    sender_role=bad.sender_role,
                    model_profile=bad.model_profile,
                    adapter_version=bad.adapter_version + 1,
                    alignment_profile_id=bad.alignment_profile_id,
                    reference_dataset_id=bad.reference_dataset_id,
                    reference_dataset_hash=bad.reference_dataset_hash,
                    top_k=bad.top_k,
                    sample_ids=bad.sample_ids,
                    artifact=bad.artifact,
                    nonce=bad.nonce,
                    created_at=utc_text(restarted.now_fn()),
                )
                self.assertNotEqual(
                    nonce_replay.package_hash,
                    bad.package_hash,
                )
                nonce_files = [
                    (
                        "package",
                        (
                            "package.json",
                            canonical_json_bytes(
                                nonce_replay.model_dump(mode="json")
                            ),
                            "application/json",
                        ),
                    ),
                    (
                        "artifact",
                        (
                            "knowledge.safetensors",
                            bad_artifact,
                            "application/octet-stream",
                        ),
                    ),
                ]
                third = await coordinator.post(
                    endpoint,
                    files=nonce_files,
                )

            self.assertEqual(first.status_code, 409, first.text)
            self.assertEqual(second.status_code, 409, second.text)
            self.assertIn("replayed Knowledge Package hash", second.text)
            self.assertEqual(third.status_code, 409, third.text)
            self.assertIn("replayed Knowledge Package nonce", third.text)

            state = restarted.get_state(manifest.round_id)
            self.assertIn(bad.package_hash, state.seen_package_hashes)
            self.assertIn(bad.nonce, state.seen_nonces)

    async def test_dp_policy_rejects_a_report_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")
            await self._register(stack, app)
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
                    "dp_policy": {
                        "required": True,
                        "mechanism": "dp_sgd",
                        "max_epsilon": 0.05,
                        "delta": 1e-5,
                    },
                },
            )
            self.assertEqual(create.status_code, 201, create.text)
            round_id = create.json()["round_id"]

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                response = await client.post("/v1/participate")

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("privacy budget", response.text)
            state = stack.coordinator_service.get_state(round_id)
            self.assertEqual(state.accepted_client_ids, [])
            self.assertEqual(state.rejected_client_ids, ["client-a"])

    async def test_pending_retry_reuses_exact_package_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            await self._register(stack, app)
            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a"],
                quorum=1,
            )

            first = runtime.create_knowledge_package(manifest)
            artifact_path = runtime.package_artifact_path(first)
            artifact_before = artifact_path.read_bytes()
            second = runtime.create_knowledge_package(manifest)

            self.assertEqual(first, second)
            self.assertEqual(artifact_before, artifact_path.read_bytes())
            pending = runtime.store.path(
                f"knowledge_cache/pending/{manifest.round_id}"
            )
            self.assertEqual(
                sorted(path.name for path in pending.iterdir()),
                ["knowledge.safetensors", "package.json"],
            )

    async def test_artifact_tampering_breaks_signed_package_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            await self._register(stack, app)
            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a"],
                quorum=1,
            )
            package = runtime.create_knowledge_package(manifest)
            artifact = bytearray(
                runtime.package_artifact_path(package).read_bytes()
            )
            artifact[-1] ^= 1
            response = await stack.coordinator_request(
                "POST",
                f"/v1/rounds/{manifest.round_id}/knowledge",
                files=[
                    (
                        "package",
                        (
                            "package.json",
                            canonical_json_bytes(
                                package.model_dump(mode="json")
                            ),
                            "application/json",
                        ),
                    ),
                    (
                        "artifact",
                        (
                            "knowledge.safetensors",
                            bytes(artifact),
                            "application/octet-stream",
                        ),
                    ),
                ],
            )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("SHA-256", response.text)
            state = stack.coordinator_service.get_state(manifest.round_id)
            self.assertIn(package.package_hash, state.seen_package_hashes)
            submission = (
                Path(directory)
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "submissions"
                / "client-a"
            )
            self.assertFalse(submission.exists())
            incoming = (
                Path(directory)
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "incoming"
            )
            self.assertEqual(list(incoming.glob("*.safetensors")), [])

    async def test_client_rejects_wrong_host_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")

            await self._register(stack, app)
            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a"],
                quorum=1,
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                participate = await client.post("/v1/participate")

            self.assertEqual(participate.status_code, 201, participate.text)

            host_path = (
                Path(directory)
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "host_knowledge"
                / "package.json"
            )
            original = KnowledgePackage.model_validate(
                json.loads(host_path.read_text(encoding="utf-8"))
            )

            wrong = KnowledgePackage.create_signed(
                identity=stack.host_runtime.identity,
                round_id=original.round_id,
                manifest_hash=original.manifest_hash,
                sender_id=original.sender_id,
                sender_role="host",
                model_profile=original.model_profile,
                adapter_version=original.adapter_version + 1,
                alignment_profile_id=original.alignment_profile_id,
                reference_dataset_id=original.reference_dataset_id,
                reference_dataset_hash=original.reference_dataset_hash,
                top_k=original.top_k,
                sample_ids=original.sample_ids,
                artifact=original.artifact,
            )

            host_path.write_text(
                json.dumps(wrong.model_dump(mode="json"), sort_keys=True),
                encoding="utf-8",
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                sync = await client.post(f"/v1/rounds/{manifest.round_id}/sync")

            self.assertEqual(sync.status_code, 409, sync.text)
            self.assertIn("adapter version", sync.text)

    async def test_client_rejects_host_signature_and_artifact_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            await self._register(stack, app)
            manifest = await self._create_round(
                stack,
                selected_client_ids=["client-a"],
                quorum=1,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                participate = await client.post("/v1/participate")
            self.assertEqual(participate.status_code, 201, participate.text)

            host_package_path = (
                Path(directory)
                / "coordinator"
                / "rounds"
                / manifest.round_id
                / "host_knowledge"
                / "package.json"
            )
            host_artifact_path = host_package_path.with_name(
                "knowledge.safetensors"
            )
            package = KnowledgePackage.model_validate(
                json.loads(host_package_path.read_text(encoding="utf-8"))
            )
            status = stack.coordinator_service.get_state(manifest.round_id)
            self.assertIsNotNone(status.host_adapter_after)
            accepted_host_adapter_version = int(
                status.host_adapter_after
            )
            invalid_signature = package.model_copy(
                update={
                    "signature": stack.coordinator_service.identity.sign_json(
                        {}
                    )
                }
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                runtime.apply_host_knowledge(
                    manifest=manifest,
                    host_package=invalid_signature,
                    host_artifact_path=host_artifact_path,
                    host_public_key=stack.host_runtime.identity.public_key_b64,
                    expected_host_id=stack.host_runtime.host_id,
                    accepted_host_adapter_version=(
                        accepted_host_adapter_version
                    ),
                    adapter_promoted=bool(status.adapter_promoted),
                )

            tampered = bytearray(host_artifact_path.read_bytes())
            tampered[-1] ^= 1
            tampered_path = Path(directory) / "tampered-host.safetensors"
            tampered_path.write_bytes(bytes(tampered))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runtime.apply_host_knowledge(
                    manifest=manifest,
                    host_package=package,
                    host_artifact_path=tampered_path,
                    host_public_key=stack.host_runtime.identity.public_key_b64,
                    expected_host_id=stack.host_runtime.host_id,
                    accepted_host_adapter_version=(
                        accepted_host_adapter_version
                    ),
                    adapter_promoted=bool(status.adapter_promoted),
                )

            host_cache = (
                Path(directory)
                / "client-a"
                / "knowledge_cache"
                / "host"
                / manifest.round_id
            )
            self.assertFalse(host_cache.exists())


class ProtocolValidationTests(unittest.TestCase):
    def test_client_real_backend_is_lazy_and_unpinned_host_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lazy_import = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "from pathlib import Path; "
                        "from client.model_profiles import "
                        "QWEN_PROFILE_ID, pinned_client_profile; "
                        "from client.runtime import ClientRuntime; "
                        f"ClientRuntime(data_dir=Path({str(Path(directory) / 'client')!r}), "
                        "model_profile=pinned_client_profile(QWEN_PROFILE_ID)); "
                        "unexpected = sorted({'torch', 'transformers', 'peft'} "
                        "& set(sys.modules)); "
                        "raise SystemExit(','.join(unexpected) if unexpected else 0)"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                lazy_import.returncode,
                0,
                lazy_import.stderr or lazy_import.stdout,
            )

            host_data = default_host_profile().model_dump(mode="json")
            host_data["training_backend"] = "transformers"
            host_data["model_revision"] = "0" * 40
            host_data["tokenizer_revision"] = "0" * 40
            host_data["tokenizer_chat_template_hash"] = "0" * 64
            host_data["vocabulary_size"] = 1
            host_data["chat_template_mode"] = "standard"

            with self.assertRaisesRegex(
                HostRuntimeError,
                "exact pinned Granite Host profile",
            ):
                HostRuntime(
                    data_dir=Path(directory) / "host",
                    model_profile=ModelProfile.model_validate(host_data),
                )

    def test_dp_contracts_reject_incomplete_values(self) -> None:
        with self.assertRaises(ValidationError):
            DifferentialPrivacyPolicy(
                required=True,
                mechanism="dp_sgd",
            )

        with self.assertRaises(ValidationError):
            DifferentialPrivacyReport(
                enabled=True,
                mechanism="dp_sgd",
            )

        with self.assertRaises(ValidationError):
            DifferentialPrivacyReport(
                enabled=False,
                mechanism="none",
                epsilon_spent=0.1,
            )


if __name__ == "__main__":
    unittest.main()
