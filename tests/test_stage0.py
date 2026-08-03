from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from pydantic import ValidationError

from client.runtime import (
    ClientRuntime,
    ClientRuntimeError,
    default_client_profile,
)
from host.runtime import (
    HostRuntime,
    HostRuntimeError,
    default_host_profile,
)
from shared.crypto import sha256_hex
from shared.protocol import (
    DifferentialPrivacyPolicy,
    DifferentialPrivacyReport,
    KnowledgePackage,
    KnowledgeSample,
    ModelProfile,
    RoundManifest,
)
from tests.test_round import Stack


class StageZeroTests(unittest.IsolatedAsyncioTestCase):
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
                    f"knowledge_cache/accepted/{manifest.round_id}.json"
                )
                before = accepted_path.read_bytes()

                trained = await client.post(
                    "/v1/local-train",
                    json={"examples": ["new private example"]},
                )
                self.assertEqual(trained.status_code, 200, trained.text)

                duplicate = await client.post("/v1/participate")

            self.assertEqual(duplicate.status_code, 409, duplicate.text)
            self.assertEqual(before, accepted_path.read_bytes())

            snapshot = runtime_a.store.read_json(
                f"adapter_snapshots/accepted/{manifest.round_id}.json"
            )
            accepted = KnowledgePackage.model_validate(
                runtime_a.store.read_json(
                    f"knowledge_cache/accepted/{manifest.round_id}.json"
                )
            )

            self.assertEqual(snapshot["adapter_version"], accepted.adapter_version)
            self.assertEqual(snapshot["package_hash"], accepted.artifact_sha256)

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
            bad_samples = []

            for sample in original.samples:
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
                samples=bad_samples,
            )

            payload = bad.model_dump(mode="json")
            endpoint = f"/v1/rounds/{manifest.round_id}/knowledge"

            first = await stack.coordinator_request(
                "POST",
                endpoint,
                content=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            second = await stack.coordinator_request(
                "POST",
                endpoint,
                content=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            self.assertEqual(first.status_code, 409, first.text)
            self.assertEqual(second.status_code, 409, second.text)
            self.assertIn("replayed Knowledge Package hash", second.text)

            state = stack.coordinator_service.get_state(manifest.round_id)
            self.assertIn(bad.artifact_sha256, state.seen_package_hashes)
            self.assertIn(bad.nonce, state.seen_nonces)

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
                / "host_knowledge.json"
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
                samples=original.samples,
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


class StageZeroValidationTests(unittest.TestCase):
    def test_real_training_backends_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client_data = default_client_profile().model_dump(mode="json")
            client_data["training_backend"] = "transformers"

            with self.assertRaises(ClientRuntimeError):
                ClientRuntime(
                    data_dir=Path(directory) / "client",
                    model_profile=ModelProfile.model_validate(client_data),
                )

            host_data = default_host_profile().model_dump(mode="json")
            host_data["training_backend"] = "transformers"

            with self.assertRaises(HostRuntimeError):
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
