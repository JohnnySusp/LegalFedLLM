from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from client.main import CoordinatorGateway, create_app as create_client_app
from client.runtime import ClientRuntime
from coordinator.main import create_app as create_coordinator_app
from coordinator.service import CoordinatorService, HostGateway
from host.main import create_app as create_host_app
from host.runtime import HostRuntime
from shared.crypto import sha256_hex


class Stack:
    def __init__(
        self,
        root: str,
        *,
        force_validation_failure: bool = False,
        now_fn=None,
    ):
        self.root = Path(root)
        self.internal_token = "test-internal"
        self.registration_token = "test-registration"
        self.admin_token = "test-admin"
        self.client_admin_token = "test-client-admin"

        self.client_headers = {
            "X-Client-Admin-Token": self.client_admin_token
        }
        self.host_runtime = HostRuntime(
            data_dir=self.root / "host",
            force_validation_failure=force_validation_failure,
        )
        self.host_app = create_host_app(
            self.host_runtime, internal_token_override=self.internal_token
        )
        self.host_gateway = HostGateway(
            "http://host",
            self.internal_token,
            transport=httpx.ASGITransport(app=self.host_app),
        )
        kwargs = {}
        if now_fn is not None:
            kwargs["now_fn"] = now_fn
        self.coordinator_service = CoordinatorService(
            data_dir=self.root / "coordinator",
            host_gateway=self.host_gateway,
            registration_token=self.registration_token,
            admin_token=self.admin_token,
            **kwargs,
        )
        self.coordinator_app = create_coordinator_app(self.coordinator_service)
        self.coordinator_transport = httpx.ASGITransport(app=self.coordinator_app)

    def client(self, client_id: str):
        runtime = ClientRuntime(data_dir=self.root / client_id, client_id=client_id)
        gateway = CoordinatorGateway(
            "http://coordinator",
            self.registration_token,
            transport=self.coordinator_transport,
        )
        app = create_client_app(
            runtime,
            gateway,
            admin_token_override=self.client_admin_token,
        )
        return runtime, app

    async def coordinator_request(self, method: str, path: str, **kwargs):
        async with httpx.AsyncClient(
            transport=self.coordinator_transport, base_url="http://coordinator"
        ) as client:
            return await client.request(method, path, **kwargs)


class ProtocolFirstRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_asynchronous_round_persists_and_syncs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime_a, app_a = stack.client("client-a")
            runtime_b, app_b = stack.client("client-b")

            for app in (app_a, app_b):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://client",
                    headers=stack.client_headers,
                ) as client:
                    response = await client.post("/v1/register")
                    self.assertEqual(response.status_code, 201, response.text)

            create = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a", "client-b"],
                    "trusted_client_quorum": 2,
                    "reference_dataset_id": "legal-reference-v1",
                    "reference_dataset_hash": sha256_hex(b"legal-reference-v1"),
                    "sample_ids": ["contract-001", "contract-002", "contract-003"],
                    "prompt_template": "Question: {question}\nAnswer: {answer}",
                    "top_k": 4,
                    "maximum_sequence_length": 64,
                    "submission_window_seconds": 3600,
                },
            )
            self.assertEqual(create.status_code, 201, create.text)
            round_id = create.json()["round_id"]

            private_marker = "confidential legal memo matter 42"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                trained = await client.post(
                    "/v1/local-train", json={"examples": [private_marker]}
                )
                first = await client.post("/v1/participate")
            self.assertEqual(trained.status_code, 200, trained.text)
            self.assertEqual(first.status_code, 201, first.text)
            self.assertEqual(first.json()["state"], "COLLECTING")

            submission_path = (
                Path(directory)
                / "coordinator"
                / "rounds"
                / round_id
                / "submissions"
                / "client-a.json"
            )
            self.assertNotIn(private_marker, submission_path.read_text(encoding="utf-8"))

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_b),
                base_url="http://client-b",
                headers=stack.client_headers,
            ) as client:
                await client.post(
                    "/v1/local-train", json={"examples": ["private client B example"]}
                )
                second = await client.post("/v1/participate")
            self.assertEqual(second.status_code, 201, second.text)
            self.assertEqual(second.json()["state"], "COMPLETED")

            status_response = await stack.coordinator_request(
                "GET", f"/v1/rounds/{round_id}/status"
            )
            self.assertEqual(status_response.status_code, 200)
            status_payload = status_response.json()
            self.assertEqual(status_payload["state"], "COMPLETED")
            self.assertEqual(status_payload["sealed_client_ids"], ["client-a", "client-b"])
            self.assertTrue(status_payload["adapter_promoted"])
            self.assertEqual(stack.host_runtime.adapter_version, 1)

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                sync = await client.post(f"/v1/rounds/{round_id}/sync")
            self.assertEqual(sync.status_code, 200, sync.text)
            self.assertEqual(sync.json()["last_completed_round"], round_id)

            restarted = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=stack.host_gateway,
                registration_token=stack.registration_token,
                admin_token=stack.admin_token,
            )
            restarted_state = await restarted.round_status(round_id)
            self.assertEqual(restarted_state.state, "COMPLETED")
            self.assertEqual(restarted_state.host_adapter_after, 1)

    async def test_duplicate_client_submission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app_a = stack.client("client-a")
            _, app_b = stack.client("client-b")
            for app in (app_a, app_b):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://client",
                    headers=stack.client_headers,
                ) as client:
                    self.assertEqual((await client.post("/v1/register")).status_code, 201)
            create = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a", "client-b"],
                    "trusted_client_quorum": 2,
                    "reference_dataset_id": "reference",
                    "reference_dataset_hash": sha256_hex(b"reference"),
                    "sample_ids": ["s1"],
                    "prompt_template": "{question} {answer}",
                    "top_k": 2,
                },
            )
            self.assertEqual(create.status_code, 201, create.text)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                first = await client.post("/v1/participate")
                duplicate = await client.post("/v1/participate")
            self.assertEqual(first.status_code, 201, first.text)
            self.assertEqual(duplicate.status_code, 409, duplicate.text)

    async def test_deadline_without_quorum_skips_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clock = [datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc)]
            stack = Stack(directory, now_fn=lambda: clock[0])
            _, app_a = stack.client("client-a")
            _, app_b = stack.client("client-b")
            for app in (app_a, app_b):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://client",
                    headers=stack.client_headers,
                ) as client:
                    self.assertEqual((await client.post("/v1/register")).status_code, 201)
            create = await stack.coordinator_request(
                "POST",
                "/v1/rounds",
                headers={"X-Admin-Token": stack.admin_token},
                json={
                    "selected_client_ids": ["client-a", "client-b"],
                    "trusted_client_quorum": 2,
                    "reference_dataset_id": "reference",
                    "reference_dataset_hash": sha256_hex(b"reference"),
                    "sample_ids": ["s1"],
                    "prompt_template": "{question} {answer}",
                    "top_k": 2,
                    "submission_window_seconds": 1,
                },
            )
            round_id = create.json()["round_id"]
            clock[0] += timedelta(seconds=2)
            status_response = await stack.coordinator_request(
                "GET", f"/v1/rounds/{round_id}/status"
            )
            self.assertEqual(status_response.json()["state"], "SKIPPED")
            self.assertEqual(stack.host_runtime.adapter_version, 0)

    async def test_failed_candidate_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory, force_validation_failure=True)
            _, app = stack.client("client-a")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                self.assertEqual((await client.post("/v1/register")).status_code, 201)
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
            round_id = create.json()["round_id"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                participate = await client.post("/v1/participate")
            self.assertEqual(participate.status_code, 201, participate.text)
            status_response = await stack.coordinator_request(
                "GET", f"/v1/rounds/{round_id}/status"
            )
            self.assertEqual(status_response.json()["state"], "COMPLETED")
            self.assertFalse(status_response.json()["adapter_promoted"])
            self.assertEqual(status_response.json()["host_adapter_after"], 0)
            self.assertEqual(stack.host_runtime.adapter_version, 0)


if __name__ == "__main__":
    unittest.main()
