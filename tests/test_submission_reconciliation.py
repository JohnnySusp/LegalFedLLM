from __future__ import annotations

import asyncio
import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE
from shared.protocol import RoundManifest
from tests.test_round import Stack


class LoseKnowledgeAcknowledgementTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport):
        self.inner = inner
        self.lost_acknowledgements = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        if (
            request.method == "POST"
            and request.url.path.endswith("/knowledge")
            and self.lost_acknowledgements == 0
        ):
            self.lost_acknowledgements += 1
            await response.aclose()
            raise httpx.ReadTimeout(
                "simulated lost submission acknowledgement",
                request=request,
            )
        return response


class DropKnowledgeSubmissionTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport):
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/knowledge"):
            raise httpx.ConnectError(
                "simulated connection loss before submission",
                request=request,
            )
        return await self.inner.handle_async_request(request)


class SubmissionReconciliationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_private_data(runtime, directory: str) -> None:
        path = Path(directory) / "client-a-private.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "example_id": "private-001",
                    "prompt": "Private question",
                    "answer": "Private answer",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.private_data_path = path

    async def _register_client(self, stack: Stack, app) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://client-a",
            headers=stack.client_headers,
        ) as client:
            response = await client.post("/v1/register")
        self.assertEqual(response.status_code, 201, response.text)

    async def _create_round(self, stack: Stack) -> RoundManifest:
        response = await stack.coordinator_request(
            "POST",
            "/v1/rounds",
            headers={"X-Admin-Token": stack.admin_token},
            json={
                "selected_client_ids": ["client-a"],
                "trusted_client_quorum": 1,
                "reference_dataset_id": "legal-reference-v1",
                "reference_dataset_hash": sha256_hex(b"legal-reference-v1"),
                "sample_ids": ["contract-001", "contract-002"],
                "prompt_template": PROMPT_TEMPLATE,
                "label_format": "chat_sft_answer_only_v1",
                "truncation_policy": "reject",
                "top_k": 2,
                "maximum_sequence_length": 64,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return RoundManifest.model_validate(response.json())

    async def test_lost_acknowledgement_reconciles_exact_accepted_package_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            self._write_private_data(runtime, directory)
            await self._register_client(stack, app)
            manifest = await self._create_round(stack)

            lossy_transport = LoseKnowledgeAcknowledgementTransport(
                stack.coordinator_transport
            )
            app.state.gateway.transport = lossy_transport
            app.state.gateway.reconciliation_timeout_seconds = 0.2
            app.state.gateway.reconciliation_poll_seconds = 0.01

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                trained = await client.post(
                    f"/v1/rounds/{manifest.round_id}/local-train"
                )
                response = await client.post(
                    f"/v1/rounds/{manifest.round_id}/participate"
                )

            self.assertEqual(trained.status_code, 200, trained.text)
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(lossy_transport.lost_acknowledgements, 1)

            receipt = response.json()
            state = stack.coordinator_service.get_state(manifest.round_id)
            self.assertEqual(state.accepted_client_ids, ["client-a"])
            self.assertEqual(state.submission_hashes, [receipt["package_hash"]])

            self.assertFalse(
                runtime.store.exists(runtime._pending_package_path(manifest.round_id))
            )
            self.assertFalse(
                runtime.store.exists(runtime._pending_artifact_path(manifest.round_id))
            )
            self.assertFalse(
                runtime.store.exists(runtime._pending_snapshot_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._accepted_package_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._accepted_artifact_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._accepted_snapshot_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._receipt_path(manifest.round_id))
            )

    async def test_reconciliation_fails_closed_for_nonmatching_authority_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            self._write_private_data(runtime, directory)
            await self._register_client(stack, app)
            manifest = await self._create_round(stack)

            package = runtime.create_knowledge_package(manifest)
            artifact_path = runtime.package_artifact_path(package)
            receipt = await app.state.gateway.submit(package, artifact_path)
            runtime.commit_knowledge_submission(
                manifest=manifest,
                package=package,
                receipt=receipt,
            )

            exact = await app.state.gateway.accepted_submission_receipt(package)
            self.assertIsNotNone(exact)
            self.assertEqual(exact.package_hash, package.package_hash)

            mismatched_hash = package.model_copy(
                update={"package_hash": "f" * 64}
            )
            with self.assertRaises(HTTPException) as mismatch:
                await app.state.gateway.accepted_submission_receipt(mismatched_hash)
            self.assertEqual(mismatch.exception.status_code, 409)

            wrong_client = package.model_copy(update={"sender_id": "client-b"})
            with self.assertRaises(HTTPException) as wrong_identity:
                await app.state.gateway.accepted_submission_receipt(wrong_client)
            self.assertEqual(wrong_identity.exception.status_code, 401)

            wrong_round = package.model_copy(update={"round_id": "round-999999"})
            self.assertIsNone(
                await app.state.gateway.accepted_submission_receipt(wrong_round)
            )

            next_manifest = await self._create_round(stack)
            pending = runtime.create_knowledge_package(next_manifest)
            self.assertIsNone(
                await app.state.gateway.accepted_submission_receipt(pending)
            )
            self.assertTrue(
                runtime.store.exists(
                    runtime._pending_package_path(next_manifest.round_id)
                )
            )
            self.assertFalse(
                runtime.store.exists(
                    runtime._accepted_package_path(next_manifest.round_id)
                )
            )


    async def test_submission_safety_probe_runs_off_event_loop_until_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            self._write_private_data(runtime, directory)
            await self._register_client(stack, app)
            manifest = await self._create_round(stack)

            package = runtime.create_knowledge_package(manifest)
            artifact_path = runtime.package_artifact_path(package)
            main_thread_id = threading.get_ident()
            started = threading.Event()
            release = threading.Event()
            probe_thread_ids: list[int] = []

            from coordinator import service as coordinator_service_module

            original_probe = coordinator_service_module.inspect_knowledge_package

            def slow_probe(package_arg, samples):
                probe_thread_ids.append(threading.get_ident())
                started.set()
                release.wait(timeout=0.5)
                return original_probe(package_arg, samples)

            with patch(
                "coordinator.service.inspect_knowledge_package",
                side_effect=slow_probe,
            ):
                submit_task = asyncio.create_task(
                    app.state.gateway.submit(package, artifact_path)
                )
                probe_started = await asyncio.to_thread(started.wait, 0.2)
                self.assertTrue(probe_started)
                self.assertNotEqual(probe_thread_ids, [main_thread_id])

                pending = await asyncio.wait_for(
                    app.state.gateway.accepted_submission_receipt(package),
                    timeout=0.1,
                )
                self.assertIsNone(pending)

                state = stack.coordinator_service.get_state(manifest.round_id)
                self.assertEqual(state.seen_package_hashes, [package.package_hash])
                self.assertEqual(state.accepted_client_ids, [])
                self.assertEqual(state.submission_hashes, [])

                release.set()
                receipt = await submit_task

            self.assertEqual(receipt.package_hash, package.package_hash)
            state = stack.coordinator_service.get_state(manifest.round_id)
            self.assertEqual(state.accepted_client_ids, ["client-a"])
            self.assertEqual(state.submission_hashes, [package.package_hash])

    async def test_unconfirmed_network_failure_keeps_pending_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            runtime, app = stack.client("client-a")
            self._write_private_data(runtime, directory)
            await self._register_client(stack, app)
            manifest = await self._create_round(stack)

            app.state.gateway.transport = DropKnowledgeSubmissionTransport(
                stack.coordinator_transport
            )
            app.state.gateway.reconciliation_timeout_seconds = 0

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client-a",
                headers=stack.client_headers,
            ) as client:
                trained = await client.post(
                    f"/v1/rounds/{manifest.round_id}/local-train"
                )
                response = await client.post(
                    f"/v1/rounds/{manifest.round_id}/participate"
                )

            self.assertEqual(trained.status_code, 200, trained.text)
            self.assertEqual(response.status_code, 503, response.text)
            self.assertIn("not confirmed accepted", response.text)
            state = stack.coordinator_service.get_state(manifest.round_id)
            self.assertEqual(state.accepted_client_ids, [])
            self.assertEqual(state.submission_hashes, [])
            self.assertTrue(
                runtime.store.exists(runtime._pending_package_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._pending_artifact_path(manifest.round_id))
            )
            self.assertTrue(
                runtime.store.exists(runtime._pending_snapshot_path(manifest.round_id))
            )
            self.assertFalse(
                runtime.store.exists(runtime._accepted_package_path(manifest.round_id))
            )


if __name__ == "__main__":
    unittest.main()
