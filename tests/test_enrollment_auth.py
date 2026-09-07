from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import httpx

from client.main import CoordinatorGateway
from client.main import create_app as create_client_app
from client.runtime import ClientRuntime, default_client_profile
from coordinator.main import create_app as create_coordinator_app
from coordinator.service import CoordinatorService
from shared.crypto import Ed25519Identity, sha256_hex
from shared.protocol import (
    ClientRegistrationRequest,
    ClientRequestAuthentication,
    RoundManifest,
    utc_text,
)
from tests.test_round import Stack


def _auth_headers(authentication: ClientRequestAuthentication) -> dict[str, str]:
    return {
        "X-Client-ID": authentication.client_id,
        "X-Client-Timestamp": authentication.timestamp,
        "X-Client-Nonce": authentication.nonce,
        "X-Client-Signature": authentication.signature,
    }


def _mock_profile():
    with mock.patch.dict(
        os.environ,
        {
            "CLIENT_MODEL_PROFILE": "mock",
            "CLIENT_TRAINING_BACKEND": "mock",
            "CLIENT_SERVING_BACKEND": "mock",
        },
    ):
        return default_client_profile()


class EnrollmentAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def _issue_token(self, stack: Stack) -> str:
        response = await stack.coordinator_request(
            "POST",
            "/v1/enrollment-tokens",
            headers={"X-Admin-Token": stack.admin_token},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["token"]

    async def _register(
        self,
        stack: Stack,
        *,
        client_id: str,
        identity: Ed25519Identity,
        token: str,
    ) -> httpx.Response:
        return await stack.coordinator_request(
            "POST",
            "/v1/clients/register",
            headers={"X-Registration-Token": token},
            json=ClientRegistrationRequest(
                client_id=client_id,
                public_key=identity.public_key_b64,
                model_profile=_mock_profile(),
            ).model_dump(mode="json"),
        )

    async def test_admin_issues_single_use_tokens_and_consumption_survives_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)

            denied = await stack.coordinator_request(
                "POST",
                "/v1/enrollment-tokens",
            )
            self.assertEqual(denied.status_code, 401)

            token = await self._issue_token(stack)
            token_hash = sha256_hex(token.encode("utf-8"))
            token_record_path = Path(directory) / "coordinator" / "enrollment_tokens" / f"{token_hash}.json"
            self.assertTrue(token_record_path.is_file())
            self.assertNotIn(token, token_record_path.read_text(encoding="utf-8"))

            identity_a = Ed25519Identity.load_or_create(Path(directory) / "client-a.pem")
            registered = await self._register(
                stack,
                client_id="client-a",
                identity=identity_a,
                token=token,
            )
            self.assertEqual(registered.status_code, 201, registered.text)

            replay_same = await self._register(
                stack,
                client_id="client-a",
                identity=identity_a,
                token=token,
            )
            self.assertEqual(replay_same.status_code, 409)

            identity_b = Ed25519Identity.load_or_create(Path(directory) / "client-b.pem")
            replay_other = await self._register(
                stack,
                client_id="client-b",
                identity=identity_b,
                token=token,
            )
            self.assertEqual(replay_other.status_code, 401)

            restarted = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=stack.host_gateway,
                admin_token=stack.admin_token,
                now_fn=stack.coordinator_service.now_fn,
            )
            restarted_app = create_coordinator_app(restarted)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://coordinator",
            ) as client:
                after_restart = await client.post(
                    "/v1/clients/register",
                    headers={"X-Registration-Token": token},
                    json=ClientRegistrationRequest(
                        client_id="client-b",
                        public_key=identity_b.public_key_b64,
                        model_profile=_mock_profile(),
                    ).model_dump(mode="json"),
                )
            self.assertEqual(after_restart.status_code, 401)
            self.assertEqual(restarted.get_registration("client-a").public_key, identity_a.public_key_b64)

            token_b = restarted.issue_enrollment_token().token
            registered_b = restarted.register_client(
                ClientRegistrationRequest(
                    client_id="client-b",
                    public_key=identity_b.public_key_b64,
                    model_profile=_mock_profile(),
                ),
                token_b,
            )
            self.assertEqual(registered_b.client_id, "client-b")

    async def test_signed_client_request_auth_rejects_missing_forged_stale_and_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            identity = Ed25519Identity.load_or_create(Path(directory) / "client-a.pem")
            token = await self._issue_token(stack)
            registered = await self._register(
                stack,
                client_id="client-a",
                identity=identity,
                token=token,
            )
            self.assertEqual(registered.status_code, 201, registered.text)

            created = await stack.coordinator_request(
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
                    "top_k": 1,
                },
            )
            self.assertEqual(created.status_code, 201, created.text)
            round_id = created.json()["round_id"]
            path = f"/v1/rounds/{round_id}/reference-dataset"

            authentication = ClientRequestAuthentication.create_signed(
                identity=identity,
                client_id="client-a",
                method="GET",
                path=path,
                nonce="signed-request-nonce-0001",
            )
            accepted = await stack.coordinator_request(
                "GET",
                path,
                headers=_auth_headers(authentication),
            )
            self.assertEqual(accepted.status_code, 204, accepted.text)

            replay = await stack.coordinator_request(
                "GET",
                path,
                headers=_auth_headers(authentication),
            )
            self.assertEqual(replay.status_code, 401)

            restarted = CoordinatorService(
                data_dir=Path(directory) / "coordinator",
                host_gateway=stack.host_gateway,
                admin_token=stack.admin_token,
                now_fn=stack.coordinator_service.now_fn,
            )
            restarted_app = create_coordinator_app(restarted)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://coordinator",
            ) as client:
                replay_after_restart = await client.get(
                    path,
                    headers=_auth_headers(authentication),
                )
            self.assertEqual(replay_after_restart.status_code, 401)

            missing = await stack.coordinator_request(
                "GET",
                path,
                headers={"X-Client-ID": "client-a"},
            )
            self.assertEqual(missing.status_code, 401)

            old_token_only = await stack.coordinator_request(
                "GET",
                path,
                headers={
                    "X-Client-ID": "client-a",
                    "X-Registration-Token": token,
                },
            )
            self.assertEqual(old_token_only.status_code, 401)

            attacker = Ed25519Identity.load_or_create(Path(directory) / "attacker.pem")
            forged = ClientRequestAuthentication.create_signed(
                identity=attacker,
                client_id="client-a",
                method="GET",
                path=path,
                nonce="signed-request-nonce-0002",
            )
            forged_response = await stack.coordinator_request(
                "GET",
                path,
                headers=_auth_headers(forged),
            )
            self.assertEqual(forged_response.status_code, 401)

            stale = ClientRequestAuthentication.create_signed(
                identity=identity,
                client_id="client-a",
                method="GET",
                path=path,
                timestamp=utc_text(stack.coordinator_service.now_fn() - timedelta(days=1)),
                nonce="signed-request-nonce-0003",
            )
            stale_response = await stack.coordinator_request(
                "GET",
                path,
                headers=_auth_headers(stale),
            )
            self.assertEqual(stale_response.status_code, 401)

            wrong_path = ClientRequestAuthentication.create_signed(
                identity=identity,
                client_id="client-a",
                method="GET",
                path=f"{path}/other",
                nonce="signed-request-nonce-0004",
            )
            wrong_path_response = await stack.coordinator_request(
                "GET",
                path,
                headers=_auth_headers(wrong_path),
            )
            self.assertEqual(wrong_path_response.status_code, 401)

    async def test_client_identity_and_enrollment_record_persist_without_token_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = Stack(directory)
            runtime, app = stack.client("client-a")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                first = await client.post("/v1/register")
            self.assertEqual(first.status_code, 201, first.text)
            first_public_key = runtime.identity.public_key_b64
            self.assertIsNotNone(runtime.registration_record())

            restarted_runtime = ClientRuntime(
                data_dir=root / "client-a",
                client_id="client-a",
                model_profile=runtime.model_profile,
            )
            gateway = CoordinatorGateway(
                "http://coordinator",
                None,
                transport=stack.coordinator_transport,
            )
            restarted_app = create_client_app(
                restarted_runtime,
                gateway,
                admin_token_override=stack.client_admin_token,
            )
            self.assertEqual(restarted_runtime.identity.public_key_b64, first_public_key)

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restarted_app),
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                health = await client.get("/health")
                local_retry = await client.post("/v1/register")
            self.assertEqual(health.status_code, 200, health.text)
            self.assertTrue(health.json()["enrolled"])
            self.assertEqual(local_retry.status_code, 201, local_retry.text)
            self.assertEqual(local_retry.json(), first.json())

            created = await stack.coordinator_request(
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
                    "top_k": 1,
                },
            )
            self.assertEqual(created.status_code, 201, created.text)
            manifest = RoundManifest.model_validate(created.json())
            self.assertIsNone(
                await restarted_app.state.gateway.reference_dataset(
                    manifest.round_id,
                    "client-a",
                )
            )


if __name__ == "__main__":
    unittest.main()
