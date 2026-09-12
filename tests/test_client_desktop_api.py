from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from client.main import CoordinatorGateway, create_app as create_client_app
from client.runtime import ClientRuntime
from client.tunnel import SshTunnelConfig, SshTunnelManager
from shared.crypto import sha256_hex
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import LoraProfile, ModelProfile
from tests.test_round import Stack


def mock_profile() -> ModelProfile:
    return ModelProfile(
        profile_id="desktop-mock-client-v1",
        role="client",
        model_id="legalfedllm/mock-client",
        model_revision="mock-v1",
        tokenizer_id="legalfedllm/mock-tokenizer",
        tokenizer_revision="mock-v1",
        tokenizer_class="MockTokenizer",
        training_backend="mock",
        serving_backend="mock",
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        lora=LoraProfile(rank=8),
    )


class DesktopClientApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_facade_exposes_local_and_host_only_and_local_answer_becomes_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ClientRuntime(
                data_dir=Path(directory) / "client",
                client_id="client-desktop",
                model_profile=mock_profile(),
                private_data_path=Path(directory) / "private" / "train.jsonl",
            )
            app = create_client_app(
                runtime,
                CoordinatorGateway("http://unused", None),
                admin_token_override="local-provider-secret",
                tunnel_manager=SshTunnelManager(SshTunnelConfig(enabled=False, target="")),
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers={"Authorization": "Bearer local-provider-secret"},
            ) as client:
                models = await client.get("/v1/models")
                self.assertEqual(models.status_code, 200, models.text)
                self.assertEqual(
                    [item["id"] for item in models.json()["data"]],
                    ["legalfedllm-local", "legalfedllm-host"],
                )
                self.assertNotIn("collaborative", models.text.lower())

                with mock.patch(
                    "starlette.requests.Request.is_disconnected",
                    new=mock.AsyncMock(return_value=False),
                ):
                    completion = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "legalfedllm-local",
                            "messages": [{"role": "user", "content": "Explain this locally."}],
                            "temperature": 0.2,
                        },
                    )
                self.assertEqual(completion.status_code, 200, completion.text)
                body = completion.json()
                self.assertIn("legalfedllm_learning_suggestion_id", body)
                self.assertIn("Explain this locally.", body["choices"][0]["message"]["content"])

            admin_headers = {"X-Client-Admin-Token": "local-provider-secret"}
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers=admin_headers,
            ) as client:
                suggestions = await client.get("/v1/learning/suggestions")
                self.assertEqual(suggestions.status_code, 200)
                suggestion = suggestions.json()[0]
                accepted = await client.post(
                    f"/v1/learning/suggestions/{suggestion['suggestion_id']}",
                    json={"learn": True},
                )
                self.assertEqual(accepted.status_code, 200, accepted.text)
                self.assertEqual(accepted.json()["queued_example_count"], 1)
                self.assertEqual(runtime.learning_queue_status()["queued_example_count"], 1)

    async def test_openai_streaming_facade_returns_sse_and_records_local_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = ClientRuntime(
                data_dir=Path(directory) / "client",
                client_id="client-desktop",
                model_profile=mock_profile(),
                private_data_path=Path(directory) / "private" / "train.jsonl",
            )
            app = create_client_app(
                runtime,
                CoordinatorGateway("http://unused", None),
                admin_token_override="local-provider-secret",
                tunnel_manager=SshTunnelManager(SshTunnelConfig(enabled=False, target="")),
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers={"Authorization": "Bearer local-provider-secret"},
            ) as client:
                with mock.patch(
                    "starlette.requests.Request.is_disconnected",
                    new=mock.AsyncMock(return_value=False),
                ):
                    completion = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "legalfedllm-local",
                            "messages": [
                                {"role": "user", "content": "Stream this locally."}
                            ],
                            "max_tokens": 32,
                            "stream": True,
                        },
                    )

            self.assertEqual(completion.status_code, 200, completion.text)
            self.assertTrue(
                completion.headers["content-type"].startswith("text/event-stream")
            )
            events = [
                line.removeprefix("data: ")
                for line in completion.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(events[-1], "[DONE]")
            chunks = [json.loads(event) for event in events[:-1]]
            self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
            streamed_text = "".join(
                chunk["choices"][0]["delta"].get("content", "")
                for chunk in chunks
            )
            self.assertIn("Stream this locally.", streamed_text)
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
            suggestions = runtime.pending_learning_suggestions()
            self.assertEqual(len(suggestions), 1)
            self.assertEqual(suggestions[0]["prompt"], "Stream this locally.")

    async def test_disconnected_local_request_does_not_create_learning_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_data_path = Path(directory) / "private" / "train.jsonl"
            runtime = ClientRuntime(
                data_dir=Path(directory) / "client",
                client_id="client-desktop",
                model_profile=mock_profile(),
                private_data_path=private_data_path,
            )
            app = create_client_app(
                runtime,
                CoordinatorGateway("http://unused", None),
                admin_token_override="local-provider-secret",
                tunnel_manager=SshTunnelManager(SshTunnelConfig(enabled=False, target="")),
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers={"Authorization": "Bearer local-provider-secret"},
            ) as client:
                with mock.patch(
                    "starlette.requests.Request.is_disconnected",
                    new=mock.AsyncMock(return_value=True),
                ):
                    completion = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "legalfedllm-local",
                            "messages": [
                                {"role": "user", "content": "This request was abandoned."}
                            ],
                        },
                    )

            self.assertEqual(completion.status_code, 200, completion.text)
            self.assertNotIn(
                "legalfedllm_learning_suggestion_id",
                completion.json(),
            )
            self.assertEqual(runtime.pending_learning_suggestions(), [])
            self.assertEqual(runtime.learning_queue_status()["queued_example_count"], 0)
            self.assertFalse(private_data_path.exists())

    async def test_host_inference_is_forwarded_with_registered_client_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                registered = await client.post("/v1/register")
                self.assertEqual(registered.status_code, 201, registered.text)
                generated = await client.post(
                    "/v1/generate/host",
                    json={"prompt": "Host question", "max_new_tokens": 12},
                )
                self.assertEqual(generated.status_code, 200, generated.text)
                self.assertIn("Host question", generated.json()["text"])

            unauthenticated = await stack.coordinator_request(
                "POST",
                "/v1/generate",
                json={"prompt": "bypass", "max_new_tokens": 12},
            )
            self.assertEqual(unauthenticated.status_code, 401, unauthenticated.text)

    async def test_ui_status_reports_connected_when_no_round_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stack = Stack(directory)
            _, app = stack.client("client-a")
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://client",
                headers=stack.client_headers,
            ) as client:
                registered = await client.post("/v1/register")
                self.assertEqual(registered.status_code, 201, registered.text)

                ui_status = await client.get("/v1/ui/status")
                self.assertEqual(ui_status.status_code, 200, ui_status.text)
                payload = ui_status.json()
                self.assertTrue(payload["coordinator_connected"])
                self.assertIsNone(payload["round"])
                self.assertNotIn("coordinator_error", payload)


if __name__ == "__main__":
    unittest.main()
