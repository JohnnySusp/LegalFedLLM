from __future__ import annotations

import json
import unittest

import httpx
from pydantic import ValidationError

from client.main import HostGateway, create_app as create_client_app
from client.runtime import create_mock_client_runtime
from host.main import create_app as create_host_app
from host.runtime import create_mock_host_runtime
from shared.adapter import AdapterUpdate


class AdapterContractTests(unittest.TestCase):
    def test_tampered_update_is_rejected(self) -> None:
        host = create_mock_host_runtime()
        client = create_mock_client_runtime()
        update = client.prepare_update(host.get_global_adapter(), ["private example"])
        payload = update.model_dump(mode="json")
        first_tensor = next(iter(payload["delta"]))
        payload["delta"][first_tensor][0] += 1.0

        with self.assertRaises(ValidationError):
            AdapterUpdate.model_validate(payload)


class HostApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_host_accepts_one_compatible_update(self) -> None:
        host_runtime = create_mock_host_runtime()
        local_runtime = create_mock_client_runtime()
        update = local_runtime.prepare_update(
            host_runtime.get_global_adapter(), ["local court record"]
        )

        transport = httpx.ASGITransport(app=create_host_app(host_runtime))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://host:8000"
        ) as api:
            response = await api.post(
                "/v1/adapter-updates", json=update.model_dump(mode="json")
            )
            health = await api.get("/health")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["version"], 1)
        self.assertEqual(health.json()["adapter_version"], 1)


class MockRoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_host_http_round_does_not_send_raw_examples(self) -> None:
        host_runtime = create_mock_host_runtime()
        client_runtime = create_mock_client_runtime()
        submitted_bodies: list[bytes] = []

        def host_handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/v1/adapters/global":
                return httpx.Response(
                    200,
                    json=host_runtime.get_global_adapter().model_dump(mode="json"),
                )
            if request.method == "POST" and request.url.path == "/v1/adapter-updates":
                submitted_bodies.append(request.content)
                update = AdapterUpdate.model_validate(json.loads(request.content))
                adapter = host_runtime.submit_update(update)
                return httpx.Response(201, json=adapter.model_dump(mode="json"))
            return httpx.Response(404)

        gateway = HostGateway(
            "http://host:8000", transport=httpx.MockTransport(host_handler)
        )
        private_example = "confidential legal memo: matter 42"

        transport = httpx.ASGITransport(app=create_client_app(client_runtime, gateway))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://client:8001"
        ) as api:
            response = await api.post(
                "/v1/rounds", json={"examples": [private_example]}
            )
            generation = await api.post(
                "/v1/generate", json={"prompt": "Summarize locally"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["global_adapter_version"], 1)
        self.assertEqual(host_runtime.get_global_adapter().version, 1)
        self.assertEqual(client_runtime.get_local_adapter().version, 1)
        self.assertNotIn(private_example.encode(), submitted_bodies[0])
        self.assertEqual(generation.status_code, 200)
        self.assertEqual(generation.json()["adapter_version"], 1)


if __name__ == "__main__":
    unittest.main()
