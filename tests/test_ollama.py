from __future__ import annotations

import json
import unittest

import httpx

from shared.ollama import OllamaClient


class OllamaBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_inspect_and_generate(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "qwen-test:1b", "digest": "sha256:mock"}
                        ]
                    },
                )
            if request.method == "POST" and request.url.path == "/api/show":
                payload = json.loads(request.content)
                return httpx.Response(200, json={"model": payload["model"], "family": "qwen"})
            if request.method == "POST" and request.url.path == "/api/generate":
                return httpx.Response(200, json={"response": "mock Ollama answer"})
            return httpx.Response(404)

        client = OllamaClient(
            "http://ollama", transport=httpx.MockTransport(handler)
        )
        models = await client.list_models()
        inspected = await client.show_model("qwen-test:1b")
        generated = await client.generate("qwen-test:1b", "Hello")
        self.assertEqual(models[0]["name"], "qwen-test:1b")
        self.assertEqual(inspected["family"], "qwen")
        self.assertEqual(generated, "mock Ollama answer")


if __name__ == "__main__":
    unittest.main()
