from __future__ import annotations

from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.request(method, path, json=json)
        if response.status_code >= 400:
            raise OllamaError(
                f"Ollama returned {response.status_code}: {response.text[:500]}"
            )
        if not response.content:
            return {}
        return response.json()

    async def health(self) -> bool:
        try:
            await self._request("GET", "/api/tags")
            return True
        except (httpx.HTTPError, OllamaError):
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/tags")
        return list(payload.get("models", []))

    async def show_model(self, model: str) -> dict[str, Any]:
        return await self._request("POST", "/api/show", json={"model": model})

    async def generate(
        self, model: str, prompt: str, max_new_tokens: int = 256
    ) -> str:
        payload = await self._request(
            "POST",
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_new_tokens},
            },
        )
        response = payload.get("response")
        if not isinstance(response, str):
            raise OllamaError("Ollama response did not contain generated text")
        return response
