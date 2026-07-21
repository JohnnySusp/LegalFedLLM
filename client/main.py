from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from client.runtime import ClientRuntime, create_mock_client_runtime
from shared.adapter import AdapterSnapshot, AdapterUpdate


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoundRequest(ApiModel):
    examples: list[str] = Field(min_length=1, max_length=1_000)

    @field_validator("examples")
    @classmethod
    def examples_not_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("examples must not contain blank text")
        if any(len(value) > 100_000 for value in values):
            raise ValueError("an example exceeds the 100000 character limit")
        return values


class RoundResponse(ApiModel):
    client_id: str
    round_id: int
    examples_used: int
    parent_adapter_hash: str
    update_hash: str
    global_adapter_hash: str
    global_adapter_version: int


class GenerateRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=256, ge=1, le=4_096)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class GenerateResponse(ApiModel):
    text: str
    model: str
    adapter_version: int


class HostGateway:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def fetch_global_adapter(self) -> AdapterSnapshot:
        async with self._client() as client:
            response = await client.get("/v1/adapters/global")
            response.raise_for_status()
            return AdapterSnapshot.model_validate(response.json())

    async def submit_update(self, update: AdapterUpdate) -> AdapterSnapshot:
        async with self._client() as client:
            response = await client.post(
                "/v1/adapter-updates",
                json=update.model_dump(mode="json"),
            )
            response.raise_for_status()
            return AdapterSnapshot.model_validate(response.json())

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        )


def runtime_from_environment() -> ClientRuntime:
    backend = os.getenv("MODEL_BACKEND", "mock").strip().lower()
    if backend != "mock":
        raise RuntimeError("this milestone supports MODEL_BACKEND=mock only")

    target_modules = tuple(
        item.strip()
        for item in os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")
        if item.strip()
    )
    return create_mock_client_runtime(
        client_id=os.getenv("CLIENT_ID", "legal-client-1"),
        base_model=os.getenv("CLIENT_MODEL_ID", "legal-poc-mock"),
        rank=int(os.getenv("LORA_RANK", "2")),
        target_modules=target_modules,
        integration_policy=os.getenv("CLIENT_ADAPTER_POLICY", "replace").strip().lower(),
    )


def gateway_from_environment() -> HostGateway:
    return HostGateway(
        base_url=os.getenv("HOST_URL", "http://host:8000"),
        timeout_seconds=float(os.getenv("HOST_TIMEOUT_SECONDS", "10")),
    )


def create_app(
    runtime: ClientRuntime | None = None,
    gateway: HostGateway | None = None,
) -> FastAPI:
    client_runtime = runtime or runtime_from_environment()
    host_gateway = gateway or gateway_from_environment()
    app = FastAPI(title="LegalFedLLM Client", version="0.1.0")
    app.state.runtime = client_runtime
    app.state.host_gateway = host_gateway

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        adapter = client_runtime.get_local_adapter()
        return {
            "status": "ok",
            "service": "legal-fed-llm-client",
            "backend": client_runtime.backend_name,
            "client_id": client_runtime.client_id,
            "adapter_version": adapter.version,
        }

    @app.post("/v1/rounds", response_model=RoundResponse)
    async def run_round(request: RoundRequest) -> RoundResponse:
        try:
            global_adapter = await host_gateway.fetch_global_adapter()
            update = client_runtime.prepare_update(global_adapter, request.examples)
            updated_global = await host_gateway.submit_update(update)
            client_runtime.install_global_adapter(updated_global)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Host rejected the round with HTTP {exc.response.status_code}",
            ) from exc
        except (httpx.RequestError, ValidationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Host is unavailable or returned an invalid adapter",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        return RoundResponse(
            client_id=client_runtime.client_id,
            round_id=update.round_id,
            examples_used=update.num_examples,
            parent_adapter_hash=update.parent_adapter_hash,
            update_hash=update.update_hash,
            global_adapter_hash=updated_global.adapter_hash,
            global_adapter_version=updated_global.version,
        )

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        adapter = client_runtime.get_local_adapter()
        return GenerateResponse(
            text=client_runtime.generate(request.prompt, request.max_new_tokens),
            model=adapter.spec.base_model,
            adapter_version=adapter.version,
        )

    return app


app = create_app()
