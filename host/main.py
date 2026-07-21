from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from host.runtime import HostRuntime, create_mock_host_runtime
from shared.adapter import AdapterCompatibilityError, AdapterSnapshot, AdapterUpdate


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


def runtime_from_environment() -> HostRuntime:
    backend = os.getenv("MODEL_BACKEND", "mock").strip().lower()
    if backend != "mock":
        raise RuntimeError("this milestone supports MODEL_BACKEND=mock only")

    target_modules = tuple(
        item.strip()
        for item in os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")
        if item.strip()
    )
    return create_mock_host_runtime(
        base_model=os.getenv("HOST_MODEL_ID", "legal-poc-mock"),
        rank=int(os.getenv("LORA_RANK", "2")),
        target_modules=target_modules,
    )


def create_app(runtime: HostRuntime | None = None) -> FastAPI:
    host_runtime = runtime or runtime_from_environment()
    app = FastAPI(title="LegalFedLLM Host", version="0.1.0")
    app.state.runtime = host_runtime

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        adapter = host_runtime.get_global_adapter()
        return {
            "status": "ok",
            "service": "legal-fed-llm-host",
            "backend": host_runtime.backend_name,
            "adapter_version": adapter.version,
        }

    @app.get("/v1/adapters/global", response_model=AdapterSnapshot)
    async def global_adapter() -> AdapterSnapshot:
        return host_runtime.get_global_adapter()

    @app.post(
        "/v1/adapter-updates",
        response_model=AdapterSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def submit_update(update: AdapterUpdate) -> AdapterSnapshot:
        try:
            return host_runtime.submit_update(update)
        except AdapterCompatibilityError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        adapter = host_runtime.get_global_adapter()
        return GenerateResponse(
            text=host_runtime.generate(request.prompt, request.max_new_tokens),
            model=adapter.spec.base_model,
            adapter_version=adapter.version,
        )

    return app


app = create_app()
