from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from host.runtime import HostRuntime, HostRuntimeError
from shared.protocol import (
    DistillationJob,
    DistillationResult,
    KnowledgePackage,
    RoundManifest,
    ServiceIdentity,
    HostReferenceDatasetBundle,
    HostReferenceDatasetReceipt,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerateRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)

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
    return HostRuntime(
        data_dir=os.getenv("HOST_DATA_DIR", "data/host"),
        host_id=os.getenv("HOST_ID", "legalfedllm-host"),
        force_validation_failure=os.getenv(
            "MOCK_FORCE_VALIDATION_FAILURE", "false"
        ).lower()
        in {"1", "true", "yes"},
    )


def create_app(
    runtime: HostRuntime | None = None, internal_token_override: str | None = None
) -> FastAPI:
    host_runtime = runtime or runtime_from_environment()
    internal_token = internal_token_override or os.getenv(
        "INTERNAL_API_TOKEN", "development-internal-token"
    )
    app = FastAPI(title="LegalFedLLM Host Runtime", version="0.2.0")
    app.state.runtime = host_runtime

    def require_internal_token(
        x_internal_token: str | None = Header(default=None),
    ) -> None:
        if x_internal_token != internal_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid internal API token",
            )

    @app.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "service": "legalfedllm-host-runtime",
            "adapter_version": host_runtime.adapter_version,
            "training_backend": host_runtime.model_profile.training_backend,
            "serving_backend": host_runtime.model_profile.serving_backend,
        }

    @app.get(
        "/internal/v1/identity",
        response_model=ServiceIdentity,
        dependencies=[Depends(require_internal_token)],
    )
    async def identity() -> ServiceIdentity:
        return ServiceIdentity.model_validate(host_runtime.service_identity())

    @app.post(
        "/internal/v1/reference-data",
        response_model=HostReferenceDatasetReceipt,
        dependencies=[Depends(require_internal_token)],
    )
    async def load_reference_data(
        bundle: HostReferenceDatasetBundle,
    ) -> HostReferenceDatasetReceipt:
        try:
            return host_runtime.load_reference_data(bundle)
        except HostRuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    @app.post(
        "/internal/v1/reference-knowledge",
        response_model=KnowledgePackage,
        dependencies=[Depends(require_internal_token)],
    )
    async def reference_knowledge(manifest: RoundManifest) -> KnowledgePackage:
        try:
            return host_runtime.generate_reference_knowledge(manifest)
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/distill",
        response_model=DistillationResult,
        dependencies=[Depends(require_internal_token)],
    )
    async def distill(job: DistillationJob) -> DistillationResult:
        try:
            return host_runtime.distill(job)
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            text=await host_runtime.generate(request.prompt, request.max_new_tokens),
            model=host_runtime.model_profile.model_id,
            adapter_version=host_runtime.adapter_version,
        )

    return app


app = create_app()
