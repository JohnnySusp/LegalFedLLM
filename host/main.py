from __future__ import annotations

import asyncio
import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from host.runtime import HostRuntime, HostRuntimeError
from shared.knowledge_transport import (
    KnowledgeTransportError,
    KnowledgeTransportTooLarge,
    knowledge_transfer_response,
    receive_knowledge_transfer,
)
from shared.protocol import (
    DistillationJob,
    DistillationResult,
    HostCandidateTrainingResult,
    HostCandidateValidationResult,
    KnowledgePackage,
    RoundManifest,
    ServiceIdentity,
    HostReferenceDatasetBundle,
    HostReferenceDatasetReceipt,
    HostTrainingJob,
    HostTrainingJobReceipt,
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
    app.state.ml_lock = asyncio.Lock()

    async def run_exclusive_ml(call, *args):
        if app.state.ml_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="another Host ML job is already running",
            )
        async with app.state.ml_lock:
            return await asyncio.to_thread(call, *args)

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
        dependencies=[Depends(require_internal_token)],
    )
    async def reference_knowledge(manifest: RoundManifest):
        try:
            package = await run_exclusive_ml(
                host_runtime.generate_reference_knowledge,
                manifest,
            )
            return knowledge_transfer_response(
                metadata=package,
                artifact_path=host_runtime.knowledge_artifact_path(
                    manifest,
                    enforce_manifest_parent=True,
                ),
                metadata_part_name="package",
            )
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/distill",
        dependencies=[Depends(require_internal_token)],
    )
    async def distill(job: DistillationJob):
        try:
            result = await run_exclusive_ml(host_runtime.distill, job)
            return knowledge_transfer_response(
                metadata=result,
                artifact_path=host_runtime.knowledge_artifact_path(
                    job.manifest,
                    enforce_manifest_parent=False,
                ),
                metadata_part_name="result",
            )
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/training-job",
        response_model=HostTrainingJobReceipt,
        dependencies=[Depends(require_internal_token)],
    )
    async def load_training_job(request: Request) -> HostTrainingJobReceipt:
        incoming = host_runtime.store.path(
            f"incoming/training-job-{secrets.token_hex(8)}.safetensors"
        )
        maximum = int(
            os.getenv(
                "HOST_MAXIMUM_TRAINING_JOB_BYTES",
                str(512 * 1024 * 1024),
            )
        )
        try:
            received = await receive_knowledge_transfer(
                content_type=request.headers.get("content-type", ""),
                chunks=request.stream(),
                artifact_path=incoming,
                metadata_part_name="job",
                maximum_content_bytes=maximum,
            )
            job = HostTrainingJob.model_validate(received.metadata)
            return await run_exclusive_ml(
                host_runtime.load_training_job,
                job,
                received.artifact_path,
            )
        except KnowledgeTransportTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (KnowledgeTransportError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            incoming.unlink(missing_ok=True)

    @app.post(
        "/internal/v1/train-candidate",
        response_model=HostCandidateTrainingResult,
        dependencies=[Depends(require_internal_token)],
    )
    async def train_candidate(
        job: HostTrainingJob,
    ) -> HostCandidateTrainingResult:
        try:
            return await run_exclusive_ml(host_runtime.train_candidate, job)
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/validate-candidate",
        response_model=HostCandidateValidationResult,
        dependencies=[Depends(require_internal_token)],
    )
    async def validate_candidate(
        job: HostTrainingJob,
    ) -> HostCandidateValidationResult:
        try:
            return await run_exclusive_ml(
                host_runtime.validate_candidate_and_decide,
                job,
            )
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/internal/v1/post-decision-knowledge",
        dependencies=[Depends(require_internal_token)],
    )
    async def post_decision_knowledge(manifest: RoundManifest):
        try:
            package = await run_exclusive_ml(
                host_runtime.generate_post_decision_reference_knowledge,
                manifest,
            )
            return knowledge_transfer_response(
                metadata=package,
                artifact_path=host_runtime.knowledge_artifact_path(
                    manifest,
                    enforce_manifest_parent=False,
                ),
                metadata_part_name="package",
            )
        except HostRuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        if host_runtime.model_profile.serving_backend == "transformers":
            text = await run_exclusive_ml(
                host_runtime.generate_transformers,
                [{"role": "user", "content": request.prompt}],
                request.max_new_tokens,
            )
        else:
            text = await host_runtime.generate(
                request.prompt,
                request.max_new_tokens,
            )
        return GenerateResponse(
            text=text,
            model=host_runtime.model_profile.model_id,
            adapter_version=host_runtime.adapter_version,
        )

    return app


app = create_app()
