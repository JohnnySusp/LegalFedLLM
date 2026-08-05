from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from coordinator.service import CoordinatorError, CoordinatorService, HostGateway
from shared.protocol import (
    ClientRegistrationRequest,
    KnowledgePackage,
    RegistrationRecord,
    RoundCreateRequest,
    RoundManifest,
    RoundState,
    ServiceIdentity,
    SubmissionReceipt,
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


def service_from_environment() -> CoordinatorService:
    internal_token = os.getenv(
        "INTERNAL_API_TOKEN", "development-internal-token"
    )
    gateway = HostGateway(
        os.getenv("HOST_RUNTIME_URL", "http://host:8002"),
        internal_token,
        timeout_seconds=float(os.getenv("HOST_TIMEOUT_SECONDS", "60")),
    )

    reference_dataset_path = (
        os.getenv("COORDINATOR_REFERENCE_DATASET_PATH") or None
    )
    validation_dataset_path = (
        os.getenv("COORDINATOR_VALIDATION_DATASET_PATH") or None
    )

    return CoordinatorService(
        data_dir=os.getenv(
            "COORDINATOR_DATA_DIR",
            "data/coordinator",
        ),
        host_gateway=gateway,
        coordinator_id=os.getenv(
            "COORDINATOR_ID",
            "legalfedllm-coordinator",
        ),
        registration_token=os.getenv(
            "REGISTRATION_TOKEN",
            "development-registration-token",
        ),
        admin_token=os.getenv(
            "ADMIN_TOKEN",
            "development-admin-token",
        ),
        maximum_clock_skew_seconds=int(
            os.getenv("MAXIMUM_CLOCK_SKEW_SECONDS", "900")
        ),
        reference_dataset_path=reference_dataset_path,
        validation_dataset_path=validation_dataset_path,
    )


def create_app(service: CoordinatorService | None = None) -> FastAPI:
    coordinator = service or service_from_environment()
    monitor_interval = float(os.getenv("ROUND_MONITOR_INTERVAL_SECONDS", "2"))

    async def monitor_rounds() -> None:
        while True:
            try:
                await coordinator.monitor_once()
            except Exception:
                pass
            await asyncio.sleep(monitor_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(monitor_rounds())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="LegalFedLLM Federated Coordinator",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.service = coordinator

    @app.exception_handler(CoordinatorError)
    async def coordinator_error_handler(
        request: Request, exc: CoordinatorError
    ) -> HTTPException:
        return __import__("fastapi").responses.JSONResponse(
            status_code=exc.status_code, content={"detail": str(exc)}
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "legalfedllm-coordinator"}

    @app.get("/v1/identity", response_model=ServiceIdentity)
    async def identity() -> ServiceIdentity:
        return await coordinator.service_identity()

    @app.post(
        "/v1/clients/register",
        response_model=RegistrationRecord,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_client(
        request: ClientRegistrationRequest,
        x_registration_token: str | None = Header(default=None),
    ) -> RegistrationRecord:
        coordinator.require_registration_token(x_registration_token)
        return coordinator.register_client(request)

    @app.post(
        "/v1/rounds",
        response_model=RoundManifest,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_round(
        request: RoundCreateRequest,
        x_admin_token: str | None = Header(default=None),
    ) -> RoundManifest:
        coordinator.require_admin_token(x_admin_token)
        return await coordinator.create_round(request)

    @app.get("/v1/rounds/current", response_model=RoundManifest)
    async def current_round() -> RoundManifest:
        return await coordinator.current_manifest()

    @app.get("/v1/rounds/{round_id}/manifest", response_model=RoundManifest)
    async def manifest(round_id: str) -> RoundManifest:
        return coordinator.get_manifest(round_id)

    @app.get("/v1/rounds/{round_id}/status", response_model=RoundState)
    async def round_status(round_id: str) -> RoundState:
        return await coordinator.round_status(round_id)

    @app.post(
        "/v1/rounds/{round_id}/knowledge",
        response_model=SubmissionReceipt,
        status_code=status.HTTP_201_CREATED,
    )
    async def submit_knowledge(round_id: str, request: Request) -> SubmissionReceipt:
        manifest = coordinator.get_manifest(round_id)
        raw = await request.body()
        if len(raw) > manifest.maximum_knowledge_package_bytes:
            raise HTTPException(status_code=413, detail="Knowledge Package is too large")
        try:
            payload = json.loads(raw)
            package = KnowledgePackage.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if package.round_id != round_id:
            raise HTTPException(status_code=409, detail="round ID path mismatch")
        return await coordinator.submit_knowledge(package, len(raw))

    @app.get(
        "/v1/rounds/{round_id}/host-knowledge", response_model=KnowledgePackage
    )
    async def host_knowledge(round_id: str) -> KnowledgePackage:
        return coordinator.get_host_knowledge(round_id)

    @app.post("/v1/generate")
    async def generate(request: GenerateRequest) -> dict:
        return await coordinator.host.generate(request.prompt, request.max_new_tokens)

    @app.get(
        "/v1/rounds/{round_id}/reference-dataset"
    )
    async def reference_dataset(
        round_id: str,
        x_client_id: str | None = Header(default=None),
        x_registration_token: str | None = Header(default=None),
    ) -> Response:
        coordinator.require_registration_token(
            x_registration_token
        )

        path = coordinator.get_reference_dataset_path(
            round_id,
            x_client_id,
        )

        if path is None:
            return Response(status_code=204)

        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename=f"{round_id}-reference.jsonl",
        )

    return app


app = create_app()
