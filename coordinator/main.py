from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from coordinator.quorum import TrustedClientQuorumPolicy
from coordinator.service import CoordinatorError, CoordinatorService, HostGateway
from shared.knowledge_transport import (
    KnowledgeTransportError,
    KnowledgeTransportTooLarge,
    knowledge_transfer_response,
    receive_knowledge_transfer,
)
from shared.protocol import (
    ClientRegistrationRequest,
    ClientRequestAuthentication,
    EnrollmentTokenIssue,
    KnowledgePackage,
    RegistrationRecord,
    RoundCreateRequest,
    RoundManifest,
    RoundState,
    SafetyReport,
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
    quorum_policy_name = os.getenv(
        "COORDINATOR_QUORUM_POLICY",
        "explicit",
    ).strip().lower()
    if quorum_policy_name not in {"explicit", "majority"}:
        raise ValueError(
            "COORDINATOR_QUORUM_POLICY must be explicit or majority"
        )
    quorum_override_text = os.getenv(
        "COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE",
        "",
    ).strip()
    quorum_policy = (
        TrustedClientQuorumPolicy(
            minimum=int(
                os.getenv(
                    "COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM",
                    "2",
                )
            ),
            override=(
                int(quorum_override_text)
                if quorum_override_text
                else None
            ),
        )
        if quorum_policy_name == "majority"
        else None
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
        initial_enrollment_token=(
            os.getenv("REGISTRATION_TOKEN", "").strip() or None
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
        quorum_policy=quorum_policy,
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
        policy = coordinator.quorum_policy
        return {
            "status": "ok",
            "service": "legalfedllm-coordinator",
            "quorum_policy": "majority" if policy is not None else "explicit",
            "minimum_trusted_client_quorum": (
                str(policy.minimum) if policy is not None else "request"
            ),
            "trusted_client_quorum_override": (
                str(policy.override)
                if policy is not None and policy.override is not None
                else "none"
            ),
        }

    @app.get("/v1/identity", response_model=ServiceIdentity)
    async def identity() -> ServiceIdentity:
        return await coordinator.service_identity()

    @app.post(
        "/v1/enrollment-tokens",
        response_model=EnrollmentTokenIssue,
        status_code=status.HTTP_201_CREATED,
    )
    async def issue_enrollment_token(
        x_admin_token: str | None = Header(default=None),
    ) -> EnrollmentTokenIssue:
        coordinator.require_admin_token(x_admin_token)
        return coordinator.issue_enrollment_token()

    def authenticate_client_request(
        request: Request,
        *,
        expected_client_id: str | None,
        x_client_id: str | None,
        x_client_timestamp: str | None,
        x_client_nonce: str | None,
        x_client_signature: str | None,
    ) -> RegistrationRecord:
        if not all(
            (
                x_client_id,
                x_client_timestamp,
                x_client_nonce,
                x_client_signature,
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing signed Client request authentication",
            )
        try:
            authentication = ClientRequestAuthentication(
                client_id=x_client_id,
                method=request.method,
                path=request.url.path,
                timestamp=x_client_timestamp,
                nonce=x_client_nonce,
                signature=x_client_signature,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid signed Client request authentication",
            ) from exc
        return coordinator.authenticate_client_request(
            authentication,
            method=request.method,
            path=request.url.path,
            expected_client_id=expected_client_id,
        )

    @app.post(
        "/v1/clients/register",
        response_model=RegistrationRecord,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_client(
        request: ClientRegistrationRequest,
        x_registration_token: str | None = Header(default=None),
    ) -> RegistrationRecord:
        return coordinator.register_client(request, x_registration_token)

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

    @app.get(
        "/v1/rounds/{round_id}/submissions/{client_id}/receipt",
        response_model=SubmissionReceipt,
    )
    async def submission_receipt(
        request: Request,
        round_id: str,
        client_id: str,
        x_client_id: str | None = Header(default=None),
        x_client_timestamp: str | None = Header(default=None),
        x_client_nonce: str | None = Header(default=None),
        x_client_signature: str | None = Header(default=None),
    ) -> SubmissionReceipt:
        authenticate_client_request(
            request,
            expected_client_id=client_id,
            x_client_id=x_client_id,
            x_client_timestamp=x_client_timestamp,
            x_client_nonce=x_client_nonce,
            x_client_signature=x_client_signature,
        )
        return coordinator.get_submission_receipt(round_id, client_id)

    @app.get(
        "/v1/rounds/{round_id}/safety",
        response_model=dict[str, SafetyReport],
    )
    async def round_safety(
        round_id: str,
        x_admin_token: str | None = Header(default=None),
    ) -> dict[str, SafetyReport]:
        coordinator.require_admin_token(x_admin_token)
        return coordinator.get_safety_reports(round_id)

    @app.post(
        "/v1/rounds/{round_id}/knowledge",
        response_model=SubmissionReceipt,
        status_code=status.HTTP_201_CREATED,
    )
    async def submit_knowledge(round_id: str, request: Request) -> SubmissionReceipt:
        manifest = coordinator.get_manifest(round_id)
        incoming = coordinator.incoming_artifact_path(
            round_id,
            "client-submission",
        )
        try:
            received = await receive_knowledge_transfer(
                content_type=request.headers.get("content-type", ""),
                chunks=request.stream(),
                artifact_path=incoming,
                metadata_part_name="package",
                maximum_content_bytes=(
                    manifest.maximum_knowledge_package_bytes
                ),
            )
            package = KnowledgePackage.model_validate(
                received.metadata
            )
        except KnowledgeTransportTooLarge as exc:
            incoming.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (KnowledgeTransportError, ValidationError) as exc:
            incoming.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            if package.round_id != round_id:
                raise HTTPException(status_code=409, detail="round ID path mismatch")
            return await coordinator.submit_knowledge(
                package,
                received.artifact_path,
                received.content_size,
            )
        finally:
            incoming.unlink(missing_ok=True)

    @app.get("/v1/rounds/{round_id}/host-knowledge")
    async def host_knowledge(round_id: str):
        package, artifact_path = coordinator.get_host_knowledge(round_id)
        return knowledge_transfer_response(
            metadata=package,
            artifact_path=artifact_path,
            metadata_part_name="package",
        )

    @app.post("/v1/generate")
    async def generate(request: GenerateRequest) -> dict:
        return await coordinator.host.generate(request.prompt, request.max_new_tokens)

    @app.get(
        "/v1/rounds/{round_id}/reference-dataset"
    )
    async def reference_dataset(
        request: Request,
        round_id: str,
        x_client_id: str | None = Header(default=None),
        x_client_timestamp: str | None = Header(default=None),
        x_client_nonce: str | None = Header(default=None),
        x_client_signature: str | None = Header(default=None),
    ) -> Response:
        authenticate_client_request(
            request,
            expected_client_id=x_client_id,
            x_client_id=x_client_id,
            x_client_timestamp=x_client_timestamp,
            x_client_nonce=x_client_nonce,
            x_client_signature=x_client_signature,
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
