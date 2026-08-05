from __future__ import annotations

import hmac
import os
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from client.runtime import ClientRuntime, ClientRuntimeError
from shared.ollama import OllamaError
from shared.protocol import (
    ClientRegistrationRequest,
    KnowledgePackage,
    RegistrationRecord,
    RoundManifest,
    RoundState,
    ServiceIdentity,
    SubmissionReceipt,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalTrainRequest(ApiModel):
    examples: list[str] = Field(min_length=1, max_length=100_000)


class GenerateRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class OllamaInspectRequest(ApiModel):
    model: str = Field(min_length=1, max_length=256)


class CoordinatorGateway:
    def __init__(
        self,
        base_url: str,
        registration_token: str,
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.registration_token = registration_token
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.request(method, path, json=json, headers=headers)
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Coordinator returned {response.status_code}: {response.text[:500]}",
            )
        return response.json()

    async def identity(self) -> ServiceIdentity:
        return ServiceIdentity.model_validate(await self._request("GET", "/v1/identity"))

    async def register(self, request: ClientRegistrationRequest) -> RegistrationRecord:
        payload = await self._request(
            "POST",
            "/v1/clients/register",
            json=request.model_dump(mode="json"),
            headers={"X-Registration-Token": self.registration_token},
        )
        return RegistrationRecord.model_validate(payload)

    async def current_manifest(self) -> RoundManifest:
        return RoundManifest.model_validate(
            await self._request("GET", "/v1/rounds/current")
        )

    async def manifest(self, round_id: str) -> RoundManifest:
        return RoundManifest.model_validate(
            await self._request(
                "GET",
                f"/v1/rounds/{round_id}/manifest",
            )
        )

    async def reference_dataset(
        self,
        round_id: str,
        client_id: str,
    ) -> bytes | None:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.get(
                f"/v1/rounds/{round_id}/reference-dataset",
                headers={
                    "X-Client-ID": client_id,
                    "X-Registration-Token": (
                        self.registration_token
                    ),
                },
            )

        if response.status_code == 204:
            return None

        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    f"Coordinator returned "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                ),
            )

        return response.content

    async def submit(self, package: KnowledgePackage) -> SubmissionReceipt:
        payload = await self._request(
            "POST",
            f"/v1/rounds/{package.round_id}/knowledge",
            json=package.model_dump(mode="json"),
        )
        return SubmissionReceipt.model_validate(payload)

    async def status(self, round_id: str) -> RoundState:
        return RoundState.model_validate(
            await self._request("GET", f"/v1/rounds/{round_id}/status")
        )

    async def host_knowledge(self, round_id: str) -> KnowledgePackage:
        return KnowledgePackage.model_validate(
            await self._request(
                "GET", f"/v1/rounds/{round_id}/host-knowledge"
            )
        )


def runtime_from_environment() -> ClientRuntime:
    return ClientRuntime(
        data_dir=os.getenv("CLIENT_DATA_DIR", "data/client"),
        client_id=os.getenv("CLIENT_ID", "legal-client-1"),
        maximum_clock_skew_seconds=int(
            os.getenv("MAXIMUM_CLOCK_SKEW_SECONDS", "900")
        ),
    )


def gateway_from_environment() -> CoordinatorGateway:
    return CoordinatorGateway(
        os.getenv("COORDINATOR_URL", "http://coordinator:8000"),
        os.getenv("REGISTRATION_TOKEN", "development-registration-token"),
        timeout_seconds=float(os.getenv("COORDINATOR_TIMEOUT_SECONDS", "60")),
    )


def create_app(
    runtime: ClientRuntime | None = None,
    gateway: CoordinatorGateway | None = None,
    admin_token_override: str | None = None,
) -> FastAPI:
    client_runtime = runtime or runtime_from_environment()
    coordinator = gateway or gateway_from_environment()

    admin_token = admin_token_override or os.getenv(
        "CLIENT_ADMIN_TOKEN",
        "development-client-admin-token",
    )

    app = FastAPI(title="LegalFedLLM Client Agent", version="0.2.0")
    app.state.runtime = client_runtime
    app.state.gateway = coordinator

    def require_client_admin_token(
        x_client_admin_token: str | None = Header(default=None),
    ) -> None:
        if x_client_admin_token is None or not hmac.compare_digest(
            x_client_admin_token,
            admin_token,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid Client admin token",
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = client_runtime.state()
        return {
            "status": "ok",
            "service": "legalfedllm-client-agent",
            "client_id": client_runtime.client_id,
            "training_backend": client_runtime.model_profile.training_backend,
            "serving_backend": client_runtime.model_profile.serving_backend,
            **state,
        }

    @app.post(
        "/v1/register",
        response_model=RegistrationRecord,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_client_admin_token)],
    )
    async def register() -> RegistrationRecord:
        return await coordinator.register(
            ClientRegistrationRequest(
                client_id=client_runtime.client_id,
                public_key=client_runtime.identity.public_key_b64,
                model_profile=client_runtime.model_profile,
            )
        )

    @app.post(
        "/v1/local-train",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def local_train(request: LocalTrainRequest) -> dict[str, Any]:
        try:
            return client_runtime.local_train(request.examples)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/v1/participate",
        response_model=SubmissionReceipt,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_client_admin_token)],
    )
    async def participate() -> SubmissionReceipt:
        identity = await coordinator.identity()
        manifest = await coordinator.current_manifest()

        if not manifest.verify_signature(identity.public_key):
            raise HTTPException(
                status_code=401,
                detail="round manifest signature is invalid",
            )

        try:
            reference_content = (
                await coordinator.reference_dataset(
                    manifest.round_id,
                    client_runtime.client_id,
                )
            )

            if reference_content is not None:
                client_runtime.cache_reference_dataset(
                    manifest=manifest,
                    content=reference_content,
                )

            package = client_runtime.create_knowledge_package(manifest)

            receipt = await coordinator.submit(package)

            client_runtime.commit_knowledge_submission(
                manifest=manifest,
                package=package,
                receipt=receipt,
            )

            return receipt

        except ClientRuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    @app.post(
        "/v1/rounds/{round_id}/sync",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def sync(round_id: str) -> dict[str, Any]:
        identity = await coordinator.identity()

        manifest = await coordinator.manifest(round_id)

        if not manifest.verify_signature(identity.public_key):
            raise HTTPException(
                status_code=401,
                detail="round manifest signature is invalid",
            )

        status_record = await coordinator.status(round_id)

        if status_record.state != "COMPLETED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"round is not completed: {status_record.state}",
            )

        if (
            identity.host_public_key is None
            or identity.host_service_id is None
            or status_record.host_adapter_after is None
        ):
            raise HTTPException(
                status_code=409,
                detail="Host identity or accepted adapter version is missing",
            )

        package = await coordinator.host_knowledge(round_id)

        try:
            return client_runtime.apply_host_knowledge(
                manifest=manifest,
                host_package=package,
                host_public_key=identity.host_public_key,
                expected_host_id=identity.host_service_id,
                accepted_host_adapter_version=(
                    status_record.host_adapter_after
                ),
                adapter_promoted=bool(status_record.adapter_promoted),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    @app.post("/v1/generate")
    async def generate(request: GenerateRequest) -> dict[str, Any]:
        return {
            "text": await client_runtime.generate(
                request.prompt, request.max_new_tokens
            ),
            "model": client_runtime.model_profile.model_id,
            "adapter_version": client_runtime.state()["serving_adapter_version"],
        }

    @app.get(
        "/v1/ollama/models",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def ollama_models() -> list[dict[str, Any]]:
        if client_runtime.ollama is None:
            raise HTTPException(status_code=409, detail="Ollama serving is not enabled")
        try:
            return await client_runtime.ollama.list_models()
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post(
        "/v1/ollama/inspect",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def ollama_inspect(request: OllamaInspectRequest) -> dict[str, Any]:
        if client_runtime.ollama is None:
            raise HTTPException(status_code=409, detail="Ollama serving is not enabled")
        try:
            return await client_runtime.ollama.show_model(request.model)
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
