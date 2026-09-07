from __future__ import annotations

import asyncio
import hmac
import os
from functools import partial
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from client.runtime import ClientRuntime, ClientRuntimeError
from shared.crypto import Ed25519Identity, canonical_json_bytes
from shared.knowledge_transport import receive_knowledge_transfer
from shared.ollama import OllamaError
from shared.protocol import (
    ClientRegistrationRequest,
    ClientRequestAuthentication,
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
        registration_token: str | None,
        *,
        timeout_seconds: float = 60.0,
        reconciliation_timeout_seconds: float = 30.0,
        reconciliation_poll_seconds: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("Coordinator timeout must be positive")
        if reconciliation_timeout_seconds < 0:
            raise ValueError("Coordinator reconciliation timeout cannot be negative")
        if reconciliation_poll_seconds <= 0:
            raise ValueError("Coordinator reconciliation poll interval must be positive")
        self.base_url = base_url.rstrip("/")
        self.registration_token = (registration_token or "").strip() or None
        self.client_id: str | None = None
        self.client_identity: Ed25519Identity | None = None
        self.timeout_seconds = timeout_seconds
        self.reconciliation_timeout_seconds = reconciliation_timeout_seconds
        self.reconciliation_poll_seconds = reconciliation_poll_seconds
        self.transport = transport

    def bind_client_identity(
        self,
        client_id: str,
        identity: Ed25519Identity,
    ) -> None:
        self.client_id = client_id
        self.client_identity = identity

    def _client_auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self.client_id is None or self.client_identity is None:
            raise RuntimeError("Coordinator gateway is not bound to a Client identity")
        authentication = ClientRequestAuthentication.create_signed(
            identity=self.client_identity,
            client_id=self.client_id,
            method=method,
            path=path,
        )
        return {
            "X-Client-ID": authentication.client_id,
            "X-Client-Timestamp": authentication.timestamp,
            "X-Client-Nonce": authentication.nonce,
            "X-Client-Signature": authentication.signature,
        }

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
        if self.registration_token is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Client enrollment token is not configured",
            )
        payload = await self._request(
            "POST",
            "/v1/clients/register",
            json=request.model_dump(mode="json"),
            headers={"X-Registration-Token": self.registration_token},
        )
        record = RegistrationRecord.model_validate(payload)
        self.registration_token = None
        return record

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
            path = f"/v1/rounds/{round_id}/reference-dataset"
            if self.client_id != client_id:
                raise RuntimeError("Coordinator gateway is bound to another Client ID")
            response = await client.get(
                path,
                headers=self._client_auth_headers("GET", path),
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

    async def submit(
        self,
        package: KnowledgePackage,
        artifact_path: str | Path,
    ) -> SubmissionReceipt:
        with Path(artifact_path).open("rb") as artifact:
            files = [
                (
                    "package",
                    (
                        "package.json",
                        canonical_json_bytes(package.model_dump(mode="json")),
                        "application/json",
                    ),
                ),
                (
                    "artifact",
                    (
                        "knowledge.safetensors",
                        artifact,
                        "application/octet-stream",
                    ),
                ),
            ]
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"/v1/rounds/{package.round_id}/knowledge",
                    files=files,
                )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    f"Coordinator returned {response.status_code}: "
                    f"{response.text[:500]}"
                ),
            )
        return SubmissionReceipt.model_validate(response.json())

    async def accepted_submission_receipt(
        self,
        package: KnowledgePackage,
        *,
        timeout_seconds: float | None = None,
    ) -> SubmissionReceipt | None:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=(
                self.timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
            transport=self.transport,
        ) as client:
            path = (
                f"/v1/rounds/{package.round_id}/submissions/"
                f"{package.sender_id}/receipt"
            )
            response = await client.get(
                path,
                headers=self._client_auth_headers("GET", path),
            )

        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=(
                    f"Coordinator returned {response.status_code}: "
                    f"{response.text[:500]}"
                ),
            )

        receipt = SubmissionReceipt.model_validate(response.json())
        if (
            receipt.round_id != package.round_id
            or receipt.client_id != package.sender_id
            or receipt.package_hash != package.package_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Coordinator reports a different accepted Client "
                    "submission for this round"
                ),
            )
        return receipt

    async def reconcile_submission(
        self,
        package: KnowledgePackage,
    ) -> SubmissionReceipt | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.reconciliation_timeout_seconds
        poll_seconds = self.reconciliation_poll_seconds

        while True:
            remaining = deadline - loop.time()
            request_timeout = min(
                self.timeout_seconds,
                max(0.1, remaining),
            )
            try:
                receipt = await self.accepted_submission_receipt(
                    package,
                    timeout_seconds=request_timeout,
                )
            except httpx.RequestError:
                receipt = None
            if receipt is not None:
                return receipt

            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(poll_seconds, remaining))

    async def status(self, round_id: str) -> RoundState:
        return RoundState.model_validate(
            await self._request("GET", f"/v1/rounds/{round_id}/status")
        )

    async def host_knowledge(
        self,
        round_id: str,
        artifact_path: str | Path,
        maximum_bytes: int,
    ) -> KnowledgePackage:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            async with client.stream(
                "GET", f"/v1/rounds/{round_id}/host-knowledge"
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=(
                            f"Coordinator returned {response.status_code}: "
                            f"{response.text[:500]}"
                        ),
                    )
                received = await receive_knowledge_transfer(
                    content_type=response.headers.get("content-type", ""),
                    chunks=response.aiter_bytes(),
                    artifact_path=artifact_path,
                    metadata_part_name="package",
                    maximum_content_bytes=maximum_bytes,
                )
        return KnowledgePackage.model_validate(received.metadata)


def runtime_from_environment() -> ClientRuntime:
    return ClientRuntime(
        data_dir=os.getenv("CLIENT_DATA_DIR", "data/client"),
        client_id=os.getenv("CLIENT_ID", "legal-client-1"),
        maximum_clock_skew_seconds=int(
            os.getenv("MAXIMUM_CLOCK_SKEW_SECONDS", "900")
        ),
        force_reverse_validation_failure=os.getenv(
            "CLIENT_FORCE_REVERSE_VALIDATION_FAILURE",
            "false",
        ).strip().lower()
        in {"1", "true", "yes"},
    )


def gateway_from_environment() -> CoordinatorGateway:
    return CoordinatorGateway(
        os.getenv("COORDINATOR_URL", "http://coordinator:8000"),
        os.getenv("REGISTRATION_TOKEN", "").strip() or None,
        timeout_seconds=float(os.getenv("COORDINATOR_TIMEOUT_SECONDS", "60")),
        reconciliation_timeout_seconds=float(
            os.getenv("COORDINATOR_RECONCILIATION_TIMEOUT_SECONDS", "30")
        ),
        reconciliation_poll_seconds=float(
            os.getenv("COORDINATOR_RECONCILIATION_POLL_SECONDS", "1")
        ),
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
    coordinator.bind_client_identity(
        client_runtime.client_id,
        client_runtime.identity,
    )
    app.state.ml_lock = asyncio.Lock()
    app.state.training_lock = app.state.ml_lock

    async def run_exclusive_ml(call, *args):
        if app.state.ml_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="another local ML job is already running",
            )
        async with app.state.ml_lock:
            return await asyncio.to_thread(call, *args)

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
            "enrolled": client_runtime.registration_record() is not None,
            **state,
        }

    @app.post(
        "/v1/register",
        response_model=RegistrationRecord,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_client_admin_token)],
    )
    async def register() -> RegistrationRecord:
        current = client_runtime.registration_record()
        if current is not None:
            return current
        record = await coordinator.register(
            ClientRegistrationRequest(
                client_id=client_runtime.client_id,
                public_key=client_runtime.identity.public_key_b64,
                model_profile=client_runtime.model_profile,
            )
        )
        client_runtime.commit_registration(record)
        return record

    @app.post(
        "/v1/local-train",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def local_train(request: LocalTrainRequest) -> dict[str, Any]:
        try:
            return await run_exclusive_ml(
                client_runtime.local_train,
                request.examples,
            )
        except (ClientRuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def verified_manifest(round_id: str) -> RoundManifest:
        identity = await coordinator.identity()
        manifest = await coordinator.manifest(round_id)
        if not manifest.verify_signature(identity.public_key):
            raise HTTPException(
                status_code=401,
                detail="round manifest signature is invalid",
            )
        return manifest

    @app.post(
        "/v1/rounds/{round_id}/local-train",
        dependencies=[Depends(require_client_admin_token)],
    )
    async def local_train_round(round_id: str) -> dict[str, Any]:
        manifest = await verified_manifest(round_id)
        try:
            return await run_exclusive_ml(
                client_runtime.local_train_round,
                manifest,
            )
        except (ClientRuntimeError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def submit_participation(
        manifest: RoundManifest,
        *,
        require_round_training: bool,
    ) -> SubmissionReceipt:
        if (
            require_round_training
            and client_runtime.model_profile.training_backend != "transformers"
        ):
            client_runtime.require_round_training(manifest)

        reference_content = await coordinator.reference_dataset(
            manifest.round_id,
            client_runtime.client_id,
        )
        if reference_content is not None:
            client_runtime.cache_reference_dataset(
                manifest=manifest,
                content=reference_content,
            )

        if client_runtime.model_profile.training_backend == "transformers":
            package = await run_exclusive_ml(
                client_runtime.create_knowledge_package,
                manifest,
            )
        else:
            package = client_runtime.create_knowledge_package(manifest)
        try:
            receipt = await coordinator.submit(
                package,
                client_runtime.package_artifact_path(package),
            )
        except httpx.RequestError as exc:
            receipt = await coordinator.reconcile_submission(package)
            if receipt is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Coordinator submission acknowledgement was ambiguous "
                        "and the exact package was not confirmed accepted; "
                        "the pending Client package was retained"
                    ),
                ) from exc
        client_runtime.commit_knowledge_submission(
            manifest=manifest,
            package=package,
            receipt=receipt,
        )
        return receipt

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
            return await submit_participation(
                manifest,
                require_round_training=False,
            )
        except ClientRuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    @app.post(
        "/v1/rounds/{round_id}/participate",
        response_model=SubmissionReceipt,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_client_admin_token)],
    )
    async def participate_round(round_id: str) -> SubmissionReceipt:
        manifest = await verified_manifest(round_id)
        try:
            return await submit_participation(
                manifest,
                require_round_training=True,
            )
        except (ClientRuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

        incoming_artifact = client_runtime.incoming_host_artifact_path(
            round_id
        )

        try:
            package = await coordinator.host_knowledge(
                round_id,
                incoming_artifact,
                manifest.maximum_knowledge_package_bytes,
            )
            return await run_exclusive_ml(
                partial(
                    client_runtime.apply_host_knowledge,
                    manifest=manifest,
                    host_package=package,
                    host_artifact_path=incoming_artifact,
                    host_public_key=identity.host_public_key,
                    expected_host_id=identity.host_service_id,
                    accepted_host_adapter_version=(
                        status_record.host_adapter_after
                    ),
                    adapter_promoted=bool(status_record.adapter_promoted),
                )
            )

        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        finally:
            incoming_artifact.unlink(missing_ok=True)

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
