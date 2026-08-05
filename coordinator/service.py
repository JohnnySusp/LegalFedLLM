from __future__ import annotations

import asyncio
import hmac
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import httpx

from coordinator.reference_data import CoordinatorReferenceData

from shared.crypto import Ed25519Identity
from shared.fedmkt_core import dual_min_ce_select, inspect_knowledge_package
from shared.protocol import (
    ClientRegistrationRequest,
    DistillationJob,
    DistillationResult,
    KnowledgePackage,
    RegistrationRecord,
    RoundCreateRequest,
    RoundManifest,
    RoundState,
    SafetyReport,
    ServiceIdentity,
    SubmissionReceipt,
    parse_utc,
    utc_now,
    utc_text,
)
from shared.storage import JsonFileStore


class CoordinatorError(RuntimeError):
    status_code = 400


class NotFoundError(CoordinatorError):
    status_code = 404


class ConflictError(CoordinatorError):
    status_code = 409


class AuthenticationError(CoordinatorError):
    status_code = 401

class AuthorizationError(CoordinatorError):
    status_code = 403

class HostGateway:
    def __init__(
        self,
        base_url: str,
        internal_token: str,
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            headers={"X-Internal-Token": self.internal_token},
        ) as client:
            response = await client.request(method, path, json=json)
        if response.status_code >= 400:
            raise ConflictError(
                f"Host runtime returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    async def identity(self) -> ServiceIdentity:
        payload = await self._request("GET", "/internal/v1/identity")
        return ServiceIdentity.model_validate(payload)

    async def reference_knowledge(self, manifest: RoundManifest) -> KnowledgePackage:
        payload = await self._request(
            "POST",
            "/internal/v1/reference-knowledge",
            json=manifest.model_dump(mode="json"),
        )
        return KnowledgePackage.model_validate(payload)

    async def distill(self, job: DistillationJob) -> DistillationResult:
        payload = await self._request(
            "POST", "/internal/v1/distill", json=job.model_dump(mode="json")
        )
        return DistillationResult.model_validate(payload)

    async def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        )


class CoordinatorService:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        host_gateway: HostGateway,
        coordinator_id: str = "legalfedllm-coordinator",
        registration_token: str = "development-registration-token",
        admin_token: str = "development-admin-token",
        maximum_clock_skew_seconds: int = 900,
        reference_dataset_path: str | Path | None = None,
        validation_dataset_path: str | Path | None = None,
        now_fn: Callable[[], Any] = utc_now,
    ):
        self.coordinator_id = coordinator_id
        self.store = JsonFileStore(data_dir)
        self.identity = Ed25519Identity.load_or_create(
            self.store.path("identity/private_key.pem")
        )
        self.host = host_gateway
        self.registration_token = registration_token
        self.admin_token = admin_token
        self.maximum_clock_skew_seconds = maximum_clock_skew_seconds
        self.now_fn = now_fn

        if (
            reference_dataset_path is None
            and validation_dataset_path is not None
        ) or (
            reference_dataset_path is not None
            and validation_dataset_path is None
        ):
            raise ValueError(
                "reference and validation dataset paths "
                "must be configured together"
            )

        self.reference_data = (
            CoordinatorReferenceData.load(
                reference_dataset_path,
                validation_dataset_path,
            )
            if reference_dataset_path is not None
            and validation_dataset_path is not None
            else None
        )

        self._lock = asyncio.Lock()

    def require_registration_token(self, value: str | None) -> None:
        if value is None or not hmac.compare_digest(value, self.registration_token):
            raise AuthenticationError("invalid registration token")

    def require_admin_token(self, value: str | None) -> None:
        if value is None or not hmac.compare_digest(value, self.admin_token):
            raise AuthenticationError("invalid admin token")

    async def service_identity(self) -> ServiceIdentity:
        host_identity = await self._host_identity()
        return ServiceIdentity(
            service_id=self.coordinator_id,
            public_key=self.identity.public_key_b64,
            host_public_key=host_identity.public_key,
            host_service_id=host_identity.service_id,
        )

    async def _host_identity(self, refresh: bool = False) -> ServiceIdentity:
        path = "host/identity.json"
        if not refresh and self.store.exists(path):
            return ServiceIdentity.model_validate(self.store.read_json(path))
        identity = await self.host.identity()
        self.store.write_json(path, identity.model_dump(mode="json"))
        return identity

    def register_client(
        self, request: ClientRegistrationRequest
    ) -> RegistrationRecord:
        path = f"clients/{request.client_id}.json"
        if self.store.exists(path):
            current = RegistrationRecord.model_validate(self.store.read_json(path))
            if (
                current.public_key != request.public_key
                or current.model_profile != request.model_profile
            ):
                raise ConflictError("Client ID is already registered with another profile")
            return current
        record = RegistrationRecord(
            **request.model_dump(mode="json"), registered_at=utc_text(self.now_fn())
        )
        self.store.write_json(path, record.model_dump(mode="json"))
        self._audit("client_registered", {"client_id": request.client_id})
        return record

    def get_registration(self, client_id: str) -> RegistrationRecord:
        path = f"clients/{client_id}.json"
        if not self.store.exists(path):
            raise NotFoundError(f"Client {client_id!r} is not registered")
        return RegistrationRecord.model_validate(self.store.read_json(path))

    def _resolve_round_request(
        self,
        request: RoundCreateRequest,
    ) -> RoundCreateRequest:
        if self.reference_data is not None:
            identity = self.reference_data.reference_identity

            return request.model_copy(
                update={
                    "reference_dataset_id": identity.dataset_id,
                    "reference_dataset_hash": identity.dataset_hash,
                    "sample_ids": self.reference_data.sample_ids,
                }
            )

        if (
            request.reference_dataset_id is None
            or request.reference_dataset_hash is None
            or request.sample_ids is None
        ):
            raise ConflictError(
                "the Coordinator has no real reference dataset configured "
                "and the mock request contains no dataset metadata"
            )

        return request

    async def create_round(self, request: RoundCreateRequest) -> RoundManifest:
        async with self._lock:
            request = self._resolve_round_request(request)

            if request.alignment.strategy != "mock_identity":
                raise ConflictError(
                    "only alignment.strategy=mock_identity is available "
                    "in the protocol-first milestone"
                )

            for client_id in request.selected_client_ids:
                self.get_registration(client_id)
            if self.store.exists("rounds/current.json"):
                current = self.store.read_json("rounds/current.json")
                state = self.get_state(current["round_id"])
                if state.state in {"COLLECTING", "SEALED", "DISTILLING"}:
                    raise ConflictError("another round is still active")

            host_identity = await self._host_identity(refresh=True)
            if host_identity.model_profile is None or host_identity.adapter_version is None:
                raise ConflictError("Host identity is missing its model profile or adapter")
            counter = 0
            if self.store.exists("rounds/counter.json"):
                counter = int(self.store.read_json("rounds/counter.json")["value"])
            counter += 1
            self.store.write_json("rounds/counter.json", {"value": counter})
            round_id = f"round-{counter:06d}"
            deadline = self.now_fn() + timedelta(
                seconds=request.submission_window_seconds
            )
            manifest = RoundManifest.create_signed(
                identity=self.identity,
                round_id=round_id,
                coordinator_id=self.coordinator_id,
                current_host_adapter_version=host_identity.adapter_version,
                host_model_profile=host_identity.model_profile,
                request=request,
                submission_deadline=utc_text(deadline),
            )

            self._snapshot_reference_data(round_id)

            state = RoundState(
                round_id=round_id,
                state="COLLECTING",
                host_adapter_before=host_identity.adapter_version,
                updated_at=utc_text(self.now_fn()),
                message="waiting for Client Knowledge Packages",
            )
            self.store.write_json(
                f"rounds/{round_id}/manifest.json", manifest.model_dump(mode="json")
            )
            self._write_state(state)
            self.store.write_json("rounds/current.json", {"round_id": round_id})
            self._audit(
                "round_created",
                {
                    "round_id": round_id,
                    "clients": request.selected_client_ids,
                    "quorum": request.trusted_client_quorum,
                },
            )
            return manifest

    def get_manifest(self, round_id: str) -> RoundManifest:
        path = f"rounds/{round_id}/manifest.json"
        if not self.store.exists(path):
            raise NotFoundError(f"round {round_id!r} does not exist")
        return RoundManifest.model_validate(self.store.read_json(path))

    def get_state(self, round_id: str) -> RoundState:
        path = f"rounds/{round_id}/state.json"
        if not self.store.exists(path):
            raise NotFoundError(f"round {round_id!r} does not exist")
        return RoundState.model_validate(self.store.read_json(path))

    async def monitor_once(self) -> None:
        if not self.store.exists("rounds/current.json"):
            return
        round_id = self.store.read_json("rounds/current.json")["round_id"]
        await self.advance(round_id)

    async def current_manifest(self) -> RoundManifest:
        if not self.store.exists("rounds/current.json"):
            raise NotFoundError("no round has been created")
        round_id = self.store.read_json("rounds/current.json")["round_id"]
        await self.advance(round_id)
        return self.get_manifest(round_id)

    async def round_status(self, round_id: str) -> RoundState:
        await self.advance(round_id)
        return self.get_state(round_id)

    async def advance(self, round_id: str) -> None:
        async with self._lock:
            state = self.get_state(round_id)
            manifest = self.get_manifest(round_id)
            if state.state == "COLLECTING" and self.now_fn() > parse_utc(
                manifest.submission_deadline
            ):
                if len(state.accepted_client_ids) < manifest.trusted_client_quorum:
                    state.state = "SKIPPED"
                    state.message = "submission deadline passed without trusted quorum"
                    state.updated_at = utc_text(self.now_fn())
                    self._write_state(state)
                    self._audit(
                        "round_skipped",
                        {"round_id": round_id, "reason": state.message},
                    )
                    return
            if state.state in {"SEALED", "DISTILLING"}:
                await self._process_sealed_round(manifest, state)

    async def submit_knowledge(
        self, package: KnowledgePackage, raw_size: int
    ) -> SubmissionReceipt:
        async with self._lock:
            manifest = self.get_manifest(package.round_id)
            state = self.get_state(package.round_id)
            if state.state != "COLLECTING":
                raise ConflictError(f"round is not collecting packages: {state.state}")
            if self.now_fn() > parse_utc(manifest.submission_deadline):
                state.state = "SKIPPED"
                state.message = "package arrived after the submission deadline"
                state.updated_at = utc_text(self.now_fn())
                self._write_state(state)
                raise ConflictError(state.message)
            if raw_size > manifest.maximum_knowledge_package_bytes:
                raise ConflictError("Knowledge Package exceeds the manifest size limit")
            if package.sender_role != "client":
                raise ConflictError("only Client Knowledge Packages may be submitted")
            if package.sender_id not in manifest.selected_client_ids:
                raise ConflictError("Client is not selected for this round")
            if package.sender_id in state.accepted_client_ids:
                raise ConflictError("Client has already submitted for this round")
            if package.manifest_hash != manifest.manifest_hash:
                raise ConflictError("Knowledge Package has a stale manifest hash")
            if package.reference_dataset_id != manifest.reference_dataset_id:
                raise ConflictError("reference dataset ID does not match the manifest")
            if package.reference_dataset_hash != manifest.reference_dataset_hash:
                raise ConflictError("reference dataset hash does not match the manifest")
            if package.sample_ids != manifest.sample_ids:
                raise ConflictError("sample order does not match the manifest")
            if package.top_k != manifest.top_k:
                raise ConflictError("top-k does not match the manifest")
            expected_alignment = (
                f"{manifest.alignment.strategy}:{manifest.alignment.profile_version}"
            )
            if package.alignment_profile_id != expected_alignment:
                raise ConflictError("alignment profile does not match the manifest")

            registration = self.get_registration(package.sender_id)
            if package.model_profile != registration.model_profile:
                raise ConflictError("model profile differs from Client registration")
            if not package.verify_signature(registration.public_key):
                raise AuthenticationError("Client Knowledge Package signature is invalid")
            if package.artifact_sha256 in state.seen_package_hashes:
                raise ConflictError("replayed Knowledge Package hash")

            if package.nonce in state.seen_nonces:
                raise ConflictError("replayed Knowledge Package nonce")

            created = parse_utc(package.created_at)
            skew = abs((self.now_fn() - created).total_seconds())
            if skew > self.maximum_clock_skew_seconds:
                raise ConflictError("Knowledge Package timestamp is outside the allowed skew")
            state.seen_package_hashes.append(package.artifact_sha256)
            state.seen_nonces.append(package.nonce)
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            try:
                self._verify_dp(manifest, package)
            except CoordinatorError as exc:
                self._record_package_rejection(
                    state,
                    package,
                    [str(exc)],
                )
                raise
            safety = inspect_knowledge_package(package)
            self.store.write_json(
                f"rounds/{manifest.round_id}/safety/{package.sender_id}.json",
                safety.model_dump(mode="json"),
            )
            if not safety.accepted:
                self._record_package_rejection(
                    state,
                    package,
                    safety.reasons,
                )
                raise ConflictError(
                    "Knowledge Package failed the safety probe"
                )

            self.store.write_json(
                f"rounds/{manifest.round_id}/submissions/{package.sender_id}.json",
                package.model_dump(mode="json"),
            )
            state.accepted_client_ids.append(package.sender_id)
            state.used_nonces.append(package.nonce)
            state.submission_hashes.append(package.artifact_sha256)
            state.updated_at = utc_text(self.now_fn())
            state.message = "package accepted; waiting for trusted quorum"
            self._write_state(state)
            self._audit(
                "package_accepted",
                {
                    "round_id": manifest.round_id,
                    "client_id": package.sender_id,
                    "hash": package.artifact_sha256,
                },
            )

            if len(state.accepted_client_ids) >= manifest.trusted_client_quorum:
                state.state = "SEALED"
                state.sealed_client_ids = sorted(state.accepted_client_ids)
                state.message = "trusted quorum reached; submission set sealed"
                state.updated_at = utc_text(self.now_fn())
                self._write_state(state)
                await self._process_sealed_round(manifest, state)
                state = self.get_state(manifest.round_id)

            return SubmissionReceipt(
                round_id=manifest.round_id,
                client_id=package.sender_id,
                package_hash=package.artifact_sha256,
                state=state.state,
                accepted_count=len(state.accepted_client_ids),
                quorum=manifest.trusted_client_quorum,
            )

    def _record_package_rejection(
        self,
        state: RoundState,
        package: KnowledgePackage,
        reasons: list[str],
    ) -> None:
        if package.sender_id not in state.rejected_client_ids:
            state.rejected_client_ids.append(package.sender_id)

        state.updated_at = utc_text(self.now_fn())
        self._write_state(state)

        self._audit(
            "package_rejected",
            {
                "round_id": package.round_id,
                "client_id": package.sender_id,
                "hash": package.artifact_sha256,
                "nonce": package.nonce,
                "reasons": reasons,
            },
        )

    def _verify_dp(
        self,
        manifest: RoundManifest,
        package: KnowledgePackage,
    ) -> None:
        policy = manifest.dp_policy
        report = package.dp_report

        if not policy.required:
            return

        if not report.enabled:
            raise ConflictError(
                "the round requires differential privacy"
            )

        if report.mechanism != policy.mechanism:
            raise ConflictError(
                "Client DP mechanism does not match the manifest"
            )

        if report.epsilon_spent is None:
            raise ConflictError(
                "Client DP report is missing epsilon_spent"
            )

        if report.delta is None:
            raise ConflictError(
                "Client DP report is missing delta"
            )

        if policy.max_epsilon is None or policy.delta is None:
            raise ConflictError(
                "the signed DP policy is incomplete"
            )

        if report.epsilon_spent > policy.max_epsilon:
            raise ConflictError(
                "Client cumulative privacy budget exceeds the policy"
            )

        if abs(report.delta - policy.delta) > 1e-15:
            raise ConflictError(
                "Client DP delta does not match the manifest"
            )

    def _verify_host_package(
        self,
        *,
        manifest: RoundManifest,
        package: KnowledgePackage,
        host_identity: ServiceIdentity,
        expected_adapter_version: int,
    ) -> None:
        if host_identity.model_profile is None:
            raise ConflictError(
                "Host identity is missing its model profile"
            )

        if package.sender_role != "host":
            raise ConflictError(
                "Host package has an invalid sender role"
            )

        if package.sender_id != host_identity.service_id:
            raise ConflictError(
                "Host package sender ID differs from Host identity"
            )

        if package.model_profile != host_identity.model_profile:
            raise ConflictError(
                "Host package model profile differs from Host identity"
            )

        if package.model_profile != manifest.host_model_profile:
            raise ConflictError(
                "Host package model profile differs from the manifest"
            )

        if not package.verify_signature(host_identity.public_key):
            raise AuthenticationError(
                "Host Knowledge Package signature is invalid"
            )

        if package.round_id != manifest.round_id:
            raise ConflictError(
                "Host Knowledge Package belongs to another round"
            )

        if package.manifest_hash != manifest.manifest_hash:
            raise ConflictError(
                "Host Knowledge Package is bound to another manifest"
            )

        if (
            package.reference_dataset_id
            != manifest.reference_dataset_id
        ):
            raise ConflictError(
                "Host package reference dataset ID differs"
            )

        if (
            package.reference_dataset_hash
            != manifest.reference_dataset_hash
        ):
            raise ConflictError(
                "Host package reference dataset hash differs"
            )

        if package.sample_ids != manifest.sample_ids:
            raise ConflictError(
                "Host package sample order differs from the manifest"
            )

        if package.top_k != manifest.top_k:
            raise ConflictError(
                "Host package top-k differs from the manifest"
            )

        expected_alignment = (
            f"{manifest.alignment.strategy}:"
            f"{manifest.alignment.profile_version}"
        )

        if package.alignment_profile_id != expected_alignment:
            raise ConflictError(
                "Host package alignment profile differs from the manifest"
            )

        if package.adapter_version != expected_adapter_version:
            raise ConflictError(
                "Host package adapter version is unexpected"
            )

        created = parse_utc(package.created_at)
        skew = abs(
            (self.now_fn() - created).total_seconds()
        )

        if skew > self.maximum_clock_skew_seconds:
            raise ConflictError(
                "Host package timestamp is outside the allowed skew"
            )

    async def _process_sealed_round(
        self, manifest: RoundManifest, state: RoundState
    ) -> None:
        if state.state not in {"SEALED", "DISTILLING"}:
            return
        state.state = "DISTILLING"
        state.message = "constructing the validated Host distillation dataset"
        state.updated_at = utc_text(self.now_fn())
        self._write_state(state)
        try:
            host_identity = await self._host_identity(refresh=True)
            baseline = await self.host.reference_knowledge(manifest)

            self._verify_host_package(
                manifest=manifest,
                package=baseline,
                host_identity=host_identity,
                expected_adapter_version=(
                    manifest.current_host_adapter_version
                ),
            )

            packages = [
                KnowledgePackage.model_validate(
                    self.store.read_json(
                        f"rounds/{manifest.round_id}/submissions/{client_id}.json"
                    )
                )
                for client_id in state.sealed_client_ids
            ]
            reports = {
                client_id: SafetyReport.model_validate(
                    self.store.read_json(
                        f"rounds/{manifest.round_id}/safety/{client_id}.json"
                    )
                )
                for client_id in state.sealed_client_ids
            }
            dataset = dual_min_ce_select(
                host_package=baseline,
                client_packages=packages,
                safety_reports=reports,
            )
            self.store.write_json(
                f"rounds/{manifest.round_id}/validated_distillation_dataset.json",
                dataset.model_dump(mode="json"),
            )
            result = await self.host.distill(
                DistillationJob(
                    manifest=manifest,
                    dataset=dataset,
                )
            )

            if result.round_id != manifest.round_id:
                raise ConflictError(
                    "Host distillation result belongs to another round"
                )

            if (
                result.previous_adapter_version
                != manifest.current_host_adapter_version
            ):
                raise ConflictError(
                    "Host distillation started from an unexpected adapter"
                )

            if (
                result.candidate_adapter_version
                != result.previous_adapter_version + 1
            ):
                raise ConflictError(
                    "Host candidate adapter version is invalid"
                )

            if result.adapter_promoted:
                if (
                    result.accepted_adapter_version
                    != result.candidate_adapter_version
                ):
                    raise ConflictError(
                        "promoted Host candidate version is inconsistent"
                    )
            else:
                if (
                    result.accepted_adapter_version
                    != result.previous_adapter_version
                ):
                    raise ConflictError(
                        "Host rollback did not retain the previous adapter"
                    )

            host_package = result.host_knowledge_package

            self._verify_host_package(
                manifest=manifest,
                package=host_package,
                host_identity=host_identity,
                expected_adapter_version=result.accepted_adapter_version,
            )

            if (
                host_package.artifact_sha256
                not in state.host_package_hashes
            ):
                state.host_package_hashes.append(
                    host_package.artifact_sha256
                )

            if host_package.nonce not in state.host_nonces:
                state.host_nonces.append(host_package.nonce)
            self.store.write_json(
                f"rounds/{manifest.round_id}/distillation_result.json",
                result.model_dump(mode="json"),
            )
            self.store.write_json(
                f"rounds/{manifest.round_id}/host_knowledge.json",
                host_package.model_dump(mode="json"),
            )
            state.state = "COMPLETED"
            state.host_adapter_after = result.accepted_adapter_version
            state.adapter_promoted = result.adapter_promoted
            state.message = (
                "candidate Host adapter accepted"
                if result.adapter_promoted
                else "candidate Host adapter rejected; previous adapter retained"
            )
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            self._audit(
                "round_completed",
                {
                    "round_id": manifest.round_id,
                    "adapter_promoted": result.adapter_promoted,
                    "adapter_version": result.accepted_adapter_version,
                    "selected_samples": len(dataset.samples),
                },
            )
        except Exception as exc:
            state.state = "ABORTED"
            state.message = str(exc)
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            self._audit(
                "round_aborted", {"round_id": manifest.round_id, "reason": str(exc)}
            )
            raise

    def get_host_knowledge(self, round_id: str) -> KnowledgePackage:
        state = self.get_state(round_id)
        if state.state != "COMPLETED":
            raise ConflictError("Host Knowledge Package is not available yet")
        path = f"rounds/{round_id}/host_knowledge.json"
        if not self.store.exists(path):
            raise NotFoundError("Host Knowledge Package was not published")
        return KnowledgePackage.model_validate(self.store.read_json(path))

    def _write_state(self, state: RoundState) -> None:
        self.store.write_json(
            f"rounds/{state.round_id}/state.json", state.model_dump(mode="json")
        )

    def _audit(self, event: str, details: dict[str, Any]) -> None:
        self.store.append_jsonl(
            "audit/events.jsonl",
            {"timestamp": utc_text(self.now_fn()), "event": event, **details},
        )

    def _snapshot_reference_data(
        self,
        round_id: str,
    ) -> None:
        if self.reference_data is None:
            return

        reference_path = self.store.path(
            f"rounds/{round_id}/datasets/reference.jsonl"
        )
        validation_path = self.store.path(
            f"rounds/{round_id}/datasets/validation.jsonl"
        )

        self.reference_data.write_snapshot(
            reference_path,
            validation_path,
        )

        self.store.write_json(
            f"rounds/{round_id}/datasets/identity.json",
            {
                "reference": (
                    self.reference_data.reference_identity.model_dump(
                        mode="json"
                    )
                ),
                "validation": (
                    self.reference_data.validation_identity.model_dump(
                        mode="json"
                    )
                ),
            },
        )

    def get_reference_dataset_path(
        self,
        round_id: str,
        client_id: str | None,
    ) -> Path:
        if client_id is None:
            raise AuthenticationError("missing Client ID")

        manifest = self.get_manifest(round_id)
        self.get_registration(client_id)

        if client_id not in manifest.selected_client_ids:
            raise AuthorizationError(
                "Client is not selected for this round"
            )

        relative_path = (
            f"rounds/{round_id}/datasets/reference.jsonl"
        )

        if not self.store.exists(relative_path):
            raise NotFoundError(
                "this round does not publish a real reference dataset"
            )

        return self.store.path(relative_path)
