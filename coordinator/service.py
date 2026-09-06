from __future__ import annotations

import asyncio
import hmac
import os
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import httpx

from coordinator.reference_data import CoordinatorReferenceData
from coordinator.quorum import TrustedClientQuorumPolicy

from shared.alignment_profiles import (
    UnsupportedAlignmentProfile,
    resolve_alignment_profile,
    resolve_alignment_profile_id_for_pair,
    validate_alignment_pair,
)
from shared.crypto import Ed25519Identity, canonical_json_bytes
from shared.distillation_artifact import (
    load_host_training_artifact,
    write_host_training_artifact,
)
from shared.fedmkt_core import (
    dual_min_ce_select,
    finalize_aligned_safety_reports,
    inspect_knowledge_package,
    is_eligible_for_distillation,
    new_client_trust_history,
    update_client_trust_history,
)
from shared.knowledge_artifact import load_package_samples
from shared.knowledge_transport import receive_knowledge_transfer
from shared.protocol import (
    ClientRegistrationRequest,
    ClientTrustHistory,
    DistillationJob,
    DistillationResult,
    KnowledgePackage,
    KnowledgeSample,
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
    HostReferenceDatasetBundle,
    HostReferenceDatasetReceipt,
    HostCandidateTrainingResult,
    HostCandidateValidationResult,
    HostTrainingJob,
    HostTrainingJobReceipt,
)
from shared.storage import JsonFileStore
from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    verify_reference_dataset,
)
from shared.reference_knowledge import encode_reference_samples
from shared.tokenizer_validation import load_pinned_tokenizer
from shared.vocabulary_mapping import VocabularyMappingCache


class CoordinatorError(RuntimeError):
    status_code = 400


class NotFoundError(CoordinatorError):
    status_code = 404


class ConflictError(CoordinatorError):
    status_code = 409


class TrustedQuorumError(ConflictError):
    pass


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

    async def load_reference_data(
        self,
        bundle: HostReferenceDatasetBundle,
    ) -> HostReferenceDatasetReceipt:
        payload = await self._request(
            "POST",
            "/internal/v1/reference-data",
            json=bundle.model_dump(mode="json"),
        )

        return HostReferenceDatasetReceipt.model_validate(
            payload
        )

    async def _knowledge_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any],
        artifact_path: str | Path,
        metadata_part_name: str,
        maximum_bytes: int,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            headers={"X-Internal-Token": self.internal_token},
        ) as client:
            async with client.stream(
                method,
                path,
                json=json,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ConflictError(
                        f"Host runtime returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                received = await receive_knowledge_transfer(
                    content_type=response.headers.get("content-type", ""),
                    chunks=response.aiter_bytes(),
                    artifact_path=artifact_path,
                    metadata_part_name=metadata_part_name,
                    maximum_content_bytes=maximum_bytes,
                )
        return received.metadata

    async def reference_knowledge(
        self,
        manifest: RoundManifest,
        artifact_path: str | Path,
    ) -> KnowledgePackage:
        payload = await self._knowledge_request(
            "POST",
            "/internal/v1/reference-knowledge",
            json=manifest.model_dump(mode="json"),
            artifact_path=artifact_path,
            metadata_part_name="package",
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        return KnowledgePackage.model_validate(payload)

    async def distill(
        self,
        job: DistillationJob,
        artifact_path: str | Path,
    ) -> DistillationResult:
        payload = await self._knowledge_request(
            "POST",
            "/internal/v1/distill",
            json=job.model_dump(mode="json"),
            artifact_path=artifact_path,
            metadata_part_name="result",
            maximum_bytes=job.manifest.maximum_knowledge_package_bytes,
        )
        return DistillationResult.model_validate(payload)

    async def load_training_job(
        self,
        job: HostTrainingJob,
        artifact_path: str | Path,
    ) -> HostTrainingJobReceipt:
        with Path(artifact_path).open("rb") as artifact:
            files = [
                (
                    "job",
                    (
                        "job.json",
                        canonical_json_bytes(job.model_dump(mode="json")),
                        "application/json",
                    ),
                ),
                (
                    "artifact",
                    (
                        "trainer_inputs.safetensors",
                        artifact,
                        "application/octet-stream",
                    ),
                ),
            ]
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
                headers={"X-Internal-Token": self.internal_token},
            ) as client:
                response = await client.post(
                    "/internal/v1/training-job",
                    files=files,
                )
        if response.status_code >= 400:
            raise ConflictError(
                f"Host runtime returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        return HostTrainingJobReceipt.model_validate(response.json())

    async def train_candidate(
        self,
        job: HostTrainingJob,
    ) -> HostCandidateTrainingResult:
        payload = await self._request(
            "POST",
            "/internal/v1/train-candidate",
            json=job.model_dump(mode="json"),
        )
        return HostCandidateTrainingResult.model_validate(payload)

    async def validate_candidate(
        self,
        job: HostTrainingJob,
    ) -> HostCandidateValidationResult:
        payload = await self._request(
            "POST",
            "/internal/v1/validate-candidate",
            json=job.model_dump(mode="json"),
        )
        return HostCandidateValidationResult.model_validate(payload)

    async def post_decision_knowledge(
        self,
        manifest: RoundManifest,
        artifact_path: str | Path,
    ) -> KnowledgePackage:
        payload = await self._knowledge_request(
            "POST",
            "/internal/v1/post-decision-knowledge",
            json=manifest.model_dump(mode="json"),
            artifact_path=artifact_path,
            metadata_part_name="package",
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        return KnowledgePackage.model_validate(payload)

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
        quorum_policy: TrustedClientQuorumPolicy | None = None,
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
        self.quorum_policy = quorum_policy
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

    def incoming_artifact_path(self, round_id: str, purpose: str) -> Path:
        return self.store.path(
            f"rounds/{round_id}/incoming/{purpose}."
            f"{secrets.token_hex(8)}.safetensors"
        )

    def _persist_knowledge_package(
        self,
        *,
        package_path: str,
        artifact_path: str,
        package: KnowledgePackage,
        source_artifact: str | Path,
        maximum_bytes: int,
    ) -> None:
        existing = (
            self.store.exists(package_path),
            self.store.exists(artifact_path),
        )
        if any(existing) and not all(existing):
            raise ConflictError("persisted Knowledge Package is incomplete")
        if all(existing):
            stored = KnowledgePackage.model_validate(
                self.store.read_json(package_path)
            )
            if stored.package_hash != package.package_hash:
                raise ConflictError("persisted Knowledge Package is immutable")
            load_package_samples(
                self.store.path(artifact_path),
                stored,
                maximum_bytes=maximum_bytes,
            )
            return
        try:
            self.store.copy_file_if_absent(
                artifact_path,
                source_artifact,
            )
            self.store.write_json_if_absent(
                package_path,
                package.model_dump(mode="json"),
            )
        except Exception:
            if not self.store.exists(package_path):
                self.store.delete(artifact_path)
            raise

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

            if self.quorum_policy is not None:
                try:
                    resolved_quorum = self.quorum_policy.resolve(
                        len(request.selected_client_ids)
                    )
                except ValueError as exc:
                    raise ConflictError(str(exc)) from exc
                request = request.model_copy(
                    update={"trusted_client_quorum": resolved_quorum}
                )

            registrations = {
                client_id: self.get_registration(client_id)
                for client_id in request.selected_client_ids
            }
            if self.store.exists("rounds/current.json"):
                current = self.store.read_json("rounds/current.json")
                state = self.get_state(current["round_id"])
                if state.state in {"COLLECTING", "SEALED", "DISTILLING"}:
                    raise ConflictError("another round is still active")

            host_identity = await self._host_identity(refresh=True)
            if host_identity.model_profile is None or host_identity.adapter_version is None:
                raise ConflictError("Host identity is missing its model profile or adapter")

            try:
                selected_client_alignment_profiles = {
                    client_id: resolve_alignment_profile_id_for_pair(
                        client_profile=registration.model_profile,
                        host_profile=host_identity.model_profile,
                    )
                    for client_id, registration in registrations.items()
                }
            except UnsupportedAlignmentProfile as exc:
                raise ConflictError(str(exc)) from exc

            alignment_strategies = {
                profile_id.split(":", 1)[0]
                for profile_id in selected_client_alignment_profiles.values()
            }
            if len(alignment_strategies) != 1:
                raise ConflictError(
                    "selected Client alignment profiles use incompatible strategies"
                )
            alignment_strategy = next(iter(alignment_strategies))

            if alignment_strategy == "dtw":
                try:
                    for client_id, registration in registrations.items():
                        validate_alignment_pair(
                            selected_client_alignment_profiles[client_id],
                            client_profile=registration.model_profile,
                            host_profile=host_identity.model_profile,
                        )
                except UnsupportedAlignmentProfile as exc:
                    raise ConflictError(str(exc)) from exc
                if host_identity.model_profile.training_backend != "transformers":
                    raise ConflictError(
                        "real DTW rounds require the Transformers Host backend"
                    )
                fixed_distillation = request.distillation
                if (
                    fixed_distillation.loss_type != "ce"
                    or fixed_distillation.lm_loss_weight != 0.9
                    or fixed_distillation.temperature != 1.0
                    or fixed_distillation.minimum_validation_improvement != 0.001
                ):
                    raise ConflictError(
                        "real DTW rounds require CE, temperature 1.0, "
                        "language-model loss weight 0.9 and validation "
                        "improvement 0.001"
                    )
                if request.truncation_policy != "reject":
                    raise ConflictError(
                        "real DTW rounds require reject-without-truncation"
                    )

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
                selected_client_profile_hashes={
                    client_id: registration.model_profile.profile_hash()
                    for client_id, registration in registrations.items()
                },
                selected_client_alignment_profiles=(
                    selected_client_alignment_profiles
                ),
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
                    "selected_client_alignment_profiles": (
                        selected_client_alignment_profiles
                    ),
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

    def get_submission_receipt(
        self,
        round_id: str,
        client_id: str,
    ) -> SubmissionReceipt:
        manifest = self.get_manifest(round_id)
        state = self.get_state(round_id)
        if client_id not in manifest.selected_client_ids:
            raise NotFoundError("accepted Client submission does not exist")

        package_path = (
            f"rounds/{round_id}/submissions/{client_id}/package.json"
        )
        if not self.store.exists(package_path):
            raise NotFoundError("accepted Client submission does not exist")

        package = KnowledgePackage.model_validate(
            self.store.read_json(package_path)
        )
        if (
            package.round_id != round_id
            or package.sender_id != client_id
            or client_id not in state.accepted_client_ids
            or package.package_hash not in state.submission_hashes
        ):
            raise ConflictError(
                "persisted Client submission is inconsistent with round state"
            )

        return SubmissionReceipt(
            round_id=round_id,
            client_id=client_id,
            package_hash=package.package_hash,
            state=state.state,
            accepted_count=len(state.accepted_client_ids),
            quorum=manifest.trusted_client_quorum,
        )

    def get_safety_reports(self, round_id: str) -> dict[str, SafetyReport]:
        manifest = self.get_manifest(round_id)
        reports: dict[str, SafetyReport] = {}
        for client_id in manifest.selected_client_ids:
            path = f"rounds/{round_id}/safety/{client_id}.json"
            if self.store.exists(path):
                reports[client_id] = SafetyReport.model_validate(
                    self.store.read_json(path)
                )
        return reports

    def _load_client_trust_history(self, client_id: str) -> ClientTrustHistory:
        path = f"trust_history/{client_id}.json"
        if not self.store.exists(path):
            return new_client_trust_history(client_id)
        history = ClientTrustHistory.model_validate(self.store.read_json(path))
        if history.client_id != client_id:
            raise ConflictError("persisted Client trust history has another identity")
        return history

    def _historical_reliability(
        self, client_ids: list[str]
    ) -> dict[str, float]:
        return {
            client_id: self._load_client_trust_history(client_id).reliability
            for client_id in client_ids
        }

    def _persist_final_safety_reports(
        self, round_id: str, reports: dict[str, SafetyReport]
    ) -> None:
        for client_id, report in reports.items():
            if report.probe_stage != "post_alignment":
                continue
            report_path = f"rounds/{round_id}/safety/{client_id}.json"
            prior_stage = None
            if self.store.exists(report_path):
                prior_stage = SafetyReport.model_validate(
                    self.store.read_json(report_path)
                ).probe_stage
            self.store.write_json(report_path, report.model_dump(mode="json"))
            history = self._load_client_trust_history(client_id)
            updated = update_client_trust_history(
                history,
                round_id=round_id,
                sample_risks=report.sample_risks,
            )
            self.store.write_json(
                f"trust_history/{client_id}.json",
                updated.model_dump(mode="json"),
            )
            if prior_stage != "post_alignment":
                self._audit(
                    "package_trust_finalized",
                    {
                        "round_id": round_id,
                        "client_id": client_id,
                        "trust_score": report.trust_score,
                        "accepted": report.accepted,
                        "score_components": report.score_components,
                    },
                )

    async def monitor_once(self) -> None:
        if not self.store.exists("rounds/current.json"):
            return
        round_id = self.store.read_json("rounds/current.json")["round_id"]
        await self.advance(round_id, process_dtw=True)

    async def current_manifest(self) -> RoundManifest:
        if not self.store.exists("rounds/current.json"):
            raise NotFoundError("no round has been created")
        round_id = self.store.read_json("rounds/current.json")["round_id"]
        await self.advance(round_id)
        return self.get_manifest(round_id)

    async def round_status(self, round_id: str) -> RoundState:
        await self.advance(round_id)
        return self.get_state(round_id)

    async def advance(self, round_id: str, *, process_dtw: bool = False) -> None:
        deferred_dtw: tuple[RoundManifest, RoundState] | None = None
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
                if manifest.alignment_strategy == "dtw":
                    if process_dtw:
                        deferred_dtw = manifest, state
                else:
                    await self._process_sealed_round(manifest, state)
        if deferred_dtw is not None:
            await self._process_sealed_round(*deferred_dtw)

    async def submit_knowledge(
        self,
        package: KnowledgePackage,
        artifact_path: str | Path,
        content_size: int,
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
            if content_size > manifest.maximum_knowledge_package_bytes:
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
            expected_alignment = manifest.alignment_profile_id_for(
                package.sender_id
            )
            if package.alignment_profile_id != expected_alignment:
                raise ConflictError("alignment profile does not match the manifest")

            registration = self.get_registration(package.sender_id)
            expected_profile_hash = manifest.selected_client_profile_hashes[
                package.sender_id
            ]
            if package.model_profile.profile_hash() != expected_profile_hash:
                raise ConflictError(
                    "model profile differs from the signed round manifest"
                )
            if package.model_profile != registration.model_profile:
                raise ConflictError("model profile differs from Client registration")
            if not package.verify_signature(registration.public_key):
                raise AuthenticationError("Client Knowledge Package signature is invalid")
            if package.package_hash in state.seen_package_hashes:
                raise ConflictError("replayed Knowledge Package hash")

            if package.nonce in state.seen_nonces:
                raise ConflictError("replayed Knowledge Package nonce")

            created = parse_utc(package.created_at)
            skew = abs((self.now_fn() - created).total_seconds())
            if skew > self.maximum_clock_skew_seconds:
                raise ConflictError("Knowledge Package timestamp is outside the allowed skew")
            state.seen_package_hashes.append(package.package_hash)
            state.seen_nonces.append(package.nonce)
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            try:
                self._verify_dp(manifest, package)
                safety = await asyncio.to_thread(
                    self._inspect_submitted_knowledge_package,
                    manifest,
                    package,
                    artifact_path,
                )
            except CoordinatorError as exc:
                self._record_package_rejection(
                    state,
                    package,
                    [str(exc)],
                )
                raise
            except (OSError, ValueError) as exc:
                self._record_package_rejection(
                    state,
                    package,
                    [str(exc)],
                )
                raise ConflictError(str(exc)) from exc
            self.store.write_json(
                f"rounds/{manifest.round_id}/safety/{package.sender_id}.json",
                safety.model_dump(mode="json"),
            )
            if not safety.accepted:
                self._record_package_rejection(
                    state,
                    package,
                    safety.reasons or ["Knowledge Package failed hard safety checks"],
                )
                raise ConflictError("Knowledge Package failed the safety probe")

            self._persist_knowledge_package(
                package_path=(
                    f"rounds/{manifest.round_id}/submissions/"
                    f"{package.sender_id}/package.json"
                ),
                artifact_path=(
                    f"rounds/{manifest.round_id}/submissions/"
                    f"{package.sender_id}/knowledge.safetensors"
                ),
                package=package,
                source_artifact=artifact_path,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            state.accepted_client_ids.append(package.sender_id)
            state.used_nonces.append(package.nonce)
            state.submission_hashes.append(package.package_hash)
            state.updated_at = utc_text(self.now_fn())
            state.message = "package accepted; waiting for trusted quorum"
            self._write_state(state)
            self._audit(
                "package_accepted",
                {
                    "round_id": manifest.round_id,
                    "client_id": package.sender_id,
                    "package_hash": package.package_hash,
                    "artifact_sha256": package.artifact.sha256,
                },
            )

            if len(state.accepted_client_ids) >= manifest.trusted_client_quorum:
                state.state = "SEALED"
                accepted = set(state.accepted_client_ids)
                state.sealed_client_ids = [
                    client_id
                    for client_id in manifest.selected_client_ids
                    if client_id in accepted
                ]
                state.message = "trusted quorum reached; submission set sealed"
                state.updated_at = utc_text(self.now_fn())
                self._write_state(state)
                if manifest.alignment_strategy != "dtw":
                    await self._process_sealed_round(manifest, state)
                state = self.get_state(manifest.round_id)

            return SubmissionReceipt(
                round_id=manifest.round_id,
                client_id=package.sender_id,
                package_hash=package.package_hash,
                state=state.state,
                accepted_count=len(state.accepted_client_ids),
                quorum=manifest.trusted_client_quorum,
            )

    @staticmethod
    def _inspect_submitted_knowledge_package(
        manifest: RoundManifest,
        package: KnowledgePackage,
        artifact_path: str | Path,
    ) -> SafetyReport:
        samples = load_package_samples(
            artifact_path,
            package,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        return inspect_knowledge_package(package, samples)

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
                "package_hash": package.package_hash,
                "artifact_sha256": package.artifact.sha256,
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
        artifact_path: str | Path,
        host_identity: ServiceIdentity,
        expected_adapter_version: int,
    ) -> list[KnowledgeSample]:
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

        expected_alignment = manifest.host_package_alignment_profile_id

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

        try:
            return load_package_samples(
                artifact_path,
                package,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
        except (OSError, ValueError) as exc:
            raise ConflictError(
                f"Host Knowledge Package artifact is invalid: {exc}"
            ) from exc

    def _load_host_training_job(
        self,
        manifest: RoundManifest,
    ) -> HostTrainingJob | None:
        from shared.fedmkt_core.integration import DistillationIntegrationAudit

        root = f"rounds/{manifest.round_id}/host_training_job"
        job_path = f"{root}/job.json"
        audit_path = f"{root}/integration_audit.json"
        artifact_path = f"{root}/trainer_inputs.safetensors"
        existing = (
            self.store.exists(job_path),
            self.store.exists(audit_path),
            self.store.exists(artifact_path),
        )
        if not any(existing):
            return None
        if not all(existing):
            raise ConflictError("persisted Host training job is incomplete")
        try:
            job = HostTrainingJob.model_validate(self.store.read_json(job_path))
            audit = DistillationIntegrationAudit.model_validate(
                self.store.read_json(audit_path)
            )
            if job.manifest.manifest_hash != manifest.manifest_hash:
                raise ValueError("Host training job belongs to another manifest")
            if job.integration_audit_hash != audit.audit_hash:
                raise ValueError("Host training job has another integration audit")
            if job.dataset_hash != audit.dataset_hash:
                raise ValueError("Host training job has another dataset hash")
            if (
                job.artifact.trainer_inputs_sha256
                != audit.trainer_inputs_sha256
            ):
                raise ValueError("Host training job has another tensor hash")
            load_host_training_artifact(
                self.store.path(artifact_path),
                job.artifact,
                job.sample_ids,
                maximum_bytes=manifest.maximum_host_training_job_bytes,
                vocabulary_size=int(
                    manifest.host_model_profile.vocabulary_size or 0
                ),
            )
            self._persist_final_safety_reports(
                manifest.round_id,
                audit.safety_reports,
            )
            return job
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise ConflictError(
                f"persisted Host training job is invalid: {exc}"
            ) from exc

    def _prepare_host_training_job(
        self,
        *,
        manifest: RoundManifest,
        baseline: KnowledgePackage,
        baseline_samples: list[KnowledgeSample],
        packages: list[KnowledgePackage],
        client_samples: dict[str, list[KnowledgeSample]],
        reports: dict[str, SafetyReport],
    ) -> HostTrainingJob:
        from shared.fedmkt_core.integration import (
            TrustedClientQuorumError,
            integrate_distillation_round,
        )

        root = f"rounds/{manifest.round_id}/host_training_job"
        job_path = f"{root}/job.json"
        audit_path = f"{root}/integration_audit.json"
        artifact_path = f"{root}/trainer_inputs.safetensors"
        cached = self._load_host_training_job(manifest)
        if cached is not None:
            return cached

        package_client_ids = {package.sender_id for package in packages}
        ordered_profile_ids = list(
            dict.fromkeys(
                manifest.alignment_profile_id_for(client_id)
                for client_id in manifest.selected_client_ids
                if client_id in package_client_ids
            )
        )
        alignment_profiles = {
            profile_id: resolve_alignment_profile(profile_id)
            for profile_id in ordered_profile_ids
        }
        host_anchor = resolve_alignment_profile(
            manifest.host_package_alignment_profile_id
        )
        host_endpoints = {
            profile.host for profile in alignment_profiles.values()
        }
        if len(host_endpoints) != 1:
            raise ConflictError(
                "selected Client alignment profiles do not share one Host endpoint"
            )

        cache_dir = os.getenv("HF_HOME") or None
        token = os.getenv("HF_TOKEN") or None
        host_tokenizer = load_pinned_tokenizer(
            host_anchor.host,
            cache_dir=cache_dir,
            token=token,
        )
        client_tokenizers = {
            profile_id: load_pinned_tokenizer(
                profile.client,
                cache_dir=cache_dir,
                token=token,
            )
            for profile_id, profile in alignment_profiles.items()
        }
        bundle = self._host_reference_dataset_bundle(manifest)
        encoded = encode_reference_samples(
            bundle.reference_samples,
            tokenizer=host_tokenizer.tokenizer,
            model_profile=manifest.host_model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=manifest.sample_ids,
        )
        try:
            batch = integrate_distillation_round(
                alignment_profiles=alignment_profiles,
                client_tokenizers=client_tokenizers,
                host_tokenizer=host_tokenizer,
                mapping_cache=VocabularyMappingCache(
                    self.store.path("vocabulary_mappings")
                ),
                host_package=baseline,
                host_samples=baseline_samples,
                client_packages=packages,
                client_samples=client_samples,
                safety_reports=reports,
                selected_client_ids=manifest.selected_client_ids,
                trusted_client_quorum=manifest.trusted_client_quorum,
                labels_by_sample={item.sample_id: item.labels for item in encoded},
                temperature=manifest.distillation.temperature,
                loss_type=manifest.distillation.loss_type,
                historical_reliability=self._historical_reliability(
                    list(manifest.selected_client_ids)
                ),
            )
        except TrustedClientQuorumError as exc:
            raise TrustedQuorumError(str(exc)) from exc
        pad_token_id = host_anchor.host.pad_token_id
        if pad_token_id is None:
            raise ConflictError("approved Host tokenizer has no padding token")
        target = self.store.path(artifact_path)
        try:
            descriptor = write_host_training_artifact(
                target,
                batch,
                pad_token_id=pad_token_id,
                maximum_bytes=manifest.maximum_host_training_job_bytes,
            )
            job = HostTrainingJob.create(
                manifest=manifest,
                dataset_hash=batch.dataset.dataset_hash,
                integration_audit_hash=batch.audit.audit_hash,
                accepted_client_ids=batch.dataset.accepted_client_ids,
                sample_ids=list(batch.sample_ids),
                host_adapter_version=batch.dataset.host_adapter_version,
                host_model_profile_hash=manifest.host_model_profile.profile_hash(),
                host_public_data_epochs=manifest.host_public_data_epochs,
                distillation=manifest.distillation,
                artifact=descriptor,
                created_at=utc_text(self.now_fn()),
            )
            self.store.write_json_if_absent(
                audit_path,
                batch.audit.model_dump(mode="json"),
            )
            self.store.write_json_if_absent(
                job_path,
                job.model_dump(mode="json"),
            )
            self._persist_final_safety_reports(
                manifest.round_id,
                batch.audit.safety_reports,
            )
            return job
        except Exception:
            if not self.store.exists(job_path):
                self.store.delete(audit_path)
                self.store.delete(artifact_path)
            raise

    async def _deliver_host_training_job(
        self,
        manifest: RoundManifest,
        job: HostTrainingJob,
    ) -> HostTrainingJobReceipt:
        receipt_path = (
            f"rounds/{manifest.round_id}/host_training_job/receipt.json"
        )
        if self.store.exists(receipt_path):
            receipt = HostTrainingJobReceipt.model_validate(
                self.store.read_json(receipt_path)
            )
        else:
            receipt = await self.host.load_training_job(
                job,
                self.store.path(
                    f"rounds/{manifest.round_id}/host_training_job/"
                    "trainer_inputs.safetensors"
                ),
            )
            self.store.write_json_if_absent(
                receipt_path,
                receipt.model_dump(mode="json"),
            )
        expected = HostTrainingJobReceipt(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            job_hash=job.job_hash,
            artifact_sha256=job.artifact.sha256,
            artifact_byte_size=job.artifact.byte_size,
            accepted_client_ids=job.accepted_client_ids,
        )
        if receipt != expected:
            raise ConflictError("Host acknowledged another training job")
        return receipt

    async def _train_host_candidate(
        self,
        manifest: RoundManifest,
        job: HostTrainingJob,
    ) -> HostCandidateTrainingResult:
        result_path = (
            f"rounds/{manifest.round_id}/host_training_job/"
            "candidate_result.json"
        )
        persisted = None
        if self.store.exists(result_path):
            persisted = HostCandidateTrainingResult.model_validate(
                self.store.read_json(result_path)
            )
        result = await self.host.train_candidate(job)
        if persisted is not None and persisted != result:
            raise ConflictError("Host candidate result changed after persistence")
        expected = {
            "round_id": manifest.round_id,
            "manifest_hash": manifest.manifest_hash,
            "job_hash": job.job_hash,
            "parent_adapter_version": manifest.current_host_adapter_version,
            "candidate_adapter_version": manifest.current_host_adapter_version + 1,
            "host_model_profile_hash": manifest.host_model_profile.profile_hash(),
            "host_public_data_epochs": manifest.host_public_data_epochs,
            "loss_type": "ce",
            "temperature": 1.0,
            "supervised_loss_weight": 0.9,
            "distillation_loss_weight": 0.1,
        }
        payload = result.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ConflictError(
                "Host candidate result differs from the signed round: "
                + ", ".join(mismatches)
            )
        if not self.store.exists(result_path):
            self.store.write_json_if_absent(
                result_path,
                result.model_dump(mode="json"),
            )
            self._audit(
                "host_candidate_trained",
                {
                    "round_id": manifest.round_id,
                    "job_hash": job.job_hash,
                    "candidate_adapter_version": (
                        result.candidate_adapter_version
                    ),
                    "candidate_adapter_hash": result.candidate_adapter_hash,
                    "optimizer_step_count": result.optimizer_step_count,
                    "optimizer_loss": result.optimizer_loss,
                },
            )
        return result

    async def _validate_host_candidate(
        self,
        manifest: RoundManifest,
        job: HostTrainingJob,
        candidate: HostCandidateTrainingResult,
    ) -> HostCandidateValidationResult:
        path = (
            f"rounds/{manifest.round_id}/host_training_job/"
            "validation_decision.json"
        )
        persisted = None
        if self.store.exists(path):
            persisted = HostCandidateValidationResult.model_validate(
                self.store.read_json(path)
            )
        decision = await self.host.validate_candidate(job)
        if persisted is not None and persisted != decision:
            raise ConflictError("Host validation decision changed after persistence")
        expected = {
            "round_id": manifest.round_id,
            "manifest_hash": manifest.manifest_hash,
            "job_hash": job.job_hash,
            "candidate_result_hash": candidate.result_hash,
            "previous_adapter_version": candidate.parent_adapter_version,
            "previous_adapter_hash": candidate.parent_adapter_hash,
            "candidate_adapter_version": candidate.candidate_adapter_version,
            "candidate_adapter_hash": candidate.candidate_adapter_hash,
            "required_improvement": (
                manifest.distillation.minimum_validation_improvement
            ),
        }
        payload = decision.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ConflictError(
                "Host validation decision differs from the signed round: "
                + ", ".join(mismatches)
            )
        if not self.store.exists(path):
            self.store.write_json_if_absent(
                path,
                decision.model_dump(mode="json"),
            )
            self._audit(
                "host_candidate_validated",
                {
                    "round_id": manifest.round_id,
                    "decision_hash": decision.decision_hash,
                    "baseline_validation_record_hash": (
                        decision.baseline_validation_record_hash
                    ),
                    "candidate_validation_record_hash": (
                        decision.candidate_validation_record_hash
                    ),
                    "previous_macro_mean_answer_token_ce": (
                        decision.previous_macro_mean_answer_token_ce
                    ),
                    "candidate_macro_mean_answer_token_ce": (
                        decision.candidate_macro_mean_answer_token_ce
                    ),
                    "previous_token_weighted_answer_token_ce": (
                        decision.previous_token_weighted_answer_token_ce
                    ),
                    "candidate_token_weighted_answer_token_ce": (
                        decision.candidate_token_weighted_answer_token_ce
                    ),
                    "required_improvement": decision.required_improvement,
                    "observed_improvement": decision.observed_improvement,
                    "adapter_promoted": decision.adapter_promoted,
                    "decision_reason": decision.decision_reason,
                    "rejected_candidate_discarded": (
                        decision.rejected_candidate_discarded
                    ),
                },
            )
        return decision

    async def _complete_real_round(
        self,
        manifest: RoundManifest,
        state: RoundState,
        job: HostTrainingJob,
        candidate: HostCandidateTrainingResult,
    ) -> None:
        state.state = "DISTILLING"
        state.message = "Host candidate private validation is running"
        state.updated_at = utc_text(self.now_fn())
        self._write_state(state)
        decision = await self._validate_host_candidate(
            manifest,
            job,
            candidate,
        )
        state.message = "post-decision Host reference inference is running"
        state.updated_at = utc_text(self.now_fn())
        self._write_state(state)
        incoming = self.incoming_artifact_path(
            manifest.round_id,
            "host-result",
        )
        try:
            previous_host_identity = await self._host_identity()
            package = await self.host.post_decision_knowledge(
                manifest,
                incoming,
            )
            host_identity = await self._host_identity(refresh=True)
            if (
                host_identity.service_id != previous_host_identity.service_id
                or host_identity.public_key != previous_host_identity.public_key
                or host_identity.model_profile
                != previous_host_identity.model_profile
            ):
                raise ConflictError(
                    "Host identity changed during the signed round"
                )
            if host_identity.adapter_version != decision.accepted_adapter_version:
                raise ConflictError(
                    "Host identity differs from the validation decision"
                )
            self._verify_host_package(
                manifest=manifest,
                package=package,
                artifact_path=incoming,
                host_identity=host_identity,
                expected_adapter_version=decision.accepted_adapter_version,
            )
            self._persist_knowledge_package(
                package_path=(
                    f"rounds/{manifest.round_id}/host_knowledge/package.json"
                ),
                artifact_path=(
                    f"rounds/{manifest.round_id}/host_knowledge/"
                    "knowledge.safetensors"
                ),
                package=package,
                source_artifact=incoming,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            result = DistillationResult(
                round_id=manifest.round_id,
                previous_adapter_version=decision.previous_adapter_version,
                candidate_adapter_version=decision.candidate_adapter_version,
                accepted_adapter_version=decision.accepted_adapter_version,
                previous_validation_loss=(
                    decision.previous_macro_mean_answer_token_ce
                ),
                candidate_validation_loss=(
                    decision.candidate_macro_mean_answer_token_ce
                ),
                required_improvement=decision.required_improvement,
                adapter_promoted=decision.adapter_promoted,
                candidate_artifact_hash=decision.candidate_adapter_hash,
                host_knowledge_package=package,
            )
            result_path = f"rounds/{manifest.round_id}/distillation_result.json"
            if self.store.exists(result_path):
                persisted = DistillationResult.model_validate(
                    self.store.read_json(result_path)
                )
                if persisted != result:
                    raise ConflictError(
                        "real Host distillation result changed after persistence"
                    )
            else:
                self.store.write_json_if_absent(
                    result_path,
                    result.model_dump(mode="json"),
                )
            if package.package_hash not in state.host_package_hashes:
                state.host_package_hashes.append(package.package_hash)
            if package.nonce not in state.host_nonces:
                state.host_nonces.append(package.nonce)
            state.state = "COMPLETED"
            state.host_adapter_after = decision.accepted_adapter_version
            state.adapter_promoted = decision.adapter_promoted
            state.message = (
                "candidate Host adapter accepted"
                if decision.adapter_promoted
                else "candidate Host adapter rejected; previous adapter retained"
            )
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            self._audit(
                "round_completed",
                {
                    "round_id": manifest.round_id,
                    "adapter_promoted": decision.adapter_promoted,
                    "adapter_version": decision.accepted_adapter_version,
                    "decision_hash": decision.decision_hash,
                    "host_package_hash": package.package_hash,
                    "selected_samples": len(job.sample_ids),
                },
            )
        finally:
            incoming.unlink(missing_ok=True)

    async def _process_sealed_round(
        self, manifest: RoundManifest, state: RoundState
    ) -> None:
        if state.state not in {"SEALED", "DISTILLING"}:
            return
        if manifest.alignment_strategy == "dtw":
            cached_job = self._load_host_training_job(manifest)
            if cached_job is not None:
                state.state = "DISTILLING"
                state.message = "Host candidate optimization is running"
                state.updated_at = utc_text(self.now_fn())
                self._write_state(state)
                await self._deliver_host_training_job(manifest, cached_job)
                result = await self._train_host_candidate(manifest, cached_job)
                await self._complete_real_round(
                    manifest,
                    state,
                    cached_job,
                    result,
                )
                return
        state.state = "DISTILLING"
        state.message = "constructing the validated Host distillation dataset"
        state.updated_at = utc_text(self.now_fn())
        self._write_state(state)
        incoming_paths: list[Path] = []
        try:
            host_identity = await self._host_identity(refresh=True)

            identity_path = (
                f"rounds/{manifest.round_id}/datasets/"
                "identity.json"
            )

            if self.store.exists(identity_path):
                bundle = self._host_reference_dataset_bundle(
                    manifest
                )

                receipt = await self.host.load_reference_data(
                    bundle
                )

                if (
                    receipt.round_id != manifest.round_id
                    or receipt.manifest_hash
                    != manifest.manifest_hash
                ):
                    raise ConflictError(
                        "Host loaded data for another round"
                    )

                if (
                    receipt.reference_identity.dataset_hash
                    != manifest.reference_dataset_hash
                ):
                    raise ConflictError(
                        "Host loaded a different reference dataset"
                    )

            baseline_artifact = self.incoming_artifact_path(
                manifest.round_id,
                "host-baseline",
            )
            incoming_paths.append(baseline_artifact)
            baseline = await self.host.reference_knowledge(
                manifest,
                baseline_artifact,
            )

            baseline_samples = self._verify_host_package(
                manifest=manifest,
                package=baseline,
                artifact_path=baseline_artifact,
                host_identity=host_identity,
                expected_adapter_version=(
                    manifest.current_host_adapter_version
                ),
            )

            self._persist_knowledge_package(
                package_path=(
                    f"rounds/{manifest.round_id}/host_baseline/package.json"
                ),
                artifact_path=(
                    f"rounds/{manifest.round_id}/host_baseline/"
                    "knowledge.safetensors"
                ),
                package=baseline,
                source_artifact=baseline_artifact,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )

            packages: list[KnowledgePackage] = []
            client_samples: dict[str, list[KnowledgeSample]] = {}
            for client_id in state.sealed_client_ids:
                package = KnowledgePackage.model_validate(
                    self.store.read_json(
                        f"rounds/{manifest.round_id}/submissions/"
                        f"{client_id}/package.json"
                    )
                )
                packages.append(package)
                client_samples[client_id] = load_package_samples(
                    self.store.path(
                        f"rounds/{manifest.round_id}/submissions/"
                        f"{client_id}/knowledge.safetensors"
                    ),
                    package,
                    maximum_bytes=manifest.maximum_knowledge_package_bytes,
                )
            reports = {
                client_id: SafetyReport.model_validate(
                    self.store.read_json(
                        f"rounds/{manifest.round_id}/safety/{client_id}.json"
                    )
                )
                for client_id in state.sealed_client_ids
            }
            if manifest.alignment_strategy == "dtw":
                job = await asyncio.to_thread(
                    self._prepare_host_training_job,
                    manifest=manifest,
                    baseline=baseline,
                    baseline_samples=baseline_samples,
                    packages=packages,
                    client_samples=client_samples,
                    reports=reports,
                )
                await self._deliver_host_training_job(manifest, job)
                result = await self._train_host_candidate(manifest, job)
                self._audit(
                    "host_training_job_prepared",
                    {
                        "round_id": manifest.round_id,
                        "job_hash": job.job_hash,
                        "dataset_hash": job.dataset_hash,
                        "integration_audit_hash": (
                            job.integration_audit_hash
                        ),
                        "artifact_sha256": job.artifact.sha256,
                        "artifact_byte_size": job.artifact.byte_size,
                        "accepted_client_ids": job.accepted_client_ids,
                        "candidate_adapter_version": (
                            result.candidate_adapter_version
                        ),
                        "candidate_adapter_hash": (
                            result.candidate_adapter_hash
                        ),
                        "optimizer_step_count": result.optimizer_step_count,
                    },
                )
                await self._complete_real_round(
                    manifest,
                    state,
                    job,
                    result,
                )
                return
            aligned_by_client = {
                client_id: {
                    sample.sample_id: (
                        sample.top_k_token_ids,
                        sample.top_k_logits,
                        0,
                    )
                    for sample in client_samples[client_id]
                }
                for client_id in state.sealed_client_ids
            }
            reports = finalize_aligned_safety_reports(
                host_samples=baseline_samples,
                client_samples=client_samples,
                aligned_by_client=aligned_by_client,
                pre_alignment_reports=reports,
                selected_client_ids=state.sealed_client_ids,
                historical_reliability=self._historical_reliability(
                    state.sealed_client_ids
                ),
            )
            self._persist_final_safety_reports(manifest.round_id, reports)
            trusted_ids = [
                client_id
                for client_id in state.sealed_client_ids
                if is_eligible_for_distillation(reports[client_id])
            ]
            if len(trusted_ids) < manifest.trusted_client_quorum:
                state.state = "SKIPPED"
                state.message = (
                    "trusted Client quorum was lost after post-alignment trust scoring"
                )
                state.updated_at = utc_text(self.now_fn())
                self._write_state(state)
                self._audit(
                    "round_skipped",
                    {
                        "round_id": manifest.round_id,
                        "reason": state.message,
                        "trusted_client_ids": trusted_ids,
                    },
                )
                return
            dataset = dual_min_ce_select(
                host_package=baseline,
                host_samples=baseline_samples,
                client_packages=packages,
                client_samples=client_samples,
                safety_reports=reports,
                selected_client_ids=manifest.selected_client_ids,
            )
            self.store.write_json(
                f"rounds/{manifest.round_id}/validated_distillation_dataset.json",
                dataset.model_dump(mode="json"),
            )
            host_artifact = self.incoming_artifact_path(
                manifest.round_id,
                "host-result",
            )
            incoming_paths.append(host_artifact)
            result = await self.host.distill(
                DistillationJob(manifest=manifest, dataset=dataset),
                host_artifact,
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
                artifact_path=host_artifact,
                host_identity=host_identity,
                expected_adapter_version=result.accepted_adapter_version,
            )

            if (
                host_package.package_hash
                not in state.host_package_hashes
            ):
                state.host_package_hashes.append(
                    host_package.package_hash
                )

            if host_package.nonce not in state.host_nonces:
                state.host_nonces.append(host_package.nonce)
            self.store.write_json(
                f"rounds/{manifest.round_id}/distillation_result.json",
                result.model_dump(mode="json"),
            )
            self._persist_knowledge_package(
                package_path=(
                    f"rounds/{manifest.round_id}/host_knowledge/package.json"
                ),
                artifact_path=(
                    f"rounds/{manifest.round_id}/host_knowledge/"
                    "knowledge.safetensors"
                ),
                package=host_package,
                source_artifact=host_artifact,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
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
        except TrustedQuorumError as exc:
            state.state = "SKIPPED"
            state.message = str(exc)
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            self._audit(
                "round_skipped", {"round_id": manifest.round_id, "reason": str(exc)}
            )
            return
        except Exception as exc:
            state.state = "ABORTED"
            state.message = str(exc)
            state.updated_at = utc_text(self.now_fn())
            self._write_state(state)
            self._audit(
                "round_aborted", {"round_id": manifest.round_id, "reason": str(exc)}
            )
            raise
        finally:
            for path in incoming_paths:
                path.unlink(missing_ok=True)

    def get_host_knowledge(self, round_id: str) -> tuple[KnowledgePackage, Path]:
        state = self.get_state(round_id)
        if state.state != "COMPLETED":
            raise ConflictError("Host Knowledge Package is not available yet")
        package_path = f"rounds/{round_id}/host_knowledge/package.json"
        artifact_path = (
            f"rounds/{round_id}/host_knowledge/knowledge.safetensors"
        )
        if not self.store.exists(package_path) or not self.store.exists(
            artifact_path
        ):
            raise NotFoundError("Host Knowledge Package was not published")
        package = KnowledgePackage.model_validate(
            self.store.read_json(package_path)
        )
        path = self.store.path(artifact_path)
        manifest = self.get_manifest(round_id)
        load_package_samples(
            path,
            package,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        return package, path

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
    ) -> Path | None:
        if client_id is None:
            raise AuthenticationError("missing Client ID")

        manifest = self.get_manifest(round_id)
        self.get_registration(client_id)

        if client_id not in manifest.selected_client_ids:
            raise AuthorizationError(
                "Client is not selected for this round"
            )

        identity_path = (
            f"rounds/{round_id}/datasets/identity.json"
        )
        reference_path = (
            f"rounds/{round_id}/datasets/reference.jsonl"
        )

        if not self.store.exists(identity_path):
            return None

        if not self.store.exists(reference_path):
            raise NotFoundError(
                "the round reference dataset snapshot is missing"
            )

        return self.store.path(reference_path)

    def _host_reference_dataset_bundle(
        self,
        manifest: RoundManifest,
    ) -> HostReferenceDatasetBundle:
        reference_path = self.store.path(
            f"rounds/{manifest.round_id}/datasets/"
            "reference.jsonl"
        )
        validation_path = self.store.path(
            f"rounds/{manifest.round_id}/datasets/"
            "validation.jsonl"
        )
        identity_path = (
            f"rounds/{manifest.round_id}/datasets/"
            "identity.json"
        )

        if (
            not reference_path.is_file()
            or not validation_path.is_file()
            or not self.store.exists(identity_path)
        ):
            raise ConflictError(
                "round dataset snapshots are incomplete"
            )

        try:
            snapshot = CoordinatorReferenceData.load(
                reference_path,
                validation_path,
            )

            record = self.store.read_json(identity_path)

            recorded_reference = (
                ReferenceDatasetIdentity.model_validate(
                    record["reference"]
                )
            )
            recorded_validation = (
                ReferenceDatasetIdentity.model_validate(
                    record["validation"]
                )
            )

            verified_reference = verify_reference_dataset(
                snapshot.reference_samples,
                expected_dataset_id=(
                    manifest.reference_dataset_id
                ),
                expected_dataset_hash=(
                    manifest.reference_dataset_hash
                ),
                expected_sample_ids=manifest.sample_ids,
            )

        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise ConflictError(
                f"round dataset snapshots are invalid: {exc}"
            ) from exc

        if verified_reference != recorded_reference:
            raise ConflictError(
                "round reference snapshot identity differs"
            )

        if (
            snapshot.validation_identity
            != recorded_validation
        ):
            raise ConflictError(
                "round validation snapshot identity differs"
            )

        return HostReferenceDatasetBundle(
            manifest=manifest,
            reference_samples=list(
                snapshot.reference_samples
            ),
            validation_samples=list(
                snapshot.validation_samples
            ),
            validation_identity=(
                snapshot.validation_identity
            ),
        )
