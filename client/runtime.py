from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.ollama import OllamaClient
from shared.protocol import (
    DifferentialPrivacyReport,
    KnowledgePackage,
    LoraProfile,
    ModelProfile,
    OllamaProfile,
    RoundManifest,
    SubmissionReceipt,
    parse_utc,
    utc_now,
    utc_text,
)
from shared.storage import JsonFileStore

from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    load_reference_jsonl,
    verify_reference_dataset,
)

class ClientRuntimeError(RuntimeError):
    pass

def default_client_profile() -> ModelProfile:
    serving_backend = os.getenv("CLIENT_SERVING_BACKEND", "mock").strip().lower()
    ollama_model = os.getenv("CLIENT_OLLAMA_MODEL", "llama3.2:1b")
    return ModelProfile(
        profile_id=os.getenv("CLIENT_PROFILE_ID", "client-mock-v1"),
        role="client",
        model_id=os.getenv("CLIENT_MODEL_ID", "legalfedllm/mock-client"),
        model_revision=os.getenv("CLIENT_MODEL_REVISION", "mock-v1"),
        tokenizer_id=os.getenv("CLIENT_TOKENIZER_ID", "legalfedllm/mock-tokenizer"),
        tokenizer_revision=os.getenv("CLIENT_TOKENIZER_REVISION", "mock-v1"),
        tokenizer_class=os.getenv("CLIENT_TOKENIZER_CLASS", "MockTokenizer"),
        training_backend=os.getenv("CLIENT_TRAINING_BACKEND", "mock"),
        serving_backend=serving_backend,
        prompt_template_hash=sha256_hex(b"legalfedllm-default-prompt"),
        lora=LoraProfile(
            rank=int(os.getenv("CLIENT_LORA_RANK", "8")),
            alpha=float(os.getenv("CLIENT_LORA_ALPHA", "16")),
            target_modules=tuple(
                item.strip()
                for item in os.getenv(
                    "CLIENT_LORA_TARGET_MODULES", "q_proj,v_proj"
                ).split(",")
                if item.strip()
            ),
        ),
        ollama=(
            OllamaProfile(model=ollama_model, digest=os.getenv("CLIENT_OLLAMA_DIGEST"))
            if serving_backend == "ollama"
            else None
        ),
    )


class ClientRuntime:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        client_id: str = "legal-client-1",
        model_profile: ModelProfile | None = None,
        ollama_client: OllamaClient | None = None,
        maximum_clock_skew_seconds: int = 900,
        now_fn: Callable[[], Any] = utc_now,
    ):
        self.client_id = client_id
        self.store = JsonFileStore(data_dir)
        self.identity = Ed25519Identity.load_or_create(
            self.store.path("identity/private_key.pem")
        )
        self.model_profile = model_profile or default_client_profile()

        if self.model_profile.role != "client":
            raise ValueError("Client runtime requires a Client model profile")

        if self.model_profile.training_backend != "mock":
            raise ClientRuntimeError(
                "CLIENT_TRAINING_BACKEND=transformers is not integrated yet; "
                "use mock until the real FedMKT runtime is connected"
            )

        self.maximum_clock_skew_seconds = maximum_clock_skew_seconds
        self.now_fn = now_fn

        self.ollama = ollama_client
        if self.model_profile.serving_backend == "ollama" and self.ollama is None:
            self.ollama = OllamaClient(
                os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            )
        self._ensure_state()

    def _ensure_state(self) -> None:
        if self.store.exists("state.json"):
            return
        self.store.write_json(
            "state.json",
            {
                "serving_adapter_version": 0,
                "candidate_adapter_version": 0,
                "local_training_runs": 0,
                "last_completed_round": None,
            },
        )

    def state(self) -> dict[str, Any]:
        return self.store.read_json("state.json")

    def local_train(self, examples: list[str]) -> dict[str, Any]:
        if not examples or any(not item.strip() for item in examples):
            raise ValueError("local training requires non-blank examples")
        state = self.state()
        state["candidate_adapter_version"] += 1
        state["local_training_runs"] += 1
        state["last_local_batch"] = {
            "example_count": len(examples),
            "content_hash": sha256_hex([item.encode("utf-8").hex() for item in examples]),
        }
        self.store.write_json("state.json", state)
        return state

    @staticmethod
    def _pending_package_path(round_id: str) -> str:
        return f"knowledge_cache/pending/{round_id}.json"

    @staticmethod
    def _accepted_package_path(round_id: str) -> str:
        return f"knowledge_cache/accepted/{round_id}.json"

    @staticmethod
    def _pending_snapshot_path(round_id: str) -> str:
        return f"adapter_snapshots/pending/{round_id}.json"

    @staticmethod
    def _accepted_snapshot_path(round_id: str) -> str:
        return f"adapter_snapshots/accepted/{round_id}.json"

    @staticmethod
    def _receipt_path(round_id: str) -> str:
        return f"knowledge_cache/receipts/{round_id}.json"


    @staticmethod
    def _reference_dataset_path(round_id: str) -> str:
        return (
            f"reference_datasets/{round_id}/reference.jsonl"
        )


    @staticmethod
    def _reference_dataset_identity_path(
        round_id: str,
    ) -> str:
        return (
            f"reference_datasets/{round_id}/identity.json"
        )


    def verify_cached_reference_dataset(
        self,
        manifest: RoundManifest,
    ) -> ReferenceDatasetIdentity:
        dataset_path = self._reference_dataset_path(
            manifest.round_id
        )
        identity_path = (
            self._reference_dataset_identity_path(
                manifest.round_id
            )
        )

        dataset_exists = self.store.exists(dataset_path)
        identity_exists = self.store.exists(identity_path)

        if not dataset_exists or not identity_exists:
            raise ClientRuntimeError(
                "verified reference dataset cache is incomplete"
            )

        try:
            record = self.store.read_json(identity_path)

            if record.get("round_id") != manifest.round_id:
                raise ValueError(
                    "cached dataset belongs to another round"
                )

            if (
                record.get("manifest_hash")
                != manifest.manifest_hash
            ):
                raise ValueError(
                    "cached dataset belongs to another manifest"
                )

            recorded_identity = (
                ReferenceDatasetIdentity.model_validate(
                    record["identity"]
                )
            )

            samples = load_reference_jsonl(
                self.store.path(dataset_path)
            )

            identity = verify_reference_dataset(
                samples,
                expected_dataset_id=(
                    manifest.reference_dataset_id
                ),
                expected_dataset_hash=(
                    manifest.reference_dataset_hash
                ),
                expected_sample_ids=manifest.sample_ids,
            )

            if recorded_identity != identity:
                raise ValueError(
                    "cached dataset identity record "
                    "is inconsistent"
                )

            return identity

        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise ClientRuntimeError(
                f"cached reference dataset is invalid: {exc}"
            ) from exc

    def cache_reference_dataset(
        self,
        *,
        manifest: RoundManifest,
        content: bytes,
    ) -> ReferenceDatasetIdentity:
        dataset_path = self._reference_dataset_path(
            manifest.round_id
        )
        identity_path = (
            self._reference_dataset_identity_path(
                manifest.round_id
            )
        )

        dataset_exists = self.store.exists(dataset_path)
        identity_exists = self.store.exists(identity_path)

        if dataset_exists and identity_exists:
            return self.verify_cached_reference_dataset(
                manifest
            )

        if dataset_exists or identity_exists:
            raise ClientRuntimeError(
                "reference dataset cache is incomplete"
            )

        target = self.store.path(dataset_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.tmp"
        )

        try:
            temporary.write_bytes(content)

            samples = load_reference_jsonl(temporary)

            identity = verify_reference_dataset(
                samples,
                expected_dataset_id=(
                    manifest.reference_dataset_id
                ),
                expected_dataset_hash=(
                    manifest.reference_dataset_hash
                ),
                expected_sample_ids=manifest.sample_ids,
            )

            temporary.replace(target)

            try:
                self.store.write_json_if_absent(
                    identity_path,
                    {
                        "round_id": manifest.round_id,
                        "manifest_hash": (
                            manifest.manifest_hash
                        ),
                        "identity": identity.model_dump(
                            mode="json"
                        ),
                    },
                )
            except Exception:
                target.unlink(missing_ok=True)
                raise

            return identity

        except (
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            temporary.unlink(missing_ok=True)

            if (
                target.exists()
                and not self.store.exists(identity_path)
            ):
                target.unlink(missing_ok=True)

            raise ClientRuntimeError(
                f"downloaded reference dataset is invalid: {exc}"
            ) from exc


    def create_knowledge_package(
        self,
        manifest: RoundManifest,
    ) -> KnowledgePackage:
        if self.client_id not in manifest.selected_client_ids:
            raise ValueError("Client is not selected for this round")

        dataset_path = self._reference_dataset_path(
            manifest.round_id
        )
        identity_path = (
            self._reference_dataset_identity_path(
                manifest.round_id
            )
        )

        if (
            self.store.exists(dataset_path)
            or self.store.exists(identity_path)
        ):
            self.verify_cached_reference_dataset(manifest)

        accepted_path = self._accepted_package_path(manifest.round_id)

        if self.store.exists(accepted_path):
            accepted = KnowledgePackage.model_validate(
                self.store.read_json(accepted_path)
            )

            if accepted.manifest_hash != manifest.manifest_hash:
                raise ClientRuntimeError(
                    "an accepted package already exists for this round "
                    "with another manifest"
                )

            return accepted

        pending_path = self._pending_package_path(manifest.round_id)

        if self.store.exists(pending_path):
            pending = KnowledgePackage.model_validate(
                self.store.read_json(pending_path)
            )

            if pending.manifest_hash != manifest.manifest_hash:
                raise ClientRuntimeError(
                    "a pending package already exists for this round "
                    "with another manifest"
                )

            return pending

        state = self.state()
        adapter_version = int(state["candidate_adapter_version"])

        samples = deterministic_knowledge_samples(
            manifest=manifest,
            participant_id=self.client_id,
            role="client",
            adapter_version=adapter_version,
        )

        package = KnowledgePackage.create_signed(
            identity=self.identity,
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            sender_id=self.client_id,
            sender_role="client",
            model_profile=self.model_profile,
            adapter_version=adapter_version,
            alignment_profile_id=(
                f"{manifest.alignment.strategy}:"
                f"{manifest.alignment.profile_version}"
            ),
            reference_dataset_id=manifest.reference_dataset_id,
            reference_dataset_hash=manifest.reference_dataset_hash,
            top_k=manifest.top_k,
            samples=samples,
            dp_report=DifferentialPrivacyReport(
                enabled=manifest.dp_policy.required,
                mechanism=(
                    manifest.dp_policy.mechanism
                    if manifest.dp_policy.required
                    else "none"
                ),
                epsilon_spent=(
                    float(os.getenv("CLIENT_MOCK_DP_EPSILON", "0.1"))
                    if manifest.dp_policy.required
                    else None
                ),
                delta=(
                    manifest.dp_policy.delta
                    if manifest.dp_policy.required
                    else None
                ),
            ),
        )

        snapshot = {
            "round_id": manifest.round_id,
            "manifest_hash": manifest.manifest_hash,
            "client_id": self.client_id,
            "model_profile_id": self.model_profile.profile_id,
            "adapter_version": adapter_version,
            "local_training_runs": int(state["local_training_runs"]),
            "state_hash": sha256_hex(state),
            "package_hash": package.artifact_sha256,
            "created_at": utc_text(self.now_fn()),
        }

        self.store.write_json_if_absent(
            self._pending_snapshot_path(manifest.round_id),
            snapshot,
        )

        self.store.write_json_if_absent(
            pending_path,
            package.model_dump(mode="json"),
        )

        return package

    def commit_knowledge_submission(
        self,
        *,
        manifest: RoundManifest,
        package: KnowledgePackage,
        receipt: SubmissionReceipt,
    ) -> None:
        if (
            receipt.round_id != manifest.round_id
            or receipt.client_id != self.client_id
        ):
            raise ClientRuntimeError(
                "submission receipt identity does not match the Client"
            )

        if receipt.package_hash != package.artifact_sha256:
            raise ClientRuntimeError(
                "submission receipt hash does not match the package"
            )

        pending_path = self._pending_package_path(manifest.round_id)
        snapshot_path = self._pending_snapshot_path(manifest.round_id)

        if (
            not self.store.exists(pending_path)
            or not self.store.exists(snapshot_path)
        ):
            raise ClientRuntimeError(
                "pending package or adapter snapshot is missing"
            )

        pending = KnowledgePackage.model_validate(
            self.store.read_json(pending_path)
        )
        snapshot = self.store.read_json(snapshot_path)

        if pending.artifact_sha256 != package.artifact_sha256:
            raise ClientRuntimeError(
                "pending package changed before acceptance"
            )

        if snapshot.get("package_hash") != package.artifact_sha256:
            raise ClientRuntimeError(
                "pending adapter snapshot is bound to another package"
            )

        if snapshot.get("adapter_version") != package.adapter_version:
            raise ClientRuntimeError(
                "pending adapter snapshot version does not match"
            )

        accepted_package_path = self._accepted_package_path(
            manifest.round_id
        )
        accepted_snapshot_path = self._accepted_snapshot_path(
            manifest.round_id
        )

        if self.store.exists(accepted_package_path):
            existing = KnowledgePackage.model_validate(
                self.store.read_json(accepted_package_path)
            )

            if existing.artifact_sha256 != package.artifact_sha256:
                raise ClientRuntimeError(
                    "accepted package is immutable and cannot be replaced"
                )
        else:
            self.store.write_json_if_absent(
                accepted_package_path,
                package.model_dump(mode="json"),
            )

        if self.store.exists(accepted_snapshot_path):
            existing_snapshot = self.store.read_json(
                accepted_snapshot_path
            )

            if existing_snapshot != snapshot:
                raise ClientRuntimeError(
                    "accepted adapter snapshot is immutable"
                )
        else:
            self.store.write_json_if_absent(
                accepted_snapshot_path,
                snapshot,
            )

        self.store.write_json(
            self._receipt_path(manifest.round_id),
            receipt.model_dump(mode="json"),
        )

        self.store.delete(pending_path)
        self.store.delete(snapshot_path)

    def apply_host_knowledge(
        self,
        *,
        manifest: RoundManifest,
        host_package: KnowledgePackage,
        host_public_key: str,
        expected_host_id: str,
        accepted_host_adapter_version: int,
        adapter_promoted: bool,
    ) -> dict[str, Any]:
        round_id = manifest.round_id

        cache_path = self._accepted_package_path(round_id)
        snapshot_path = self._accepted_snapshot_path(round_id)

        if (
            not self.store.exists(cache_path)
            or not self.store.exists(snapshot_path)
        ):
            raise ValueError(
                "accepted Client package or adapter snapshot is missing"
            )

        cached = KnowledgePackage.model_validate(
            self.store.read_json(cache_path)
        )
        snapshot = self.store.read_json(snapshot_path)

        if cached.artifact_sha256 != snapshot.get("package_hash"):
            raise ValueError(
                "accepted Client cache is not bound to its adapter snapshot"
            )

        if cached.adapter_version != snapshot.get("adapter_version"):
            raise ValueError(
                "accepted Client adapter snapshot version differs"
            )

        if (
            host_package.sender_role != "host"
            or host_package.sender_id != expected_host_id
        ):
            raise ValueError("Host package identity is invalid")

        if not host_package.verify_signature(host_public_key):
            raise ValueError("Host package signature is invalid")

        if host_package.model_profile != manifest.host_model_profile:
            raise ValueError(
                "Host package model profile differs from the manifest"
            )

        if (
            host_package.round_id != round_id
            or cached.round_id != round_id
        ):
            raise ValueError(
                "Host package or cache belongs to another round"
            )

        if host_package.manifest_hash != manifest.manifest_hash:
            raise ValueError(
                "Host package is bound to another manifest"
            )

        if (
            host_package.reference_dataset_id
            != manifest.reference_dataset_id
        ):
            raise ValueError(
                "Host package reference dataset ID differs"
            )

        if (
            host_package.reference_dataset_hash
            != manifest.reference_dataset_hash
        ):
            raise ValueError(
                "Host package reference dataset hash differs"
            )

        if host_package.sample_ids != manifest.sample_ids:
            raise ValueError(
                "Host package sample order differs from the manifest"
            )

        if host_package.sample_ids != cached.sample_ids:
            raise ValueError(
                "Host and Client sample order differs"
            )

        if host_package.top_k != manifest.top_k:
            raise ValueError(
                "Host package top-k differs from the manifest"
            )

        expected_alignment = (
            f"{manifest.alignment.strategy}:"
            f"{manifest.alignment.profile_version}"
        )

        if host_package.alignment_profile_id != expected_alignment:
            raise ValueError(
                "Host package alignment profile differs"
            )

        if (
            host_package.adapter_version
            != accepted_host_adapter_version
        ):
            raise ValueError(
                "Host package adapter version differs from round state"
            )

        created = parse_utc(host_package.created_at)
        skew = abs(
            (self.now_fn() - created).total_seconds()
        )

        if skew > self.maximum_clock_skew_seconds:
            raise ValueError(
                "Host package timestamp is outside the allowed skew"
            )

        host_by_id = {
            sample.sample_id: sample
            for sample in host_package.samples
        }

        client_by_id = {
            sample.sample_id: sample
            for sample in cached.samples
        }

        selected = [
            sample_id
            for sample_id in cached.sample_ids
            if (
                host_by_id[sample_id].ce_loss
                < client_by_id[sample_id].ce_loss
            )
        ]

        state = self.state()

        if adapter_promoted and selected:
            state["candidate_adapter_version"] += 1
            state["serving_adapter_version"] = (
                state["candidate_adapter_version"]
            )

        state["last_completed_round"] = round_id
        state["last_host_adapter_version"] = (
            host_package.adapter_version
        )
        state["host_distillation_samples"] = selected

        self.store.write_json("state.json", state)

        return state

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        if self.model_profile.serving_backend == "ollama":
            assert self.ollama is not None and self.model_profile.ollama is not None
            return await self.ollama.generate(
                self.model_profile.ollama.model, prompt, max_new_tokens
            )
        version = self.state()["serving_adapter_version"]
        return (
            f"[mock client:{self.model_profile.model_id} theta-v{version}] "
            f"{prompt.strip()}"
        )
