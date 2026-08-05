from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from shared.crypto import Ed25519Identity, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.ollama import OllamaClient
from shared.protocol import (
    DistillationJob,
    DistillationResult,
    KnowledgePackage,
    LoraProfile,
    ModelProfile,
    OllamaProfile,
    RoundManifest,
    HostReferenceDatasetBundle,
    HostReferenceDatasetReceipt,
)
from shared.storage import JsonFileStore
from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    ReferenceSample,
    load_reference_jsonl,
    reference_dataset_identity,
    verify_reference_dataset,
    write_reference_jsonl,
)


class HostRuntimeError(RuntimeError):
    pass


def default_host_profile() -> ModelProfile:
    serving_backend = os.getenv("HOST_SERVING_BACKEND", "mock").strip().lower()
    ollama_model = os.getenv("HOST_OLLAMA_MODEL", "llama3.2:3b")
    return ModelProfile(
        profile_id=os.getenv("HOST_PROFILE_ID", "host-mock-v1"),
        role="host",
        model_id=os.getenv("HOST_MODEL_ID", "legalfedllm/mock-host"),
        model_revision=os.getenv("HOST_MODEL_REVISION", "mock-v1"),
        tokenizer_id=os.getenv("HOST_TOKENIZER_ID", "legalfedllm/mock-tokenizer"),
        tokenizer_revision=os.getenv("HOST_TOKENIZER_REVISION", "mock-v1"),
        tokenizer_class=os.getenv("HOST_TOKENIZER_CLASS", "MockTokenizer"),
        training_backend=os.getenv("HOST_TRAINING_BACKEND", "mock"),
        serving_backend=serving_backend,
        prompt_template_hash=sha256_hex(b"legalfedllm-default-prompt"),
        lora=LoraProfile(
            rank=int(os.getenv("HOST_LORA_RANK", "8")),
            alpha=float(os.getenv("HOST_LORA_ALPHA", "16")),
            target_modules=tuple(
                item.strip()
                for item in os.getenv(
                    "HOST_LORA_TARGET_MODULES", "q_proj,v_proj"
                ).split(",")
                if item.strip()
            ),
        ),
        ollama=(
            OllamaProfile(model=ollama_model, digest=os.getenv("HOST_OLLAMA_DIGEST"))
            if serving_backend == "ollama"
            else None
        ),
    )


class HostRuntime:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        host_id: str = "legalfedllm-host",
        model_profile: ModelProfile | None = None,
        force_validation_failure: bool = False,
        ollama_client: OllamaClient | None = None,
    ):
        self.host_id = host_id
        self.store = JsonFileStore(data_dir)
        self.identity = Ed25519Identity.load_or_create(
            self.store.path("identity/private_key.pem")
        )
        self.model_profile = model_profile or default_host_profile()

        if self.model_profile.role != "host":
            raise ValueError("Host runtime requires a host model profile")

        if self.model_profile.training_backend != "mock":
            raise HostRuntimeError(
                "HOST_TRAINING_BACKEND=transformers is not integrated yet; "
                "use mock until the real FedMKT runtime is connected"
            )

        self.force_validation_failure = force_validation_failure
        self.ollama = ollama_client
        if self.model_profile.serving_backend == "ollama" and self.ollama is None:
            self.ollama = OllamaClient(
                os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            )
        self._ensure_initial_adapter()

    def _ensure_initial_adapter(self) -> None:
        if self.store.exists("adapters/active.json"):
            return
        adapter = {
            "version": 0,
            "validation_loss": 1.0,
            "parent_version": None,
            "round_id": None,
            "dataset_hash": None,
        }
        adapter["artifact_hash"] = sha256_hex(adapter)
        self.store.write_json("adapters/omega-v0.json", adapter)
        self.store.write_json("adapters/active.json", adapter)

    def active_adapter(self) -> dict[str, Any]:
        return self.store.read_json("adapters/active.json")

    @property
    def adapter_version(self) -> int:
        return int(self.active_adapter()["version"])

    def service_identity(self) -> dict[str, Any]:
        return {
            "service_id": self.host_id,
            "public_key": self.identity.public_key_b64,
            "model_profile": self.model_profile.model_dump(mode="json"),
            "adapter_version": self.adapter_version,
        }

    def generate_reference_knowledge(
        self, manifest: RoundManifest, *, enforce_manifest_parent: bool = True
    ) -> KnowledgePackage:
        identity_path = self._dataset_identity_path(
            manifest.round_id
        )

        if self.store.exists(identity_path):
            self.verify_cached_reference_data(manifest)
        
        cache_name = (
            "baseline_knowledge.json"
            if enforce_manifest_parent
            else "host_knowledge.json"
        )
        cache_path = f"rounds/{manifest.round_id}/{cache_name}"
        if self.store.exists(cache_path):
            return KnowledgePackage.model_validate(self.store.read_json(cache_path))

        active = self.active_adapter()
        if enforce_manifest_parent and manifest.current_host_adapter_version != active["version"]:
            raise HostRuntimeError("manifest is bound to a different Host adapter version")
        samples = deterministic_knowledge_samples(
            manifest=manifest,
            participant_id=self.host_id,
            role="host",
            adapter_version=active["version"],
        )
        package = KnowledgePackage.create_signed(
            identity=self.identity,
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            sender_id=self.host_id,
            sender_role="host",
            model_profile=self.model_profile,
            adapter_version=active["version"],
            alignment_profile_id=(
                f"{manifest.alignment.strategy}:{manifest.alignment.profile_version}"
            ),
            reference_dataset_id=manifest.reference_dataset_id,
            reference_dataset_hash=manifest.reference_dataset_hash,
            top_k=manifest.top_k,
            samples=samples,
        )
        self.store.write_json(cache_path, package.model_dump(mode="json"))
        return package

    def distill(self, job: DistillationJob) -> DistillationResult:
        manifest = job.manifest

        identity_path = self._dataset_identity_path(
            manifest.round_id
        )

        if self.store.exists(identity_path):
            self.verify_cached_reference_data(manifest)

        result_path = f"rounds/{manifest.round_id}/distillation_result.json"
        if self.store.exists(result_path):
            return DistillationResult.model_validate(self.store.read_json(result_path))
        dataset = job.dataset
        active = self.active_adapter()
        if dataset.round_id != manifest.round_id:
            raise HostRuntimeError("distillation dataset belongs to a different round")
        if dataset.manifest_hash != manifest.manifest_hash:
            raise HostRuntimeError("distillation dataset has a stale manifest hash")
        if dataset.host_adapter_version != active["version"]:
            raise HostRuntimeError("Host adapter changed before distillation")

        previous_version = int(active["version"])
        previous_loss = float(active["validation_loss"])
        candidate_version = previous_version + 1
        useful_samples = len(dataset.samples)
        expected_improvement = min(0.25, 0.01 + useful_samples * 0.0125)
        if self.force_validation_failure:
            candidate_loss = previous_loss + 0.01
        else:
            candidate_loss = max(0.001, previous_loss - expected_improvement)

        candidate = {
            "version": candidate_version,
            "validation_loss": round(candidate_loss, 8),
            "parent_version": previous_version,
            "round_id": manifest.round_id,
            "dataset_hash": dataset.dataset_hash,
            "selected_samples": useful_samples,
        }
        candidate["artifact_hash"] = sha256_hex(candidate)
        self.store.write_json(
            f"adapters/candidates/{manifest.round_id}/omega-v{candidate_version}.json",
            candidate,
        )

        required = manifest.distillation.minimum_validation_improvement
        promoted = useful_samples > 0 and candidate_loss <= previous_loss - required
        accepted = candidate if promoted else active
        if promoted:
            self.store.write_json(f"adapters/omega-v{candidate_version}.json", candidate)
            self.store.write_json("adapters/active.json", candidate)

        host_package = self.generate_reference_knowledge(
            manifest, enforce_manifest_parent=False
        )
        result = DistillationResult(
            round_id=manifest.round_id,
            previous_adapter_version=previous_version,
            candidate_adapter_version=candidate_version,
            accepted_adapter_version=int(accepted["version"]),
            previous_validation_loss=previous_loss,
            candidate_validation_loss=candidate_loss,
            required_improvement=required,
            adapter_promoted=promoted,
            candidate_artifact_hash=candidate["artifact_hash"],
            host_knowledge_package=host_package,
        )
        self.store.write_json(result_path, result.model_dump(mode="json"))
        return result

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        if self.model_profile.serving_backend == "ollama":
            assert self.ollama is not None and self.model_profile.ollama is not None
            return await self.ollama.generate(
                self.model_profile.ollama.model, prompt, max_new_tokens
            )
        return (
            f"[mock host:{self.model_profile.model_id} omega-v{self.adapter_version}] "
            f"{prompt.strip()}"
        )

    @staticmethod
    def _reference_dataset_path(round_id: str) -> str:
        return (
            f"rounds/{round_id}/datasets/reference.jsonl"
        )


    @staticmethod
    def _validation_dataset_path(round_id: str) -> str:
        return (
            f"rounds/{round_id}/datasets/validation.jsonl"
        )


    @staticmethod
    def _dataset_identity_path(round_id: str) -> str:
        return (
            f"rounds/{round_id}/datasets/identity.json"
        )

    @staticmethod
    def _validate_reference_pair(
        reference_samples: list[ReferenceSample],
        validation_samples: list[ReferenceSample],
        reference_identity: ReferenceDatasetIdentity,
        validation_identity: ReferenceDatasetIdentity,
    ) -> None:
        if (
            reference_identity.dataset_id
            != validation_identity.dataset_id
        ):
            raise HostRuntimeError(
                "reference and validation dataset IDs differ"
            )

        if (
            reference_identity.dataset_version
            != validation_identity.dataset_version
        ):
            raise HostRuntimeError(
                "reference and validation dataset versions differ"
            )

        reference_ids = {
            sample.sample_id
            for sample in reference_samples
        }
        validation_ids = {
            sample.sample_id
            for sample in validation_samples
        }

        if reference_ids & validation_ids:
            raise HostRuntimeError(
                "reference and validation datasets overlap"
            )

    def verify_cached_reference_data(
        self,
        manifest: RoundManifest,
    ) -> HostReferenceDatasetReceipt:
        reference_path = self._reference_dataset_path(
            manifest.round_id
        )
        validation_path = self._validation_dataset_path(
            manifest.round_id
        )
        identity_path = self._dataset_identity_path(
            manifest.round_id
        )

        if (
            not self.store.exists(reference_path)
            or not self.store.exists(validation_path)
            or not self.store.exists(identity_path)
        ):
            raise HostRuntimeError(
                "Host reference dataset cache is incomplete"
            )

        try:
            record = self.store.read_json(identity_path)

            if record.get("round_id") != manifest.round_id:
                raise ValueError(
                    "Host datasets belong to another round"
                )

            if (
                record.get("manifest_hash")
                != manifest.manifest_hash
            ):
                raise ValueError(
                    "Host datasets belong to another manifest"
                )

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

            reference_samples = load_reference_jsonl(
                self.store.path(reference_path)
            )
            validation_samples = load_reference_jsonl(
                self.store.path(validation_path)
            )

            reference_identity = verify_reference_dataset(
                reference_samples,
                expected_dataset_id=(
                    manifest.reference_dataset_id
                ),
                expected_dataset_hash=(
                    manifest.reference_dataset_hash
                ),
                expected_sample_ids=manifest.sample_ids,
            )

            validation_identity = (
                reference_dataset_identity(
                    validation_samples
                )
            )

            if recorded_reference != reference_identity:
                raise ValueError(
                    "Host reference identity record differs"
                )

            if recorded_validation != validation_identity:
                raise ValueError(
                    "Host validation identity record differs"
                )

            self._validate_reference_pair(
                reference_samples,
                validation_samples,
                reference_identity,
                validation_identity,
            )

            return HostReferenceDatasetReceipt(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                reference_identity=reference_identity,
                validation_identity=validation_identity,
            )

        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise HostRuntimeError(
                f"Host reference data is invalid: {exc}"
            ) from exc

    def load_reference_data(
        self,
        bundle: HostReferenceDatasetBundle,
    ) -> HostReferenceDatasetReceipt:
        manifest = bundle.manifest

        if manifest.host_model_profile != self.model_profile:
            raise HostRuntimeError(
                "reference data is bound to another Host profile"
            )

        reference_path = self._reference_dataset_path(
            manifest.round_id
        )
        validation_path = self._validation_dataset_path(
            manifest.round_id
        )
        identity_path = self._dataset_identity_path(
            manifest.round_id
        )

        existing = (
            self.store.exists(reference_path),
            self.store.exists(validation_path),
            self.store.exists(identity_path),
        )

        if any(existing):
            if not all(existing):
                raise HostRuntimeError(
                    "Host reference dataset cache is incomplete"
                )

            receipt = self.verify_cached_reference_data(
                manifest
            )

            if (
                receipt.validation_identity
                != bundle.validation_identity
            ):
                raise HostRuntimeError(
                    "cached validation identity differs "
                    "from the Coordinator identity"
                )

            return receipt

        try:
            reference_identity = verify_reference_dataset(
                bundle.reference_samples,
                expected_dataset_id=(
                    manifest.reference_dataset_id
                ),
                expected_dataset_hash=(
                    manifest.reference_dataset_hash
                ),
                expected_sample_ids=manifest.sample_ids,
            )

            validation_identity = (
                reference_dataset_identity(
                    bundle.validation_samples
                )
            )

        except ValueError as exc:
            raise HostRuntimeError(
                f"received Host reference data is invalid: {exc}"
            ) from exc

        if validation_identity != bundle.validation_identity:
            raise HostRuntimeError(
                "validation dataset identity differs "
                "from the Coordinator identity"
            )

        self._validate_reference_pair(
            bundle.reference_samples,
            bundle.validation_samples,
            reference_identity,
            validation_identity,
        )

        reference_target = self.store.path(
            reference_path
        )
        validation_target = self.store.path(
            validation_path
        )

        reference_target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        reference_temporary = reference_target.with_name(
            f".{reference_target.name}.{os.getpid()}.tmp"
        )
        validation_temporary = validation_target.with_name(
            f".{validation_target.name}.{os.getpid()}.tmp"
        )

        try:
            write_reference_jsonl(
                reference_temporary,
                bundle.reference_samples,
            )
            write_reference_jsonl(
                validation_temporary,
                bundle.validation_samples,
            )

            reference_temporary.replace(reference_target)
            validation_temporary.replace(validation_target)

            try:
                self.store.write_json_if_absent(
                    identity_path,
                    {
                        "round_id": manifest.round_id,
                        "manifest_hash": (
                            manifest.manifest_hash
                        ),
                        "reference": (
                            reference_identity.model_dump(
                                mode="json"
                            )
                        ),
                        "validation": (
                            validation_identity.model_dump(
                                mode="json"
                            )
                        ),
                    },
                )
            except Exception:
                reference_target.unlink(missing_ok=True)
                validation_target.unlink(missing_ok=True)
                raise

        except (
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            reference_temporary.unlink(missing_ok=True)
            validation_temporary.unlink(missing_ok=True)

            raise HostRuntimeError(
                f"failed to cache Host reference data: {exc}"
            ) from exc

        return HostReferenceDatasetReceipt(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            reference_identity=reference_identity,
            validation_identity=validation_identity,
        )
