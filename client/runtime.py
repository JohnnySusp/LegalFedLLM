from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from client.learning_queue import LearningQueue
from client.model_profiles import (
    ollama_model_for_profile,
    pinned_client_profile,
    supported_ollama_models,
)
from client.reverse_training import (
    CLIENT_QUALITY_NON_REGRESSION_TOLERANCE,
    ClientReverseCandidateResult,
    ClientReverseDecision,
    ClientValidationRecord,
    ClientValidationSampleMetric,
)
from client.training import (
    BackendTrainingResult,
    LocalTrainingRecord,
    PrivateTrainingExample,
    TrainingExecutionProfile,
    execution_profile_from_environment,
    load_private_examples,
    private_dataset_semantic_hash,
)
from shared.adapter_checkpoint import AdapterCheckpointStore
from shared.alignment_profiles import (
    MOCK_IDENTITY_PROFILE_ID,
    UnsupportedAlignmentProfile,
    resolve_alignment_profile,
    validate_alignment_pair,
)
from shared.client_reverse_artifact import (
    load_client_reverse_training_artifact,
    write_client_reverse_training_artifact,
)
from shared.crypto import Ed25519Identity, canonical_json_bytes, sha256_hex
from shared.fedmkt_runtime import deterministic_knowledge_samples
from shared.knowledge_artifact import load_package_samples, write_knowledge_artifact
from shared.ollama import OllamaClient
from shared.prompt import PROMPT_TEMPLATE, PROMPT_TEMPLATE_ID
from shared.protocol import (
    ClientPublicDataPartition,
    ClientReverseTrainingJob,
    DifferentialPrivacyReport,
    KnowledgePackage,
    KnowledgeSample,
    LoraProfile,
    ModelProfile,
    RegistrationRecord,
    OllamaProfile,
    RoundManifest,
    SubmissionReceipt,
    parse_utc,
    utc_now,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    client_public_data_partition,
    load_reference_jsonl,
    verify_reference_dataset,
)
from shared.reference_knowledge import encode_reference_samples
from shared.storage import JsonFileStore
from shared.tokenizer_validation import (
    TokenizerValidationError,
    load_pinned_tokenizer,
)
from shared.vocabulary_mapping import VocabularyMappingCache


class ClientRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ClientTrainingState:
    kind: Literal["base", "peft"]
    version: int
    state_hash: str
    checkpoint_hash: str | None


def default_client_profile() -> ModelProfile:
    serving_backend = os.getenv("CLIENT_SERVING_BACKEND", "mock").strip().lower()
    selected_profile = os.getenv("CLIENT_MODEL_PROFILE", "mock").strip()
    if selected_profile != "mock":
        requested_backend = os.getenv(
            "CLIENT_TRAINING_BACKEND", "transformers"
        ).strip().lower()
        if requested_backend != "transformers":
            raise ClientRuntimeError(
                "pinned real Client profiles require "
                "CLIENT_TRAINING_BACKEND=transformers"
            )
        return pinned_client_profile(
            selected_profile,
            serving_backend=serving_backend,
        )

    ollama_model = os.getenv("CLIENT_OLLAMA_MODEL", "qwen3:1.7b")
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
        prompt_template_id=PROMPT_TEMPLATE_ID,
        prompt_template_hash=sha256_hex(PROMPT_TEMPLATE.encode("utf-8")),
        lora=LoraProfile(
            rank=int(os.getenv("CLIENT_LORA_RANK", "8")),
            alpha=float(os.getenv("CLIENT_LORA_ALPHA", "16")),
            dropout=float(os.getenv("CLIENT_LORA_DROPOUT", "0.05")),
            target_modules=tuple(
                item.strip()
                for item in os.getenv(
                    "CLIENT_LORA_TARGET_MODULES",
                    "q_proj,k_proj,v_proj,o_proj",
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
        private_data_path: str | Path | None = None,
        private_dataset_id: str | None = None,
        training_execution_profile: TrainingExecutionProfile | None = None,
        knowledge_batch_size: int | None = None,
        reverse_backend: Any | None = None,
        force_reverse_validation_failure: bool = False,
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

        configured_private_path = (
            str(private_data_path)
            if private_data_path is not None
            else os.getenv("CLIENT_PRIVATE_DATA_PATH", "").strip()
        )
        self.private_data_path = (
            Path(configured_private_path).resolve()
            if configured_private_path
            else self.store.path("private/train.jsonl").resolve()
        )
        self.learning_queue = LearningQueue(self.private_data_path)
        self.private_dataset_id = private_dataset_id or os.getenv(
            "CLIENT_PRIVATE_DATASET_ID", "client-private-v1"
        )
        if not self.private_dataset_id.strip():
            raise ValueError("private dataset ID must not be blank")
        self.training_execution_profile = (
            training_execution_profile
            or execution_profile_from_environment(
                self.model_profile.training_backend
            )
        )
        if self.training_execution_profile.backend != (
            self.model_profile.training_backend
        ):
            raise ValueError(
                "training execution backend differs from the ModelProfile"
            )
        self.knowledge_batch_size = (
            int(os.getenv("CLIENT_KNOWLEDGE_BATCH_SIZE", "1"))
            if knowledge_batch_size is None
            else knowledge_batch_size
        )
        if self.knowledge_batch_size < 1:
            raise ValueError("knowledge batch size must be positive")
        self.reverse_backend = reverse_backend
        self.force_reverse_validation_failure = (
            force_reverse_validation_failure
        )

        self.maximum_clock_skew_seconds = maximum_clock_skew_seconds
        self.now_fn = now_fn

        self.ollama = ollama_client
        if (
            self.ollama is None
            and (
                self.model_profile.training_backend == "transformers"
                or self.model_profile.serving_backend == "ollama"
            )
        ):
            self.ollama = OllamaClient(
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "10")),
            )
        self._ensure_state()
        self.adapter_store: AdapterCheckpointStore | None = None
        if self.model_profile.training_backend == "transformers":
            self.adapter_store = AdapterCheckpointStore(
                self.store.path("adapters"),
                self.model_profile,
            )
            current = self.adapter_store.current()
            if current is not None:
                metadata, _ = current
                state = self.state()
                state["candidate_adapter_version"] = metadata.version
                state["training_adapter_version"] = metadata.version
                state["training_checkpoint_hash"] = metadata.checkpoint_hash
                self.store.write_json("state.json", state)

    def _ensure_state(self) -> None:
        defaults = {
            "serving_adapter_version": 0,
            "candidate_adapter_version": 0,
            "training_adapter_version": 0,
            "training_checkpoint_hash": None,
            "local_training_runs": 0,
            "last_completed_round": None,
            "last_training_round": None,
            "last_reverse_training_job_hash": None,
            "last_reverse_candidate_result_hash": None,
            "last_reverse_decision_hash": None,
            "last_reverse_decision_reason": None,
            "last_reverse_consent": None,
            "last_learning_receipt": None,
        }
        if self.store.exists("state.json"):
            state = self.store.read_json("state.json")
            changed = False
            for key, value in defaults.items():
                if key not in state:
                    state[key] = value
                    changed = True
            if changed:
                self.store.write_json("state.json", state)
            return
        self.store.write_json("state.json", defaults)

    @staticmethod
    def _registration_path() -> str:
        return "identity/registration.json"

    def registration_record(self) -> RegistrationRecord | None:
        path = self._registration_path()
        if not self.store.exists(path):
            return None
        record = RegistrationRecord.model_validate(self.store.read_json(path))
        if (
            record.client_id != self.client_id
            or record.public_key != self.identity.public_key_b64
            or record.model_profile != self.model_profile
        ):
            raise ClientRuntimeError(
                "persisted Client registration does not match the active identity/profile"
            )
        return record

    def commit_registration(self, record: RegistrationRecord) -> None:
        if (
            record.client_id != self.client_id
            or record.public_key != self.identity.public_key_b64
            or record.model_profile != self.model_profile
        ):
            raise ClientRuntimeError(
                "Coordinator registration does not match the active Client identity/profile"
            )
        current = self.registration_record()
        if current is not None and current != record:
            raise ClientRuntimeError("persisted Client registration is immutable")
        if current is None:
            self.store.write_json_if_absent(
                self._registration_path(),
                record.model_dump(mode="json"),
            )

    def state(self) -> dict[str, Any]:
        return self.store.read_json("state.json")

    def local_train(self, examples: list[str]) -> dict[str, Any]:
        if self.model_profile.training_backend != "mock":
            raise ClientRuntimeError(
                "/v1/local-train is mock-only; use the round-specific endpoint"
            )
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
    def _local_training_record_path(round_id: str) -> str:
        return f"local_training/rounds/{round_id}.json"

    def local_train_round(
        self,
        manifest: RoundManifest,
        private_data_path: str | Path | None = None,
    ) -> dict[str, Any]:
        self._validate_training_manifest(manifest)
        source_path = Path(private_data_path or self.private_data_path)
        examples = load_private_examples(source_path)
        dataset_hash = private_dataset_semantic_hash(examples)
        record_path = self._local_training_record_path(manifest.round_id)
        if self.store.exists(record_path):
            record = LocalTrainingRecord.model_validate(
                self.store.read_json(record_path)
            )
            self._validate_local_training_record(
                record,
                manifest,
                dataset_hash=dataset_hash,
            )
            return record.model_dump(mode="json")

        started_at = utc_text(self.now_fn())
        if self.model_profile.training_backend == "mock":
            state = self.state()
            parent_version = int(state["candidate_adapter_version"])
            parent_hash = state.get("training_checkpoint_hash")
            result_version = parent_version + 1
            result_hash = sha256_hex(
                {
                    "backend": "mock",
                    "round_id": manifest.round_id,
                    "manifest_hash": manifest.manifest_hash,
                    "profile_hash": self.model_profile.profile_hash(),
                    "dataset_hash": dataset_hash,
                    "parent_version": parent_version,
                    "parent_hash": parent_hash,
                }
            )
            result = BackendTrainingResult(
                parent_version=parent_version,
                parent_checkpoint_hash=parent_hash,
                result_version=result_version,
                result_checkpoint_hash=result_hash,
                checkpoint_format="mock-json",
                dependency_versions={},
                trainable_parameter_count=0,
                total_parameter_count=0,
                optimizer_step_count=0,
                training_loss=None,
            )
        else:
            from client.peft_backend import TransformersPeftTrainingBackend

            backend = TransformersPeftTrainingBackend(
                data_dir=self.store.root,
                model_profile=self.model_profile,
                execution_profile=self.training_execution_profile,
            )
            current_checkpoint = (
                self.adapter_store.current()
                if self.adapter_store is not None
                else None
            )
            result = backend.train(
                examples,
                manifest,
                base_parent_hash=(
                    sha256_hex(self.state())
                    if current_checkpoint is None
                    else None
                ),
            )

        record = LocalTrainingRecord.create(
            round_id=manifest.round_id,
            manifest_hash=manifest.manifest_hash,
            client_model_profile_hash=self.model_profile.profile_hash(),
            training_execution_profile=self.training_execution_profile,
            training_execution_profile_hash=(
                self.training_execution_profile.profile_hash()
            ),
            private_dataset_id=self.private_dataset_id,
            private_dataset_semantic_hash=dataset_hash,
            private_example_count=len(examples),
            parent_adapter_version=result.parent_version,
            parent_checkpoint_hash=result.parent_checkpoint_hash,
            result_adapter_version=result.result_version,
            result_checkpoint_hash=result.result_checkpoint_hash,
            checkpoint_format=result.checkpoint_format,
            label_format="chat_sft_answer_only_v1",
            maximum_sequence_length=manifest.maximum_sequence_length,
            truncation_policy="reject",
            started_at=started_at,
            completed_at=utc_text(self.now_fn()),
            dependency_versions=result.dependency_versions,
            trainable_parameter_count=result.trainable_parameter_count,
            total_parameter_count=result.total_parameter_count,
            optimizer_step_count=result.optimizer_step_count,
            training_loss=result.training_loss,
        )
        self.store.write_json_if_absent(
            record_path,
            record.model_dump(mode="json"),
        )
        state = self.state()
        state["candidate_adapter_version"] = result.result_version
        state["training_adapter_version"] = result.result_version
        state["training_checkpoint_hash"] = result.result_checkpoint_hash
        if self.model_profile.serving_backend == "transformers":
            state["serving_adapter_version"] = result.result_version
        state["local_training_runs"] += 1
        state["last_training_round"] = manifest.round_id
        state["last_local_batch"] = {
            "example_count": len(examples),
            "content_hash": dataset_hash,
        }
        self.store.write_json("state.json", state)
        return record.model_dump(mode="json")

    def learning_queue_status(self) -> dict[str, Any]:
        return self.learning_queue.status()

    def add_learning_example(self, prompt: str, answer: str) -> dict[str, Any]:
        example = self.learning_queue.append(prompt, answer)
        status = self.learning_queue.status()
        return {
            "example_id": example.example_id,
            **status,
        }

    def train_queued_learning(
        self,
        manifest: RoundManifest,
    ) -> dict[str, Any] | None:
        batch = self.learning_queue.begin()
        if batch is None:
            return None
        try:
            payload = self.local_train_round(manifest, batch.path)
            record = LocalTrainingRecord.model_validate(payload)
            if record.private_dataset_semantic_hash != batch.semantic_hash:
                raise ClientRuntimeError(
                    "local learning batch hash differs from the training record"
                )
            receipt = self.learning_queue.complete(
                batch,
                adapter_version=record.result_adapter_version,
                checkpoint_hash=record.result_checkpoint_hash,
                round_id=manifest.round_id,
                training_record_hash=record.record_hash,
            )
            state = self.state()
            state["last_learning_receipt"] = receipt
            self.store.write_json("state.json", state)
            return receipt
        except Exception:
            self.learning_queue.restore(batch)
            raise

    def _learning_suggestion_path(self, suggestion_id: str) -> str:
        if Path(suggestion_id).name != suggestion_id or suggestion_id in {".", ".."}:
            raise ValueError("learning suggestion ID is not safe")
        return f"learning_suggestions/{suggestion_id}.json"

    def record_learning_suggestion(
        self,
        prompt: str,
        answer: str,
    ) -> dict[str, Any]:
        if not prompt.strip() or not answer.strip():
            raise ValueError("learning suggestion requires prompt and answer")
        suggestion_id = f"suggest-{secrets.token_hex(12)}"
        payload = {
            "suggestion_id": suggestion_id,
            "prompt": prompt,
            "answer": answer,
            "created_at": utc_text(self.now_fn()),
        }
        self.store.write_json_if_absent(
            self._learning_suggestion_path(suggestion_id),
            payload,
        )
        return payload

    def pending_learning_suggestions(self) -> list[dict[str, Any]]:
        root = self.store.path("learning_suggestions")
        if not root.exists():
            return []
        values: list[dict[str, Any]] = []
        for path in root.glob("suggest-*.json"):
            try:
                value = self.store.read_json(
                    str(path.relative_to(self.store.root))
                )
            except (OSError, ValueError):
                continue
            values.append(value)
        values.sort(key=lambda value: (value.get("created_at", ""), value["suggestion_id"]))
        return values

    def resolve_learning_suggestion(
        self,
        suggestion_id: str,
        *,
        learn: bool,
    ) -> dict[str, Any]:
        path = self._learning_suggestion_path(suggestion_id)
        if not self.store.exists(path):
            raise ValueError("learning suggestion does not exist")
        suggestion = self.store.read_json(path)
        result: dict[str, Any] = {
            "suggestion_id": suggestion_id,
            "learned": learn,
        }
        if learn:
            result.update(
                self.add_learning_example(
                    str(suggestion["prompt"]),
                    str(suggestion["answer"]),
                )
            )
        self.store.delete(path)
        return result

    def require_round_training(
        self,
        manifest: RoundManifest,
    ) -> LocalTrainingRecord:
        self._validate_training_manifest(manifest)
        path = self._local_training_record_path(manifest.round_id)
        if not self.store.exists(path):
            raise ClientRuntimeError(
                "the Client has not trained for this signed round manifest"
            )
        record = LocalTrainingRecord.model_validate(self.store.read_json(path))
        current_dataset_hash = private_dataset_semantic_hash(
            load_private_examples(self.private_data_path)
        )
        self._validate_local_training_record(
            record,
            manifest,
            dataset_hash=current_dataset_hash,
        )
        return record

    def _validate_training_manifest(self, manifest: RoundManifest) -> None:
        if self.client_id not in manifest.selected_client_ids:
            raise ClientRuntimeError("Client is not selected for this round")
        expected_profile_hash = manifest.selected_client_profile_hashes.get(
            self.client_id
        )
        if expected_profile_hash != self.model_profile.profile_hash():
            raise ClientRuntimeError(
                "signed manifest is bound to another Client model profile"
            )
        alignment_profile_id = manifest.alignment_profile_id_for(
            self.client_id
        )
        if alignment_profile_id == MOCK_IDENTITY_PROFILE_ID:
            if (
                self.model_profile.training_backend != "mock"
                or manifest.host_model_profile.training_backend != "mock"
            ):
                raise ClientRuntimeError(
                    "mock identity alignment requires mock Client and Host profiles"
                )
        else:
            try:
                validate_alignment_pair(
                    alignment_profile_id,
                    client_profile=self.model_profile,
                    host_profile=manifest.host_model_profile,
                )
            except UnsupportedAlignmentProfile as exc:
                raise ClientRuntimeError(str(exc)) from exc
        if manifest.prompt_template_hash != self.model_profile.prompt_template_hash:
            raise ClientRuntimeError(
                "signed manifest prompt differs from the Client ModelProfile"
            )
        if manifest.label_format != "chat_sft_answer_only_v1":
            raise ClientRuntimeError(
                "signed manifest requires another private label format"
            )
        if manifest.truncation_policy != "reject":
            raise ClientRuntimeError(
                "signed manifest requires another truncation policy"
            )
        if (
            self.model_profile.training_backend == "transformers"
            and manifest.dp_policy.required
        ):
            raise ClientRuntimeError(
                "real DP-SGD is not implemented for Client PEFT training"
            )

    def ensure_alignment_tokenizers(
        self,
        manifest: RoundManifest,
    ) -> tuple[Any, Any] | None:
        self._validate_training_manifest(manifest)
        alignment_profile_id = manifest.alignment_profile_id_for(self.client_id)
        if alignment_profile_id == MOCK_IDENTITY_PROFILE_ID:
            return None

        profile = resolve_alignment_profile(alignment_profile_id)
        cache_dir = os.getenv("CLIENT_TOKENIZER_CACHE_DIR") or None
        token = os.getenv("HF_TOKEN") or None

        loaded = []
        for endpoint in (profile.client, profile.host):
            try:
                validated = load_pinned_tokenizer(
                    endpoint,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=True,
                )
            except TokenizerValidationError as exc:
                raise ClientRuntimeError(
                    f"cached alignment tokenizer {endpoint.tokenizer_id!r} failed "
                    f"pinned validation: {exc}"
                ) from exc
            except Exception:
                try:
                    validated = load_pinned_tokenizer(
                        endpoint,
                        cache_dir=cache_dir,
                        token=token,
                        local_files_only=False,
                    )
                except Exception as exc:
                    raise ClientRuntimeError(
                        "required pinned alignment tokenizer "
                        f"{endpoint.tokenizer_id!r} at revision "
                        f"{endpoint.tokenizer_revision!r} is not available locally "
                        "and could not be downloaded and validated"
                    ) from exc
            loaded.append(validated)

        return loaded[0], loaded[1]

    def _validate_local_training_record(
        self,
        record: LocalTrainingRecord,
        manifest: RoundManifest,
        *,
        dataset_hash: str | None = None,
    ) -> None:
        if record.round_id != manifest.round_id:
            raise ClientRuntimeError("local training record belongs to another round")
        if record.manifest_hash != manifest.manifest_hash:
            raise ClientRuntimeError(
                "local training record belongs to another signed manifest"
            )
        if record.client_model_profile_hash != self.model_profile.profile_hash():
            raise ClientRuntimeError(
                "local training record belongs to another Client profile"
            )
        if dataset_hash is not None and (
            record.private_dataset_semantic_hash != dataset_hash
        ):
            raise ClientRuntimeError(
                "private dataset changed after round-bound local training"
            )
        state = self.state()
        if int(state["training_adapter_version"]) != (
            record.result_adapter_version
        ):
            raise ClientRuntimeError(
                "current training adapter belongs to another local training run"
            )
        if state.get("training_checkpoint_hash") != record.result_checkpoint_hash:
            raise ClientRuntimeError(
                "current training checkpoint differs from the round record"
            )
        if self.adapter_store is not None:
            current = self.adapter_store.current()
            if current is None:
                raise ClientRuntimeError("current PEFT adapter checkpoint is missing")
            metadata, _ = current
            if (
                metadata.version != record.result_adapter_version
                or metadata.checkpoint_hash != record.result_checkpoint_hash
            ):
                raise ClientRuntimeError(
                    "current PEFT checkpoint differs from the round record"
                )

    @staticmethod
    def _pending_package_path(round_id: str) -> str:
        return f"knowledge_cache/pending/{round_id}/package.json"

    @staticmethod
    def _pending_artifact_path(round_id: str) -> str:
        return f"knowledge_cache/pending/{round_id}/knowledge.safetensors"

    @staticmethod
    def _accepted_package_path(round_id: str) -> str:
        return f"knowledge_cache/accepted/{round_id}/package.json"

    @staticmethod
    def _accepted_artifact_path(round_id: str) -> str:
        return f"knowledge_cache/accepted/{round_id}/knowledge.safetensors"

    @staticmethod
    def _host_package_path(round_id: str) -> str:
        return f"knowledge_cache/host/{round_id}/package.json"

    @staticmethod
    def _host_artifact_path(round_id: str) -> str:
        return f"knowledge_cache/host/{round_id}/knowledge.safetensors"

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
    def _reverse_job_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/job.json"

    @staticmethod
    def _reverse_artifact_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/trainer_inputs.safetensors"

    @staticmethod
    def _reverse_audit_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/integration_audit.json"

    @staticmethod
    def _reverse_partition_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/public_partition.json"

    @staticmethod
    def _reverse_candidate_result_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/candidate/result.json"

    @staticmethod
    def _reverse_parent_validation_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/validation/parent.json"

    @staticmethod
    def _reverse_candidate_validation_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/validation/candidate.json"

    @staticmethod
    def _reverse_decision_path(round_id: str) -> str:
        return f"reverse_distillation/rounds/{round_id}/decision.json"

    def _validate_package_snapshot(
        self,
        *,
        manifest: RoundManifest,
        package: KnowledgePackage,
        snapshot_path: str,
    ) -> dict[str, Any]:
        if not self.store.exists(snapshot_path):
            raise ClientRuntimeError(
                "Knowledge Package adapter snapshot is missing"
            )
        snapshot = self.store.read_json(snapshot_path)
        expected = {
            "round_id": manifest.round_id,
            "manifest_hash": manifest.manifest_hash,
            "client_id": self.client_id,
            "model_profile_id": self.model_profile.profile_id,
            "model_profile_hash": self.model_profile.profile_hash(),
            "adapter_version": package.adapter_version,
            "package_hash": package.package_hash,
            "artifact_sha256": package.artifact.sha256,
        }
        mismatched = [
            name
            for name, value in expected.items()
            if snapshot.get(name) != value
        ]
        if mismatched:
            raise ClientRuntimeError(
                "Knowledge Package adapter snapshot differs: "
                + ", ".join(sorted(mismatched))
            )
        if (
            package.round_id != manifest.round_id
            or package.manifest_hash != manifest.manifest_hash
            or package.sender_id != self.client_id
            or package.sender_role != "client"
            or package.model_profile != self.model_profile
            or package.reference_dataset_id != manifest.reference_dataset_id
            or package.reference_dataset_hash != manifest.reference_dataset_hash
            or package.sample_ids != manifest.sample_ids
            or package.top_k != manifest.top_k
        ):
            raise ClientRuntimeError(
                "cached Knowledge Package differs from its signed round"
            )
        if not package.verify_signature(self.identity.public_key_b64):
            raise ClientRuntimeError(
                "cached Knowledge Package signature is invalid"
            )

        training_path = self._local_training_record_path(manifest.round_id)
        if not self.store.exists(training_path):
            if self.model_profile.training_backend == "transformers":
                version, checkpoint_hash = self.current_training_adapter()
                if package.adapter_version != version:
                    raise ClientRuntimeError(
                        "Knowledge Package adapter differs from the active Client adapter"
                    )
                if snapshot.get("training_checkpoint_hash") != checkpoint_hash:
                    raise ClientRuntimeError(
                        "Knowledge Package active checkpoint binding differs"
                    )
                return snapshot
            if (
                "local_training_record_hash" in snapshot
                or snapshot.get("training_checkpoint_hash") is not None
            ):
                raise ClientRuntimeError(
                    "Knowledge Package snapshot names a missing training record"
                )
            return snapshot

        try:
            record = LocalTrainingRecord.model_validate(
                self.store.read_json(training_path)
            )
        except (OSError, ValueError) as exc:
            raise ClientRuntimeError(
                f"Knowledge Package training record is invalid: {exc}"
            ) from exc
        if (
            record.round_id != manifest.round_id
            or record.manifest_hash != manifest.manifest_hash
            or record.client_model_profile_hash
            != self.model_profile.profile_hash()
            or record.result_adapter_version != package.adapter_version
            or snapshot.get("local_training_record_hash")
            != record.record_hash
            or snapshot.get("training_checkpoint_hash")
            != record.result_checkpoint_hash
        ):
            raise ClientRuntimeError(
                "Knowledge Package training provenance differs"
            )

        if self.adapter_store is not None:
            try:
                metadata, _ = self.adapter_store.version(
                    record.result_adapter_version
                )
            except (OSError, ValueError) as exc:
                raise ClientRuntimeError(
                    f"Knowledge Package adapter checkpoint is invalid: {exc}"
                ) from exc
            if (
                metadata.checkpoint_hash != record.result_checkpoint_hash
                or metadata.round_id != manifest.round_id
                or metadata.manifest_hash != manifest.manifest_hash
            ):
                raise ClientRuntimeError(
                    "Knowledge Package adapter checkpoint differs"
                )
        return snapshot


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


    def package_artifact_path(self, package: KnowledgePackage) -> Path:
        round_id = package.round_id
        candidates = (
            (
                self._pending_package_path(round_id),
                self._pending_artifact_path(round_id),
            ),
            (
                self._accepted_package_path(round_id),
                self._accepted_artifact_path(round_id),
            ),
        )
        for metadata_path, artifact_path in candidates:
            if not self.store.exists(metadata_path):
                continue
            stored = KnowledgePackage.model_validate(
                self.store.read_json(metadata_path)
            )
            if stored.package_hash != package.package_hash:
                continue
            path = self.store.path(artifact_path)
            load_package_samples(
                path,
                stored,
                maximum_bytes=stored.artifact.byte_size,
            )
            return path
        raise ClientRuntimeError("Knowledge Package artifact is missing")

    def current_training_adapter(self) -> tuple[int, str | None]:
        if self.adapter_store is None:
            state = self.state()
            return (
                int(state["training_adapter_version"]),
                state.get("training_checkpoint_hash"),
            )
        current = self.adapter_store.current()
        if current is None:
            return 0, None
        metadata, _ = current
        return metadata.version, metadata.checkpoint_hash

    def generate_knowledge_samples(
        self,
        manifest: RoundManifest,
        *,
        require_round_training: bool = True,
        expected_adapter_version: int | None = None,
        expected_checkpoint_hash: str | None = None,
    ) -> list[KnowledgeSample]:
        record: LocalTrainingRecord | None = None
        if require_round_training:
            record = self.require_round_training(manifest)
            expected_adapter_version = record.result_adapter_version
            expected_checkpoint_hash = record.result_checkpoint_hash
        elif expected_adapter_version is None:
            expected_adapter_version, expected_checkpoint_hash = (
                self.current_training_adapter()
            )

        self.verify_cached_reference_dataset(manifest)
        reference_samples = load_reference_jsonl(
            self.store.path(self._reference_dataset_path(manifest.round_id))
        )

        if self.model_profile.training_backend == "mock":
            adapter_version = (
                record.result_adapter_version
                if record is not None
                else int(expected_adapter_version or 0)
            )
            return deterministic_knowledge_samples(
                manifest=manifest,
                participant_id=self.client_id,
                role="client",
                adapter_version=adapter_version,
            )

        from client.peft_backend import TransformersPeftTrainingBackend

        backend = TransformersPeftTrainingBackend(
            data_dir=self.store.root,
            model_profile=self.model_profile,
            execution_profile=self.training_execution_profile,
            knowledge_batch_size=self.knowledge_batch_size,
        )
        try:
            return backend.generate_knowledge(
                reference_samples,
                manifest,
                expected_adapter_version=int(expected_adapter_version or 0),
                expected_checkpoint_hash=expected_checkpoint_hash,
            )
        except (RuntimeError, ValueError) as exc:
            raise ClientRuntimeError(
                f"real Client knowledge generation failed: {exc}"
            ) from exc

    def incoming_host_artifact_path(self, round_id: str) -> Path:
        return self.store.path(
            f"knowledge_cache/incoming/{round_id}."
            f"{secrets.token_hex(8)}.safetensors"
        )

    def create_knowledge_package(
        self,
        manifest: RoundManifest,
        require_round_training: bool = True,
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
        accepted_artifact_path = self._accepted_artifact_path(
            manifest.round_id
        )
        accepted_snapshot_path = self._accepted_snapshot_path(
            manifest.round_id
        )

        accepted_exists = (
            self.store.exists(accepted_path),
            self.store.exists(accepted_artifact_path),
            self.store.exists(accepted_snapshot_path),
        )
        if any(accepted_exists) and not all(accepted_exists):
            raise ClientRuntimeError(
                "accepted Knowledge Package cache is incomplete"
            )

        if all(accepted_exists):
            accepted = KnowledgePackage.model_validate(
                self.store.read_json(accepted_path)
            )

            if accepted.manifest_hash != manifest.manifest_hash:
                raise ClientRuntimeError(
                    "an accepted package already exists for this round "
                    "with another manifest"
                )

            load_package_samples(
                self.store.path(accepted_artifact_path),
                accepted,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            self._validate_package_snapshot(
                manifest=manifest,
                package=accepted,
                snapshot_path=accepted_snapshot_path,
            )

            return accepted

        pending_path = self._pending_package_path(manifest.round_id)
        pending_artifact_path = self._pending_artifact_path(
            manifest.round_id
        )
        pending_snapshot_path = self._pending_snapshot_path(
            manifest.round_id
        )

        pending_exists = (
            self.store.exists(pending_path),
            self.store.exists(pending_artifact_path),
            self.store.exists(pending_snapshot_path),
        )
        if any(pending_exists) and not all(pending_exists):
            raise ClientRuntimeError(
                "pending Knowledge Package cache is incomplete"
            )

        if all(pending_exists):
            pending = KnowledgePackage.model_validate(
                self.store.read_json(pending_path)
            )

            if pending.manifest_hash != manifest.manifest_hash:
                raise ClientRuntimeError(
                    "a pending package already exists for this round "
                    "with another manifest"
                )

            load_package_samples(
                self.store.path(pending_artifact_path),
                pending,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            self._validate_package_snapshot(
                manifest=manifest,
                package=pending,
                snapshot_path=pending_snapshot_path,
            )

            return pending

        state = self.state()
        training_record: LocalTrainingRecord | None = None
        active_checkpoint_hash: str | None = None
        if self.model_profile.training_backend == "transformers":
            if require_round_training:
                training_record = self.require_round_training(manifest)
                adapter_version = training_record.result_adapter_version
                active_checkpoint_hash = training_record.result_checkpoint_hash
            else:
                adapter_version, active_checkpoint_hash = (
                    self.current_training_adapter()
                )
            samples = self.generate_knowledge_samples(
                manifest,
                require_round_training=require_round_training,
                expected_adapter_version=adapter_version,
                expected_checkpoint_hash=active_checkpoint_hash,
            )
        else:
            adapter_version = int(state["candidate_adapter_version"])
            samples = deterministic_knowledge_samples(
                manifest=manifest,
                participant_id=self.client_id,
                role="client",
                adapter_version=adapter_version,
            )

        try:
            descriptor = write_knowledge_artifact(
                self.store.path(pending_artifact_path),
                samples,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
            package = KnowledgePackage.create_signed(
                identity=self.identity,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                sender_id=self.client_id,
                sender_role="client",
                model_profile=self.model_profile,
                adapter_version=adapter_version,
                alignment_profile_id=manifest.alignment_profile_id_for(
                    self.client_id
                ),
                reference_dataset_id=manifest.reference_dataset_id,
                reference_dataset_hash=manifest.reference_dataset_hash,
                top_k=manifest.top_k,
                sample_ids=[sample.sample_id for sample in samples],
                artifact=descriptor,
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
                created_at=utc_text(self.now_fn()),
            )
            metadata_size = len(
                canonical_json_bytes(package.model_dump(mode="json"))
            )
            if metadata_size + descriptor.byte_size > (
                manifest.maximum_knowledge_package_bytes
            ):
                raise ClientRuntimeError(
                    "Knowledge Package exceeds the manifest size limit"
                )
        except Exception:
            self.store.delete(pending_artifact_path)
            raise

        snapshot = {
            "round_id": manifest.round_id,
            "manifest_hash": manifest.manifest_hash,
            "client_id": self.client_id,
            "model_profile_id": self.model_profile.profile_id,
            "model_profile_hash": self.model_profile.profile_hash(),
            "adapter_version": adapter_version,
            "local_training_runs": int(state["local_training_runs"]),
            "state_hash": sha256_hex(state),
            "package_hash": package.package_hash,
            "artifact_sha256": descriptor.sha256,
            "created_at": utc_text(self.now_fn()),
        }
        if active_checkpoint_hash is not None:
            snapshot["training_checkpoint_hash"] = active_checkpoint_hash
        if training_record is None:
            training_record_path = self._local_training_record_path(
                manifest.round_id
            )
            if self.store.exists(training_record_path):
                training_record = LocalTrainingRecord.model_validate(
                    self.store.read_json(training_record_path)
                )
        if training_record is not None:
            snapshot.update(
                {
                    "local_training_record_hash": training_record.record_hash,
                    "training_checkpoint_hash": (
                        training_record.result_checkpoint_hash
                    ),
                }
            )

        try:
            self.store.write_json_if_absent(
                pending_snapshot_path,
                snapshot,
            )
            self.store.write_json_if_absent(
                pending_path,
                package.model_dump(mode="json"),
            )
            self._validate_package_snapshot(
                manifest=manifest,
                package=package,
                snapshot_path=pending_snapshot_path,
            )
        except Exception:
            self.store.delete(pending_path)
            self.store.delete(pending_artifact_path)
            self.store.delete(pending_snapshot_path)
            raise

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

        if receipt.package_hash != package.package_hash:
            raise ClientRuntimeError(
                "submission receipt hash does not match the package"
            )

        pending_path = self._pending_package_path(manifest.round_id)
        pending_artifact_path = self._pending_artifact_path(
            manifest.round_id
        )
        snapshot_path = self._pending_snapshot_path(manifest.round_id)

        if (
            not self.store.exists(pending_path)
            or not self.store.exists(pending_artifact_path)
            or not self.store.exists(snapshot_path)
        ):
            raise ClientRuntimeError(
                "pending package or adapter snapshot is missing"
            )

        pending = KnowledgePackage.model_validate(
            self.store.read_json(pending_path)
        )
        load_package_samples(
            self.store.path(pending_artifact_path),
            pending,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        snapshot = self.store.read_json(snapshot_path)
        self._validate_package_snapshot(
            manifest=manifest,
            package=pending,
            snapshot_path=snapshot_path,
        )

        if pending.package_hash != package.package_hash:
            raise ClientRuntimeError(
                "pending package changed before acceptance"
            )

        if snapshot.get("package_hash") != package.package_hash:
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
        accepted_artifact_path = self._accepted_artifact_path(
            manifest.round_id
        )
        accepted_snapshot_path = self._accepted_snapshot_path(
            manifest.round_id
        )

        accepted_exists = (
            self.store.exists(accepted_package_path),
            self.store.exists(accepted_artifact_path),
        )
        if any(accepted_exists) and not all(accepted_exists):
            raise ClientRuntimeError(
                "accepted Knowledge Package cache is incomplete"
            )

        if all(accepted_exists):
            existing = KnowledgePackage.model_validate(
                self.store.read_json(accepted_package_path)
            )

            if existing.package_hash != package.package_hash:
                raise ClientRuntimeError(
                    "accepted package is immutable and cannot be replaced"
                )
            load_package_samples(
                self.store.path(accepted_artifact_path),
                existing,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
        else:
            try:
                self.store.copy_file_if_absent(
                    accepted_artifact_path,
                    self.store.path(pending_artifact_path),
                )
                self.store.write_json_if_absent(
                    accepted_package_path,
                    package.model_dump(mode="json"),
                )
            except Exception:
                if not self.store.exists(accepted_package_path):
                    self.store.delete(accepted_artifact_path)
                raise

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

        self._validate_package_snapshot(
            manifest=manifest,
            package=package,
            snapshot_path=accepted_snapshot_path,
        )

        self.store.write_json(
            self._receipt_path(manifest.round_id),
            receipt.model_dump(mode="json"),
        )

        self.store.delete(pending_path)
        self.store.delete(pending_artifact_path)
        self.store.delete(snapshot_path)

    def _reverse_backend_instance(self) -> Any:
        if self.model_profile.training_backend != "transformers":
            raise ClientRuntimeError("real Client reverse training is unavailable")
        if self.reverse_backend is None:
            from client.reverse_training import TransformersPeftReverseBackend

            self.reverse_backend = TransformersPeftReverseBackend(
                data_dir=self.store.root,
                model_profile=self.model_profile,
                execution_profile=self.training_execution_profile,
                knowledge_batch_size=self.knowledge_batch_size,
            )
        return self.reverse_backend

    def _reverse_job_parent_state(
        self,
        job: ClientReverseTrainingJob,
    ) -> _ClientTrainingState:
        snapshot_path = self._accepted_snapshot_path(job.manifest.round_id)
        if not self.store.exists(snapshot_path):
            raise ClientRuntimeError("Client parent snapshot is missing")
        snapshot = self.store.read_json(snapshot_path)
        expected = {
            "round_id": job.manifest.round_id,
            "manifest_hash": job.manifest.manifest_hash,
            "client_id": self.client_id,
            "model_profile_hash": self.model_profile.profile_hash(),
            "adapter_version": job.parent_adapter_version,
        }
        mismatches = [
            key for key, value in expected.items()
            if snapshot.get(key) != value
        ]
        checkpoint_hash = snapshot.get("training_checkpoint_hash")
        parent_hash = checkpoint_hash or snapshot.get("state_hash")
        if mismatches or parent_hash != job.parent_adapter_hash:
            raise ClientRuntimeError(
                "Client parent snapshot differs from the reverse job"
            )
        if checkpoint_hash is None:
            if job.parent_adapter_version != 0:
                raise ClientRuntimeError(
                    "base-only Client reverse parent must be adapter version 0"
                )
            return _ClientTrainingState(
                kind="base",
                version=0,
                state_hash=job.parent_adapter_hash,
                checkpoint_hash=None,
            )
        return _ClientTrainingState(
            kind="peft",
            version=job.parent_adapter_version,
            state_hash=str(checkpoint_hash),
            checkpoint_hash=str(checkpoint_hash),
        )

    def _current_reverse_state(
        self,
        job: ClientReverseTrainingJob,
    ) -> _ClientTrainingState:
        if self.model_profile.training_backend == "transformers":
            if self.adapter_store is None:
                raise ClientRuntimeError("Client adapter store is unavailable")
            current = self.adapter_store.current()
            if current is not None:
                metadata, _ = current
                return _ClientTrainingState(
                    kind="peft",
                    version=metadata.version,
                    state_hash=metadata.checkpoint_hash,
                    checkpoint_hash=metadata.checkpoint_hash,
                )
            parent_state = self._reverse_job_parent_state(job)
            if parent_state.kind != "base":
                raise ClientRuntimeError("current Client PEFT checkpoint is missing")
            return parent_state

        state = self.state()
        checkpoint_hash = state.get("training_checkpoint_hash")
        state_hash = str(checkpoint_hash or sha256_hex(state))
        return _ClientTrainingState(
            kind="peft",
            version=int(state["candidate_adapter_version"]),
            state_hash=state_hash,
            checkpoint_hash=(str(checkpoint_hash) if checkpoint_hash else None),
        )

    def _validate_reverse_candidate_result(
        self,
        job: ClientReverseTrainingJob,
        result: ClientReverseCandidateResult,
    ) -> None:
        expected = {
            "round_id": job.manifest.round_id,
            "manifest_hash": job.manifest.manifest_hash,
            "job_hash": job.job_hash,
            "parent_adapter_version": job.parent_adapter_version,
            "parent_adapter_hash": job.parent_adapter_hash,
            "candidate_adapter_version": job.parent_adapter_version + 1,
            "client_model_profile_hash": self.model_profile.profile_hash(),
            "execution_profile_hash": self.training_execution_profile.profile_hash(),
            "client_public_data_epochs": job.client_public_data_epochs,
        }
        payload = result.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ClientRuntimeError(
                "Client reverse candidate differs from its job: "
                + ", ".join(mismatches)
            )

    def _validate_client_validation_record(
        self,
        job: ClientReverseTrainingJob,
        record: ClientValidationRecord,
        *,
        adapter_role: str,
        adapter_version: int,
        checkpoint_hash: str,
    ) -> None:
        expected = {
            "round_id": job.manifest.round_id,
            "manifest_hash": job.manifest.manifest_hash,
            "job_hash": job.job_hash,
            "partition_hash": job.public_data_partition.partition_hash,
            "validation_sample_ids_hash": (
                job.public_data_partition.validation_sample_ids_sha256
            ),
            "client_model_profile_hash": self.model_profile.profile_hash(),
            "adapter_role": adapter_role,
            "adapter_version": adapter_version,
            "checkpoint_hash": checkpoint_hash,
        }
        payload = record.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ClientRuntimeError(
                "Client validation record differs from its job: "
                + ", ".join(mismatches)
            )
        if [sample.sample_id for sample in record.samples] != (
            job.public_data_partition.validation_sample_ids
        ):
            raise ClientRuntimeError("Client validation record changed sample order")

    @staticmethod
    def _reverse_decision_reason(
        *,
        stale_parent: bool,
        forced_rejection: bool,
        quality_passed: bool,
    ) -> str:
        if stale_parent:
            return "stale_parent"
        if forced_rejection:
            return "forced_validation_rejection"
        if quality_passed:
            return "candidate_accepted"
        return "quality_gate_failed"

    def _mock_reverse_candidate(
        self,
        job: ClientReverseTrainingJob,
    ) -> ClientReverseCandidateResult:
        candidate_hash = sha256_hex(
            {
                "backend": "mock",
                "job_hash": job.job_hash,
                "parent_adapter_hash": job.parent_adapter_hash,
                "candidate_adapter_version": job.parent_adapter_version + 1,
            }
        )
        return ClientReverseCandidateResult.create(
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            parent_adapter_version=job.parent_adapter_version,
            parent_adapter_hash=job.parent_adapter_hash,
            candidate_adapter_version=job.parent_adapter_version + 1,
            candidate_adapter_hash=candidate_hash,
            client_model_profile_hash=self.model_profile.profile_hash(),
            execution_profile_hash=self.training_execution_profile.profile_hash(),
            client_public_data_epochs=1,
            supervised_loss_weight=0.9,
            distillation_loss_weight=0.1,
            loss_type="ce",
            temperature=1.0,
            optimizer_step_count=1,
            optimizer_loss=0.9,
            supervised_answer_loss=1.0,
            distillation_answer_loss=0.0,
            trainable_parameter_count=1,
            total_parameter_count=2,
            lora_tensors_changed=True,
            frozen_base_unchanged=True,
            reload_verified=True,
            dependency_versions={"backend": "deterministic-mock"},
            created_at=utc_text(self.now_fn()),
        )

    def _mock_validation_record(
        self,
        job: ClientReverseTrainingJob,
        result: ClientReverseCandidateResult,
        *,
        adapter_role: str,
    ) -> ClientValidationRecord:
        if not job.public_data_partition.validation_sample_ids:
            raise ClientRuntimeError(
                "Client public validation split contains no samples"
            )
        if adapter_role == "parent":
            version = result.parent_adapter_version
            checkpoint_hash = result.parent_adapter_hash
            ce_loss = 1.0
        else:
            version = result.candidate_adapter_version
            checkpoint_hash = result.candidate_adapter_hash
            ce_loss = 0.999
        metrics = [
            ClientValidationSampleMetric(
                sample_id=sample_id,
                answer_token_count=1,
                answer_token_ce=ce_loss,
                teacher_forced_exact_match=1.0,
                teacher_forced_rouge_l=1.0,
            )
            for sample_id in job.public_data_partition.validation_sample_ids
        ]
        return ClientValidationRecord.create(
            backend="mock",
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            partition_hash=job.public_data_partition.partition_hash,
            client_model_profile_hash=self.model_profile.profile_hash(),
            adapter_role=adapter_role,
            adapter_version=version,
            checkpoint_hash=checkpoint_hash,
            samples=metrics,
            created_at=utc_text(self.now_fn()),
        )

    def _load_or_create_reverse_candidate(
        self,
        job: ClientReverseTrainingJob,
    ) -> ClientReverseCandidateResult:
        path = self._reverse_candidate_result_path(job.manifest.round_id)
        if self.store.exists(path):
            result = ClientReverseCandidateResult.model_validate(
                self.store.read_json(path)
            )
            self._validate_reverse_candidate_result(job, result)
            if self.model_profile.training_backend == "transformers":
                self._reverse_backend_instance().validate_candidate(result)
            return result
        if self.model_profile.training_backend == "transformers":
            if self.adapter_store is None:
                raise ClientRuntimeError("Client adapter store is unavailable")
            candidate_version = job.parent_adapter_version + 1
            incomplete_path = self.adapter_store.candidate_path(
                job.manifest.round_id,
                candidate_version,
            )
            if incomplete_path.exists():
                metadata, _ = self.adapter_store.candidate(
                    job.manifest.round_id,
                    candidate_version,
                )
                expected = {
                    "round_id": job.manifest.round_id,
                    "manifest_hash": job.manifest.manifest_hash,
                    "version": candidate_version,
                    "parent_version": job.parent_adapter_version,
                    "parent_checkpoint_hash": job.parent_adapter_hash,
                    "profile_hash": self.model_profile.profile_hash(),
                    "execution_profile_hash": (
                        self.training_execution_profile.profile_hash()
                    ),
                    "parent_state_kind": self._reverse_job_parent_state(job).kind,
                }
                metadata_payload = metadata.model_dump(mode="json")
                mismatches = [
                    key
                    for key, value in expected.items()
                    if metadata_payload[key] != value
                ]
                if mismatches:
                    raise ClientRuntimeError(
                        "incomplete Client reverse candidate differs from its "
                        "job: "
                        + ", ".join(mismatches)
                    )
                self.adapter_store.discard_candidate(
                    job.manifest.round_id,
                    candidate_version,
                    metadata.checkpoint_hash,
                )
            parent_state = self._reverse_job_parent_state(job)
            result = self._reverse_backend_instance().train_candidate(
                job,
                self.store.path(self._reverse_artifact_path(job.manifest.round_id)),
                parent_state_kind=parent_state.kind,
            )
            self._reverse_backend_instance().validate_candidate(result)
        else:
            result = self._mock_reverse_candidate(job)
        self._validate_reverse_candidate_result(job, result)
        self.store.write_json_if_absent(path, result.model_dump(mode="json"))
        return result

    def _load_or_create_reverse_validation(
        self,
        job: ClientReverseTrainingJob,
        result: ClientReverseCandidateResult,
        *,
        adapter_role: str,
        validation_samples: list[Any],
    ) -> ClientValidationRecord:
        path = (
            self._reverse_parent_validation_path(job.manifest.round_id)
            if adapter_role == "parent"
            else self._reverse_candidate_validation_path(job.manifest.round_id)
        )
        version = (
            result.parent_adapter_version
            if adapter_role == "parent"
            else result.candidate_adapter_version
        )
        checkpoint_hash = (
            result.parent_adapter_hash
            if adapter_role == "parent"
            else result.candidate_adapter_hash
        )
        if self.store.exists(path):
            record = ClientValidationRecord.model_validate(
                self.store.read_json(path)
            )
        elif self.model_profile.training_backend == "transformers":
            record = self._reverse_backend_instance().validation_record(
                job=job,
                result=result,
                validation_samples=validation_samples,
                adapter_role=adapter_role,
            )
            self.store.write_json_if_absent(path, record.model_dump(mode="json"))
        else:
            record = self._mock_validation_record(
                job,
                result,
                adapter_role=adapter_role,
            )
            self.store.write_json_if_absent(path, record.model_dump(mode="json"))
        self._validate_client_validation_record(
            job,
            record,
            adapter_role=adapter_role,
            adapter_version=version,
            checkpoint_hash=checkpoint_hash,
        )
        return record

    def _validate_reverse_decision(
        self,
        job: ClientReverseTrainingJob,
        decision: ClientReverseDecision,
    ) -> None:
        expected = {
            "round_id": job.manifest.round_id,
            "manifest_hash": job.manifest.manifest_hash,
            "job_hash": job.job_hash,
            "host_teacher_sample_count": len(job.host_teacher_sample_ids),
            "parent_adapter_version": job.parent_adapter_version,
            "parent_adapter_hash": job.parent_adapter_hash,
        }
        payload = decision.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ClientRuntimeError(
                "Client reverse decision differs from its job: "
                + ", ".join(mismatches)
            )
        if decision.candidate_result_hash is not None:
            result = ClientReverseCandidateResult.model_validate(
                self.store.read_json(
                    self._reverse_candidate_result_path(job.manifest.round_id)
                )
            )
            self._validate_reverse_candidate_result(job, result)
            if result.result_hash != decision.candidate_result_hash:
                raise ClientRuntimeError(
                    "Client reverse decision uses another candidate result"
                )
            for path, expected_hash in (
                (
                    self._reverse_parent_validation_path(job.manifest.round_id),
                    decision.parent_validation_record_hash,
                ),
                (
                    self._reverse_candidate_validation_path(job.manifest.round_id),
                    decision.candidate_validation_record_hash,
                ),
            ):
                record = ClientValidationRecord.model_validate(
                    self.store.read_json(path)
                )
                if record.record_hash != expected_hash:
                    raise ClientRuntimeError(
                        "Client reverse decision uses another validation record"
                    )

    def _apply_reverse_decision(
        self,
        job: ClientReverseTrainingJob,
        decision: ClientReverseDecision,
    ) -> None:
        if decision.candidate_result_hash is None:
            return
        result = ClientReverseCandidateResult.model_validate(
            self.store.read_json(
                self._reverse_candidate_result_path(job.manifest.round_id)
            )
        )
        if self.model_profile.training_backend != "transformers":
            return
        backend = self._reverse_backend_instance()
        current_state = self._current_reverse_state(job)
        if decision.adapter_promoted:
            if (
                current_state.kind == "peft"
                and current_state.version == result.candidate_adapter_version
                and current_state.state_hash == result.candidate_adapter_hash
            ):
                return
            if (
                current_state.version == result.parent_adapter_version
                and current_state.state_hash == result.parent_adapter_hash
            ):
                backend.promote_candidate(result)
                return
            if current_state.version > result.candidate_adapter_version:
                if self.adapter_store is None:
                    raise ClientRuntimeError("Client adapter store is unavailable")
                archived, _ = self.adapter_store.version(
                    result.candidate_adapter_version
                )
                if archived.checkpoint_hash != result.candidate_adapter_hash:
                    raise ClientRuntimeError(
                        "archived promoted Client candidate hash differs"
                    )
                return
            raise ClientRuntimeError(
                "active Client adapter differs from the promotion decision"
            )
        backend.discard_candidate(result)

    def _complete_reverse_distillation(
        self,
        job: ClientReverseTrainingJob,
    ) -> ClientReverseDecision:
        decision_path = self._reverse_decision_path(job.manifest.round_id)
        if self.store.exists(decision_path):
            decision = ClientReverseDecision.model_validate(
                self.store.read_json(decision_path)
            )
            self._validate_reverse_decision(job, decision)
            self._apply_reverse_decision(job, decision)
            return decision

        active_state = self._current_reverse_state(job)
        active_version = active_state.version
        active_hash = active_state.state_hash
        stale_parent = (
            active_version != job.parent_adapter_version
            or active_hash != job.parent_adapter_hash
        )
        if not job.host_teacher_sample_ids:
            decision = ClientReverseDecision.create(
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                job_hash=job.job_hash,
                host_teacher_sample_count=0,
                parent_adapter_version=job.parent_adapter_version,
                parent_adapter_hash=job.parent_adapter_hash,
                candidate_result_hash=None,
                candidate_adapter_version=None,
                candidate_adapter_hash=None,
                active_adapter_version_before_decision=active_version,
                active_adapter_hash_before_decision=active_hash,
                accepted_adapter_version=active_version,
                accepted_adapter_hash=active_hash,
                parent_validation_record_hash=None,
                candidate_validation_record_hash=None,
                parent_macro_mean_answer_token_ce=None,
                candidate_macro_mean_answer_token_ce=None,
                observed_ce_regression=None,
                quality_non_regression_tolerance=(
                    CLIENT_QUALITY_NON_REGRESSION_TOLERANCE
                ),
                quality_gate_passed=None,
                stale_parent=stale_parent,
                forced_rejection=False,
                adapter_promoted=False,
                decision_reason="no_host_teacher_samples",
                rejected_candidate_discarded=False,
                created_at=utc_text(self.now_fn()),
            )
            self.store.write_json_if_absent(
                decision_path,
                decision.model_dump(mode="json"),
            )
            return decision

        if not job.public_data_partition.validation_sample_ids:
            raise ClientRuntimeError(
                "Client reverse adoption requires a non-empty held-out "
                "public validation split"
            )

        if self.model_profile.training_backend == "transformers":
            self.verify_cached_reference_dataset(job.manifest)
            reference_samples = load_reference_jsonl(
                self.store.path(
                    self._reference_dataset_path(job.manifest.round_id)
                )
            )
            by_id = {sample.sample_id: sample for sample in reference_samples}
            validation_samples = [
                by_id[sample_id]
                for sample_id in job.public_data_partition.validation_sample_ids
            ]
        else:
            validation_samples = []

        result = self._load_or_create_reverse_candidate(job)
        parent_validation = self._load_or_create_reverse_validation(
            job,
            result,
            adapter_role="parent",
            validation_samples=validation_samples,
        )
        candidate_validation = self._load_or_create_reverse_validation(
            job,
            result,
            adapter_role="candidate",
            validation_samples=validation_samples,
        )
        active_state = self._current_reverse_state(job)
        active_version = active_state.version
        active_hash = active_state.state_hash
        stale_parent = (
            active_version != job.parent_adapter_version
            or active_hash != job.parent_adapter_hash
        )
        observed_regression = (
            candidate_validation.macro_mean_answer_token_ce
            - parent_validation.macro_mean_answer_token_ce
        )
        quality_passed = (
            observed_regression <= CLIENT_QUALITY_NON_REGRESSION_TOLERANCE
        )
        forced_rejection = self.force_reverse_validation_failure
        promoted = (
            quality_passed
            and not stale_parent
            and not forced_rejection
        )
        reason = self._reverse_decision_reason(
            stale_parent=stale_parent,
            forced_rejection=forced_rejection,
            quality_passed=quality_passed,
        )
        decision = ClientReverseDecision.create(
            round_id=job.manifest.round_id,
            manifest_hash=job.manifest.manifest_hash,
            job_hash=job.job_hash,
            host_teacher_sample_count=len(job.host_teacher_sample_ids),
            parent_adapter_version=result.parent_adapter_version,
            parent_adapter_hash=result.parent_adapter_hash,
            candidate_result_hash=result.result_hash,
            candidate_adapter_version=result.candidate_adapter_version,
            candidate_adapter_hash=result.candidate_adapter_hash,
            active_adapter_version_before_decision=active_version,
            active_adapter_hash_before_decision=active_hash,
            accepted_adapter_version=(
                result.candidate_adapter_version if promoted else active_version
            ),
            accepted_adapter_hash=(
                result.candidate_adapter_hash if promoted else active_hash
            ),
            parent_validation_record_hash=parent_validation.record_hash,
            candidate_validation_record_hash=candidate_validation.record_hash,
            parent_macro_mean_answer_token_ce=(
                parent_validation.macro_mean_answer_token_ce
            ),
            candidate_macro_mean_answer_token_ce=(
                candidate_validation.macro_mean_answer_token_ce
            ),
            observed_ce_regression=observed_regression,
            quality_non_regression_tolerance=(
                CLIENT_QUALITY_NON_REGRESSION_TOLERANCE
            ),
            quality_gate_passed=quality_passed,
            stale_parent=stale_parent,
            forced_rejection=forced_rejection,
            adapter_promoted=promoted,
            decision_reason=reason,
            rejected_candidate_discarded=not promoted,
            created_at=utc_text(self.now_fn()),
        )
        self.store.write_json_if_absent(
            decision_path,
            decision.model_dump(mode="json"),
        )
        self._apply_reverse_decision(job, decision)
        return decision

    def _require_verified_host_preview_cache(
        self,
        *,
        manifest: RoundManifest,
        host_package: KnowledgePackage,
        host_artifact_path: str | Path,
    ) -> None:
        round_id = manifest.round_id
        job_path = self._reverse_job_path(round_id)
        if not self.store.exists(job_path):
            raise ClientRuntimeError(
                "verified Host Knowledge preview is missing"
            )

        job = ClientReverseTrainingJob.model_validate(
            self.store.read_json(job_path)
        )
        if (
            job.manifest.manifest_hash != manifest.manifest_hash
            or job.host_package_hash != host_package.package_hash
            or job.accepted_host_adapter_version
            != host_package.adapter_version
        ):
            raise ClientRuntimeError(
                "verified Host Knowledge preview binding differs"
            )

        cached_package, cached_artifact_path = self.cached_host_knowledge(
            round_id,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        if cached_package.package_hash != host_package.package_hash:
            raise ClientRuntimeError(
                "cached Host Knowledge Package differs from the verified preview"
            )
        if Path(host_artifact_path).resolve() != cached_artifact_path.resolve():
            raise ClientRuntimeError(
                "Host Knowledge consent must use the verified cached artifact"
            )

    def apply_host_knowledge(
        self,
        *,
        manifest: RoundManifest,
        host_package: KnowledgePackage,
        host_artifact_path: str | Path,
        host_public_key: str,
        expected_host_id: str,
        accepted_host_adapter_version: int,
        adapter_promoted: bool,
        complete_reverse: bool = True,
        require_fresh_host_package: bool = True,
    ) -> dict[str, Any]:
        from shared.fedmkt_core.reverse_integration import (
            ReverseDistillationIntegrationAudit,
            integrate_reverse_distillation,
        )

        round_id = manifest.round_id

        cache_path = self._accepted_package_path(round_id)
        cache_artifact_path = self._accepted_artifact_path(round_id)
        snapshot_path = self._accepted_snapshot_path(round_id)

        if (
            not self.store.exists(cache_path)
            or not self.store.exists(cache_artifact_path)
            or not self.store.exists(snapshot_path)
        ):
            raise ValueError(
                "accepted Client package or adapter snapshot is missing"
            )

        cached = KnowledgePackage.model_validate(
            self.store.read_json(cache_path)
        )
        cached_samples = load_package_samples(
            self.store.path(cache_artifact_path),
            cached,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )
        snapshot = self.store.read_json(snapshot_path)

        if cached.package_hash != snapshot.get("package_hash"):
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

        expected_alignment = manifest.host_package_alignment_profile_id

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

        if require_fresh_host_package:
            created = parse_utc(host_package.created_at)
            skew = abs(
                (self.now_fn() - created).total_seconds()
            )

            if skew > self.maximum_clock_skew_seconds:
                raise ValueError(
                    "Host package timestamp is outside the allowed skew"
                )
        else:
            self._require_verified_host_preview_cache(
                manifest=manifest,
                host_package=host_package,
                host_artifact_path=host_artifact_path,
            )

        host_samples = load_package_samples(
            host_artifact_path,
            host_package,
            maximum_bytes=manifest.maximum_knowledge_package_bytes,
        )

        partition = client_public_data_partition(
            reference_dataset_id=manifest.reference_dataset_id,
            reference_dataset_hash=manifest.reference_dataset_hash,
            sample_ids=manifest.sample_ids,
        )
        if partition.validation_fraction != manifest.client_public_validation_fraction:
            raise ValueError("Client public-data partition differs from the manifest")

        parent_hash = snapshot.get("training_checkpoint_hash") or snapshot.get(
            "state_hash"
        )
        if not isinstance(parent_hash, str) or len(parent_hash) != 64:
            raise ValueError("accepted Client parent snapshot hash is missing")
        parent_version = int(snapshot["adapter_version"])

        transfer_set = set(partition.transfer_sample_ids)
        transfer_client_samples = [
            sample for sample in cached_samples if sample.sample_id in transfer_set
        ]
        labels_by_sample: dict[str, list[int]]
        profile = None
        host_tokenizer = None
        client_tokenizer = None
        mapping_cache = None
        if self.model_profile.training_backend == "mock":
            labels_by_sample = {}
            for sample in transfer_client_samples:
                labels = [-100] * len(sample.source_input_ids)
                labels[-1] = sample.source_input_ids[-1]
                labels_by_sample[sample.sample_id] = labels
        else:
            self.verify_cached_reference_dataset(manifest)
            reference_samples = load_reference_jsonl(
                self.store.path(self._reference_dataset_path(round_id))
            )
            reference_by_id = {sample.sample_id: sample for sample in reference_samples}
            selected_reference = [
                reference_by_id[sample_id]
                for sample_id in partition.transfer_sample_ids
            ]
            profile = resolve_alignment_profile(
                manifest.alignment_profile_id_for(self.client_id)
            )
            tokenizers = self.ensure_alignment_tokenizers(manifest)
            if tokenizers is None:
                raise ClientRuntimeError(
                    "real reverse distillation requires real alignment tokenizers"
                )
            client_tokenizer, host_tokenizer = tokenizers
            encoded = encode_reference_samples(
                selected_reference,
                tokenizer=client_tokenizer.tokenizer,
                model_profile=self.model_profile,
                maximum_sequence_length=manifest.maximum_sequence_length,
                expected_sample_ids=partition.transfer_sample_ids,
                dataset_label="signed Client transfer split",
            )
            cached_transfer_by_id = {
                sample.sample_id: sample for sample in transfer_client_samples
            }
            for sample in encoded:
                cached_sample = cached_transfer_by_id[sample.sample_id]
                if (
                    sample.input_ids != cached_sample.source_input_ids
                    or sum(sample.attention_mask)
                    != cached_sample.attention_length
                ):
                    raise ClientRuntimeError(
                        "accepted Client package differs from the freshly "
                        "encoded transfer split"
                    )
            labels_by_sample = {
                sample.sample_id: sample.labels for sample in encoded
            }
            mapping_cache = VocabularyMappingCache(
                self.store.path("vocabulary_mappings")
            )

        batch = integrate_reverse_distillation(
            client_id=self.client_id,
            parent_adapter_version=parent_version,
            parent_adapter_hash=parent_hash,
            host_adapter_promoted=adapter_promoted,
            partition_hash=partition.partition_hash,
            transfer_sample_ids=partition.transfer_sample_ids,
            host_package=host_package,
            host_samples=host_samples,
            client_package=cached,
            client_samples=cached_samples,
            labels_by_sample=labels_by_sample,
            profile=profile,
            host_tokenizer=host_tokenizer,
            client_tokenizer=client_tokenizer,
            mapping_cache=mapping_cache,
        )

        job_path = self._reverse_job_path(round_id)
        artifact_path = self._reverse_artifact_path(round_id)
        audit_path = self._reverse_audit_path(round_id)
        partition_path = self._reverse_partition_path(round_id)
        reverse_exists = tuple(
            self.store.exists(path)
            for path in (job_path, artifact_path, audit_path, partition_path)
        )
        if any(reverse_exists) and not all(reverse_exists):
            raise ClientRuntimeError("Client reverse-training job cache is incomplete")

        pad_token_id = 0 if profile is None else profile.client.pad_token_id
        if pad_token_id is None:
            raise ClientRuntimeError("Client tokenizer profile has no padding token")
        vocabulary_size = (
            max(
                int(batch.input_ids.max().item()),
                int(batch.sparse_targets.token_ids.max().item()),
            ) + 1
            if profile is None
            else profile.client.vocabulary_size
        )
        if all(reverse_exists):
            job = ClientReverseTrainingJob.model_validate(
                self.store.read_json(job_path)
            )
            if (
                job.manifest.manifest_hash != manifest.manifest_hash
                or job.parent_adapter_version != parent_version
                or job.parent_adapter_hash != parent_hash
                or job.host_package_hash != host_package.package_hash
                or job.host_adapter_promoted != adapter_promoted
                or job.public_data_partition != partition
                or job.integration_audit_hash != batch.audit.audit_hash
            ):
                raise ClientRuntimeError("Client reverse-training job is immutable")
            stored_audit = ReverseDistillationIntegrationAudit.model_validate(
                self.store.read_json(audit_path)
            )
            stored_partition = ClientPublicDataPartition.model_validate(
                self.store.read_json(partition_path)
            )
            if stored_audit != batch.audit or stored_partition != partition:
                raise ClientRuntimeError("Client reverse-training audit is immutable")
            load_client_reverse_training_artifact(
                self.store.path(artifact_path),
                job.artifact,
                partition.transfer_sample_ids,
                maximum_bytes=manifest.maximum_client_reverse_training_job_bytes,
                vocabulary_size=vocabulary_size,
            )
        else:
            try:
                artifact = write_client_reverse_training_artifact(
                    self.store.path(artifact_path),
                    batch,
                    pad_token_id=pad_token_id,
                    maximum_bytes=manifest.maximum_client_reverse_training_job_bytes,
                )
                job = ClientReverseTrainingJob.create(
                    manifest=manifest,
                    client_id=self.client_id,
                    client_model_profile_hash=self.model_profile.profile_hash(),
                    parent_adapter_version=parent_version,
                    parent_adapter_hash=parent_hash,
                    accepted_host_adapter_version=host_package.adapter_version,
                    host_package_hash=host_package.package_hash,
                    host_adapter_promoted=adapter_promoted,
                    public_data_partition=partition,
                    host_teacher_sample_ids=batch.audit.host_teacher_sample_ids,
                    client_public_data_epochs=manifest.client_public_data_epochs,
                    distillation=manifest.distillation,
                    integration_audit_hash=batch.audit.audit_hash,
                    artifact=artifact,
                    created_at=utc_text(self.now_fn()),
                )
                self.store.write_json_if_absent(
                    audit_path, batch.audit.model_dump(mode="json")
                )
                self.store.write_json_if_absent(
                    partition_path, partition.model_dump(mode="json")
                )
                self.store.write_json_if_absent(
                    job_path, job.model_dump(mode="json")
                )
            except Exception:
                if not self.store.exists(job_path):
                    self.store.delete(artifact_path)
                    self.store.delete(audit_path)
                    self.store.delete(partition_path)
                raise

        host_cache_package = self._host_package_path(round_id)
        host_cache_artifact = self._host_artifact_path(round_id)
        host_cache_exists = (
            self.store.exists(host_cache_package),
            self.store.exists(host_cache_artifact),
        )
        if any(host_cache_exists) and not all(host_cache_exists):
            raise ValueError("Host Knowledge Package cache is incomplete")
        if all(host_cache_exists):
            existing = KnowledgePackage.model_validate(
                self.store.read_json(host_cache_package)
            )
            if existing.package_hash != host_package.package_hash:
                raise ValueError("Host Knowledge Package cache is immutable")
            load_package_samples(
                self.store.path(host_cache_artifact),
                existing,
                maximum_bytes=manifest.maximum_knowledge_package_bytes,
            )
        else:
            try:
                self.store.copy_file_if_absent(
                    host_cache_artifact,
                    host_artifact_path,
                )
                self.store.write_json_if_absent(
                    host_cache_package,
                    host_package.model_dump(mode="json"),
                )
            except Exception:
                if not self.store.exists(host_cache_package):
                    self.store.delete(host_cache_artifact)
                raise

        if not complete_reverse:
            return {
                "round_id": round_id,
                "host_adapter_version": host_package.adapter_version,
                "host_teacher_sample_count": len(batch.audit.host_teacher_sample_ids),
                "host_teacher_sample_ids": list(batch.audit.host_teacher_sample_ids),
                "requires_consent": bool(batch.audit.host_teacher_sample_ids),
                "reverse_job_hash": job.job_hash,
            }

        decision = self._complete_reverse_distillation(job)
        state = self.state()
        if self.model_profile.training_backend == "transformers":
            current_state = self._current_reverse_state(job)
            state["candidate_adapter_version"] = current_state.version
            state["training_adapter_version"] = current_state.version
            state["training_checkpoint_hash"] = current_state.checkpoint_hash
            if self.model_profile.serving_backend in {"mock", "transformers"}:
                state["serving_adapter_version"] = current_state.version
        elif decision.adapter_promoted:
            if (
                int(state["candidate_adapter_version"])
                == decision.parent_adapter_version
            ):
                state["candidate_adapter_version"] = (
                    decision.accepted_adapter_version
                )
                state["training_adapter_version"] = (
                    decision.accepted_adapter_version
                )
                state["training_checkpoint_hash"] = (
                    decision.accepted_adapter_hash
                )
                state["serving_adapter_version"] = (
                    decision.accepted_adapter_version
                )
        state["last_completed_round"] = round_id
        state["last_host_adapter_version"] = host_package.adapter_version
        state["host_distillation_samples"] = batch.audit.host_teacher_sample_ids
        state["last_reverse_training_job_hash"] = job.job_hash
        state["last_reverse_candidate_result_hash"] = (
            decision.candidate_result_hash
        )
        state["last_reverse_decision_hash"] = decision.decision_hash
        state["last_reverse_decision_reason"] = decision.decision_reason
        state["last_reverse_consent"] = "accepted"

        self.store.write_json("state.json", state)

        return state

    def cached_host_knowledge(
        self,
        round_id: str,
        *,
        maximum_bytes: int,
    ) -> tuple[KnowledgePackage, Path]:
        package_path = self._host_package_path(round_id)
        artifact_path = self._host_artifact_path(round_id)
        if not self.store.exists(package_path) or not self.store.exists(artifact_path):
            raise ClientRuntimeError("cached Host Knowledge Package is missing")
        package = KnowledgePackage.model_validate(self.store.read_json(package_path))
        path = self.store.path(artifact_path)
        load_package_samples(path, package, maximum_bytes=maximum_bytes)
        return package, path

    def decline_host_knowledge(self, round_id: str) -> dict[str, Any]:
        job_path = self._reverse_job_path(round_id)
        if not self.store.exists(job_path):
            raise ClientRuntimeError("reverse Host Knowledge preview is missing")
        job = ClientReverseTrainingJob.model_validate(self.store.read_json(job_path))
        state = self.state()
        state["last_completed_round"] = round_id
        state["last_host_adapter_version"] = job.accepted_host_adapter_version
        state["host_distillation_samples"] = list(job.host_teacher_sample_ids)
        state["last_reverse_training_job_hash"] = job.job_hash
        state["last_reverse_candidate_result_hash"] = None
        state["last_reverse_decision_hash"] = None
        state["last_reverse_decision_reason"] = (
            "no_host_teacher_samples"
            if not job.host_teacher_sample_ids
            else "user_declined"
        )
        state["last_reverse_consent"] = (
            "not_needed" if not job.host_teacher_sample_ids else "declined"
        )
        self.store.write_json("state.json", state)
        return state

    async def ollama_compatibility(self) -> dict[str, Any]:
        supported = list(supported_ollama_models())
        if self.model_profile.training_backend != "transformers":
            return {
                "required": False,
                "compatible": True,
                "selected_model": None,
                "supported_models": supported,
                "installed_models": [],
            }
        selected = ollama_model_for_profile(self.model_profile.profile_id)
        if self.ollama is None:
            return {
                "required": True,
                "compatible": False,
                "selected_model": selected,
                "supported_models": supported,
                "installed_models": [],
                "error": "Ollama is unavailable",
            }
        try:
            values = await self.ollama.list_models()
        except Exception as exc:
            return {
                "required": True,
                "compatible": False,
                "selected_model": selected,
                "supported_models": supported,
                "installed_models": [],
                "error": str(exc),
            }
        installed: list[str] = []
        for value in values:
            for key in ("name", "model"):
                name = value.get(key)
                if isinstance(name, str) and name not in installed:
                    installed.append(name)
        compatible = selected in installed
        result: dict[str, Any] = {
            "required": True,
            "compatible": compatible,
            "selected_model": selected,
            "supported_models": supported,
            "installed_models": installed,
        }
        if not compatible:
            result["error"] = (
                "Error: currently used model is not supported or is not installed.\n"
                "Supported models include:\n- " + "\n- ".join(supported)
            )
        return result

    def generate_transformers(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        if self.model_profile.training_backend != "transformers":
            raise ClientRuntimeError(
                "Transformers serving requires a real Client profile"
            )
        from client.peft_backend import TransformersPeftTrainingBackend

        backend = TransformersPeftTrainingBackend(
            data_dir=self.store.root,
            model_profile=self.model_profile,
            execution_profile=self.training_execution_profile,
            knowledge_batch_size=self.knowledge_batch_size,
        )
        return backend.generate_text(messages, max_new_tokens)

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        if self.model_profile.serving_backend == "transformers":
            return self.generate_transformers(
                [{"role": "user", "content": prompt}],
                max_new_tokens,
            )
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
