from __future__ import annotations

import gc
import math
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from host.training import (
    HostAdapterInitializationRecord,
    HostValidationRecord,
    HostValidationSampleMetric,
    GraniteHostTrainingContract,
    HostTrainingExecutionProfile,
)
from shared.answer_only import AnswerOnlyCollator
from shared.adapter_checkpoint import (
    AdapterCheckpointMetadata,
    AdapterCheckpointStore,
    write_atomic_json,
)
from shared.alignment_profiles import POC_DTW_PROFILE
from shared.distillation_artifact import TENSOR_NAMES, load_host_training_artifact
from shared.protocol import (
    HostCandidateTrainingResult,
    HostTrainingJob,
    KnowledgeSample,
    ModelProfile,
    RoundManifest,
    utc_text,
)
from shared.reference_dataset import (
    ReferenceDatasetIdentity,
    ReferenceSample,
    reference_dataset_identity,
)
from shared.reference_knowledge import (
    FedMKTGenerationArguments,
    encode_reference_samples,
    knowledge_sample_from_rows,
)
from shared.tokenizer_validation import load_pinned_tokenizer


HOST_INITIALIZATION_RECORD = "host_initialization.json"
HOST_INITIALIZATION_PROBE = "LegalFedLLM Host adapter initialization probe."
HOST_LOSS_SEQUENCE_CHUNK_SIZE = 64


@dataclass(frozen=True, slots=True)
class InitializedHostAdapter:
    metadata: AdapterCheckpointMetadata
    checkpoint_path: Path
    initialization: HostAdapterInitializationRecord


@dataclass(frozen=True, slots=True)
class HostBaselineResult:
    knowledge_samples: list[KnowledgeSample]
    validation: HostValidationRecord


class _TensorRowDataset:
    def __init__(self, tensors: dict[str, Any]):
        self.tensors = tensors
        self.length = int(next(iter(tensors.values())).shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {name: value[index] for name, value in self.tensors.items()}


def collate_host_training_rows(
    torch: Any,
    features: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not features:
        raise ValueError("Host training batch is empty")
    if any(set(feature) != TENSOR_NAMES for feature in features):
        raise ValueError("Host training rows contain unexpected tensors")

    batch = {
        name: torch.stack([feature[name] for feature in features])
        for name in sorted(TENSOR_NAMES)
    }
    attention_mask = batch["attention_mask"]
    if attention_mask.ndim != 2:
        raise ValueError("Host attention mask must have [batch, sequence] shape")
    active = attention_mask.bool().any(dim=0)
    active_positions = torch.nonzero(active, as_tuple=False)
    if active_positions.numel() == 0:
        raise ValueError("Host training batch has no attended token")
    sequence_length = int(active_positions[-1].item()) + 1

    for name in ("input_ids", "attention_mask", "labels"):
        batch[name] = batch[name][:, :sequence_length]
    for name in (
        "sparse_target_token_ids",
        "sparse_target_probabilities",
        "sparse_target_valid_mask",
    ):
        batch[name] = batch[name][:, :sequence_length, :]
    return batch


class _TrimmedHostTrainingCollator:
    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return collate_host_training_rows(torch, features)


def selective_host_loss(
    torch: Any,
    logits: Any,
    inputs: dict[str, Any],
    *,
    sequence_chunk_size: int = HOST_LOSS_SEQUENCE_CHUNK_SIZE,
) -> tuple[Any, Any, Any]:
    from shared.fedmkt_core.ml.sparse_targets import (
        SparseTargetBatch,
        SparseTargetError,
        validate_sparse_target_batch,
    )

    if type(sequence_chunk_size) is not int or sequence_chunk_size < 1:
        raise ValueError("Host loss sequence chunk size must be positive")
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    if labels.shape != logits.shape[:2] or attention_mask.shape != labels.shape:
        raise SparseTargetError(
            "labels and attention mask must match model batch and sequence shape"
        )
    if labels.device != logits.device or attention_mask.device != logits.device:
        raise SparseTargetError(
            "model logits, labels and attention mask must share one device"
        )
    targets = SparseTargetBatch(
        token_ids=inputs["sparse_target_token_ids"].long(),
        probabilities=inputs["sparse_target_probabilities"].to(
            dtype=logits.dtype
        ),
        valid_mask=inputs["sparse_target_valid_mask"].bool(),
    )
    validate_sparse_target_batch(targets, vocab_size=logits.shape[-1])
    if logits.shape[:2] != targets.token_ids.shape[:2]:
        raise SparseTargetError(
            "model logits and sparse targets differ in batch or sequence shape"
        )
    if logits.device != targets.token_ids.device:
        raise SparseTargetError(
            "model logits and sparse targets must share one device"
        )

    shifted_labels = labels[..., 1:]
    shifted_attention = attention_mask[..., 1:].bool()
    supervised_mask = shifted_labels.ne(-100)
    distillation_mask = supervised_mask & shifted_attention
    supervised_positions = supervised_mask.sum()
    distillation_positions = distillation_mask.sum()
    if supervised_positions == 0:
        raise SparseTargetError(
            "answer-only supervision requires at least one supervised target token"
        )
    if distillation_positions == 0:
        raise SparseTargetError(
            "answer-only distillation requires at least one supervised target token"
        )

    supervised_sum = None
    distillation_sum = None
    shifted_sequence_length = logits.shape[1] - 1
    for start in range(0, shifted_sequence_length, sequence_chunk_size):
        end = min(start + sequence_chunk_size, shifted_sequence_length)
        chunk_logits = logits[:, start:end, :]
        normalizer = torch.logsumexp(chunk_logits, dim=-1)

        chunk_supervised_mask = supervised_mask[:, start:end]
        chunk_labels = shifted_labels[:, start:end]
        safe_labels = torch.where(
            chunk_supervised_mask,
            chunk_labels,
            torch.zeros_like(chunk_labels),
        )
        selected_labels = torch.gather(
            chunk_logits,
            -1,
            safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        chunk_supervised = (
            (normalizer - selected_labels)
            * chunk_supervised_mask.to(normalizer.dtype)
        ).sum()
        supervised_sum = (
            chunk_supervised
            if supervised_sum is None
            else supervised_sum + chunk_supervised
        )

        chunk_token_ids = targets.token_ids[:, start:end, :]
        chunk_probabilities = targets.probabilities[:, start:end, :]
        chunk_valid_mask = targets.valid_mask[:, start:end, :]
        selected_sparse_logits = torch.gather(
            chunk_logits,
            -1,
            chunk_token_ids,
        )
        selected_sparse_log_probabilities = (
            selected_sparse_logits - normalizer.unsqueeze(-1)
        )
        per_position_distillation = (
            -chunk_probabilities * selected_sparse_log_probabilities
            * chunk_valid_mask.to(chunk_probabilities.dtype)
        ).sum(dim=-1)
        chunk_distillation_mask = distillation_mask[:, start:end]
        chunk_distillation = (
            per_position_distillation
            * chunk_distillation_mask.to(per_position_distillation.dtype)
        ).sum()
        distillation_sum = (
            chunk_distillation
            if distillation_sum is None
            else distillation_sum + chunk_distillation
        )

    if supervised_sum is None or distillation_sum is None:
        raise SparseTargetError("Host loss requires at least two sequence tokens")
    supervised = supervised_sum / supervised_positions.to(supervised_sum.dtype)
    distillation = distillation_sum / distillation_positions.to(
        distillation_sum.dtype
    )
    return 0.9 * supervised + 0.1 * distillation, supervised, distillation


def _selective_host_trainer_class(transformers: Any, torch: Any) -> type:
    class SelectiveHostTrainer(transformers.Trainer):
        supervised_losses: list[float]
        distillation_losses: list[float]

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.supervised_losses = []
            self.distillation_losses = []

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            attention_mask = inputs["attention_mask"]
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            loss, supervised, distillation = selective_host_loss(
                torch,
                logits,
                inputs,
            )
            self.supervised_losses.append(float(supervised.detach().float()))
            self.distillation_losses.append(
                float(distillation.detach().float())
            )
            return (loss, outputs) if return_outputs else loss

    return SelectiveHostTrainer


class TransformersPeftHostBackend:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        model_profile: ModelProfile,
        execution_profile: HostTrainingExecutionProfile,
        knowledge_batch_size: int | None = None,
    ):
        if model_profile.training_backend != "transformers":
            raise ValueError(
                "real Host adapter lifecycle requires a Transformers profile"
            )
        self.data_dir = Path(data_dir).resolve()
        self.model_profile = model_profile
        self.execution_profile = execution_profile
        self.contract = GraniteHostTrainingContract.create(model_profile)
        self.knowledge_batch_size = (
            int(os.getenv("HOST_KNOWLEDGE_BATCH_SIZE", "1"))
            if knowledge_batch_size is None
            else knowledge_batch_size
        )
        if self.knowledge_batch_size < 1:
            raise ValueError("Host knowledge batch size must be positive")
        self.checkpoints = AdapterCheckpointStore(
            self.data_dir / "adapters",
            model_profile,
        )

    def initialize_adapter(self) -> InitializedHostAdapter:
        current = self.checkpoints.current()
        if current is not None:
            metadata, path = current
            initial_metadata, initial_path = self.checkpoints.version(0)
            record = self._load_initialization_record(initial_path)
            self._validate_initialization(initial_metadata, record)
            if metadata.version > 0:
                self._validate_promoted_adapter(metadata)
            return InitializedHostAdapter(metadata, path, record)

        torch, transformers, peft, safetensors = self._dependencies()
        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        validated_tokenizer = load_pinned_tokenizer(
            POC_DTW_PROFILE.host,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer = validated_tokenizer.tokenizer
        probe_ids = tokenizer.encode(
            HOST_INITIALIZATION_PROBE,
            add_special_tokens=True,
        )
        if not probe_ids:
            raise RuntimeError("Host initialization probe produced no tokens")
        probe_ids = list(probe_ids[:32])

        staging_path: Path | None = None
        base_model = None
        model = None
        reloaded_base = None
        reloaded = None
        try:
            base_model = self._load_base_model(torch, transformers)
            base_logits = self._probe_logits(torch, base_model, probe_ids)
            model = self._create_adapter(peft, base_model)
            trainable_count, total_count = self._assert_trainable_parameters(
                model
            )
            initialized_logits = self._probe_logits(torch, model, probe_ids)
            self._require_same_logits(
                torch,
                base_logits,
                initialized_logits,
                label="fresh Host LoRA is not zero-effect",
                absolute_tolerance=1e-6,
            )

            staging_path = self.checkpoints.staging_path(
                f"initial-{secrets.token_hex(8)}"
            )
            model.save_pretrained(
                staging_path,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            self._validate_adapter_tensors(safetensors, staging_path)

            del model, base_model
            model = None
            base_model = None
            self._release_memory(torch)

            reloaded_base = self._load_base_model(torch, transformers)
            reloaded = peft.PeftModel.from_pretrained(
                reloaded_base,
                staging_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(reloaded)
            if any(parameter.requires_grad for parameter in reloaded.parameters()):
                raise RuntimeError(
                    "reloaded initial Host adapter has trainable parameters"
                )
            reloaded_logits = self._probe_logits(torch, reloaded, probe_ids)
            self._require_same_logits(
                torch,
                initialized_logits,
                reloaded_logits,
                label="reloaded Host adapter changed the initialization probe",
                absolute_tolerance=1e-4,
            )

            record = HostAdapterInitializationRecord.create(
                contract=self.contract,
                contract_hash=self.contract.contract_hash,
                model_profile_hash=self.model_profile.profile_hash(),
                execution_profile=self.execution_profile,
                execution_profile_hash=self.execution_profile.profile_hash(),
                adapter_version=0,
                initialization_policy=self.contract.initial_adapter_policy,
                zero_effect_verified=True,
                reload_verified=True,
                trainable_parameter_count=trainable_count,
                total_parameter_count=total_count,
                dependency_versions={
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                    "safetensors": safetensors.__version__,
                    "cuda": torch.version.cuda or "none",
                },
                created_at=utc_text(),
            )
            write_atomic_json(
                staging_path / HOST_INITIALIZATION_RECORD,
                record.model_dump(mode="json"),
            )
            metadata = self.checkpoints.seal(
                staging_path,
                version=0,
                parent=None,
                round_id=None,
                manifest_hash=None,
                execution_profile_hash=self.execution_profile.profile_hash(),
            )
            checkpoint_path = self.checkpoints.promote(
                staging_path,
                metadata,
            )
            staging_path = None
            self._validate_initialization(metadata, record)
            return InitializedHostAdapter(
                metadata,
                checkpoint_path,
                record,
            )
        except Exception:
            if staging_path is not None and staging_path.exists():
                self.checkpoints.discard_staging(staging_path)
            raise
        finally:
            del reloaded, reloaded_base, model, base_model
            self._release_memory(torch)

    def train_candidate(
        self,
        job: HostTrainingJob,
        artifact_path: str | Path,
    ) -> HostCandidateTrainingResult:
        torch, transformers, peft, safetensors = self._dependencies()
        self._validate_device(torch)
        if job.host_model_profile_hash != self.model_profile.profile_hash():
            raise ValueError("Host training job uses another model profile")
        if job.host_public_data_epochs != self.execution_profile.public_data_epochs:
            raise ValueError("Host training job epochs differ from execution profile")
        if (
            job.distillation.loss_type != "ce"
            or job.distillation.temperature != 1.0
            or job.distillation.lm_loss_weight != 0.9
        ):
            raise ValueError("Host training job uses an unsupported loss contract")

        current = self.checkpoints.current()
        if current is None:
            raise RuntimeError("current Host adapter checkpoint is missing")
        parent_metadata, parent_path = current
        if parent_metadata.version != job.host_adapter_version:
            raise RuntimeError("Host adapter changed before candidate training")

        arrays = load_host_training_artifact(
            artifact_path,
            job.artifact,
            job.sample_ids,
            maximum_bytes=job.manifest.maximum_host_training_job_bytes,
            vocabulary_size=int(self.model_profile.vocabulary_size or 0),
        )
        tensor_values = {}
        for name, value in arrays.items():
            tensor = torch.from_numpy(value)
            if name in {"input_ids", "labels", "sparse_target_token_ids"}:
                tensor = tensor.long()
            elif name == "sparse_target_probabilities":
                tensor = tensor.float()
            elif name == "sparse_target_valid_mask":
                tensor = tensor.bool()
            tensor_values[name] = tensor
        dataset = _TensorRowDataset(tensor_values)
        transformers.set_seed(self.execution_profile.seed)
        job_id = f"host-train-{secrets.token_hex(8)}"
        output_dir = self.data_dir / "training_jobs" / job_id
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_path: Path | None = None
        base_model = None
        model = None
        trainer = None
        reloaded_base = None
        reloaded = None
        try:
            base_model = self._load_base_model(torch, transformers)
            model = peft.PeftModel.from_pretrained(
                base_model,
                parent_path,
                is_trainable=True,
            )
            self._verify_loaded_adapter(model)
            trainable_count, total_count = self._assert_trainable_parameters(model)
            frozen_checksum = self._frozen_parameter_checksum(torch, model)
            if self.execution_profile.gradient_checkpointing:
                model.config.use_cache = False
                model.enable_input_require_grads()

            profile = self.execution_profile
            arguments = transformers.TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=job.host_public_data_epochs,
                per_device_train_batch_size=profile.micro_batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                learning_rate=profile.learning_rate,
                lr_scheduler_type=profile.learning_rate_scheduler,
                warmup_ratio=profile.warmup_ratio,
                optim=profile.optimizer,
                adam_beta1=profile.adam_beta1,
                adam_beta2=profile.adam_beta2,
                weight_decay=profile.weight_decay,
                max_grad_norm=profile.maximum_gradient_norm,
                seed=profile.seed,
                data_seed=profile.seed,
                bf16=profile.precision == "bfloat16",
                fp16=profile.precision == "float16",
                use_cpu=profile.device == "cpu",
                gradient_checkpointing=profile.gradient_checkpointing,
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="steps",
                logging_steps=1,
                report_to=[],
                remove_unused_columns=False,
                dataloader_num_workers=profile.dataloader_num_workers,
                dataloader_pin_memory=profile.device == "cuda",
            )
            trainer_class = _selective_host_trainer_class(transformers, torch)
            trainer = trainer_class(
                model=model,
                args=arguments,
                train_dataset=dataset,
                data_collator=_TrimmedHostTrainingCollator(),
            )
            train_output = trainer.train()
            optimizer_step_count = int(trainer.state.global_step)
            optimizer_loss = float(train_output.training_loss)
            if optimizer_step_count < 1:
                raise RuntimeError("Host training completed no optimizer step")
            if not math.isfinite(optimizer_loss) or optimizer_loss < 0:
                raise RuntimeError("Host training produced an invalid optimizer loss")
            if not trainer.supervised_losses or not trainer.distillation_losses:
                raise RuntimeError("Host training did not evaluate both loss terms")
            supervised_loss = math.fsum(trainer.supervised_losses) / len(
                trainer.supervised_losses
            )
            distillation_loss = math.fsum(trainer.distillation_losses) / len(
                trainer.distillation_losses
            )
            if not math.isfinite(supervised_loss) or supervised_loss < 0:
                raise RuntimeError("Host supervised loss is invalid")
            if not math.isfinite(distillation_loss) or distillation_loss < 0:
                raise RuntimeError("Host distillation loss is invalid")

            model = trainer.accelerator.unwrap_model(
                trainer.model_wrapped,
                keep_fp32_wrapper=False,
            )
            trainable_count, total_count = self._assert_trainable_parameters(model)
            if self._frozen_parameter_checksum(torch, model) != frozen_checksum:
                raise RuntimeError(
                    "a frozen Host base-model parameter changed during training"
                )
            first_length = int(arrays["attention_mask"][0].sum())
            probe_ids = arrays["input_ids"][0, :first_length].tolist()[:32]
            expected_logits = self._probe_logits(torch, model, probe_ids)

            candidate_path = self.checkpoints.staging_path(job_id)
            model.save_pretrained(
                candidate_path,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            self._validate_adapter_tensors(safetensors, candidate_path)
            self._assert_lora_tensors_changed(
                torch,
                safetensors,
                parent_path,
                candidate_path,
            )
            candidate_metadata = self.checkpoints.seal(
                candidate_path,
                version=parent_metadata.version + 1,
                parent=parent_metadata,
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                execution_profile_hash=profile.profile_hash(),
            )

            del trainer, model, base_model
            trainer = None
            model = None
            base_model = None
            self._release_memory(torch)
            reloaded_base = self._load_base_model(torch, transformers)
            reloaded = peft.PeftModel.from_pretrained(
                reloaded_base,
                candidate_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(reloaded)
            if any(parameter.requires_grad for parameter in reloaded.parameters()):
                raise RuntimeError("reloaded Host candidate is trainable")
            actual_logits = self._probe_logits(torch, reloaded, probe_ids)
            self._require_same_logits(
                torch,
                expected_logits,
                actual_logits,
                label="reloaded Host candidate changed the training probe",
                absolute_tolerance=1e-4,
            )
            self.checkpoints.store_candidate(candidate_path, candidate_metadata)
            candidate_path = None
            return HostCandidateTrainingResult.create(
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                job_hash=job.job_hash,
                parent_adapter_version=parent_metadata.version,
                parent_adapter_hash=parent_metadata.checkpoint_hash,
                candidate_adapter_version=candidate_metadata.version,
                candidate_adapter_hash=candidate_metadata.checkpoint_hash,
                host_model_profile_hash=self.model_profile.profile_hash(),
                execution_profile_hash=profile.profile_hash(),
                host_public_data_epochs=job.host_public_data_epochs,
                supervised_loss_weight=0.9,
                distillation_loss_weight=0.1,
                loss_type="ce",
                temperature=1.0,
                optimizer_step_count=optimizer_step_count,
                optimizer_loss=optimizer_loss,
                supervised_answer_loss=supervised_loss,
                distillation_answer_loss=distillation_loss,
                trainable_parameter_count=trainable_count,
                total_parameter_count=total_count,
                lora_tensors_changed=True,
                frozen_base_unchanged=True,
                reload_verified=True,
                dependency_versions={
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                    "safetensors": safetensors.__version__,
                    "cuda": torch.version.cuda or "none",
                },
                created_at=utc_text(),
            )
        except Exception:
            if candidate_path is not None and candidate_path.exists():
                self.checkpoints.discard_staging(candidate_path)
            raise
        finally:
            del reloaded, reloaded_base, trainer, model, base_model
            shutil.rmtree(output_dir, ignore_errors=True)
            self._release_memory(torch)

    def validate_candidate(
        self,
        result: HostCandidateTrainingResult,
    ) -> Path:
        metadata, path = self.checkpoints.candidate(
            result.round_id,
            result.candidate_adapter_version,
        )
        expected = {
            "round_id": result.round_id,
            "manifest_hash": result.manifest_hash,
            "version": result.candidate_adapter_version,
            "checkpoint_hash": result.candidate_adapter_hash,
            "parent_version": result.parent_adapter_version,
            "parent_checkpoint_hash": result.parent_adapter_hash,
            "profile_hash": result.host_model_profile_hash,
            "execution_profile_hash": result.execution_profile_hash,
        }
        payload = metadata.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ValueError(
                "stored Host candidate differs from its result: "
                + ", ".join(mismatches)
            )
        return path

    def promote_candidate(
        self,
        result: HostCandidateTrainingResult,
    ) -> InitializedHostAdapter:
        metadata, path = self.checkpoints.promote_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )
        self._validate_promoted_adapter(metadata)
        initial_metadata, initial_path = self.checkpoints.version(0)
        initialization = self._load_initialization_record(initial_path)
        self._validate_initialization(initial_metadata, initialization)
        return InitializedHostAdapter(metadata, path, initialization)

    def discard_candidate(
        self,
        result: HostCandidateTrainingResult,
    ) -> None:
        self.checkpoints.discard_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )

    def generate_baseline(
        self,
        reference_samples: Sequence[ReferenceSample],
        validation_samples: Sequence[ReferenceSample],
        validation_identity: ReferenceDatasetIdentity,
        manifest: RoundManifest,
        *,
        expected_adapter_version: int,
        expected_checkpoint_hash: str,
    ) -> HostBaselineResult:
        torch, transformers, peft, _ = self._dependencies()
        from shared.fedmkt_core.ml.logit_generation import (
            generate_pub_data_logits,
        )
        from shared.fedmkt_core.ml.vars_define import (
            FULL_LOGSUMEXP,
            GOLD_TOKEN_IDS,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
        )

        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        validated_tokenizer = load_pinned_tokenizer(
            POC_DTW_PROFILE.host,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer = validated_tokenizer.tokenizer
        if tokenizer.pad_token_id is None:
            raise RuntimeError("pinned Host tokenizer has no padding token")

        reference_values = list(reference_samples)
        validation_values = list(validation_samples)
        actual_validation_identity = reference_dataset_identity(
            validation_values
        )
        if actual_validation_identity != validation_identity:
            raise ValueError("Host validation data differs from its identity")

        encoded_reference = encode_reference_samples(
            reference_values,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=manifest.sample_ids,
        )
        encoded_validation = encode_reference_samples(
            validation_values,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=[
                sample.sample_id for sample in validation_values
            ],
            dataset_label="private D^V",
        )
        vocabulary_size = int(self.model_profile.vocabulary_size or 0)
        if manifest.top_k > vocabulary_size:
            raise ValueError("manifest top_k exceeds the Host vocabulary size")

        current = self.checkpoints.current()
        if current is None:
            raise RuntimeError("current Host adapter checkpoint is missing")
        metadata, checkpoint_path = current
        if (
            metadata.version != expected_adapter_version
            or metadata.checkpoint_hash != expected_checkpoint_hash
        ):
            raise RuntimeError(
                "current Host checkpoint differs from the round record"
            )
        if metadata.profile_hash != self.model_profile.profile_hash():
            raise RuntimeError("current Host checkpoint uses another profile")

        base_model = None
        model = None
        try:
            base_model = self._load_base_model(torch, transformers)
            model = peft.PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(model)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError(
                    "Host baseline inference loaded trainable parameters"
                )
            model.eval()

            collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
            arguments = FedMKTGenerationArguments(
                top_k_logits_keep=manifest.top_k
            )
            generated: list[KnowledgeSample] = []
            for start in range(
                0,
                len(encoded_reference),
                self.knowledge_batch_size,
            ):
                batch_values = encoded_reference[
                    start : start + self.knowledge_batch_size
                ]
                result = generate_pub_data_logits(
                    self._generation_inputs(batch_values),
                    model,
                    arguments,
                    collator,
                )
                token_ids = result[PER_STEP_INDICES]
                logits = result[PER_STEP_LOGITS]
                full_logsumexp = result[FULL_LOGSUMEXP]
                gold_token_ids = result[GOLD_TOKEN_IDS]
                gold_token_logits = result[GOLD_TOKEN_LOGITS]
                gold_token_nll = result[GOLD_TOKEN_NLL]
                losses = result[METRIC]
                self._validate_fedmkt_batch(
                    len(batch_values),
                    token_ids,
                    logits,
                    losses,
                )
                for index, item in enumerate(batch_values):
                    generated.append(
                        knowledge_sample_from_rows(
                            item,
                            top_k_token_ids=token_ids[index].tolist(),
                            top_k_logits=logits[index].tolist(),
                            full_logsumexp=full_logsumexp[index].tolist(),
                            gold_token_ids=gold_token_ids[index].tolist(),
                            gold_token_logits=gold_token_logits[index].tolist(),
                            gold_token_nll=gold_token_nll[index].tolist(),
                            ce_loss=float(losses[index].item()),
                        )
                    )

            if [sample.sample_id for sample in generated] != list(
                manifest.sample_ids
            ):
                raise RuntimeError(
                    "generated Host knowledge changed the signed D^P order"
                )

            validation_metrics: list[HostValidationSampleMetric] = []
            for start in range(
                0,
                len(encoded_validation),
                self.knowledge_batch_size,
            ):
                batch_values = encoded_validation[
                    start : start + self.knowledge_batch_size
                ]
                result = generate_pub_data_logits(
                    self._generation_inputs(batch_values),
                    model,
                    arguments,
                    collator,
                )
                token_ids = result[PER_STEP_INDICES]
                logits = result[PER_STEP_LOGITS]
                full_logsumexp = result[FULL_LOGSUMEXP]
                gold_token_ids = result[GOLD_TOKEN_IDS]
                gold_token_logits = result[GOLD_TOKEN_LOGITS]
                gold_token_nll = result[GOLD_TOKEN_NLL]
                losses = result[METRIC]
                self._validate_fedmkt_batch(
                    len(batch_values),
                    token_ids,
                    logits,
                    losses,
                )
                for index, item in enumerate(batch_values):
                    answer_token_count = sum(
                        label != -100 for label in item.labels[1:]
                    )
                    validation_metrics.append(
                        HostValidationSampleMetric(
                            sample_id=item.sample_id,
                            answer_token_count=answer_token_count,
                            answer_token_ce=float(losses[index].item()),
                        )
                    )

            validation = HostValidationRecord.create(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                validation_dataset=validation_identity,
                host_model_profile_hash=self.model_profile.profile_hash(),
                adapter_version=metadata.version,
                checkpoint_hash=metadata.checkpoint_hash,
                contract_hash=self.contract.contract_hash,
                execution_profile_hash=self.execution_profile.profile_hash(),
                samples=validation_metrics,
            )
            return HostBaselineResult(generated, validation)
        finally:
            del model, base_model
            self._release_memory(torch)

    def validate_trained_candidate(
        self,
        validation_samples: Sequence[ReferenceSample],
        validation_identity: ReferenceDatasetIdentity,
        manifest: RoundManifest,
        result: HostCandidateTrainingResult,
    ) -> HostValidationRecord:
        torch, transformers, peft, _ = self._dependencies()
        from shared.fedmkt_core.ml.logit_generation import (
            generate_pub_data_logits,
        )
        from shared.fedmkt_core.ml.vars_define import (
            FULL_LOGSUMEXP,
            GOLD_TOKEN_IDS,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
        )

        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        validated_tokenizer = load_pinned_tokenizer(
            POC_DTW_PROFILE.host,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer = validated_tokenizer.tokenizer
        if tokenizer.pad_token_id is None:
            raise RuntimeError("pinned Host tokenizer has no padding token")
        values = list(validation_samples)
        if reference_dataset_identity(values) != validation_identity:
            raise ValueError("Host validation data differs from its identity")
        encoded = encode_reference_samples(
            values,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=[sample.sample_id for sample in values],
            dataset_label="private D^V",
        )
        candidate_path = self.validate_candidate(result)
        base_model = None
        model = None
        try:
            base_model = self._load_base_model(torch, transformers)
            model = peft.PeftModel.from_pretrained(
                base_model,
                candidate_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(model)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Host candidate validation loaded trainable parameters")
            model.eval()
            collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
            arguments = FedMKTGenerationArguments(
                top_k_logits_keep=manifest.top_k
            )
            metrics: list[HostValidationSampleMetric] = []
            for start in range(0, len(encoded), self.knowledge_batch_size):
                batch_values = encoded[start : start + self.knowledge_batch_size]
                generated = generate_pub_data_logits(
                    self._generation_inputs(batch_values),
                    model,
                    arguments,
                    collator,
                )
                token_ids = generated[PER_STEP_INDICES]
                logits = generated[PER_STEP_LOGITS]
                full_logsumexp = generated[FULL_LOGSUMEXP]
                gold_token_ids = generated[GOLD_TOKEN_IDS]
                gold_token_logits = generated[GOLD_TOKEN_LOGITS]
                gold_token_nll = generated[GOLD_TOKEN_NLL]
                losses = generated[METRIC]
                self._validate_fedmkt_batch(
                    len(batch_values),
                    token_ids,
                    logits,
                    losses,
                )
                for index, item in enumerate(batch_values):
                    metrics.append(
                        HostValidationSampleMetric(
                            sample_id=item.sample_id,
                            answer_token_count=sum(
                                label != -100 for label in item.labels[1:]
                            ),
                            answer_token_ce=float(losses[index].item()),
                        )
                    )
            return HostValidationRecord.create(
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                validation_dataset=validation_identity,
                host_model_profile_hash=self.model_profile.profile_hash(),
                adapter_version=result.candidate_adapter_version,
                checkpoint_hash=result.candidate_adapter_hash,
                contract_hash=self.contract.contract_hash,
                execution_profile_hash=self.execution_profile.profile_hash(),
                samples=metrics,
            )
        finally:
            del model, base_model
            self._release_memory(torch)

    def generate_post_decision_knowledge(
        self,
        reference_samples: Sequence[ReferenceSample],
        manifest: RoundManifest,
        *,
        expected_adapter_version: int,
        expected_checkpoint_hash: str,
    ) -> list[KnowledgeSample]:
        torch, transformers, peft, _ = self._dependencies()
        from shared.fedmkt_core.ml.logit_generation import (
            generate_pub_data_logits,
        )
        from shared.fedmkt_core.ml.vars_define import (
            FULL_LOGSUMEXP,
            GOLD_TOKEN_IDS,
            GOLD_TOKEN_LOGITS,
            GOLD_TOKEN_NLL,
            METRIC,
            PER_STEP_INDICES,
            PER_STEP_LOGITS,
        )

        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        validated_tokenizer = load_pinned_tokenizer(
            POC_DTW_PROFILE.host,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        tokenizer = validated_tokenizer.tokenizer
        if tokenizer.pad_token_id is None:
            raise RuntimeError("pinned Host tokenizer has no padding token")
        encoded = encode_reference_samples(
            list(reference_samples),
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=manifest.sample_ids,
        )
        current = self.checkpoints.current()
        if current is None:
            raise RuntimeError("current Host adapter checkpoint is missing")
        metadata, checkpoint_path = current
        if (
            metadata.version != expected_adapter_version
            or metadata.checkpoint_hash != expected_checkpoint_hash
        ):
            raise RuntimeError("current Host checkpoint differs from the decision")
        base_model = None
        model = None
        try:
            base_model = self._load_base_model(torch, transformers)
            model = peft.PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(model)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("post-decision Host inference loaded trainable parameters")
            model.eval()
            collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
            arguments = FedMKTGenerationArguments(
                top_k_logits_keep=manifest.top_k
            )
            samples: list[KnowledgeSample] = []
            for start in range(0, len(encoded), self.knowledge_batch_size):
                batch_values = encoded[start : start + self.knowledge_batch_size]
                generated = generate_pub_data_logits(
                    self._generation_inputs(batch_values),
                    model,
                    arguments,
                    collator,
                )
                token_ids = generated[PER_STEP_INDICES]
                logits = generated[PER_STEP_LOGITS]
                full_logsumexp = generated[FULL_LOGSUMEXP]
                gold_token_ids = generated[GOLD_TOKEN_IDS]
                gold_token_logits = generated[GOLD_TOKEN_LOGITS]
                gold_token_nll = generated[GOLD_TOKEN_NLL]
                losses = generated[METRIC]
                self._validate_fedmkt_batch(
                    len(batch_values),
                    token_ids,
                    logits,
                    losses,
                )
                for index, item in enumerate(batch_values):
                    samples.append(
                        knowledge_sample_from_rows(
                            item,
                            top_k_token_ids=token_ids[index].tolist(),
                            top_k_logits=logits[index].tolist(),
                            full_logsumexp=full_logsumexp[index].tolist(),
                            gold_token_ids=gold_token_ids[index].tolist(),
                            gold_token_logits=gold_token_logits[index].tolist(),
                            gold_token_nll=gold_token_nll[index].tolist(),
                            ce_loss=float(losses[index].item()),
                        )
                    )
            return samples
        finally:
            del model, base_model
            self._release_memory(torch)

    @staticmethod
    def _generation_inputs(values: Sequence[Any]) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [item.input_ids for item in values],
            "attention_mask": [item.attention_mask for item in values],
            "labels": [item.labels for item in values],
        }

    @staticmethod
    def _validate_fedmkt_batch(
        batch_size: int,
        token_ids: Any,
        logits: Any,
        losses: Any,
    ) -> None:
        if (
            token_ids.size(0) != batch_size
            or logits.size(0) != batch_size
            or losses.numel() != batch_size
        ):
            raise RuntimeError("FedMKT returned an invalid Host batch shape")

    def _load_initialization_record(
        self,
        checkpoint_path: Path,
    ) -> HostAdapterInitializationRecord:
        path = checkpoint_path / HOST_INITIALIZATION_RECORD
        if not path.is_file():
            raise ValueError(
                "initial Host adapter is missing its initialization record"
            )
        return HostAdapterInitializationRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def _validate_initialization(
        self,
        metadata: AdapterCheckpointMetadata,
        record: HostAdapterInitializationRecord,
    ) -> None:
        if metadata.version != 0 or metadata.parent_version is not None:
            raise ValueError("initial Host adapter must be parentless version 0")
        if metadata.round_id is not None or metadata.manifest_hash is not None:
            raise ValueError("initial Host adapter must not belong to a round")
        if record.adapter_version != metadata.version:
            raise ValueError("Host initialization version differs from checkpoint")
        if record.model_profile_hash != self.model_profile.profile_hash():
            raise ValueError("Host initialization uses another model profile")
        if record.contract_hash != self.contract.contract_hash:
            raise ValueError("Host initialization uses another training contract")
        if record.execution_profile_hash != self.execution_profile.profile_hash():
            raise ValueError(
                "Host initialization uses another execution profile"
            )
        if metadata.execution_profile_hash != record.execution_profile_hash:
            raise ValueError(
                "Host initialization execution profile differs from checkpoint"
            )

    def _validate_promoted_adapter(
        self,
        metadata: AdapterCheckpointMetadata,
    ) -> None:
        if metadata.version < 1 or metadata.parent_version is None:
            raise ValueError("promoted Host adapter has no parent version")
        if metadata.parent_checkpoint_hash is None:
            raise ValueError("promoted Host adapter has no parent checkpoint")
        if metadata.round_id is None or metadata.manifest_hash is None:
            raise ValueError("promoted Host adapter has no round binding")
        if metadata.execution_profile_hash != self.execution_profile.profile_hash():
            raise ValueError("promoted Host adapter uses another execution profile")

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            import peft
            import safetensors
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "real Host adapter initialization requires requirements.txt"
            ) from exc
        return torch, transformers, peft, safetensors

    def _validate_device(self, torch: Any) -> None:
        profile = self.execution_profile
        if profile.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("HOST_TRAINING_DEVICE=cuda but CUDA is unavailable")
        if profile.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "HOST_TRAINING_PRECISION=bfloat16 is unsupported by this GPU"
            )

    def _load_base_model(self, torch: Any, transformers: Any) -> Any:
        profile = self.model_profile
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.execution_profile.precision]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            profile.model_id,
            revision=profile.model_revision,
            dtype=dtype,
            use_safetensors=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        if model.__class__.__name__ != profile.model_class:
            raise RuntimeError(
                f"expected model class {profile.model_class}, got "
                f"{model.__class__.__name__}"
            )
        if model.config.model_type != profile.model_type:
            raise RuntimeError("loaded Host model type differs from ModelProfile")
        if model.config.vocab_size != profile.vocabulary_size:
            raise RuntimeError("loaded Host vocabulary differs from ModelProfile")
        if getattr(model.config, "_commit_hash", None) != profile.model_revision:
            raise RuntimeError(
                "loaded Host model snapshot differs from the pinned revision"
            )
        targets = set(profile.lora.target_modules)
        found = {
            name.rsplit(".", 1)[-1]
            for name, _ in model.named_modules()
            if name.rsplit(".", 1)[-1] in targets
        }
        if found != targets:
            raise RuntimeError(
                f"Host LoRA targets differ from the model: expected "
                f"{sorted(targets)}, found {sorted(found)}"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model.to(self.execution_profile.device)

    def _create_adapter(self, peft: Any, base_model: Any) -> Any:
        lora = self.model_profile.lora
        config = peft.LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules),
            bias=lora.bias,
            task_type=lora.task_type,
            modules_to_save=list(lora.modules_to_save) or None,
            base_model_name_or_path=self.model_profile.model_id,
            revision=self.model_profile.model_revision,
        )
        model = peft.get_peft_model(base_model, config)
        self._verify_loaded_adapter(model)
        return model

    def _verify_loaded_adapter(self, model: Any) -> None:
        config = model.peft_config["default"]
        lora = self.model_profile.lora
        if int(config.r) != lora.rank:
            raise RuntimeError("Host adapter rank differs from ModelProfile")
        if float(config.lora_alpha) != lora.alpha:
            raise RuntimeError("Host adapter alpha differs from ModelProfile")
        if float(config.lora_dropout) != lora.dropout:
            raise RuntimeError("Host adapter dropout differs from ModelProfile")
        if set(config.target_modules) != set(lora.target_modules):
            raise RuntimeError("Host adapter targets differ from ModelProfile")
        if config.bias != lora.bias:
            raise RuntimeError("Host adapter bias differs from ModelProfile")

    @staticmethod
    def _assert_trainable_parameters(model: Any) -> tuple[int, int]:
        trainable_names: list[str] = []
        trainable_count = 0
        total_count = 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            total_count += count
            if parameter.requires_grad:
                trainable_names.append(name)
                trainable_count += count
        if not trainable_names:
            raise RuntimeError("initial Host PEFT model has no trainable parameters")
        unexpected = [name for name in trainable_names if ".lora_" not in name]
        if unexpected:
            raise RuntimeError(
                f"non-LoRA Host parameters are trainable: {unexpected[:5]}"
            )
        return trainable_count, total_count

    @staticmethod
    def _frozen_parameter_checksum(torch: Any, model: Any) -> str:
        import hashlib

        digest = hashlib.sha256()
        frozen_count = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                continue
            frozen_count += 1
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(
                parameter.detach()
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
        if frozen_count == 0:
            raise RuntimeError("Host PEFT model has no frozen base parameters")
        return digest.hexdigest()

    def _validate_adapter_tensors(self, safetensors: Any, path: Path) -> None:
        tensor_path = path / "adapter_model.safetensors"
        with safetensors.safe_open(
            tensor_path,
            framework="pt",
            device="cpu",
        ) as handle:
            keys = list(handle.keys())
        if not keys or any(".lora_" not in key for key in keys):
            raise RuntimeError(
                "initial Host checkpoint contains unexpected tensor keys"
            )
        targets = set(self.model_profile.lora.target_modules)
        found = {
            target
            for target in targets
            if any(f".{target}." in key for key in keys)
        }
        if found != targets:
            raise RuntimeError(
                "initial Host checkpoint is missing configured LoRA targets"
            )

    @staticmethod
    def _assert_lora_tensors_changed(
        torch: Any,
        safetensors: Any,
        parent_path: Path,
        candidate_path: Path,
    ) -> None:
        with safetensors.safe_open(
            parent_path / "adapter_model.safetensors",
            framework="pt",
            device="cpu",
        ) as parent_handle:
            with safetensors.safe_open(
                candidate_path / "adapter_model.safetensors",
                framework="pt",
                device="cpu",
            ) as candidate_handle:
                parent_keys = set(parent_handle.keys())
                candidate_keys = set(candidate_handle.keys())
                if parent_keys != candidate_keys:
                    raise RuntimeError(
                        "Host candidate LoRA tensor keys differ from its parent"
                    )
                for key in sorted(parent_keys):
                    parent_tensor = parent_handle.get_tensor(key)
                    candidate_tensor = candidate_handle.get_tensor(key)
                    if (
                        parent_tensor.shape != candidate_tensor.shape
                        or parent_tensor.dtype != candidate_tensor.dtype
                    ):
                        raise RuntimeError(
                            "Host candidate LoRA tensor structure differs"
                        )
                    if not torch.equal(parent_tensor, candidate_tensor):
                        return
        raise RuntimeError("Host training did not change any LoRA tensor")

    def _probe_logits(self, torch: Any, model: Any, token_ids: list[int]) -> Any:
        model.eval()
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.execution_profile.device,
        )
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
        return logits.detach().float().cpu()

    @staticmethod
    def _require_same_logits(
        torch: Any,
        expected: Any,
        actual: Any,
        *,
        label: str,
        absolute_tolerance: float,
    ) -> None:
        if torch.allclose(
            expected,
            actual,
            rtol=0,
            atol=absolute_tolerance,
        ):
            return
        difference = (expected - actual).abs()
        raise RuntimeError(
            f"{label}: max_abs_difference={difference.max().item():.8g}, "
            f"mean_abs_difference={difference.mean().item():.8g}"
        )

    @staticmethod
    def _release_memory(torch: Any) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
