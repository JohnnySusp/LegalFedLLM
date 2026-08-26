from __future__ import annotations

import gc
import math
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from client.peft_backend import TransformersPeftTrainingBackend
from shared.adapter_checkpoint import AdapterCheckpointMetadata
from shared.alignment_profiles import resolve_alignment_profile
from shared.answer_only import AnswerOnlyCollator
from shared.client_reverse_artifact import load_client_reverse_training_artifact
from shared.crypto import sha256_hex
from shared.protocol import (
    HASH_PATTERN,
    ClientReverseTrainingJob,
    parse_utc,
    utc_text,
)
from shared.reference_dataset import ReferenceSample
from shared.reference_knowledge import (
    FedMKTGenerationArguments,
    encode_reference_samples,
)
from shared.tokenizer_validation import load_pinned_tokenizer


CLIENT_QUALITY_NON_REGRESSION_TOLERANCE = 0.001


class ReverseTrainingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ClientReverseCandidateResult(ReverseTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    parent_adapter_version: int = Field(ge=0)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    candidate_adapter_version: int = Field(ge=1)
    candidate_adapter_hash: str = Field(pattern=HASH_PATTERN)
    client_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    execution_profile_hash: str = Field(pattern=HASH_PATTERN)
    client_public_data_epochs: Literal[1]
    supervised_loss_weight: Literal[0.9] = 0.9
    distillation_loss_weight: Literal[0.1] = 0.1
    loss_type: Literal["ce"] = "ce"
    temperature: Literal[1.0] = 1.0
    optimizer_step_count: int = Field(ge=1)
    optimizer_loss: float = Field(ge=0, allow_inf_nan=False)
    supervised_answer_loss: float = Field(ge=0, allow_inf_nan=False)
    distillation_answer_loss: float = Field(ge=0, allow_inf_nan=False)
    trainable_parameter_count: int = Field(gt=0)
    total_parameter_count: int = Field(gt=0)
    lora_tensors_changed: Literal[True] = True
    frozen_base_unchanged: Literal[True] = True
    reload_verified: Literal[True] = True
    dependency_versions: dict[str, str]
    created_at: str
    result_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> "ClientReverseCandidateResult":
        parse_utc(self.created_at)
        if self.candidate_adapter_version != self.parent_adapter_version + 1:
            raise ValueError("Client reverse candidate must follow its parent")
        if self.trainable_parameter_count >= self.total_parameter_count:
            raise ValueError("Client reverse candidate did not freeze its base")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"result_hash"})
        )
        if self.result_hash != expected:
            raise ValueError("Client reverse candidate result hash differs")
        return self

    @classmethod
    def create(cls, **values: Any) -> "ClientReverseCandidateResult":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"result_hash"}
        )
        return cls(**payload, result_hash=sha256_hex(payload))


class ClientValidationSampleMetric(ReverseTrainingContract):
    sample_id: str = Field(min_length=1, max_length=256)
    answer_token_count: int = Field(ge=1)
    answer_token_ce: float = Field(ge=0, allow_inf_nan=False)
    teacher_forced_exact_match: float = Field(ge=0, le=1, allow_inf_nan=False)
    teacher_forced_rouge_l: float = Field(ge=0, le=1, allow_inf_nan=False)


class ClientValidationRecord(ReverseTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    backend: Literal["mock", "transformers"]
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    partition_hash: str = Field(pattern=HASH_PATTERN)
    validation_sample_ids_hash: str = Field(pattern=HASH_PATTERN)
    client_model_profile_hash: str = Field(pattern=HASH_PATTERN)
    adapter_role: Literal["parent", "candidate"]
    adapter_version: int = Field(ge=0)
    checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    primary_metric: Literal["macro_mean_answer_token_ce"] = (
        "macro_mean_answer_token_ce"
    )
    secondary_metric: Literal["token_weighted_answer_token_ce"] = (
        "token_weighted_answer_token_ce"
    )
    diagnostic_metrics: tuple[
        Literal["teacher_forced_exact_match"],
        Literal["teacher_forced_rouge_l"],
    ] = ("teacher_forced_exact_match", "teacher_forced_rouge_l")
    sample_count: int = Field(ge=1)
    supervised_answer_token_count: int = Field(ge=1)
    macro_mean_answer_token_ce: float = Field(ge=0, allow_inf_nan=False)
    token_weighted_answer_token_ce: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    exact_match_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_rouge_l: float = Field(ge=0, le=1, allow_inf_nan=False)
    samples: list[ClientValidationSampleMetric] = Field(min_length=1)
    created_at: str
    record_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_record(self) -> "ClientValidationRecord":
        parse_utc(self.created_at)
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Client validation sample IDs must be unique")
        if self.validation_sample_ids_hash != sha256_hex(sample_ids):
            raise ValueError("Client validation sample order hash differs")
        if self.sample_count != len(self.samples):
            raise ValueError("Client validation sample count differs")
        token_count = sum(sample.answer_token_count for sample in self.samples)
        if self.supervised_answer_token_count != token_count:
            raise ValueError("Client validation answer-token count differs")
        macro = math.fsum(
            sample.answer_token_ce for sample in self.samples
        ) / self.sample_count
        weighted = math.fsum(
            sample.answer_token_ce * sample.answer_token_count
            for sample in self.samples
        ) / token_count
        exact_match = math.fsum(
            sample.teacher_forced_exact_match for sample in self.samples
        ) / self.sample_count
        rouge_l = math.fsum(
            sample.teacher_forced_rouge_l for sample in self.samples
        ) / self.sample_count
        expected_metrics = (
            (self.macro_mean_answer_token_ce, macro, "macro CE"),
            (
                self.token_weighted_answer_token_ce,
                weighted,
                "token-weighted CE",
            ),
            (self.exact_match_accuracy, exact_match, "exact match"),
            (self.macro_rouge_l, rouge_l, "ROUGE-L"),
        )
        for actual, expected, label in expected_metrics:
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Client validation {label} differs")
        expected_hash = sha256_hex(
            self.model_dump(mode="json", exclude={"record_hash"})
        )
        if self.record_hash != expected_hash:
            raise ValueError("Client validation record hash differs")
        return self

    @classmethod
    def create(
        cls,
        *,
        backend: Literal["mock", "transformers"],
        round_id: str,
        manifest_hash: str,
        job_hash: str,
        partition_hash: str,
        client_model_profile_hash: str,
        adapter_role: Literal["parent", "candidate"],
        adapter_version: int,
        checkpoint_hash: str,
        samples: list[ClientValidationSampleMetric],
        created_at: str | None = None,
    ) -> "ClientValidationRecord":
        sample_count = len(samples)
        token_count = sum(sample.answer_token_count for sample in samples)
        if sample_count < 1 or token_count < 1:
            raise ValueError("Client validation metrics must not be empty")
        payload = {
            "schema_version": "1.0",
            "backend": backend,
            "round_id": round_id,
            "manifest_hash": manifest_hash,
            "job_hash": job_hash,
            "partition_hash": partition_hash,
            "validation_sample_ids_hash": sha256_hex(
                [sample.sample_id for sample in samples]
            ),
            "client_model_profile_hash": client_model_profile_hash,
            "adapter_role": adapter_role,
            "adapter_version": adapter_version,
            "checkpoint_hash": checkpoint_hash,
            "primary_metric": "macro_mean_answer_token_ce",
            "secondary_metric": "token_weighted_answer_token_ce",
            "diagnostic_metrics": [
                "teacher_forced_exact_match",
                "teacher_forced_rouge_l",
            ],
            "sample_count": sample_count,
            "supervised_answer_token_count": token_count,
            "macro_mean_answer_token_ce": math.fsum(
                sample.answer_token_ce for sample in samples
            )
            / sample_count,
            "token_weighted_answer_token_ce": math.fsum(
                sample.answer_token_ce * sample.answer_token_count
                for sample in samples
            )
            / token_count,
            "exact_match_accuracy": math.fsum(
                sample.teacher_forced_exact_match for sample in samples
            )
            / sample_count,
            "macro_rouge_l": math.fsum(
                sample.teacher_forced_rouge_l for sample in samples
            )
            / sample_count,
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "created_at": created_at or utc_text(),
        }
        return cls(**payload, record_hash=sha256_hex(payload))


class ClientReverseDecision(ReverseTrainingContract):
    schema_version: Literal["1.0"] = "1.0"
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    host_teacher_sample_count: int = Field(ge=0)
    parent_adapter_version: int = Field(ge=0)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    candidate_result_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    candidate_adapter_version: int | None = Field(default=None, ge=1)
    candidate_adapter_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    active_adapter_version_before_decision: int = Field(ge=0)
    active_adapter_hash_before_decision: str = Field(pattern=HASH_PATTERN)
    accepted_adapter_version: int = Field(ge=0)
    accepted_adapter_hash: str = Field(pattern=HASH_PATTERN)
    parent_validation_record_hash: str | None = Field(
        default=None,
        pattern=HASH_PATTERN,
    )
    candidate_validation_record_hash: str | None = Field(
        default=None,
        pattern=HASH_PATTERN,
    )
    safety_report_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    probe_artifact_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    parent_macro_mean_answer_token_ce: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    candidate_macro_mean_answer_token_ce: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    observed_ce_regression: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )
    quality_non_regression_tolerance: Literal[0.001] = (
        CLIENT_QUALITY_NON_REGRESSION_TOLERANCE
    )
    quality_gate_passed: bool | None = None
    maliciousness_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    safety_threshold: Literal[0.8] | None = None
    safety_gate_passed: bool | None = None
    stale_parent: bool
    forced_rejection: bool
    adapter_promoted: bool
    decision_reason: Literal[
        "candidate_accepted",
        "no_host_teacher_samples",
        "quality_gate_failed",
        "safety_gate_failed",
        "quality_and_safety_gates_failed",
        "forced_validation_rejection",
        "stale_parent",
    ]
    rejected_candidate_discarded: bool
    created_at: str
    decision_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_decision(self) -> "ClientReverseDecision":
        parse_utc(self.created_at)
        candidate_fields = (
            self.candidate_result_hash,
            self.candidate_adapter_version,
            self.candidate_adapter_hash,
            self.parent_validation_record_hash,
            self.candidate_validation_record_hash,
            self.quality_gate_passed,
            self.parent_macro_mean_answer_token_ce,
            self.candidate_macro_mean_answer_token_ce,
            self.observed_ce_regression,
            self.maliciousness_probability,
            self.safety_gate_passed,
        )
        if self.decision_reason == "no_host_teacher_samples":
            if self.host_teacher_sample_count != 0 or any(
                value is not None for value in candidate_fields
            ):
                raise ValueError("Client no-op decision contains candidate evidence")
            if self.adapter_promoted or self.rejected_candidate_discarded:
                raise ValueError("Client no-op decision changed an adapter")
            if (
                self.accepted_adapter_version
                != self.active_adapter_version_before_decision
                or self.accepted_adapter_hash
                != self.active_adapter_hash_before_decision
            ):
                raise ValueError("Client no-op decision changed the active adapter")
        else:
            if self.host_teacher_sample_count < 1 or any(
                value is None for value in candidate_fields
            ):
                raise ValueError("Client candidate decision is missing evidence")
            assert self.candidate_adapter_version is not None
            assert self.parent_macro_mean_answer_token_ce is not None
            assert self.candidate_macro_mean_answer_token_ce is not None
            assert self.observed_ce_regression is not None
            assert self.maliciousness_probability is not None
            assert self.safety_threshold is not None
            assert self.quality_gate_passed is not None
            assert self.safety_gate_passed is not None
            if self.candidate_adapter_version != self.parent_adapter_version + 1:
                raise ValueError("Client decision candidate does not follow parent")
            observed = (
                self.candidate_macro_mean_answer_token_ce
                - self.parent_macro_mean_answer_token_ce
            )
            if not math.isclose(
                self.observed_ce_regression,
                observed,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("Client CE regression differs")
            if self.quality_gate_passed != (
                observed <= self.quality_non_regression_tolerance
            ):
                raise ValueError("Client quality gate differs from its CE metrics")
            if self.safety_gate_passed != (
                self.maliciousness_probability < self.safety_threshold
            ):
                raise ValueError("Client safety gate differs from its probability")
            expected_promotion = (
                self.quality_gate_passed
                and self.safety_gate_passed
                and not self.stale_parent
                and not self.forced_rejection
            )
            if self.adapter_promoted != expected_promotion:
                raise ValueError("Client promotion differs from its gates")
            if self.stale_parent:
                expected_reason = "stale_parent"
            elif self.forced_rejection:
                expected_reason = "forced_validation_rejection"
            elif self.quality_gate_passed and self.safety_gate_passed:
                expected_reason = "candidate_accepted"
            elif not self.quality_gate_passed and not self.safety_gate_passed:
                expected_reason = "quality_and_safety_gates_failed"
            elif not self.quality_gate_passed:
                expected_reason = "quality_gate_failed"
            else:
                expected_reason = "safety_gate_failed"
            if self.decision_reason != expected_reason:
                raise ValueError("Client decision reason differs from its gates")
            if self.adapter_promoted:
                if self.rejected_candidate_discarded:
                    raise ValueError("promoted Client candidate was discarded")
                if (
                    self.accepted_adapter_version
                    != self.candidate_adapter_version
                    or self.accepted_adapter_hash != self.candidate_adapter_hash
                ):
                    raise ValueError("promoted Client decision accepted another adapter")
            else:
                if not self.rejected_candidate_discarded:
                    raise ValueError("rejected Client candidate was not discarded")
                if (
                    self.accepted_adapter_version
                    != self.active_adapter_version_before_decision
                    or self.accepted_adapter_hash
                    != self.active_adapter_hash_before_decision
                ):
                    raise ValueError("rejected Client decision accepted another adapter")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"decision_hash"})
        )
        if self.decision_hash != expected:
            raise ValueError("Client reverse decision hash differs")
        return self

    @classmethod
    def create(cls, **values: Any) -> "ClientReverseDecision":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"decision_hash"}
        )
        return cls(**payload, decision_hash=sha256_hex(payload))


class _TensorRowDataset:
    def __init__(self, tensors: dict[str, Any]):
        self.tensors = tensors
        self.length = int(next(iter(tensors.values())).shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {name: value[index] for name, value in self.tensors.items()}


def selective_client_loss(
    torch: Any,
    logits: Any,
    inputs: dict[str, Any],
) -> tuple[Any, Any, Any]:
    from shared.fedmkt_core.ml.sparse_targets import (
        SparseTargetBatch,
        answer_only_sparse_distillation_loss,
    )

    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    supervised = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    targets = SparseTargetBatch(
        token_ids=inputs["sparse_target_token_ids"].long(),
        probabilities=inputs["sparse_target_probabilities"].to(
            dtype=logits.dtype
        ),
        valid_mask=inputs["sparse_target_valid_mask"].bool(),
    )
    distillation = answer_only_sparse_distillation_loss(
        logits,
        targets,
        labels=labels,
        attention_mask=attention_mask,
        loss_type="ce",
    )
    return 0.9 * supervised + 0.1 * distillation, supervised, distillation


def _selective_client_trainer_class(transformers: Any, torch: Any) -> type:
    class SelectiveClientTrainer(transformers.Trainer):
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
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
            loss, supervised, distillation = selective_client_loss(
                torch,
                outputs.logits,
                inputs,
            )
            self.supervised_losses.append(float(supervised.detach().float()))
            self.distillation_losses.append(
                float(distillation.detach().float())
            )
            return (loss, outputs) if return_outputs else loss

    return SelectiveClientTrainer


class TransformersPeftReverseBackend(TransformersPeftTrainingBackend):
    def train_candidate(
        self,
        job: ClientReverseTrainingJob,
        artifact_path: str | Path,
    ) -> ClientReverseCandidateResult:
        torch, transformers, peft, safetensors = self._dependencies()
        self._validate_device(torch)
        if job.client_model_profile_hash != self.model_profile.profile_hash():
            raise ValueError("Client reverse job uses another model profile")
        if job.client_public_data_epochs != 1:
            raise ValueError("Client reverse training requires one public epoch")
        if (
            job.distillation.loss_type != "ce"
            or job.distillation.temperature != 1.0
            or job.distillation.lm_loss_weight != 0.9
        ):
            raise ValueError("Client reverse job uses another loss contract")

        parent_metadata, parent_path = self.checkpoints.version(
            job.parent_adapter_version
        )
        if parent_metadata.checkpoint_hash != job.parent_adapter_hash:
            raise ValueError("Client reverse parent checkpoint hash differs")
        arrays = load_client_reverse_training_artifact(
            artifact_path,
            job.artifact,
            job.public_data_partition.transfer_sample_ids,
            maximum_bytes=job.manifest.maximum_client_reverse_training_job_bytes,
            vocabulary_size=int(self.model_profile.vocabulary_size or 0),
        )
        tensor_values: dict[str, Any] = {}
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
        job_id = f"client-reverse-{secrets.token_hex(8)}"
        output_dir = self.data_dir / "reverse_training_jobs" / job_id
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_path: Path | None = None
        base_model = model = trainer = reloaded_base = reloaded = None
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
                num_train_epochs=job.client_public_data_epochs,
                per_device_train_batch_size=profile.micro_batch_size,
                gradient_accumulation_steps=profile.gradient_accumulation_steps,
                learning_rate=profile.learning_rate,
                lr_scheduler_type=profile.learning_rate_scheduler,
                optim="adamw_torch",
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
                dataloader_pin_memory=profile.device == "cuda",
            )
            trainer = _selective_client_trainer_class(
                transformers,
                torch,
            )(
                model=model,
                args=arguments,
                train_dataset=dataset,
            )
            train_output = trainer.train()
            optimizer_step_count = int(trainer.state.global_step)
            optimizer_loss = float(train_output.training_loss)
            if optimizer_step_count < 1:
                raise RuntimeError("Client reverse training completed no optimizer step")
            if not math.isfinite(optimizer_loss) or optimizer_loss < 0:
                raise RuntimeError("Client reverse training produced an invalid loss")
            if not trainer.supervised_losses or not trainer.distillation_losses:
                raise RuntimeError("Client reverse training skipped a loss term")
            supervised_loss = math.fsum(trainer.supervised_losses) / len(
                trainer.supervised_losses
            )
            distillation_loss = math.fsum(trainer.distillation_losses) / len(
                trainer.distillation_losses
            )
            if not math.isfinite(supervised_loss) or supervised_loss < 0:
                raise RuntimeError("Client supervised answer loss is invalid")
            if not math.isfinite(distillation_loss) or distillation_loss < 0:
                raise RuntimeError("Client distillation answer loss is invalid")
            model = trainer.accelerator.unwrap_model(
                trainer.model_wrapped,
                keep_fp32_wrapper=False,
            )
            trainable_count, total_count = self._assert_trainable_parameters(model)
            if self._frozen_parameter_checksum(torch, model) != frozen_checksum:
                raise RuntimeError(
                    "a frozen Client base-model parameter changed during reverse training"
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
            metadata = self.checkpoints.seal(
                candidate_path,
                version=parent_metadata.version + 1,
                parent=parent_metadata,
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                execution_profile_hash=profile.profile_hash(),
            )

            del trainer, model, base_model
            trainer = model = base_model = None
            self._release_memory(torch)
            reloaded_base = self._load_base_model(torch, transformers)
            reloaded = peft.PeftModel.from_pretrained(
                reloaded_base,
                candidate_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(reloaded)
            if any(parameter.requires_grad for parameter in reloaded.parameters()):
                raise RuntimeError("reloaded Client reverse candidate is trainable")
            actual_logits = self._probe_logits(torch, reloaded, probe_ids)
            if not torch.allclose(expected_logits, actual_logits, rtol=0, atol=1e-4):
                difference = (expected_logits - actual_logits).abs()
                raise RuntimeError(
                    "reloaded Client reverse candidate changed its probe logits: "
                    f"max_abs_difference={difference.max().item():.8g}"
                )
            result = ClientReverseCandidateResult.create(
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                job_hash=job.job_hash,
                parent_adapter_version=parent_metadata.version,
                parent_adapter_hash=parent_metadata.checkpoint_hash,
                candidate_adapter_version=metadata.version,
                candidate_adapter_hash=metadata.checkpoint_hash,
                client_model_profile_hash=self.model_profile.profile_hash(),
                execution_profile_hash=profile.profile_hash(),
                client_public_data_epochs=job.client_public_data_epochs,
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
            self.checkpoints.store_candidate(candidate_path, metadata)
            candidate_path = None
            return result
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
        result: ClientReverseCandidateResult,
    ) -> Path:
        metadata, path = self.checkpoints.candidate(
            result.round_id,
            result.candidate_adapter_version,
        )
        self._validate_candidate_metadata(metadata, result)
        return path

    def validation_record(
        self,
        *,
        job: ClientReverseTrainingJob,
        result: ClientReverseCandidateResult,
        validation_samples: Sequence[ReferenceSample],
        adapter_role: Literal["parent", "candidate"],
    ) -> ClientValidationRecord:
        if adapter_role == "parent":
            metadata, checkpoint_path = self.checkpoints.version(
                result.parent_adapter_version
            )
            if metadata.checkpoint_hash != result.parent_adapter_hash:
                raise ValueError("Client validation parent hash differs")
        else:
            checkpoint_path = self.validate_candidate(result)
            metadata, _ = self.checkpoints.candidate(
                result.round_id,
                result.candidate_adapter_version,
            )
        return self._evaluate_adapter(
            job=job,
            validation_samples=validation_samples,
            adapter_role=adapter_role,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
        )

    def promote_candidate(
        self,
        result: ClientReverseCandidateResult,
    ) -> tuple[AdapterCheckpointMetadata, Path]:
        return self.checkpoints.promote_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )

    def discard_candidate(self, result: ClientReverseCandidateResult) -> None:
        self.checkpoints.discard_candidate(
            result.round_id,
            result.candidate_adapter_version,
            result.candidate_adapter_hash,
        )

    def _evaluate_adapter(
        self,
        *,
        job: ClientReverseTrainingJob,
        validation_samples: Sequence[ReferenceSample],
        adapter_role: Literal["parent", "candidate"],
        metadata: AdapterCheckpointMetadata,
        checkpoint_path: Path,
    ) -> ClientValidationRecord:
        torch, transformers, peft, _ = self._dependencies()
        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        profile = resolve_alignment_profile(
            f"{job.manifest.alignment.strategy}:"
            f"{job.manifest.alignment.profile_version}"
        )
        if profile.client.profile_id != self.model_profile.profile_id:
            raise ValueError("Client validation alignment profile differs")
        validated_tokenizer = load_pinned_tokenizer(
            profile.client,
            cache_dir=os.getenv("CLIENT_TOKENIZER_CACHE_DIR")
            or os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
            local_files_only=os.getenv(
                "LEGALFEDLLM_TOKENIZER_LOCAL_FILES_ONLY",
                "true",
            ).strip().lower()
            not in {"0", "false", "no"},
        )
        tokenizer = validated_tokenizer.tokenizer
        values = list(validation_samples)
        expected_ids = job.public_data_partition.validation_sample_ids
        if [sample.sample_id for sample in values] != expected_ids:
            raise ValueError("Client validation samples changed partition order")
        encoded = encode_reference_samples(
            values,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=job.manifest.maximum_sequence_length,
            expected_sample_ids=expected_ids,
            dataset_label="Client public validation split",
        )
        base_model = model = None
        try:
            base_model = self._load_base_model(torch, transformers)
            model = peft.PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(model)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Client validation loaded trainable parameters")
            model.eval()
            collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
            arguments = FedMKTGenerationArguments(
                top_k_logits_keep=job.manifest.top_k
            )
            metrics: list[ClientValidationSampleMetric] = []
            from shared.fedmkt_core.ml.logit_generation import (
                generate_pub_data_logits,
            )
            from shared.fedmkt_core.ml.vars_define import (
                METRIC,
                PER_STEP_INDICES,
                PER_STEP_LOGITS,
            )

            for start in range(0, len(encoded), self.knowledge_batch_size):
                batch_values = encoded[start : start + self.knowledge_batch_size]
                generated = generate_pub_data_logits(
                    {
                        "input_ids": [item.input_ids for item in batch_values],
                        "attention_mask": [
                            item.attention_mask for item in batch_values
                        ],
                        "labels": [item.labels for item in batch_values],
                    },
                    model,
                    arguments,
                    collator,
                )
                token_ids = generated[PER_STEP_INDICES]
                logits = generated[PER_STEP_LOGITS]
                losses = generated[METRIC]
                if (
                    token_ids.size(0) != len(batch_values)
                    or logits.size(0) != len(batch_values)
                    or losses.numel() != len(batch_values)
                ):
                    raise RuntimeError("FedMKT returned invalid validation shapes")
                for index, item in enumerate(batch_values):
                    supervised_positions = [
                        position
                        for position in range(1, len(item.labels))
                        if item.labels[position] != -100
                    ]
                    if not supervised_positions:
                        raise RuntimeError("Client validation sample has no answer")
                    gold_ids = [item.labels[position] for position in supervised_positions]
                    predicted_ids = [
                        int(token_ids[index, position - 1, 0].item())
                        for position in supervised_positions
                    ]
                    gold_text = tokenizer.decode(
                        gold_ids,
                        skip_special_tokens=True,
                    )
                    predicted_text = tokenizer.decode(
                        predicted_ids,
                        skip_special_tokens=True,
                    )
                    metrics.append(
                        ClientValidationSampleMetric(
                            sample_id=item.sample_id,
                            answer_token_count=len(supervised_positions),
                            answer_token_ce=float(losses[index].item()),
                            teacher_forced_exact_match=float(
                                _normalized_text(predicted_text)
                                == _normalized_text(gold_text)
                            ),
                            teacher_forced_rouge_l=_rouge_l_f1(
                                predicted_text,
                                gold_text,
                            ),
                        )
                    )
            return ClientValidationRecord.create(
                backend="transformers",
                round_id=job.manifest.round_id,
                manifest_hash=job.manifest.manifest_hash,
                job_hash=job.job_hash,
                partition_hash=job.public_data_partition.partition_hash,
                client_model_profile_hash=self.model_profile.profile_hash(),
                adapter_role=adapter_role,
                adapter_version=metadata.version,
                checkpoint_hash=metadata.checkpoint_hash,
                samples=metrics,
            )
        finally:
            del model, base_model
            self._release_memory(torch)

    @staticmethod
    def _validate_candidate_metadata(
        metadata: AdapterCheckpointMetadata,
        result: ClientReverseCandidateResult,
    ) -> None:
        expected = {
            "round_id": result.round_id,
            "manifest_hash": result.manifest_hash,
            "version": result.candidate_adapter_version,
            "checkpoint_hash": result.candidate_adapter_hash,
            "parent_version": result.parent_adapter_version,
            "parent_checkpoint_hash": result.parent_adapter_hash,
            "profile_hash": result.client_model_profile_hash,
            "execution_profile_hash": result.execution_profile_hash,
        }
        payload = metadata.model_dump(mode="json")
        mismatches = [key for key, value in expected.items() if payload[key] != value]
        if mismatches:
            raise ValueError(
                "stored Client candidate differs from its result: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _release_memory(torch: Any) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _rouge_l_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _normalized_text(predicted).split()
    expected_tokens = _normalized_text(expected).split()
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0
    previous = [0] * (len(expected_tokens) + 1)
    for predicted_token in predicted_tokens:
        current = [0]
        for index, expected_token in enumerate(expected_tokens, start=1):
            if predicted_token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
