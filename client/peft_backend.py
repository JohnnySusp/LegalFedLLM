from __future__ import annotations

import gc
import math
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Sequence

from client.knowledge import (
    FedMKTGenerationArguments,
    encode_reference_samples,
    knowledge_sample_from_rows,
)
from client.training import (
    BackendTrainingResult,
    EncodedTrainingExample,
    PrivateTrainingExample,
    TrainingExecutionProfile,
    encode_private_examples,
)
from shared.answer_only import AnswerOnlyCollator
from shared.adapter_checkpoint import AdapterCheckpointStore
from shared.crypto import sha256_hex
from shared.protocol import KnowledgeSample, ModelProfile, RoundManifest
from shared.reference_dataset import ReferenceSample


class TransformersPeftTrainingBackend:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        model_profile: ModelProfile,
        execution_profile: TrainingExecutionProfile,
        knowledge_batch_size: int | None = None,
    ):
        if model_profile.training_backend != "transformers":
            raise ValueError("PEFT backend requires a Transformers model profile")
        if execution_profile.backend != "transformers":
            raise ValueError("PEFT backend requires a Transformers execution profile")
        self.data_dir = Path(data_dir).resolve()
        self.model_profile = model_profile
        self.execution_profile = execution_profile
        self.knowledge_batch_size = (
            int(os.getenv("CLIENT_KNOWLEDGE_BATCH_SIZE", "1"))
            if knowledge_batch_size is None
            else knowledge_batch_size
        )
        if self.knowledge_batch_size < 1:
            raise ValueError("knowledge batch size must be positive")
        self.checkpoints = AdapterCheckpointStore(
            self.data_dir / "adapters",
            model_profile,
        )

    def train(
        self,
        examples: list[PrivateTrainingExample],
        manifest: RoundManifest,
    ) -> BackendTrainingResult:
        torch, transformers, peft, safetensors = self._dependencies()
        self._validate_device(torch)
        transformers.set_seed(self.execution_profile.seed)
        tokenizer = self._load_tokenizer(transformers)
        base_model = self._load_base_model(torch, transformers)
        current = self.checkpoints.current()

        if current is None:
            model = self._create_adapter(peft, base_model)
            self._assert_trainable_parameters(model)
            initial_path = self.checkpoints.staging_path(
                f"initial-{secrets.token_hex(8)}"
            )
            try:
                model.save_pretrained(
                    initial_path,
                    safe_serialization=True,
                    save_embedding_layers=False,
                )
                self._validate_adapter_tensors(safetensors, initial_path)
                initial_metadata = self.checkpoints.seal(
                    initial_path,
                    version=self.checkpoints.next_version(0),
                    parent=None,
                    round_id=None,
                    manifest_hash=None,
                    execution_profile_hash=None,
                )
                initial_version_path = self.checkpoints.promote(
                    initial_path,
                    initial_metadata,
                )
            except Exception:
                if initial_path.exists():
                    self.checkpoints.discard_staging(initial_path)
                raise
            current = initial_metadata, initial_version_path
        else:
            metadata, checkpoint_path = current
            model = peft.PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                is_trainable=True,
            )
            self._verify_loaded_adapter(model)

        parent_metadata, parent_path = current
        self._assert_trainable_parameters(model)
        base_checksum = None
        if self.execution_profile.verify_frozen_base_checksum:
            base_checksum = self._frozen_parameter_checksum(torch, model)
        encoded = encode_private_examples(
            examples,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
        )
        dataset = _ListDataset(encoded)
        collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
        job_id = f"train-{secrets.token_hex(8)}"
        output_dir = self.data_dir / "training_jobs" / job_id
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate_path: Path | None = None

        try:
            if self.execution_profile.gradient_checkpointing:
                model.config.use_cache = False
                model.enable_input_require_grads()

            arguments = transformers.TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=manifest.training_epochs,
                per_device_train_batch_size=(
                    self.execution_profile.micro_batch_size
                ),
                gradient_accumulation_steps=(
                    self.execution_profile.gradient_accumulation_steps
                ),
                learning_rate=self.execution_profile.learning_rate,
                lr_scheduler_type=(
                    self.execution_profile.learning_rate_scheduler
                ),
                optim="adamw_torch",
                seed=self.execution_profile.seed,
                data_seed=self.execution_profile.seed,
                bf16=self.execution_profile.precision == "bfloat16",
                fp16=self.execution_profile.precision == "float16",
                use_cpu=self.execution_profile.device == "cpu",
                gradient_checkpointing=(
                    self.execution_profile.gradient_checkpointing
                ),
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="steps",
                logging_steps=1,
                report_to=[],
                remove_unused_columns=False,
                dataloader_pin_memory=self.execution_profile.device == "cuda",
            )
            trainer = transformers.Trainer(
                model=model,
                args=arguments,
                train_dataset=dataset,
                data_collator=collator,
            )

            train_output = trainer.train()
            optimizer_step_count = int(trainer.state.global_step)
            training_loss = float(train_output.training_loss)
            if optimizer_step_count < 1:
                raise RuntimeError("real PEFT training completed no optimizer step")
            if not math.isfinite(training_loss) or training_loss < 0:
                raise RuntimeError("real PEFT training produced a non-finite loss")
            model = trainer.accelerator.unwrap_model(
                trainer.model_wrapped,
                keep_fp32_wrapper=False,
            )

            trainable_count, total_count = self._assert_trainable_parameters(model)
            if base_checksum is not None and (
                self._frozen_parameter_checksum(torch, model) != base_checksum
            ):
                raise RuntimeError(
                    "a frozen base-model parameter changed during training"
                )
            probe_ids = encoded[0].input_ids[:32]
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
                version=self.checkpoints.next_version(
                    parent_metadata.version + 1
                ),
                parent=parent_metadata,
                round_id=manifest.round_id,
                manifest_hash=manifest.manifest_hash,
                execution_profile_hash=self.execution_profile.profile_hash(),
            )

            del trainer, model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reloaded_base = self._load_base_model(torch, transformers)
            reloaded = peft.PeftModel.from_pretrained(
                reloaded_base,
                candidate_path,
                is_trainable=False,
            )
            self._verify_loaded_adapter(reloaded)

            actual_logits = self._probe_logits(torch, reloaded, probe_ids)

            if not torch.allclose(
                expected_logits,
                actual_logits,
                rtol=0,
                atol=1e-4,
            ):
                difference = (expected_logits - actual_logits).abs()
                maximum_index = int(difference.argmax().item())
                expected_value = expected_logits.flatten()[maximum_index].item()
                actual_value = actual_logits.flatten()[maximum_index].item()
                raise RuntimeError(
                    "reloaded PEFT adapter does not preserve probe logits: "
                    f"max_abs_difference={difference.max().item():.8g}, "
                    f"mean_abs_difference={difference.mean().item():.8g}, "
                    f"expected_at_max={expected_value:.8g}, "
                    f"actual_at_max={actual_value:.8g}"
                )

            del reloaded, reloaded_base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.checkpoints.promote(candidate_path, candidate_metadata)
            return BackendTrainingResult(
                parent_version=parent_metadata.version,
                parent_checkpoint_hash=parent_metadata.checkpoint_hash,
                result_version=candidate_metadata.version,
                result_checkpoint_hash=candidate_metadata.checkpoint_hash,
                checkpoint_format="peft-safetensors",
                dependency_versions={
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                    "safetensors": safetensors.__version__,
                    "cuda": torch.version.cuda or "none",
                },
                trainable_parameter_count=trainable_count,
                total_parameter_count=total_count,
                optimizer_step_count=optimizer_step_count,
                training_loss=training_loss,
            )
        except Exception:
            if candidate_path is not None and candidate_path.exists():
                self.checkpoints.discard_staging(candidate_path)
            raise
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_knowledge(
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
        tokenizer = self._load_tokenizer(transformers)
        encoded = encode_reference_samples(
            reference_samples,
            tokenizer=tokenizer,
            model_profile=self.model_profile,
            maximum_sequence_length=manifest.maximum_sequence_length,
            expected_sample_ids=manifest.sample_ids,
        )
        if manifest.top_k > int(self.model_profile.vocabulary_size or 0):
            raise ValueError("manifest top_k exceeds the Client vocabulary size")

        current = self.checkpoints.current()
        if current is None:
            raise RuntimeError("current PEFT adapter checkpoint is missing")
        metadata, checkpoint_path = current
        if (
            metadata.version != expected_adapter_version
            or metadata.checkpoint_hash != expected_checkpoint_hash
        ):
            raise RuntimeError("current PEFT checkpoint differs from the round record")
        if (
            metadata.round_id != manifest.round_id
            or metadata.manifest_hash != manifest.manifest_hash
        ):
            raise RuntimeError("current PEFT checkpoint belongs to another round")

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
                raise RuntimeError("knowledge generation loaded trainable parameters")
            model.eval()

            collator = AnswerOnlyCollator(torch, tokenizer.pad_token_id)
            arguments = FedMKTGenerationArguments(
                top_k_logits_keep=manifest.top_k
            )
            generated: list[KnowledgeSample] = []
            for start in range(0, len(encoded), self.knowledge_batch_size):
                batch_values = encoded[start : start + self.knowledge_batch_size]
                inputs = {
                    "input_ids": [item.input_ids for item in batch_values],
                    "attention_mask": [
                        item.attention_mask for item in batch_values
                    ],
                    "labels": [item.labels for item in batch_values],
                }
                result = generate_pub_data_logits(
                    inputs,
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
                if (
                    token_ids.size(0) != len(batch_values)
                    or logits.size(0) != len(batch_values)
                    or losses.numel() != len(batch_values)
                ):
                    raise RuntimeError("FedMKT returned an invalid batch shape")
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
                raise RuntimeError("generated knowledge changed the signed D^P order")
            return generated
        finally:
            del model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            import peft
            import safetensors
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "real Client training requires the ML dependencies in requirements.txt"
            ) from exc
        return torch, transformers, peft, safetensors

    def _validate_device(self, torch: Any) -> None:
        profile = self.execution_profile
        if profile.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CLIENT_TRAINING_DEVICE=cuda but CUDA is unavailable")
        if profile.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "CLIENT_TRAINING_PRECISION=bfloat16 is unsupported by this GPU"
            )

    def _load_tokenizer(self, transformers: Any) -> Any:
        profile = self.model_profile
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            profile.tokenizer_id,
            revision=profile.tokenizer_revision,
            trust_remote_code=False,
            use_fast=True,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN") or None,
        )
        if tokenizer.__class__.__name__ != profile.tokenizer_class:
            raise RuntimeError(
                f"expected tokenizer class {profile.tokenizer_class}, got "
                f"{tokenizer.__class__.__name__}"
            )
        if tokenizer.padding_side != "right":
            tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            raise RuntimeError("pinned tokenizer has no padding token")
        chat_template = tokenizer.chat_template
        if not isinstance(chat_template, str):
            raise RuntimeError("pinned tokenizer has no default chat template")
        if sha256_hex(chat_template.encode("utf-8")) != (
            profile.tokenizer_chat_template_hash
        ):
            raise RuntimeError("tokenizer chat template differs from ModelProfile")
        return tokenizer

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
            raise RuntimeError("loaded model type differs from ModelProfile")
        if model.config.vocab_size != profile.vocabulary_size:
            raise RuntimeError("loaded vocabulary size differs from ModelProfile")
        resolved_revision = getattr(model.config, "_commit_hash", None)
        if resolved_revision != profile.model_revision:
            raise RuntimeError(
                "loaded model snapshot does not match the pinned revision"
            )
        targets = set(profile.lora.target_modules)
        found = {
            name.rsplit(".", 1)[-1]
            for name, _ in model.named_modules()
            if name.rsplit(".", 1)[-1] in targets
        }
        if found != targets:
            raise RuntimeError(
                f"LoRA targets differ from the model: expected {sorted(targets)}, "
                f"found {sorted(found)}"
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
            raise RuntimeError("adapter rank differs from ModelProfile")
        if float(config.lora_alpha) != lora.alpha:
            raise RuntimeError("adapter alpha differs from ModelProfile")
        if float(config.lora_dropout) != lora.dropout:
            raise RuntimeError("adapter dropout differs from ModelProfile")
        if set(config.target_modules) != set(lora.target_modules):
            raise RuntimeError("adapter targets differ from ModelProfile")
        if config.bias != lora.bias:
            raise RuntimeError("adapter bias differs from ModelProfile")

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
            raise RuntimeError("PEFT model has no trainable parameters")
        unexpected = [name for name in trainable_names if ".lora_" not in name]
        if unexpected:
            raise RuntimeError(
                f"non-LoRA parameters are trainable: {unexpected[:5]}"
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
            content = (
                parameter.detach()
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            digest.update(content)
        if frozen_count == 0:
            raise RuntimeError("PEFT model has no frozen base parameters")
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
            raise RuntimeError("adapter checkpoint contains unexpected tensor keys")
        targets = set(self.model_profile.lora.target_modules)
        found = {
            target
            for target in targets
            if any(f".{target}." in key for key in keys)
        }
        if found != targets:
            raise RuntimeError("adapter checkpoint is missing configured targets")

    @staticmethod
    def _assert_lora_tensors_changed(
        torch: Any,
        safetensors: Any,
        parent_path: Path,
        candidate_path: Path,
    ) -> None:
        parent_tensor_path = parent_path / "adapter_model.safetensors"
        candidate_tensor_path = candidate_path / "adapter_model.safetensors"
        with safetensors.safe_open(
            parent_tensor_path,
            framework="pt",
            device="cpu",
        ) as parent_handle:
            with safetensors.safe_open(
                candidate_tensor_path,
                framework="pt",
                device="cpu",
            ) as candidate_handle:
                parent_keys = set(parent_handle.keys())
                candidate_keys = set(candidate_handle.keys())
                if parent_keys != candidate_keys:
                    raise RuntimeError(
                        "candidate LoRA tensor keys differ from its parent"
                    )
                for key in sorted(parent_keys):
                    parent_tensor = parent_handle.get_tensor(key)
                    candidate_tensor = candidate_handle.get_tensor(key)
                    if (
                        parent_tensor.shape != candidate_tensor.shape
                        or parent_tensor.dtype != candidate_tensor.dtype
                    ):
                        raise RuntimeError(
                            "candidate LoRA tensor structure differs from its parent"
                        )
                    if not torch.equal(parent_tensor, candidate_tensor):
                        return
        raise RuntimeError("real PEFT training did not change any LoRA tensor")

    def _probe_logits(self, torch: Any, model: Any, token_ids: list[int]) -> Any:
        model.eval()
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.execution_profile.device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                use_cache=False,
            ).logits[:, -1, :]
        return logits.detach().float().cpu()


class _ListDataset:
    def __init__(self, examples: list[EncodedTrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }
