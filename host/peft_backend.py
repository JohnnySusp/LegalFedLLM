from __future__ import annotations

import gc
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from host.training import (
    HostAdapterInitializationRecord,
    HostStageFiveContract,
    HostTrainingExecutionProfile,
)
from shared.adapter_checkpoint import (
    AdapterCheckpointMetadata,
    AdapterCheckpointStore,
    write_atomic_json,
)
from shared.alignment_profiles import POC_DTW_PROFILE
from shared.protocol import ModelProfile, utc_text
from shared.tokenizer_validation import load_pinned_tokenizer


HOST_INITIALIZATION_RECORD = "host_initialization.json"
HOST_INITIALIZATION_PROBE = "LegalFedLLM Host adapter initialization probe."


@dataclass(frozen=True, slots=True)
class InitializedHostAdapter:
    metadata: AdapterCheckpointMetadata
    checkpoint_path: Path
    initialization: HostAdapterInitializationRecord


class TransformersPeftHostBackend:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        model_profile: ModelProfile,
        execution_profile: HostTrainingExecutionProfile,
    ):
        if model_profile.training_backend != "transformers":
            raise ValueError(
                "real Host adapter lifecycle requires a Transformers profile"
            )
        self.data_dir = Path(data_dir).resolve()
        self.model_profile = model_profile
        self.execution_profile = execution_profile
        self.contract = HostStageFiveContract.create(model_profile)
        self.checkpoints = AdapterCheckpointStore(
            self.data_dir / "adapters",
            model_profile,
        )

    def initialize_adapter(self) -> InitializedHostAdapter:
        current = self.checkpoints.current()
        if current is not None:
            metadata, path = current
            record = self._load_initialization_record(path)
            self._validate_initialization(metadata, record)
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
            raise ValueError("Host initialization uses another Step 5 contract")
        if record.execution_profile_hash != self.execution_profile.profile_hash():
            raise ValueError(
                "Host initialization uses another execution profile"
            )
        if metadata.execution_profile_hash != record.execution_profile_hash:
            raise ValueError(
                "Host initialization execution profile differs from checkpoint"
            )

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
