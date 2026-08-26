from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from safetensors import SafetensorError, safe_open

from client.model_profiles import QWEN_PROFILE_ID
from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile, parse_utc, utc_text


SAFED_PROBE_THRESHOLD = 0.8
_WEIGHT_NAMES = {"linear_weight", "linear_bias"}
_QWEN_LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}
_FIRST_LAYER_B = re.compile(
    r"(?:^|\.)layers\.0\..*\.lora_B(?:\.[^.]+)?\.weight$"
)


class SafetyProbeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SafeFedProbeManifest(SafetyProbeContract):
    schema_version: Literal["1.0"] = "1.0"
    probe_id: str = Field(min_length=1, max_length=256)
    probe_version: str = Field(min_length=1, max_length=128)
    classifier_family: Literal["linear_probe"] = "linear_probe"
    feature_policy: Literal["first_layer_lora_b_delta"] = (
        "first_layer_lora_b_delta"
    )
    normalization: Literal["l2"] = "l2"
    model_profile_id: Literal["qwen3-1.7b-lora-v1"] = QWEN_PROFILE_ID
    model_profile_hash: str = Field(pattern=HASH_PATTERN)
    model_revision: str = Field(min_length=1, max_length=256)
    lora_profile_hash: str = Field(pattern=HASH_PATTERN)
    compatible_parent_policy: Literal["same_parent_lora_delta"] = (
        "same_parent_lora_delta"
    )
    ordered_parameter_keys: list[str] = Field(min_length=1, max_length=1024)
    parameter_sizes: dict[str, int]
    input_dimension: int = Field(gt=0, le=100_000_000)
    harmful_threshold: Literal[0.8] = SAFED_PROBE_THRESHOLD
    weights_file: str = Field(min_length=1, max_length=255)
    weights_byte_size: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    weights_sha256: str = Field(pattern=HASH_PATTERN)
    training_corpus_hash: str = Field(pattern=HASH_PATTERN)
    validation_corpus_hash: str = Field(pattern=HASH_PATTERN)
    created_at: str
    artifact_hash: str = Field(pattern=HASH_PATTERN)

    @field_validator("weights_file")
    @classmethod
    def validate_weights_name(cls, value: str) -> str:
        if Path(value).name != value or value != "linear_probe.safetensors":
            raise ValueError(
                "SafeFed probe weights_file must be linear_probe.safetensors"
            )
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> "SafeFedProbeManifest":
        parse_utc(self.created_at)
        if len(self.ordered_parameter_keys) != len(
            set(self.ordered_parameter_keys)
        ):
            raise ValueError("SafeFed probe parameter keys must be unique")
        if set(self.parameter_sizes) != set(self.ordered_parameter_keys):
            raise ValueError("SafeFed probe parameter sizes differ from its keys")
        for key in self.ordered_parameter_keys:
            if not _FIRST_LAYER_B.search(key):
                raise ValueError(
                    "SafeFed probe accepts only first-layer LoRA-B tensors"
                )
            if self.parameter_sizes[key] < 1:
                raise ValueError("SafeFed probe parameter sizes must be positive")
        found_targets = {
            target
            for target in _QWEN_LORA_TARGETS
            if any(f".{target}.lora_B" in key for key in self.ordered_parameter_keys)
        }
        if (
            found_targets != _QWEN_LORA_TARGETS
            or len(self.ordered_parameter_keys) != len(_QWEN_LORA_TARGETS)
        ):
            raise ValueError(
                "SafeFed probe must cover every first-layer Qwen LoRA-B target"
            )
        if sum(self.parameter_sizes.values()) != self.input_dimension:
            raise ValueError("SafeFed probe input dimension differs from its keys")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"artifact_hash"})
        )
        if self.artifact_hash != expected:
            raise ValueError("SafeFed probe artifact hash differs")
        return self

    @classmethod
    def create(cls, **values: Any) -> "SafeFedProbeManifest":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"artifact_hash"}
        )
        return cls(**payload, artifact_hash=sha256_hex(payload))


class ClientSafetyProbeReport(SafetyProbeContract):
    schema_version: Literal["1.0"] = "1.0"
    round_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(pattern=HASH_PATTERN)
    job_hash: str = Field(pattern=HASH_PATTERN)
    probe_id: str = Field(min_length=1, max_length=256)
    probe_version: str = Field(min_length=1, max_length=128)
    probe_artifact_hash: str = Field(pattern=HASH_PATTERN)
    model_profile_hash: str = Field(pattern=HASH_PATTERN)
    parent_checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    candidate_checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    delta_sha256: str = Field(pattern=HASH_PATTERN)
    maliciousness_probability: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    harmful_threshold: Literal[0.8] = SAFED_PROBE_THRESHOLD
    safety_gate_passed: bool
    created_at: str
    report_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_report(self) -> "ClientSafetyProbeReport":
        parse_utc(self.created_at)
        if self.safety_gate_passed != (
            self.maliciousness_probability < self.harmful_threshold
        ):
            raise ValueError("SafeFed probe decision differs from its probability")
        expected = sha256_hex(
            self.model_dump(mode="json", exclude={"report_hash"})
        )
        if self.report_hash != expected:
            raise ValueError("SafeFed probe report hash differs")
        return self

    @classmethod
    def create(cls, **values: Any) -> "ClientSafetyProbeReport":
        payload = cls.model_construct(**values).model_dump(
            mode="json", exclude={"report_hash"}
        )
        return cls(**payload, report_hash=sha256_hex(payload))


class SafeFedLoraProbe:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        model_profile: ModelProfile,
    ):
        path = Path(manifest_path).resolve()
        try:
            manifest = SafeFedProbeManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"SafeFed probe manifest is invalid: {exc}") from exc
        if model_profile.profile_id != QWEN_PROFILE_ID:
            raise ValueError("the current SafeFed probe contract is Qwen-specific")
        if manifest.model_profile_hash != model_profile.profile_hash():
            raise ValueError("SafeFed probe uses another Client model profile")
        if manifest.model_revision != model_profile.model_revision:
            raise ValueError("SafeFed probe uses another model revision")
        if manifest.lora_profile_hash != sha256_hex(
            model_profile.lora.model_dump(mode="json")
        ):
            raise ValueError("SafeFed probe uses another LoRA profile")

        weights_path = path.parent / manifest.weights_file
        try:
            content = weights_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"SafeFed probe weights are unavailable: {exc}") from exc
        if len(content) != manifest.weights_byte_size:
            raise ValueError("SafeFed probe weights byte size differs")
        if hashlib.sha256(content).hexdigest() != manifest.weights_sha256:
            raise ValueError("SafeFed probe weights SHA-256 differs")
        try:
            with safe_open(weights_path, framework="np") as handle:
                if handle.metadata():
                    raise ValueError("SafeFed probe weights metadata is not allowed")
                if set(handle.keys()) != _WEIGHT_NAMES:
                    raise ValueError("SafeFed probe tensor names are invalid")
                linear_weight = handle.get_tensor("linear_weight")
                linear_bias = handle.get_tensor("linear_bias")
        except (OSError, SafetensorError) as exc:
            raise ValueError("SafeFed probe safetensors are invalid") from exc
        if linear_weight.dtype != np.float32 or linear_bias.dtype != np.float32:
            raise ValueError("SafeFed probe tensors must use float32")
        if linear_weight.shape != (manifest.input_dimension,):
            raise ValueError("SafeFed probe linear weight shape differs")
        if linear_bias.shape != (1,):
            raise ValueError("SafeFed probe linear bias shape differs")
        if not np.isfinite(linear_weight).all() or not np.isfinite(
            linear_bias
        ).all():
            raise ValueError("SafeFed probe tensors must be finite")

        self.manifest = manifest
        self.linear_weight = np.ascontiguousarray(linear_weight)
        self.linear_bias = float(linear_bias[0])

    def evaluate(
        self,
        *,
        round_id: str,
        manifest_hash: str,
        job_hash: str,
        parent_checkpoint_hash: str,
        candidate_checkpoint_hash: str,
        parent_path: str | Path,
        candidate_path: str | Path,
        created_at: str | None = None,
    ) -> ClientSafetyProbeReport:
        parent_tensors = self._load_adapter_tensors(parent_path)
        candidate_tensors = self._load_adapter_tensors(candidate_path)
        if set(parent_tensors) != set(candidate_tensors):
            raise ValueError("SafeFed candidate tensor keys differ from its parent")

        features: list[np.ndarray] = []
        delta_entries: list[dict[str, Any]] = []
        for key in self.manifest.ordered_parameter_keys:
            if key not in parent_tensors:
                raise ValueError(f"SafeFed probe parameter is missing: {key}")
            parent = parent_tensors[key]
            candidate = candidate_tensors[key]
            if parent.shape != candidate.shape or parent.dtype != candidate.dtype:
                raise ValueError("SafeFed candidate tensor structure differs")
            expected_size = self.manifest.parameter_sizes[key]
            if parent.size != expected_size:
                raise ValueError("SafeFed probe parameter size differs")
            delta = np.ascontiguousarray(
                candidate.astype(np.float32) - parent.astype(np.float32)
            )
            if not np.isfinite(delta).all():
                raise ValueError("SafeFed candidate delta is not finite")
            flattened = delta.reshape(-1)
            features.append(flattened)
            delta_entries.append(
                {
                    "key": key,
                    "shape": list(delta.shape),
                    "sha256": hashlib.sha256(delta.tobytes()).hexdigest(),
                }
            )

        feature = np.concatenate(features).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(feature, ord=2))
        if norm >= 1e-12:
            feature = feature / norm
        logit = float(np.dot(self.linear_weight, feature) + self.linear_bias)
        if not math.isfinite(logit):
            raise ValueError("SafeFed probe produced a non-finite logit")
        if logit >= 0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            probability = exponential / (1.0 + exponential)
        return ClientSafetyProbeReport.create(
            round_id=round_id,
            manifest_hash=manifest_hash,
            job_hash=job_hash,
            probe_id=self.manifest.probe_id,
            probe_version=self.manifest.probe_version,
            probe_artifact_hash=self.manifest.artifact_hash,
            model_profile_hash=self.manifest.model_profile_hash,
            parent_checkpoint_hash=parent_checkpoint_hash,
            candidate_checkpoint_hash=candidate_checkpoint_hash,
            delta_sha256=sha256_hex(delta_entries),
            maliciousness_probability=probability,
            harmful_threshold=SAFED_PROBE_THRESHOLD,
            safety_gate_passed=probability < SAFED_PROBE_THRESHOLD,
            created_at=created_at or utc_text(),
        )

    @staticmethod
    def _load_adapter_tensors(path: str | Path) -> dict[str, np.ndarray]:
        tensor_path = Path(path) / "adapter_model.safetensors"
        try:
            with safe_open(tensor_path, framework="np") as handle:
                metadata = handle.metadata()
                if metadata not in (None, {}, {"format": "pt"}):
                    raise ValueError("PEFT adapter metadata is invalid")
                tensors = {
                    key: np.ascontiguousarray(handle.get_tensor(key))
                    for key in handle.keys()
                }
        except (OSError, SafetensorError) as exc:
            raise ValueError("PEFT adapter safetensors are invalid") from exc
        if not tensors or any(".lora_" not in key for key in tensors):
            raise ValueError("PEFT adapter contains unexpected tensors")
        return tensors
