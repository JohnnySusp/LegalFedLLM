from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TensorMap = dict[str, list[float]]
HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"


class AdapterError(ValueError):
    pass


class AdapterIntegrityError(AdapterError):
    pass


class AdapterCompatibilityError(AdapterError):
    pass


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _validate_tensor_map(tensors: TensorMap, label: str) -> TensorMap:
    if not tensors:
        raise ValueError(f"{label} must contain at least one tensor")

    for name, values in tensors.items():
        if not name.strip():
            raise ValueError(f"{label} contains an empty tensor name")
        if not values:
            raise ValueError(f"tensor {name!r} must contain at least one value")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"tensor {name!r} contains a non-finite value")

    return tensors


class AdapterSpec(ContractModel):
    format: Literal["legal-fed-llm.mock-lora.v1"] = "legal-fed-llm.mock-lora.v1"
    base_model: str = Field(min_length=1)
    rank: int = Field(gt=0)
    target_modules: tuple[str, ...] = Field(min_length=1)

    @field_validator("base_model")
    @classmethod
    def clean_base_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("base_model must not be blank")
        return cleaned

    @field_validator("target_modules")
    @classmethod
    def clean_target_modules(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("target_modules must not contain blank names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("target_modules must be unique")
        return cleaned


class AdapterSnapshot(ContractModel):
    spec: AdapterSpec
    version: int = Field(ge=0)
    round_id: int = Field(ge=0)
    parent_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    tensors: TensorMap
    adapter_hash: str = Field(pattern=HASH_PATTERN)

    @field_validator("tensors")
    @classmethod
    def valid_tensors(cls, tensors: TensorMap) -> TensorMap:
        return _validate_tensor_map(tensors, "adapter")

    @model_validator(mode="after")
    def valid_hash(self) -> AdapterSnapshot:
        expected = sha256_digest(self.hash_payload())
        if not hmac.compare_digest(self.adapter_hash, expected):
            raise ValueError("adapter_hash does not match the adapter payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"adapter_hash"})

    @classmethod
    def create(
        cls,
        *,
        spec: AdapterSpec,
        version: int,
        round_id: int,
        tensors: TensorMap,
        parent_hash: str | None,
    ) -> AdapterSnapshot:
        payload = {
            "spec": spec.model_dump(mode="json"),
            "version": version,
            "round_id": round_id,
            "parent_hash": parent_hash,
            "tensors": tensors,
        }
        return cls(**payload, adapter_hash=sha256_digest(payload))


class AdapterUpdate(ContractModel):
    spec: AdapterSpec
    round_id: int = Field(gt=0)
    client_id: str = Field(min_length=1, max_length=128)
    parent_adapter_hash: str = Field(pattern=HASH_PATTERN)
    num_examples: int = Field(gt=0)
    delta: TensorMap
    update_hash: str = Field(pattern=HASH_PATTERN)

    @field_validator("client_id")
    @classmethod
    def clean_client_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("client_id must not be blank")
        return cleaned

    @field_validator("delta")
    @classmethod
    def valid_delta(cls, tensors: TensorMap) -> TensorMap:
        return _validate_tensor_map(tensors, "delta")

    @model_validator(mode="after")
    def valid_hash(self) -> AdapterUpdate:
        expected = sha256_digest(self.hash_payload())
        if not hmac.compare_digest(self.update_hash, expected):
            raise ValueError("update_hash does not match the update payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"update_hash"})

    @classmethod
    def create(
        cls,
        *,
        spec: AdapterSpec,
        round_id: int,
        client_id: str,
        parent_adapter_hash: str,
        num_examples: int,
        delta: TensorMap,
    ) -> AdapterUpdate:
        payload = {
            "spec": spec.model_dump(mode="json"),
            "round_id": round_id,
            "client_id": client_id,
            "parent_adapter_hash": parent_adapter_hash,
            "num_examples": num_examples,
            "delta": delta,
        }
        return cls(**payload, update_hash=sha256_digest(payload))


def create_initial_adapter(spec: AdapterSpec) -> AdapterSnapshot:
    tensors = {
        f"adapter.{module}.lora_A": [0.0] * spec.rank
        for module in spec.target_modules
    }
    tensors.update(
        {
            f"adapter.{module}.lora_B": [0.0] * spec.rank
            for module in spec.target_modules
        }
    )
    return AdapterSnapshot.create(
        spec=spec,
        version=0,
        round_id=0,
        tensors=tensors,
        parent_hash=None,
    )


def validate_update(base: AdapterSnapshot, update: AdapterUpdate) -> None:
    if update.parent_adapter_hash != base.adapter_hash:
        raise AdapterCompatibilityError("update is based on a stale global adapter")
    if update.round_id != base.round_id + 1:
        raise AdapterCompatibilityError(
            f"expected round {base.round_id + 1}, received {update.round_id}"
        )
    if update.spec != base.spec:
        raise AdapterCompatibilityError("adapter specification does not match the host")
    if update.delta.keys() != base.tensors.keys():
        raise AdapterCompatibilityError("adapter tensor names do not match the host")

    for name, values in update.delta.items():
        if len(values) != len(base.tensors[name]):
            raise AdapterCompatibilityError(
                f"tensor {name!r} has {len(values)} values; expected {len(base.tensors[name])}"
            )


def apply_update(base: AdapterSnapshot, update: AdapterUpdate) -> AdapterSnapshot:
    validate_update(base, update)
    tensors = {
        name: [current + change for current, change in zip(values, update.delta[name])]
        for name, values in base.tensors.items()
    }
    return AdapterSnapshot.create(
        spec=base.spec,
        version=base.version + 1,
        round_id=update.round_id,
        tensors=tensors,
        parent_hash=base.adapter_hash,
    )
