from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

from shared.adapter import (
    AdapterSnapshot,
    AdapterSpec,
    AdapterUpdate,
    TensorMap,
    apply_update,
    canonical_json,
    create_initial_adapter,
)


class ClientModelBackend(Protocol):
    name: str
    model_id: str

    def load_adapter(self, adapter: AdapterSnapshot) -> None: ...

    def fine_tune(self, examples: Sequence[str], client_id: str, round_id: int) -> TensorMap: ...

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class MockClientModel:
    name = "mock"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._adapter: AdapterSnapshot | None = None

    def load_adapter(self, adapter: AdapterSnapshot) -> None:
        if adapter.spec.base_model != self.model_id:
            raise ValueError("adapter base model does not match the Client model")
        self._adapter = adapter.model_copy(deep=True)

    def fine_tune(self, examples: Sequence[str], client_id: str, round_id: int) -> TensorMap:
        if self._adapter is None:
            raise RuntimeError("no adapter is loaded")
        material = canonical_json(
            {"client_id": client_id, "examples": list(examples), "round_id": round_id}
        ).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        cursor = 0
        delta: TensorMap = {}

        for name, values in sorted(self._adapter.tensors.items()):
            changes = []
            for _ in values:
                centered = (digest[cursor % len(digest)] - 127.5) / 127.5
                changes.append(round(centered * 0.01, 8))
                cursor += 1
            delta[name] = changes

        return delta

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        if self._adapter is None:
            raise RuntimeError("no adapter is loaded")
        excerpt = " ".join(prompt.split())[:max_new_tokens]
        return f"Client mock response (adapter v{self._adapter.version}): {excerpt}"


class ClientRuntime:
    def __init__(
        self,
        client_id: str,
        backend: ClientModelBackend,
        local_adapter: AdapterSnapshot,
        integration_policy: str = "replace",
    ) -> None:
        if backend.model_id != local_adapter.spec.base_model:
            raise ValueError("Client backend and adapter use different base models")
        if integration_policy != "replace":
            raise ValueError("this milestone supports the replace integration policy only")
        self.client_id = client_id
        self.backend = backend
        self.integration_policy = integration_policy
        self._local_adapter = local_adapter.model_copy(deep=True)
        self.backend.load_adapter(self._local_adapter)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def get_local_adapter(self) -> AdapterSnapshot:
        return self._local_adapter.model_copy(deep=True)

    def install_global_adapter(self, adapter: AdapterSnapshot) -> None:
        if adapter.spec.base_model != self.backend.model_id:
            raise ValueError("Host and Client base models are incompatible")
        self.backend.load_adapter(adapter)
        self._local_adapter = adapter.model_copy(deep=True)

    def prepare_update(
        self, global_adapter: AdapterSnapshot, examples: Sequence[str]
    ) -> AdapterUpdate:
        if not examples:
            raise ValueError("at least one local example is required")
        self.install_global_adapter(global_adapter)
        round_id = global_adapter.round_id + 1
        delta = self.backend.fine_tune(examples, self.client_id, round_id)
        update = AdapterUpdate.create(
            spec=global_adapter.spec,
            round_id=round_id,
            client_id=self.client_id,
            parent_adapter_hash=global_adapter.adapter_hash,
            num_examples=len(examples),
            delta=delta,
        )
        self._local_adapter = apply_update(global_adapter, update)
        self.backend.load_adapter(self._local_adapter)
        return update

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return self.backend.generate(prompt, max_new_tokens)


def create_mock_client_runtime(
    client_id: str = "legal-client-1",
    base_model: str = "legal-poc-mock",
    rank: int = 2,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    integration_policy: str = "replace",
) -> ClientRuntime:
    spec = AdapterSpec(
        base_model=base_model,
        rank=rank,
        target_modules=target_modules,
    )
    return ClientRuntime(
        client_id=client_id,
        backend=MockClientModel(base_model),
        local_adapter=create_initial_adapter(spec),
        integration_policy=integration_policy,
    )
