from __future__ import annotations

from threading import RLock
from typing import Protocol

from shared.adapter import AdapterSnapshot, AdapterSpec, AdapterUpdate, apply_update, create_initial_adapter


class HostModelBackend(Protocol):
    name: str
    model_id: str

    def load_adapter(self, adapter: AdapterSnapshot) -> None: ...

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class MockHostModel:
    name = "mock"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._adapter: AdapterSnapshot | None = None

    def load_adapter(self, adapter: AdapterSnapshot) -> None:
        if adapter.spec.base_model != self.model_id:
            raise ValueError("adapter base model does not match the Host model")
        self._adapter = adapter.model_copy(deep=True)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        if self._adapter is None:
            raise RuntimeError("no adapter is loaded")
        excerpt = " ".join(prompt.split())[:max_new_tokens]
        return f"Host mock response (adapter v{self._adapter.version}): {excerpt}"


class HostRuntime:
    def __init__(self, backend: HostModelBackend, global_adapter: AdapterSnapshot) -> None:
        if backend.model_id != global_adapter.spec.base_model:
            raise ValueError("Host backend and adapter use different base models")
        self.backend = backend
        self._global_adapter = global_adapter.model_copy(deep=True)
        self._lock = RLock()
        self.backend.load_adapter(self._global_adapter)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def get_global_adapter(self) -> AdapterSnapshot:
        with self._lock:
            return self._global_adapter.model_copy(deep=True)

    def submit_update(self, update: AdapterUpdate) -> AdapterSnapshot:
        with self._lock:
            updated = apply_update(self._global_adapter, update)
            self.backend.load_adapter(updated)
            self._global_adapter = updated
            return updated.model_copy(deep=True)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        with self._lock:
            return self.backend.generate(prompt, max_new_tokens)


def create_mock_host_runtime(
    base_model: str = "legal-poc-mock",
    rank: int = 2,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> HostRuntime:
    spec = AdapterSpec(
        base_model=base_model,
        rank=rank,
        target_modules=target_modules,
    )
    return HostRuntime(MockHostModel(base_model), create_initial_adapter(spec))
