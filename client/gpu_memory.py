from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIB = 1024 ** 2


@dataclass(frozen=True)
class ClientMemoryBudget:
    total_bytes: int
    limit_bytes: int
    headroom_bytes: int


def client_memory_budget(
    *, total_bytes: int, free_bytes: int, reserved_bytes: int, windows: bool,
) -> ClientMemoryBudget:
    if total_bytes <= 0 or not 0 <= free_bytes <= total_bytes:
        raise ValueError("invalid CUDA memory capacity")
    if not 0 <= reserved_bytes <= total_bytes:
        raise ValueError("invalid CUDA reserved memory")
    headroom = max((1024 if windows else 768) * MIB, total_bytes // 8)
    # Leave additional space even when other processes already consume headroom.
    limit = min(total_bytes - headroom, free_bytes + reserved_bytes - 256 * MIB)
    if limit < 512 * MIB:
        raise RuntimeError("Client GPU memory admission rejected: insufficient free VRAM")
    return ClientMemoryBudget(total_bytes, limit, headroom)


class ClientGpuMemory:
    def __init__(self, torch: Any, data_dir: Path):
        self.torch = torch
        self.path = data_dir / "gpu-memory.jsonl"
        self.budget: ClientMemoryBudget | None = None

    def phase(self, name: str, **details: Any) -> None:
        cuda = self.torch.cuda
        free, total = cuda.mem_get_info()
        value = {
            "time": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "phase": name,
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(cuda.memory_allocated()),
            "reserved_bytes": int(cuda.memory_reserved()),
            "peak_allocated_bytes": int(cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(cuda.max_memory_reserved()),
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def prepare(self) -> None:
        cuda = self.torch.cuda
        free, total = cuda.mem_get_info()
        try:
            self.budget = client_memory_budget(
                total_bytes=int(total), free_bytes=int(free),
                reserved_bytes=int(cuda.memory_reserved()),
                windows=sys.platform == "win32",
            )
        except RuntimeError:
            self.phase("admission_rejected")
            raise
        cuda.set_per_process_memory_fraction(self.budget.limit_bytes / total)
        cuda.reset_peak_memory_stats()
        self.phase("admission", limit_bytes=self.budget.limit_bytes,
                   headroom_bytes=self.budget.headroom_bytes)

    def admit_model(self, model: Any) -> None:
        assert self.budget is not None
        size = sum(t.numel() * t.element_size() for t in model.parameters())
        size += sum(t.numel() * t.element_size() for t in model.buffers())
        allocated = int(self.torch.cuda.memory_allocated())
        self.phase("model_cpu_loaded", model_bytes=size,
                   limit_bytes=self.budget.limit_bytes)
        # A lower-bound check, not an estimate of every training activation.
        required = allocated + size + 512 * MIB
        if required > self.budget.limit_bytes:
            raise RuntimeError(
                "Client GPU memory admission rejected before model transfer: "
                f"model plus minimum workspace needs {required // MIB} MiB; "
                f"budget is {self.budget.limit_bytes // MIB} MiB. "
                "Close other GPU workloads and retry manually. "
                "No CPU fallback or sequence truncation was applied."
            )

    def check_headroom(self) -> None:
        free, _ = self.torch.cuda.mem_get_info()
        if free < 256 * MIB:
            self.phase("headroom_rejected")
            raise RuntimeError(
                "Client GPU memory admission rejected: less than 256 MiB is free; "
                "close other GPU workloads before retrying manually"
            )
