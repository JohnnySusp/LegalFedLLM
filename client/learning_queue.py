from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from client.training import (
    PrivateTrainingExample,
    load_private_examples,
    private_dataset_semantic_hash,
)
from shared.protocol import utc_text


@dataclass(frozen=True, slots=True)
class LearningBatch:
    batch_id: str
    path: Path
    examples: tuple[PrivateTrainingExample, ...]
    semantic_hash: str

    @property
    def example_count(self) -> int:
        return len(self.examples)


class LearningQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.inflight_dir = self.path.parent / ".learning-inflight"
        self.receipts_dir = self.path.parent / ".learning-receipts"
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self._recover_inflight()

    def _recover_inflight(self) -> None:
        inflight = sorted(self.inflight_dir.glob("*.jsonl"))
        if not inflight:
            return
        chunks = [item.read_bytes() for item in inflight]
        if self.path.exists():
            chunks.append(self.path.read_bytes())
        payload = b"".join(chunks)
        temporary = self.path.with_name(f".{self.path.name}.recover.tmp")
        temporary.write_bytes(payload)
        if payload:
            load_private_examples(temporary)
        temporary.replace(self.path)
        for item in inflight:
            item.unlink(missing_ok=True)

    def _active_examples(self) -> list[PrivateTrainingExample]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return []
        return load_private_examples(self.path)

    def status(self) -> dict[str, Any]:
        with self._lock:
            examples = self._active_examples()
            return {
                "queued_example_count": len(examples),
                "queued_dataset_hash": (
                    private_dataset_semantic_hash(examples) if examples else None
                ),
            }

    def append(self, prompt: str, answer: str) -> PrivateTrainingExample:
        example = PrivateTrainingExample(
            example_id=f"learn-{secrets.token_hex(16)}",
            prompt=prompt,
            answer=answer,
        )
        with self._lock:
            existing = self._active_examples()
            if len(existing) >= 100_000:
                raise ValueError("local learning queue exceeds 100000 examples")
            line = json.dumps(
                example.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return example

    def begin(self) -> LearningBatch | None:
        with self._lock:
            examples = self._active_examples()
            if not examples:
                return None
            batch_id = f"learning-{secrets.token_hex(12)}"
            target = self.inflight_dir / f"{batch_id}.jsonl"
            if target.exists():
                raise FileExistsError(str(target))
            self.path.replace(target)
            return LearningBatch(
                batch_id=batch_id,
                path=target,
                examples=tuple(examples),
                semantic_hash=private_dataset_semantic_hash(examples),
            )

    def restore(self, batch: LearningBatch) -> None:
        with self._lock:
            if not batch.path.is_file():
                return
            current = self.path.read_bytes() if self.path.exists() else b""
            payload = batch.path.read_bytes() + current
            temporary = self.path.with_name(f".{self.path.name}.restore.tmp")
            temporary.write_bytes(payload)
            if payload:
                load_private_examples(temporary)
            temporary.replace(self.path)
            batch.path.unlink(missing_ok=True)

    def complete(
        self,
        batch: LearningBatch,
        *,
        adapter_version: int,
        checkpoint_hash: str,
        round_id: str,
        training_record_hash: str,
    ) -> dict[str, Any]:
        with self._lock:
            if not batch.path.is_file():
                raise ValueError("local learning batch is no longer pending")
            receipt = {
                "schema_version": "1.0",
                "batch_id": batch.batch_id,
                "round_id": round_id,
                "consumed_example_count": batch.example_count,
                "training_dataset_hash": batch.semantic_hash,
                "adapter_version": adapter_version,
                "checkpoint_hash": checkpoint_hash,
                "training_record_hash": training_record_hash,
                "completed_at": utc_text(),
            }
            target = self.receipts_dir / f"{batch.batch_id}.json"
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
            batch.path.unlink()
            return receipt
