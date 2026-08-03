from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any


class JsonFileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def path(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("storage path escapes the configured root")
        return candidate

    def exists(self, relative: str | Path) -> bool:
        return self.path(relative).exists()

    def read_json(self, relative: str | Path) -> dict[str, Any]:
        with self._lock:
            return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def write_json(self, relative: str | Path, value: Any) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with self._lock:
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(target)
        return target

    def write_json_if_absent(self, relative: str | Path, value: Any) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        with self._lock:
            if target.exists():
                raise FileExistsError(str(target))
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(target)
        return target

    def delete(self, relative: str | Path, *, missing_ok: bool = True) -> None:
        target = self.path(relative)
        with self._lock:
            try:
                target.unlink()
            except FileNotFoundError:
                if not missing_ok:
                    raise

    def append_jsonl(self, relative: str | Path, value: Any) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return target

    def list_json(self, relative: str | Path) -> list[Path]:
        directory = self.path(relative)
        if not directory.exists():
            return []
        return sorted(path for path in directory.glob("*.json") if path.is_file())
