from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.crypto import sha256_hex
from shared.protocol import HASH_PATTERN, ModelProfile, utc_text


class AdapterCheckpointContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AdapterCheckpointMetadata(AdapterCheckpointContract):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str
    profile_hash: str = Field(pattern=HASH_PATTERN)
    model_profile: ModelProfile
    version: int = Field(ge=0)
    parent_version: int | None = Field(default=None, ge=0)
    parent_checkpoint_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    round_id: str | None = None
    manifest_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    execution_profile_hash: str | None = Field(default=None, pattern=HASH_PATTERN)
    checkpoint_hash: str = Field(pattern=HASH_PATTERN)
    file_count: int = Field(ge=2)
    created_at: str


def write_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


class AdapterCheckpointStore:
    def __init__(self, root: str | Path, model_profile: ModelProfile):
        if (
            Path(model_profile.profile_id).name != model_profile.profile_id
            or model_profile.profile_id in {".", ".."}
        ):
            raise ValueError("model profile ID is not safe for adapter storage")
        self.root = Path(root).resolve() / model_profile.profile_id
        self.model_profile = model_profile
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True)

    def staging_path(self, job_id: str) -> Path:
        if Path(job_id).name != job_id or job_id in {".", ".."}:
            raise ValueError("adapter job ID is not safe")
        path = self.root / "staging" / job_id
        path.mkdir(parents=False, exist_ok=False)
        return path

    def discard_staging(self, path: str | Path) -> None:
        candidate = Path(path).resolve()
        staging_root = (self.root / "staging").resolve()
        if candidate.parent != staging_root:
            raise ValueError("adapter staging directory is outside its store")
        shutil.rmtree(candidate, ignore_errors=True)

    def version_path(self, version: int) -> Path:
        return self.root / "versions" / f"v{version:06d}"

    def candidate_path(self, round_id: str, version: int) -> Path:
        if Path(round_id).name != round_id or round_id in {".", ".."}:
            raise ValueError("adapter round ID is not safe")
        return self.root / "candidates" / round_id / f"v{version:06d}"

    def next_version(self, minimum: int) -> int:
        version = minimum
        while self.version_path(version).exists():
            version += 1
        return version

    @staticmethod
    def _tree_hash(path: Path) -> tuple[str, int]:
        required = {"adapter_config.json", "adapter_model.safetensors"}
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.name != "checkpoint.json"
        )
        relative_names = {item.relative_to(path).as_posix() for item in files}
        if not required.issubset(relative_names):
            raise ValueError("PEFT checkpoint is missing its required files")
        entries: list[dict[str, Any]] = []
        for item in files:
            if item.is_symlink():
                raise ValueError("adapter checkpoints must not contain symlinks")
            content = item.read_bytes()
            entries.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        return sha256_hex(entries), len(files)

    def seal(
        self,
        staging_path: Path,
        *,
        version: int,
        parent: AdapterCheckpointMetadata | None,
        round_id: str | None,
        manifest_hash: str | None,
        execution_profile_hash: str | None,
    ) -> AdapterCheckpointMetadata:
        expected_parent = self.root / "staging"
        if staging_path.parent.resolve() != expected_parent.resolve():
            raise ValueError("adapter staging directory is outside its store")
        checkpoint_hash, file_count = self._tree_hash(staging_path)
        metadata = AdapterCheckpointMetadata(
            profile_id=self.model_profile.profile_id,
            profile_hash=self.model_profile.profile_hash(),
            model_profile=self.model_profile,
            version=version,
            parent_version=parent.version if parent else None,
            parent_checkpoint_hash=parent.checkpoint_hash if parent else None,
            round_id=round_id,
            manifest_hash=manifest_hash,
            execution_profile_hash=execution_profile_hash,
            checkpoint_hash=checkpoint_hash,
            file_count=file_count,
            created_at=utc_text(),
        )
        write_atomic_json(
            staging_path / "checkpoint.json",
            metadata.model_dump(mode="json"),
        )
        return metadata

    def promote(
        self,
        staging_path: Path,
        metadata: AdapterCheckpointMetadata,
    ) -> Path:
        self._validate_directory(staging_path, metadata)
        target = self.version_path(metadata.version)
        if target.exists():
            raise FileExistsError(str(target))
        staging_path.replace(target)
        write_atomic_json(
            self.root / "current.json",
            {
                "version": metadata.version,
                "checkpoint_hash": metadata.checkpoint_hash,
                "profile_hash": metadata.profile_hash,
            },
        )
        return target

    def store_candidate(
        self,
        staging_path: Path,
        metadata: AdapterCheckpointMetadata,
    ) -> Path:
        if metadata.round_id is None or metadata.parent_version is None:
            raise ValueError("candidate adapter must belong to a round and parent")
        self._validate_directory(staging_path, metadata)
        target = self.candidate_path(metadata.round_id, metadata.version)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(str(target))
        staging_path.replace(target)
        return target

    def candidate(
        self,
        round_id: str,
        version: int,
    ) -> tuple[AdapterCheckpointMetadata, Path]:
        path = self.candidate_path(round_id, version)
        metadata_path = path / "checkpoint.json"
        if not metadata_path.is_file():
            raise ValueError(
                f"adapter candidate {round_id!r} version {version} is missing"
            )
        metadata = AdapterCheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.round_id != round_id or metadata.version != version:
            raise ValueError("adapter candidate directory has another identity")
        self._validate_directory(path, metadata)
        return metadata, path

    def current(self) -> tuple[AdapterCheckpointMetadata, Path] | None:
        pointer_path = self.root / "current.json"
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if set(pointer) != {"version", "checkpoint_hash", "profile_hash"}:
            raise ValueError("adapter current pointer has an invalid schema")
        metadata, path = self.version(int(pointer["version"]))
        if pointer != {
            "version": metadata.version,
            "checkpoint_hash": metadata.checkpoint_hash,
            "profile_hash": metadata.profile_hash,
        }:
            raise ValueError("adapter current pointer does not match its metadata")
        return metadata, path

    def version(self, version: int) -> tuple[AdapterCheckpointMetadata, Path]:
        path = self.version_path(version)
        metadata_path = path / "checkpoint.json"
        if not metadata_path.is_file():
            raise ValueError(f"adapter checkpoint version {version} is missing")
        metadata = AdapterCheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.version != version:
            raise ValueError("adapter checkpoint directory has another version")
        self._validate_directory(path, metadata)
        return metadata, path

    def _validate_directory(
        self,
        path: Path,
        metadata: AdapterCheckpointMetadata,
    ) -> None:
        if metadata.profile_id != self.model_profile.profile_id:
            raise ValueError("adapter checkpoint has another profile ID")
        if metadata.profile_hash != self.model_profile.profile_hash():
            raise ValueError("adapter checkpoint has an incompatible model profile")
        if metadata.model_profile.profile_hash() != (
            self.model_profile.profile_hash()
        ):
            raise ValueError("adapter checkpoint model metadata is incompatible")
        checkpoint_hash, file_count = self._tree_hash(path)
        if checkpoint_hash != metadata.checkpoint_hash:
            raise ValueError("adapter checkpoint file hash does not match")
        if file_count != metadata.file_count:
            raise ValueError("adapter checkpoint file count does not match")
