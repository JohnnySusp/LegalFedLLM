from __future__ import annotations

import json
import os
import secrets
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from client.model_profiles import (
    ollama_model_for_profile,
    supported_profile_ids,
)
from shared.protocol import utc_text

DEFAULT_DESKTOP_SETTINGS = {
    "constant_learning": True,
    "debug_mode": False,
    "low_vram_mode": False,
}


class DesktopProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = "1.0"
    profile_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=128)
    client_id: str = Field(min_length=1, max_length=128)
    model_profile_id: str = Field(min_length=1, max_length=256)
    ssh_target: str = Field(min_length=1, max_length=512)
    ssh_port: int = Field(default=22, ge=1, le=65535)
    coordinator_local_port: int = Field(default=8000, ge=1, le=65535)
    coordinator_remote_host: str = Field(default="127.0.0.1", min_length=1, max_length=253)
    coordinator_remote_port: int = Field(default=8000, ge=1, le=65535)
    agent_port: int = Field(default=8001, ge=1, le=65535)
    created_at: str

    @field_validator("profile_id")
    @classmethod
    def safe_profile_id(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("profile ID is not safe")
        return value

    @field_validator("model_profile_id")
    @classmethod
    def supported_model_profile(cls, value: str) -> str:
        if value not in supported_profile_ids():
            raise ValueError(f"unsupported Client model profile: {value}")
        return value

    @property
    def ollama_model(self) -> str:
        return ollama_model_for_profile(self.model_profile_id)


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    root: Path

    @property
    def profile_json(self) -> Path:
        return self.root / "profile.json"

    @property
    def env_file(self) -> Path:
        return self.root / "profile.env"

    @property
    def client_data(self) -> Path:
        return self.root / "client-data"

    @property
    def private_train(self) -> Path:
        return self.root / "private" / "train.jsonl"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


class PortableProfileManager:
    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root or portable_data_root()).resolve()
        self.profiles_root = self.data_root / "profiles"
        self.models_root = self.data_root / "models" / "huggingface"
        self.state_path = self.data_root / "desktop-state.json"
        self.profiles_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def profile_paths(self, profile_id: str) -> ProfilePaths:
        if Path(profile_id).name != profile_id or profile_id in {".", ".."}:
            raise ValueError("profile ID is not safe")
        return ProfilePaths(self.profiles_root / profile_id)

    def list_profiles(self) -> list[DesktopProfile]:
        profiles: list[DesktopProfile] = []
        for path in sorted(self.profiles_root.glob("*/profile.json")):
            try:
                profiles.append(
                    DesktopProfile.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError):
                continue
        profiles.sort(key=lambda item: (item.display_name.lower(), item.profile_id))
        return profiles

    def load(self, profile_id: str) -> DesktopProfile:
        paths = self.profile_paths(profile_id)
        if not paths.profile_json.is_file():
            raise ValueError("profile does not exist")
        return DesktopProfile.model_validate_json(
            paths.profile_json.read_text(encoding="utf-8")
        )

    def create(
        self,
        *,
        display_name: str,
        model_profile_id: str,
        ssh_target: str,
        ssh_port: int,
        coordinator_local_port: int = 8000,
        coordinator_remote_host: str = "127.0.0.1",
        coordinator_remote_port: int = 8000,
        agent_port: int = 8001,
    ) -> DesktopProfile:
        profile_id = f"profile-{uuid.uuid4().hex[:16]}"
        profile = DesktopProfile(
            profile_id=profile_id,
            display_name=display_name.strip(),
            client_id=f"client-{uuid.uuid4().hex[:20]}",
            model_profile_id=model_profile_id,
            ssh_target=ssh_target.strip(),
            ssh_port=ssh_port,
            coordinator_local_port=coordinator_local_port,
            coordinator_remote_host=coordinator_remote_host,
            coordinator_remote_port=coordinator_remote_port,
            agent_port=agent_port,
            created_at=utc_text(),
        )
        paths = self.profile_paths(profile.profile_id)
        paths.root.mkdir(parents=False, exist_ok=False)
        paths.client_data.mkdir(parents=True, exist_ok=True)
        paths.private_train.parent.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)
        self._write_json(paths.profile_json, profile.model_dump(mode="json"))
        self._write_secret_env(
            paths.env_file,
            {"CLIENT_ADMIN_TOKEN": secrets.token_urlsafe(32)},
        )
        self.set_active(profile.profile_id)
        return profile

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def active_profile_id(self) -> str | None:
        value = self._read_state().get("active_profile_id")
        return value if isinstance(value, str) else None

    def desktop_settings(self) -> dict[str, bool]:
        payload = self._read_state().get("settings")
        stored = payload if isinstance(payload, dict) else {}
        settings: dict[str, bool] = {}
        for key, default in DEFAULT_DESKTOP_SETTINGS.items():
            value = stored.get(key, default)
            settings[key] = value if isinstance(value, bool) else default
        return settings

    def set_desktop_setting(self, key: str, value: bool) -> None:
        if key not in DEFAULT_DESKTOP_SETTINGS:
            raise ValueError(f"unsupported desktop setting: {key}")
        state = self._read_state()
        settings = self.desktop_settings()
        settings[key] = bool(value)
        state.update({"schema_version": "1.0", "settings": settings})
        self._write_json(self.state_path, state)

    def reset_desktop_settings(self) -> dict[str, bool]:
        state = self._read_state()
        settings = dict(DEFAULT_DESKTOP_SETTINGS)
        state.update({"schema_version": "1.0", "settings": settings})
        self._write_json(self.state_path, state)
        return settings

    def active_profile(self) -> DesktopProfile | None:
        profile_id = self.active_profile_id()
        if profile_id is None:
            return None
        try:
            return self.load(profile_id)
        except ValueError:
            return None

    def set_active(self, profile_id: str) -> None:
        self.load(profile_id)
        state = self._read_state()
        state.update(
            {
                "schema_version": "1.0",
                "active_profile_id": profile_id,
                "settings": self.desktop_settings(),
            }
        )
        self._write_json(self.state_path, state)

    def admin_token(self, profile_id: str) -> str:
        values = self._read_env(self.profile_paths(profile_id).env_file)
        token = values.get("CLIENT_ADMIN_TOKEN", "").strip()
        if not token:
            raise ValueError("profile Client admin token is missing")
        return token

    def agent_environment(
        self,
        profile: DesktopProfile,
        *,
        enrollment_token: str | None = None,
    ) -> dict[str, str]:
        paths = self.profile_paths(profile.profile_id)
        env = dict(os.environ)
        env.update(
            {
                "CLIENT_ID": profile.client_id,
                "CLIENT_DATA_DIR": str(paths.client_data),
                "CLIENT_PRIVATE_DATA_PATH": str(paths.private_train),
                "CLIENT_PRIVATE_DATASET_ID": f"{profile.client_id}-local-learning-v1",
                "CLIENT_MODEL_PROFILE": profile.model_profile_id,
                "CLIENT_TRAINING_BACKEND": "transformers",
                "CLIENT_SERVING_BACKEND": "transformers",
                "CLIENT_ADMIN_TOKEN": self.admin_token(profile.profile_id),
                "CLIENT_TRAINING_DEVICE": env.get("CLIENT_TRAINING_DEVICE", "cuda"),
                "CLIENT_TRAINING_PRECISION": env.get("CLIENT_TRAINING_PRECISION", "bfloat16"),
                "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                "HF_HOME": str(self.models_root),
                "COORDINATOR_URL": f"http://127.0.0.1:{profile.coordinator_local_port}",
                "CLIENT_SSH_TUNNEL_ENABLED": "true",
                "CLIENT_SSH_TARGET": profile.ssh_target,
                "CLIENT_SSH_PORT": str(profile.ssh_port),
                "CLIENT_COORDINATOR_LOCAL_PORT": str(profile.coordinator_local_port),
                "CLIENT_COORDINATOR_REMOTE_HOST": profile.coordinator_remote_host,
                "CLIENT_COORDINATOR_REMOTE_PORT": str(profile.coordinator_remote_port),
                "CLIENT_AGENT_HOST": "127.0.0.1",
                "CLIENT_AGENT_PORT": str(profile.agent_port),
            }
        )
        if self.desktop_settings()["low_vram_mode"]:
            env["CLIENT_GRADIENT_CHECKPOINTING"] = "true"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            env["CLIENT_GRADIENT_CHECKPOINTING"] = "false"
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if enrollment_token:
            env["REGISTRATION_TOKEN"] = enrollment_token
        else:
            env.pop("REGISTRATION_TOKEN", None)
        return env

    @staticmethod
    def _read_env(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        if not path.is_file():
            return values
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        return values

    @staticmethod
    def _write_secret_env(path: Path, values: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            "".join(f"{key}={value}\n" for key, value in sorted(values.items())),
            encoding="utf-8",
        )
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def portable_install_root() -> Path:
    appimage = os.getenv("APPIMAGE", "").strip()
    if appimage:
        return Path(appimage).resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def portable_data_root() -> Path:
    override = os.getenv("LEGALFEDLLM_DATA_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return portable_install_root() / "LegalFedLLM-data"
