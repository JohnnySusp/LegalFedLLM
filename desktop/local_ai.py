from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from desktop.profiles import DesktopProfile


BUNDLE_FILES = (
    "compose.yaml",
    "anythingllm.env",
    "anythingllm.docker.env",
    "legalfedllm.network.yaml",
)
NETWORK_NAME = "legalfed-ai-net"
OLLAMA_URL = "http://127.0.0.1:11434"
ANYTHINGLLM_URL = "http://127.0.0.1:3001"
_PLACEHOLDER_SECRETS = {
    "",
    "PASTE_RANDOM_SECRET_HERE",
    "GENERATED_BY_LEGALFEDLLM",
}


def bundled_local_ai_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve() / "legalfed-ai"
    return Path(__file__).resolve().parents[1] / "legalfed-ai"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _write_env(path: Path, values: dict[str, str]) -> None:
    ordered = (
        "STORAGE_DIR",
        "JWT_SECRET",
        "LLM_PROVIDER",
        "GENERIC_OPEN_AI_BASE_PATH",
        "GENERIC_OPEN_AI_MODEL_PREF",
        "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT",
        "GENERIC_OPEN_AI_MAX_TOKENS",
        "GENERIC_OPEN_AI_API_KEY",
        "GENERIC_OPENAI_STREAMING_DISABLED",
        "PROVIDER_DISABLE_NATIVE_TOOL_CALLING",
        "ANYTHINGLLM_FETCH_TIMEOUT",
        "ANYTHINGLLM_MAX_RETRIES",
        "EMBEDDING_ENGINE",
        "EMBEDDING_MODEL_PREF",
        "VECTOR_DB",
        "WHISPER_PROVIDER",
        "TTS_PROVIDER",
        "PASSWORDMINCHAR",
    )
    lines: list[str] = []
    seen: set[str] = set()
    for key in ordered:
        if key in values:
            lines.append(f"{key}={values[key]}")
            seen.add(key)
    for key in sorted(values):
        if key in seen or key.startswith("OLLAMA_"):
            continue
        lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


class LocalAiStack:
    def __init__(
        self,
        data_root: str | Path,
        *,
        bundle_root: str | Path | None = None,
        legacy_root: str | Path | None = None,
        platform: str | None = None,
    ):
        self.data_root = Path(data_root).resolve()
        self.platform = platform or sys.platform
        self.runtime_root = self.data_root / "legalfed-ai"
        self.bundle_root = Path(bundle_root or bundled_local_ai_root()).resolve()
        self.legacy_root = Path(
            legacy_root or (Path.home() / "legalfed-ai")
        ).expanduser().resolve()

    @property
    def is_windows(self) -> bool:
        return self.platform.startswith("win")

    def ensure_runtime_files(self) -> Path:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        first_seed = not any((self.runtime_root / name).exists() for name in BUNDLE_FILES)
        source_root = self.bundle_root
        if first_seed and all((self.legacy_root / name).is_file() for name in BUNDLE_FILES):
            source_root = self.legacy_root

        for name in BUNDLE_FILES:
            destination = self.runtime_root / name
            if destination.exists():
                continue
            source = source_root / name
            if not source.is_file():
                source = self.bundle_root / name
            if not source.is_file():
                raise RuntimeError(f"Bundled legalfed-ai file is missing: {name}")
            shutil.copy2(source, destination)
        return self.runtime_root

    def configure_anythingllm(
        self,
        profile: DesktopProfile,
        admin_token: str,
    ) -> None:
        self.ensure_runtime_files()
        docker_env = self.runtime_root / "anythingllm.docker.env"
        base_values = _read_env(docker_env)
        jwt_secret = base_values.get("JWT_SECRET", "").strip()
        if jwt_secret in _PLACEHOLDER_SECRETS or jwt_secret.lower().startswith("placeholder"):
            jwt_secret = secrets.token_hex(32)

        updates = {
            "STORAGE_DIR": base_values.get("STORAGE_DIR", "/app/server/storage"),
            "JWT_SECRET": jwt_secret,
            "LLM_PROVIDER": "generic-openai",
            "GENERIC_OPEN_AI_BASE_PATH": f"http://127.0.0.1:{profile.agent_port}/v1",
            "GENERIC_OPEN_AI_MODEL_PREF": "legalfedllm-local",
            "GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT": "8192",
            "GENERIC_OPEN_AI_MAX_TOKENS": "1024",
            "GENERIC_OPEN_AI_API_KEY": admin_token,
            "GENERIC_OPENAI_STREAMING_DISABLED": "true",
            "PROVIDER_DISABLE_NATIVE_TOOL_CALLING": "generic-openai",
            "ANYTHINGLLM_FETCH_TIMEOUT": "1800000",
            "ANYTHINGLLM_MAX_RETRIES": "0",
            "EMBEDDING_ENGINE": base_values.get("EMBEDDING_ENGINE", "native"),
            "EMBEDDING_MODEL_PREF": base_values.get(
                "EMBEDDING_MODEL_PREF", "Xenova/all-MiniLM-L6-v2"
            ),
            "VECTOR_DB": base_values.get("VECTOR_DB", "lancedb"),
            "WHISPER_PROVIDER": base_values.get("WHISPER_PROVIDER", "local"),
            "TTS_PROVIDER": base_values.get("TTS_PROVIDER", "native"),
            "PASSWORDMINCHAR": base_values.get("PASSWORDMINCHAR", "8"),
        }
        for key, value in base_values.items():
            if key.startswith("OLLAMA_"):
                continue
            updates.setdefault(key, value)

        _write_env(docker_env, updates)
        _write_env(self.runtime_root / "anythingllm.env", dict(updates))

    def prepare(self, profile: DesktopProfile, admin_token: str) -> dict[str, Any]:
        if self.is_windows:
            return self._prepare_windows(profile)

        self.configure_anythingllm(profile, admin_token)
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker was not found. LegalFedLLM cannot start Ollama/AnythingLLM.")

        inspect = self._run([docker, "network", "inspect", NETWORK_NAME], allow_failure=True)
        if inspect.returncode != 0:
            self._run([docker, "network", "create", NETWORK_NAME])

        compose = [docker, "compose", "-f", "compose.yaml"]
        self._run([*compose, "up", "-d", "ollama"])
        self._wait_for_ollama()
        self._verify_ollama_model(profile.ollama_model)
        self._run([*compose, "up", "-d", "--force-recreate", "anythingllm"])
        return {
            "mode": "linux-docker",
            "runtime_root": str(self.runtime_root),
            "ollama_model": profile.ollama_model,
            "anythingllm_url": ANYTHINGLLM_URL,
            "openai_base_url": f"http://127.0.0.1:{profile.agent_port}/v1",
            "anythingllm_managed": True,
        }

    def _prepare_windows(self, profile: DesktopProfile) -> dict[str, Any]:
        if shutil.which("ollama") is None:
            raise RuntimeError(
                "Native Ollama for Windows was not found on PATH. Install/start Ollama for Windows "
                "before launching LegalFedLLM."
            )
        self._wait_for_ollama()
        self._verify_ollama_model(profile.ollama_model)
        return {
            "mode": "windows-native",
            "ollama_model": profile.ollama_model,
            "openai_base_url": f"http://127.0.0.1:{profile.agent_port}/v1",
            "anythingllm_managed": False,
        }

    def running_services(self) -> set[str]:
        if self.is_windows:
            return set()
        if not self.runtime_root.is_dir():
            return set()
        docker = shutil.which("docker")
        if docker is None:
            return set()
        result = self._run(
            [
                docker,
                "compose",
                "-f",
                "compose.yaml",
                "ps",
                "--services",
                "--status",
                "running",
            ],
            allow_failure=True,
        )
        if result.returncode != 0:
            return set()
        return {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() in {"anythingllm", "ollama"}
        }

    def is_running(self) -> bool:
        return bool(self.running_services())

    def stop(self) -> None:
        if self.is_windows:
            return
        if not self.runtime_root.is_dir():
            return
        docker = shutil.which("docker")
        if docker is None:
            raise RuntimeError("Docker was not found. LegalFedLLM cannot stop Ollama/AnythingLLM.")
        self._run(
            [
                docker,
                "compose",
                "-f",
                "compose.yaml",
                "stop",
                "anythingllm",
                "ollama",
            ]
        )

    def _run(
        self,
        command: list[str],
        *,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=self.runtime_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 and not allow_failure:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Command failed ({' '.join(command)}): {detail or f'exit {result.returncode}'}"
            )
        return result

    def _wait_for_ollama(self, timeout: float = 45.0) -> None:
        deadline = time.monotonic() + timeout
        last_error = "Ollama did not answer"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"Ollama did not become ready: {last_error}")

    def _verify_ollama_model(self, expected_model: str) -> None:
        try:
            response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            response.raise_for_status()
            payload: Any = response.json()
        except Exception as exc:
            raise RuntimeError(f"Could not inspect Ollama models: {exc}") from exc

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = {
            str(item.get("name", ""))
            for item in models
            if isinstance(item, dict)
        }
        if expected_model not in names:
            raise RuntimeError(
                f"Required Ollama compatibility model '{expected_model}' is not installed. "
                "LegalFedLLM will not download it automatically."
            )
