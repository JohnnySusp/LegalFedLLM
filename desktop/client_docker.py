from __future__ import annotations

import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Mapping

_BUNDLE_FILE_NAMES = ("requirements.txt", "Dockerfile", "compose.yaml")
_BUNDLE_DIR_NAMES = ("client", "shared")


def _bool_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def bundled_client_runtime_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve() / "client-runtime"
    return Path(__file__).resolve().parents[1]


def _source_layout(root: Path) -> dict[str, Path]:
    packaged = all((root / name).exists() for name in (*_BUNDLE_FILE_NAMES, *_BUNDLE_DIR_NAMES))
    if packaged:
        return {
            "requirements.txt": root / "requirements.txt",
            "Dockerfile": root / "Dockerfile",
            "compose.yaml": root / "compose.yaml",
            "client": root / "client",
            "shared": root / "shared",
        }

    template = root / "desktop" / "client-runtime"
    return {
        "requirements.txt": root / "requirements.txt",
        "Dockerfile": template / "Dockerfile",
        "compose.yaml": template / "compose.yaml",
        "client": root / "client",
        "shared": root / "shared",
    }


def _iter_files(source: Path):
    if source.is_file():
        yield Path(source.name), source
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path.relative_to(source), path


def client_runtime_bundle_hash(bundle_root: str | Path | None = None) -> str:
    root = Path(bundle_root or bundled_client_runtime_root()).resolve()
    layout = _source_layout(root)
    digest = hashlib.sha256()
    for destination in sorted(layout):
        source = layout[destination]
        if not source.exists():
            raise RuntimeError(f"Client Docker runtime input is missing: {source}")
        if source.is_file():
            entries = [(Path(destination), source)]
        else:
            entries = [
                (Path(destination) / relative, path)
                for relative, path in _iter_files(source)
            ]
        for relative, path in entries:
            digest.update(str(relative).replace(os.sep, "/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _copy_runtime_bundle(bundle_root: Path, destination: Path) -> None:
    layout = _source_layout(bundle_root)
    for name, source in layout.items():
        target = destination / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )


def _project_name(profile_root: Path) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", profile_root.name.lower()).strip("-_")
    return f"legalfedllm-{value or 'client'}"


class ClientDockerStack:
    def __init__(
        self,
        environment: Mapping[str, str] | None = None,
        *,
        bundle_root: str | Path | None = None,
    ):
        self.environment = dict(environment or os.environ)
        self.bundle_root = Path(bundle_root or bundled_client_runtime_root()).resolve()
        self.docker = shutil.which("docker")
        self.compose_process: subprocess.Popen[bytes] | None = None
        self.runtime_dir: Path | None = None
        self.image: str | None = None

        client_data_value = self.environment.get("CLIENT_DATA_DIR", "").strip()
        private_value = self.environment.get("CLIENT_PRIVATE_DATA_PATH", "").strip()
        hf_value = self.environment.get("HF_HOME", "").strip()
        if not client_data_value or not private_value or not hf_value:
            raise RuntimeError(
                "desktop Docker Client requires CLIENT_DATA_DIR, CLIENT_PRIVATE_DATA_PATH and HF_HOME"
            )

        self.client_data = Path(client_data_value).expanduser().resolve()
        self.private_file = Path(private_value).expanduser().resolve()
        self.hf_home = Path(hf_value).expanduser().resolve()
        self.profile_root = self.client_data.parent

        explicit_data_root = self.environment.get("LEGALFEDLLM_DATA_ROOT", "").strip()
        if explicit_data_root:
            self.data_root = Path(explicit_data_root).expanduser().resolve()
        else:
            try:
                self.data_root = self.profile_root.parents[1]
            except IndexError as exc:
                raise RuntimeError("could not derive portable LegalFedLLM data root") from exc

        self.runtime_root = self.data_root / "client-runtime"
        self.project = _project_name(self.profile_root)

    def prepare_runtime(self) -> Path:
        digest = client_runtime_bundle_hash(self.bundle_root)
        version = digest[:16]
        target = self.runtime_root / version
        marker = target / ".bundle-sha256"
        if target.is_dir() and marker.is_file():
            if marker.read_text(encoding="utf-8").strip() == digest:
                self.runtime_dir = target
                self.image = f"legalfedllm-client:{version}"
                return target
            shutil.rmtree(target)

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        temporary = self.runtime_root / f".{version}.tmp-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=False)
        try:
            _copy_runtime_bundle(self.bundle_root, temporary)
            (temporary / ".bundle-sha256").write_text(digest + "\n", encoding="utf-8")
            try:
                temporary.replace(target)
            except FileExistsError:
                shutil.rmtree(temporary)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        self.runtime_dir = target
        self.image = f"legalfedllm-client:{version}"
        return target

    def _docker_environment(self) -> dict[str, str]:
        if self.runtime_dir is None or self.image is None:
            raise RuntimeError("Client Docker runtime has not been prepared")
        environment = dict(self.environment)
        self.client_data.mkdir(parents=True, exist_ok=True)
        self.private_file.parent.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "LEGALFEDLLM_CLIENT_IMAGE": self.image,
                "LEGALFEDLLM_CLIENT_DATA_DIR": str(self.client_data),
                "LEGALFEDLLM_PRIVATE_DATA_DIR": str(self.private_file.parent),
                "LEGALFEDLLM_HF_HOME": str(self.hf_home),
            }
        )
        return environment

    def _compose_command(self) -> list[str]:
        if self.docker is None:
            raise RuntimeError("Docker was not found. LegalFedLLM cannot start the Client Agent.")
        if self.runtime_dir is None:
            raise RuntimeError("Client Docker runtime has not been prepared")
        return [
            self.docker,
            "compose",
            "-p",
            self.project,
            "-f",
            str(self.runtime_dir / "compose.yaml"),
        ]

    def _image_exists(self, environment: Mapping[str, str]) -> bool:
        if self.docker is None or self.image is None:
            return False
        result = subprocess.run(
            [self.docker, "image", "inspect", self.image],
            cwd=self.runtime_dir,
            env=dict(environment),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def build_if_needed(self) -> None:
        self.prepare_runtime()
        environment = self._docker_environment()
        if self._image_exists(environment):
            print(f"Reusing LegalFedLLM Client Docker image {self.image}.", flush=True)
            return
        print(
            "Building the LegalFedLLM Client Docker image for this release. "
            "Docker will install the ML dependencies; this can take a while on first launch.",
            flush=True,
        )
        subprocess.run(
            [*self._compose_command(), "build", "client"],
            cwd=self.runtime_dir,
            env=environment,
            check=True,
        )

    def run_foreground(self) -> int:
        self.build_if_needed()
        environment = self._docker_environment()
        print(
            "Starting the LegalFedLLM Client Agent in Docker. "
            "The GUI will connect to its loopback API when it becomes healthy.",
            flush=True,
        )
        self.compose_process = subprocess.Popen(
            [
                *self._compose_command(),
                "up",
                "--no-build",
                "--force-recreate",
                "client",
            ],
            cwd=self.runtime_dir,
            env=environment,
        )
        try:
            return self.compose_process.wait()
        finally:
            self.compose_process = None

    def request_stop(self) -> None:
        process = self.compose_process
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                pass

    def stop(self) -> None:
        if self.runtime_dir is None or self.docker is None:
            return
        environment = self._docker_environment()
        subprocess.run(
            [*self._compose_command(), "down", "--remove-orphans"],
            cwd=self.runtime_dir,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.request_stop()
