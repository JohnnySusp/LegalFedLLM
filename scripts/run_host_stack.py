from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinator.quorum import TrustedClientQuorumPolicy
from shared.env_bootstrap import load_env_file


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)


def _required_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith(_UNSAFE_SECRET_PREFIXES):
        raise ValueError(f"{name} must contain a non-placeholder secret")
    return value


def _path_below_root(name: str, root: Path) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured")
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} must be below LEGALFEDLLM_RUNTIME_ROOT")
    return path


def validate_environment() -> dict[str, object]:
    root_text = os.getenv("LEGALFEDLLM_RUNTIME_ROOT", "").strip()
    if not root_text:
        raise ValueError("LEGALFEDLLM_RUNTIME_ROOT must be configured")
    root = Path(root_text).expanduser().resolve()

    if os.getenv("HOST_MODEL_PROFILE", "").strip() != (
        "mistral-nemo-instruct-2407-host-lora-v1"
    ):
        raise ValueError("HOST_MODEL_PROFILE must select the pinned Nemo Host")
    if os.getenv("HOST_TRAINING_BACKEND", "").strip() != "transformers":
        raise ValueError("HOST_TRAINING_BACKEND must be transformers")
    if os.getenv("HOST_SERVING_BACKEND", "").strip() != "mock":
        raise ValueError("the pinned Nemo Host requires mock serving")

    for name in ("ADMIN_TOKEN", "REGISTRATION_TOKEN", "INTERNAL_API_TOKEN"):
        _required_secret(name)

    if os.getenv("COORDINATOR_QUORUM_POLICY", "").strip() != "majority":
        raise ValueError("COORDINATOR_QUORUM_POLICY must be majority")
    quorum_override_text = os.getenv(
        "COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE",
        "",
    ).strip()
    quorum_policy = TrustedClientQuorumPolicy(
        minimum=int(
            os.getenv("COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM", "2")
        ),
        override=int(quorum_override_text) if quorum_override_text else None,
    )
    if quorum_policy.minimum != 2:
        raise ValueError(
            "COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM must be 2"
        )

    writable_paths = (
        _path_below_root("HOST_DATA_DIR", root),
        _path_below_root("COORDINATOR_DATA_DIR", root),
        _path_below_root("HF_HOME", root),
        _path_below_root("PIP_CACHE_DIR", root),
        _path_below_root("TORCH_HOME", root),
        _path_below_root("TORCH_EXTENSIONS_DIR", root),
        _path_below_root("TRITON_HOME", root),
        _path_below_root("XDG_CACHE_HOME", root),
        _path_below_root("CUDA_CACHE_PATH", root),
        _path_below_root("TMPDIR", root),
    )
    for path in writable_paths:
        path.mkdir(parents=True, exist_ok=True)

    for name in (
        "COORDINATOR_REFERENCE_DATASET_PATH",
        "COORDINATOR_VALIDATION_DATASET_PATH",
    ):
        path = _path_below_root(name, root)
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")

    host_bind = os.getenv("HOST_BIND_HOST", "127.0.0.1").strip()
    coordinator_bind = os.getenv(
        "COORDINATOR_BIND_HOST",
        "127.0.0.1",
    ).strip()
    if host_bind not in {"127.0.0.1", "localhost"}:
        raise ValueError("HOST_BIND_HOST must remain loopback-only")
    if coordinator_bind not in {"127.0.0.1", "localhost"}:
        raise ValueError("COORDINATOR_BIND_HOST must remain loopback-only")

    return {
        "root": root,
        "host_bind": host_bind,
        "host_port": int(os.getenv("HOST_PORT", "8002")),
        "coordinator_bind": coordinator_bind,
        "coordinator_port": int(os.getenv("COORDINATOR_PORT", "8000")),
        "host_startup_timeout": float(
            os.getenv("HOST_STARTUP_TIMEOUT_SECONDS", "900")
        ),
        "coordinator_startup_timeout": float(
            os.getenv("COORDINATOR_STARTUP_TIMEOUT_SECONDS", "120")
        ),
    }


def _uvicorn_command(module: str, bind_host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        module,
        "--host",
        bind_host,
        "--port",
        str(port),
    ]


def _wait_for_health(
    *,
    label: str,
    process: subprocess.Popen[bytes],
    url: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{label} exited during startup with {return_code}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{label} did not become healthy within {timeout_seconds}s")


def _stop(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 15
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only real Host and Coordinator stack."
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="private Host environment file created by scripts/bootstrap.py host",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)
    configuration = validate_environment()
    processes: list[subprocess.Popen[bytes]] = []

    def stop_from_signal(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_from_signal)
    signal.signal(signal.SIGINT, stop_from_signal)

    host_bind = str(configuration["host_bind"])
    host_port = int(configuration["host_port"])
    coordinator_bind = str(configuration["coordinator_bind"])
    coordinator_port = int(configuration["coordinator_port"])

    try:
        host = subprocess.Popen(
            _uvicorn_command("host.main:app", host_bind, host_port)
        )
        processes.append(host)
        _wait_for_health(
            label="Host",
            process=host,
            url=f"http://{host_bind}:{host_port}/health",
            timeout_seconds=float(configuration["host_startup_timeout"]),
        )

        coordinator_environment = os.environ.copy()
        coordinator_environment["HOST_RUNTIME_URL"] = (
            f"http://{host_bind}:{host_port}"
        )
        coordinator = subprocess.Popen(
            _uvicorn_command(
                "coordinator.main:app",
                coordinator_bind,
                coordinator_port,
            ),
            env=coordinator_environment,
        )
        processes.append(coordinator)
        _wait_for_health(
            label="Coordinator",
            process=coordinator,
            url=f"http://{coordinator_bind}:{coordinator_port}/health",
            timeout_seconds=float(configuration["coordinator_startup_timeout"]),
        )

        print(
            "Host and Coordinator are healthy on loopback; "
            "keep this process running while the SSH tunnel is active.",
            flush=True,
        )
        while True:
            for label, process in (
                ("Host", host),
                ("Coordinator", coordinator),
            ):
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"{label} exited unexpectedly with {return_code}"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        _stop(processes)


if __name__ == "__main__":
    raise SystemExit(main())
