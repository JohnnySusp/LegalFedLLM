from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.env_bootstrap import load_env_file


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)

_SLOT_PREFIXES = {
    "client-1": "QWEN",
    "client-2": "GRANITE",
}

_SLOT_DEFAULT_PORTS = {
    "client-1": 8001,
    "client-2": 8002,
}

_PROJECTED_FIELDS = {
    "CLIENT_ID": "CLIENT_ID",
    "CLIENT_MODEL_PROFILE": "CLIENT_MODEL_PROFILE",
    "CLIENT_MODEL_ID": "CLIENT_MODEL_ID",
    "CLIENT_SERVING_BACKEND": "CLIENT_SERVING_BACKEND",
    "CLIENT_OLLAMA_MODEL": "CLIENT_OLLAMA_MODEL",
    "CLIENT_PRIVATE_DATASET_ID": "CLIENT_PRIVATE_DATASET_ID",
    "CLIENT_ADMIN_TOKEN": "CLIENT_ADMIN_TOKEN",
}


def _required_value(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured")
    return value


def _required_secret(environment: Mapping[str, str], name: str) -> str:
    value = _required_value(environment, name)
    if value.startswith(_UNSAFE_SECRET_PREFIXES):
        raise ValueError(f"{name} must contain a non-placeholder secret")
    return value


def _native_private_data_path(
    environment: Mapping[str, str],
    *,
    prefix: str,
) -> str:
    directory_text = _required_value(
        environment,
        f"{prefix}_CLIENT_PRIVATE_DATA_DIR",
    )
    directory = Path(directory_text).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    return str((directory / "train.jsonl").resolve())


def project_slot_environment(
    environment: Mapping[str, str],
    slot: str,
) -> tuple[dict[str, str], int]:
    if slot not in _SLOT_PREFIXES:
        raise ValueError(f"unknown Client slot: {slot}")

    prefix = _SLOT_PREFIXES[slot]
    projected = dict(environment)

    for generic_name, suffix in _PROJECTED_FIELDS.items():
        source_name = f"{prefix}_{suffix}"
        if generic_name == "CLIENT_ADMIN_TOKEN":
            value = _required_secret(environment, source_name)
        else:
            value = _required_value(environment, source_name)
        projected[generic_name] = value

    projected["CLIENT_PRIVATE_DATA_PATH"] = _native_private_data_path(
        environment,
        prefix=prefix,
    )

    port_name = f"{prefix}_CLIENT_PORT"
    port_text = environment.get(
        port_name,
        str(_SLOT_DEFAULT_PORTS[slot]),
    ).strip()
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"{port_name} must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{port_name} must be between 1 and 65535")

    return projected, port


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one native LegalFedLLM Client service from a multi-Client "
            "environment file."
        )
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="private Client environment file created by scripts/bootstrap.py client",
    )
    parser.add_argument(
        "--slot",
        required=True,
        choices=tuple(_SLOT_PREFIXES),
        help="Client slot to project into the generic Client runtime environment",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    projected, port = project_slot_environment(os.environ, args.slot)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "client.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=projected,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
