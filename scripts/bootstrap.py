from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.env_bootstrap import load_or_create_env


ROLE_TEMPLATES = {
    "development": ROOT / ".env.example",
    "host": ROOT / "config/container.env.example",
    "client": ROOT / "config/clients.env.example",
}


def _runtime_replacements(root: Path) -> dict[str, str]:
    runtime_root = root.expanduser().resolve()
    return {
        "LEGALFEDLLM_RUNTIME_ROOT": str(runtime_root),
        "HF_HOME": str(runtime_root / "cache/huggingface"),
        "PIP_CACHE_DIR": str(runtime_root / "cache/pip"),
        "TORCH_HOME": str(runtime_root / "cache/torch"),
        "TORCH_EXTENSIONS_DIR": str(runtime_root / "cache/torch-extensions"),
        "TRITON_HOME": str(runtime_root / "cache/triton"),
        "XDG_CACHE_HOME": str(runtime_root / "cache/xdg"),
        "CUDA_CACHE_PATH": str(runtime_root / "cache/cuda"),
        "TMPDIR": str(runtime_root / "tmp"),
        "HOST_DATA_DIR": str(runtime_root / "artifacts/host-runtime"),
        "COORDINATOR_DATA_DIR": str(runtime_root / "artifacts/coordinator-runtime"),
        "COORDINATOR_REFERENCE_DATASET_PATH": str(
            runtime_root / "datasets/gld2012/reference.jsonl"
        ),
        "COORDINATOR_VALIDATION_DATASET_PATH": str(
            runtime_root / "datasets/gld2012/validation.jsonl"
        ),
    }


def _create_host_directories(values: dict[str, str]) -> None:
    for name in (
        "HOST_DATA_DIR",
        "COORDINATOR_DATA_DIR",
        "HF_HOME",
        "PIP_CACHE_DIR",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_HOME",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
        "TMPDIR",
    ):
        value = values.get(name, "").strip()
        if value:
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
    for name in (
        "COORDINATOR_REFERENCE_DATASET_PATH",
        "COORDINATOR_VALIDATION_DATASET_PATH",
    ):
        value = values.get(name, "").strip()
        if value:
            Path(value).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _create_client_directories(values: dict[str, str]) -> None:
    for name in (
        "QWEN_CLIENT_PRIVATE_DATA_DIR",
        "GRANITE_CLIENT_PRIVATE_DATA_DIR",
    ):
        value = values.get(name, "").strip()
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            path.mkdir(parents=True, exist_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one private LegalFedLLM environment file without overwriting it."
    )
    parser.add_argument("role", choices=tuple(ROLE_TEMPLATES))
    parser.add_argument(
        "--output",
        default=str(ROOT / ".env"),
        help="environment file to create; defaults to the repository .env",
    )
    parser.add_argument(
        "--registration-token",
        help="Host-issued registration token required for the client role",
    )
    parser.add_argument(
        "--runtime-root",
        help="override the Host runtime root and all scratch-backed Host paths",
    )
    parser.add_argument(
        "--trusted-quorum-override",
        type=int,
        help="temporary Host-only trusted quorum override, e.g. 1 for today's one-Client test",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    replacements: dict[str, str] = {}

    if args.role == "client":
        if not args.registration_token:
            raise SystemExit(
                "client bootstrap requires --registration-token from the Host/Coordinator installation"
            )
        replacements["REGISTRATION_TOKEN"] = args.registration_token

    if args.runtime_root:
        if args.role != "host":
            raise SystemExit("--runtime-root is valid only for the host role")
        replacements.update(_runtime_replacements(Path(args.runtime_root)))

    if args.trusted_quorum_override is not None:
        if args.role != "host":
            raise SystemExit("--trusted-quorum-override is valid only for the host role")
        if args.trusted_quorum_override < 1:
            raise SystemExit("--trusted-quorum-override must be positive")
        replacements["COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE"] = str(
            args.trusted_quorum_override
        )

    result = load_or_create_env(
        path=Path(args.output),
        template_path=ROLE_TEMPLATES[args.role],
        role=args.role,
        replacements=replacements,
    )

    if result.role == "host":
        _create_host_directories(result.values)
    elif result.role == "client":
        _create_client_directories(result.values)

    action = "created" if result.created else "reused"
    print(f"{action} {result.path} for role {result.role}")
    if result.created:
        print("generated secrets were written only to the private environment file")
    if result.role == "host":
        print("Host runtime directories are ready; copy D^P and D^V into the configured dataset paths")
    elif result.role == "client":
        print("Client data directories are ready; supply private train.jsonl when needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
