from __future__ import annotations

import struct
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 14)
EXPECTED_TORCH = "2.13.0+cu132"
EXPECTED_CUDA = "13.2"


def _requirements(path: Path) -> list[Requirement]:
    result: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            include = line.split(maxsplit=1)[1].strip()
            result.extend(_requirements((path.parent / include).resolve()))
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            result.append(requirement)
    return result


def verify() -> None:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("LegalFedLLM requires 64-bit Python on Windows")

    requirements = _requirements(ROOT / "requirements-desktop.txt")
    for requirement in requirements:
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"missing required package: {requirement.name}") from exc
        if requirement.specifier and not requirement.specifier.contains(
            installed,
            prereleases=True,
        ):
            raise RuntimeError(
                f"{requirement.name} {installed} does not satisfy {requirement.specifier}"
            )

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        detail = (pip_check.stdout + pip_check.stderr).strip()
        raise RuntimeError(f"pip dependency check failed: {detail}")

    import torch

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"expected torch {EXPECTED_TORCH}, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the pinned Windows Torch build")
    if torch.version.cuda != EXPECTED_CUDA:
        raise RuntimeError(f"expected Torch CUDA {EXPECTED_CUDA}, found {torch.version.cuda!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected Windows GPU does not report BF16 support")

    print(f"python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"bf16: {torch.cuda.is_bf16_supported()}")
    print(f"requirements verified: {len(requirements)}")


def main() -> int:
    try:
        verify()
    except Exception as exc:
        print(f"LegalFedLLM Windows runtime verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
