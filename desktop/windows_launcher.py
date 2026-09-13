from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Sequence

EXPECTED_PYTHON = (3, 14)


class LauncherError(RuntimeError):
    pass


def install_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def venv_python(root: Path) -> Path:
    return root / ".venv" / "Scripts" / "python.exe"


def verifier_script(root: Path) -> Path:
    return root / "scripts" / "verify_windows_desktop.py"


def setup_script(root: Path) -> Path:
    return root / "scripts" / "setup_windows_desktop.ps1"


def _python_probe(command: Sequence[str]) -> Path | None:
    probe = (
        "import struct,sys; "
        "ok=sys.version_info[:2]==(3,14) and struct.calcsize('P')*8==64; "
        "print(sys.executable); raise SystemExit(0 if ok else 1)"
    )
    try:
        completed = subprocess.run(
            [*command, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip().splitlines()
    if not output:
        return None
    path = Path(output[-1]).resolve()
    return path if path.is_file() else None


def find_system_python(root: Path) -> Path:
    local_python = venv_python(root).resolve()
    candidates: list[list[str]] = []

    override = os.getenv("LEGALFEDLLM_PYTHON", "").strip()
    if override:
        candidates.append([override])

    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append([py_launcher, "-3.14"])

    python = shutil.which("python")
    if python:
        candidates.append([python])

    python314 = shutil.which("python3.14")
    if python314:
        candidates.append([python314])

    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        executable = _python_probe(command)
        if executable is None or executable == local_python:
            continue
        return executable

    raise LauncherError(
        "Python 3.14 x64 was not found. Install Python 3.14 x64, then start "
        "LegalFedLLM again."
    )


def verify_local_environment(root: Path) -> bool:
    python = venv_python(root)
    verifier = verifier_script(root)
    if not python.is_file() or not verifier.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python), str(verifier)],
            cwd=root,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def bootstrap_local_environment(root: Path) -> None:
    script = setup_script(root)
    if not script.is_file():
        raise LauncherError(f"Windows setup helper is missing: {script}")

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise LauncherError("Windows PowerShell was not found.")

    python = find_system_python(root)
    print("LegalFedLLM local Python environment is missing or invalid.", flush=True)
    print(f"Using system Python: {python}", flush=True)
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-PythonCommand",
            str(python),
        ],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise LauncherError(
            f"LegalFedLLM environment setup failed with exit code {completed.returncode}."
        )


def validate_release_layout(root: Path) -> None:
    required = (
        root / "desktop" / "app.py",
        root / "requirements.txt",
        root / "requirements-desktop.txt",
        setup_script(root),
        verifier_script(root),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise LauncherError(
            "LegalFedLLM release files are incomplete: " + ", ".join(missing)
        )


def launch_desktop(root: Path) -> int:
    python = venv_python(root)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(
        [str(python), "-m", "desktop.app"],
        cwd=root,
        env=environment,
    )


def run(root: Path | None = None) -> int:
    root = Path(root or install_root()).resolve()
    validate_release_layout(root)

    if not verify_local_environment(root):
        bootstrap_local_environment(root)
        if not verify_local_environment(root):
            raise LauncherError(
                "LegalFedLLM local environment verification still fails after setup."
            )

    return launch_desktop(root)


def main() -> int:
    try:
        return run()
    except LauncherError as exc:
        print(f"LegalFedLLM startup failed: {exc}", file=sys.stderr, flush=True)
        if os.name == "nt" and sys.stdin.isatty():
            try:
                input("Press Enter to close...")
            except (EOFError, KeyboardInterrupt):
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
