from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
DESKTOP_ICON = ROOT / "desktop" / "legalfedllm.png"
CLIENT_RUNTIME_TEMPLATE = ROOT / "desktop" / "client-runtime"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _copytree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def prepare_client_runtime_bundle(destination: Path) -> Path:
    required = (
        ROOT / "requirements.txt",
        ROOT / "client",
        ROOT / "shared",
        CLIENT_RUNTIME_TEMPLATE / "Dockerfile",
        CLIENT_RUNTIME_TEMPLATE / "compose.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Client Docker runtime inputs are missing: " + ", ".join(missing)
        )

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "requirements.txt", destination / "requirements.txt")
    shutil.copy2(CLIENT_RUNTIME_TEMPLATE / "Dockerfile", destination / "Dockerfile")
    shutil.copy2(CLIENT_RUNTIME_TEMPLATE / "compose.yaml", destination / "compose.yaml")
    _copytree(ROOT / "client", destination / "client")
    _copytree(ROOT / "shared", destination / "shared")
    return destination


def build_pyinstaller(*, clean: bool, appimage: bool = False) -> Path:
    name = "LegalFedLLM"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir" if appimage else "--onefile",
        "--console",
        "--name",
        name,
    ]
    if clean:
        command.insert(4, "--clean")

    if appimage:
        command.extend(
            [
                "--hidden-import",
                "desktop.agent_entry",
            ]
        )
    else:
        for package in ("client", "coordinator", "host", "shared", "desktop"):
            command.extend(["--collect-submodules", package])

    command.extend(
        [
            "--add-data",
            f"{ROOT / 'legalfed-ai'}{os.pathsep}legalfed-ai",
        ]
    )

    if appimage:
        with tempfile.TemporaryDirectory(prefix="legalfedllm-client-runtime-") as directory:
            bundle = prepare_client_runtime_bundle(Path(directory) / "client-runtime")
            command.extend(
                [
                    "--add-data",
                    f"{bundle}{os.pathsep}client-runtime",
                    "desktop/app.py",
                ]
            )
            run(command)
    else:
        command.append("desktop/app.py")
        run(command)

    suffix = ".exe" if os.name == "nt" else ""
    artifact = DIST / name if appimage else DIST / f"{name}{suffix}"
    expected = artifact / f"{name}{suffix}" if appimage else artifact
    if not expected.is_file():
        raise RuntimeError(f"PyInstaller output is missing: {expected}")
    return artifact


def _app_run_text() -> str:
    return """#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP="$HERE/usr/lib/LegalFedLLM/LegalFedLLM"
export LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1

if [ "${LEGALFEDLLM_CONSOLE_PARENT:-}" = "1" ] || [ -t 0 ]; then
    exec "$APP" "$@"
fi

RELAUNCH="${APPIMAGE:-$APP}"

if command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e env \
        LEGALFEDLLM_CONSOLE_PARENT=1 \
        LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1 \
        "$RELAUNCH" --console-parent "$@"
fi
if command -v konsole >/dev/null 2>&1; then
    exec konsole -e env \
        LEGALFEDLLM_CONSOLE_PARENT=1 \
        LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1 \
        "$RELAUNCH" --console-parent "$@"
fi
if command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal -- env \
        LEGALFEDLLM_CONSOLE_PARENT=1 \
        LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1 \
        "$RELAUNCH" --console-parent "$@"
fi
if command -v xterm >/dev/null 2>&1; then
    exec xterm -e env \
        LEGALFEDLLM_CONSOLE_PARENT=1 \
        LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1 \
        "$RELAUNCH" --console-parent "$@"
fi

printf '%s\n' 'LegalFedLLM could not find a supported terminal emulator; starting the GUI without one.' >&2
exec "$APP" "$@"
"""


def build_appimage(bundle: Path) -> Path:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("AppImage packaging must run on Linux")
    appimagetool = shutil.which("appimagetool")
    if appimagetool is None:
        raise RuntimeError(
            "appimagetool was not found in PATH; install/provide it before building the AppImage"
        )
    if not DESKTOP_ICON.is_file():
        raise RuntimeError(f"AppImage icon is missing: {DESKTOP_ICON}")
    executable = bundle / "LegalFedLLM"
    if not bundle.is_dir() or not executable.is_file():
        raise RuntimeError(f"PyInstaller onedir bundle is missing: {bundle}")

    appdir = BUILD / "LegalFedLLM.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)

    app_bundle = appdir / "usr/lib/LegalFedLLM"
    app_bundle.parent.mkdir(parents=True)
    shutil.copytree(bundle, app_bundle)
    shutil.copy2(DESKTOP_ICON, appdir / "legalfedllm.png")

    (appdir / "AppRun").write_text(_app_run_text(), encoding="utf-8")
    (appdir / "AppRun").chmod(0o755)
    (appdir / "LegalFedLLM.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=LegalFedLLM\n"
        "Exec=LegalFedLLM\n"
        "Icon=legalfedllm\n"
        "Terminal=false\n"
        "Categories=Development;\n",
        encoding="utf-8",
    )

    output = DIST / "LegalFedLLM-x86_64.AppImage"
    environment = dict(os.environ)
    environment.setdefault("ARCH", "x86_64")
    print(f"+ {appimagetool} {appdir} {output}", flush=True)
    subprocess.run([appimagetool, str(appdir), str(output)], check=True, env=environment)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the portable LegalFedLLM desktop Client on the current OS."
    )
    parser.add_argument(
        "--appimage",
        action="store_true",
        help=(
            "on Linux, build a lightweight controller AppImage; the real Client ML "
            "runtime is built and run through Docker on first use"
        ),
    )
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    artifact = build_pyinstaller(
        clean=not args.no_clean,
        appimage=args.appimage,
    )
    print(f"Built: {artifact}")
    if args.appimage:
        artifact = build_appimage(artifact)
        print(f"Built: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
