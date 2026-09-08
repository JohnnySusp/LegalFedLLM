from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_pyinstaller(*, clean: bool) -> Path:
    name = "LegalFedLLM"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--console",
        "--name",
        name,
        "--collect-submodules",
        "client",
        "--collect-submodules",
        "coordinator",
        "--collect-submodules",
        "host",
        "--collect-submodules",
        "shared",
        "--collect-submodules",
        "desktop",
        "desktop/app.py",
    ]
    if clean:
        command.insert(4, "--clean")
    run(command)
    suffix = ".exe" if os.name == "nt" else ""
    artifact = DIST / f"{name}{suffix}"
    if not artifact.is_file():
        raise RuntimeError(f"PyInstaller output is missing: {artifact}")
    return artifact


def build_appimage(binary: Path) -> Path:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("AppImage packaging must run on Linux")
    appimagetool = shutil.which("appimagetool")
    if appimagetool is None:
        raise RuntimeError(
            "appimagetool was not found in PATH; install/provide it before building the AppImage"
        )
    appdir = BUILD / "LegalFedLLM.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    (appdir / "usr/bin").mkdir(parents=True)
    shutil.copy2(binary, appdir / "usr/bin/LegalFedLLM")
    (appdir / "AppRun").write_text(
        "#!/bin/sh\n"
        "HERE=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "exec \"$HERE/usr/bin/LegalFedLLM\" \"$@\"\n",
        encoding="utf-8",
    )
    (appdir / "AppRun").chmod(0o755)
    (appdir / "LegalFedLLM.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=LegalFedLLM\n"
        "Exec=LegalFedLLM\n"
        "Terminal=true\n"
        "Categories=Development;Utility;\n",
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
        help="on Linux, wrap the PyInstaller binary in an AppImage",
    )
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    artifact = build_pyinstaller(clean=not args.no_clean)
    print(f"Built: {artifact}")
    if args.appimage:
        artifact = build_appimage(artifact)
        print(f"Built: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
