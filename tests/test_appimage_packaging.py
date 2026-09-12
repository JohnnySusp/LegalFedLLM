from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop import app as desktop_app
from scripts import build_desktop


class AppImagePackagingTests(unittest.TestCase):
    def test_appimage_apprun_relaunches_graphical_start_in_terminal(self) -> None:
        text = build_desktop._app_run_text()
        self.assertIn("LEGALFEDLLM_DESKTOP_DOCKER_AGENT=1", text)
        self.assertIn('RELAUNCH="${APPIMAGE:-$APP}"', text)
        self.assertIn("x-terminal-emulator", text)
        self.assertIn("konsole", text)
        self.assertIn("gnome-terminal", text)
        self.assertIn("xterm", text)
        self.assertIn("--console-parent", text)

    def test_appimage_browser_launch_uses_sanitized_host_environment(self) -> None:
        appimage_environment = {
            "APPIMAGE": "/tmp/LegalFedLLM.AppImage",
            "LD_LIBRARY_PATH": "/tmp/.mount_Legal/usr/lib/LegalFedLLM/_internal",
            "LD_LIBRARY_PATH_ORIG": "/host/lib",
            "QT_PLUGIN_PATH": "/tmp/.mount_Legal/qt/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/tmp/.mount_Legal/qt/platforms",
            "QML2_IMPORT_PATH": "/tmp/.mount_Legal/qml",
            "QML_IMPORT_PATH": "/tmp/.mount_Legal/qml-old",
        }
        with (
            mock.patch.dict(os.environ, appimage_environment, clear=True),
            mock.patch.object(desktop_app.sys, "platform", "linux"),
            mock.patch.object(desktop_app.shutil, "which", return_value="/usr/bin/xdg-open"),
            mock.patch.object(desktop_app.subprocess, "Popen") as popen,
            mock.patch.object(desktop_app.webbrowser, "open") as browser_open,
        ):
            launched = desktop_app.open_default_browser("http://127.0.0.1:3001")

        self.assertTrue(launched)
        browser_open.assert_not_called()
        popen.assert_called_once()
        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(command, ["/usr/bin/xdg-open", "http://127.0.0.1:3001"])
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/host/lib")
        self.assertNotIn("LD_LIBRARY_PATH_ORIG", environment)
        self.assertNotIn("QT_PLUGIN_PATH", environment)
        self.assertNotIn("QT_QPA_PLATFORM_PLUGIN_PATH", environment)
        self.assertNotIn("QML2_IMPORT_PATH", environment)
        self.assertNotIn("QML_IMPORT_PATH", environment)

    def test_appimage_build_uses_onedir_and_not_full_ml_submodule_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_template = root / "desktop/client-runtime"
            runtime_template.mkdir(parents=True)
            (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
            (runtime_template / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
            (runtime_template / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            (root / "client").mkdir()
            (root / "client/main.py").write_text("", encoding="utf-8")
            (root / "shared").mkdir()
            (root / "shared/protocol.py").write_text("", encoding="utf-8")
            (root / "legalfed-ai").mkdir()
            (root / "desktop/app.py").write_text("", encoding="utf-8")
            dist = root / "dist/LegalFedLLM"
            dist.mkdir(parents=True)
            executable = "LegalFedLLM.exe" if os.name == "nt" else "LegalFedLLM"
            (dist / executable).write_text("binary", encoding="utf-8")

            commands: list[list[str]] = []
            def fake_run(command: list[str]) -> None:
                commands.append(command)

            with (
                mock.patch.object(build_desktop, "ROOT", root),
                mock.patch.object(build_desktop, "DIST", root / "dist"),
                mock.patch.object(build_desktop, "CLIENT_RUNTIME_TEMPLATE", runtime_template),
                mock.patch.object(build_desktop, "run", side_effect=fake_run),
            ):
                artifact = build_desktop.build_pyinstaller(clean=False, appimage=True)

            self.assertEqual(artifact, dist)
            command = commands[0]
            self.assertIn("--onedir", command)
            self.assertNotIn("--onefile", command)
            joined = " ".join(command)
            self.assertNotIn("--collect-submodules client", joined)
            self.assertNotIn("--collect-submodules host", joined)
            self.assertNotIn("--collect-submodules coordinator", joined)
            self.assertNotIn("--collect-submodules uvicorn", joined)
            self.assertIn("desktop.agent_entry", command)
            self.assertIn("client-runtime", joined)

    def test_normal_build_retains_onefile_full_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "legalfed-ai").mkdir()
            (root / "desktop").mkdir()
            (root / "desktop/app.py").write_text("", encoding="utf-8")
            dist = root / "dist"
            dist.mkdir()
            executable = "LegalFedLLM.exe" if os.name == "nt" else "LegalFedLLM"
            (dist / executable).write_text("binary", encoding="utf-8")
            commands: list[list[str]] = []
            with (
                mock.patch.object(build_desktop, "ROOT", root),
                mock.patch.object(build_desktop, "DIST", dist),
                mock.patch.object(build_desktop, "run", side_effect=lambda command: commands.append(command)),
            ):
                build_desktop.build_pyinstaller(clean=False, appimage=False)
            command = commands[0]
            self.assertIn("--onefile", command)
            self.assertIn("--collect-submodules", command)
            self.assertIn("client", command)
            self.assertIn("host", command)
            self.assertIn("coordinator", command)
            self.assertIn("uvicorn", command)


    def test_windows_normal_build_embeds_desktop_icon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "legalfed-ai").mkdir()
            (root / "desktop").mkdir()
            (root / "desktop/app.py").write_text("", encoding="utf-8")
            icon = root / "desktop/legalfedllm.ico"
            icon.write_bytes(b"ico")
            dist = root / "dist"
            dist.mkdir()
            (dist / "LegalFedLLM.exe").write_text("binary", encoding="utf-8")
            commands: list[list[str]] = []
            with (
                mock.patch.object(build_desktop, "ROOT", root),
                mock.patch.object(build_desktop, "DIST", dist),
                mock.patch.object(build_desktop, "WINDOWS_ICON", icon),
                mock.patch.object(build_desktop, "_is_windows", return_value=True),
                mock.patch.object(build_desktop, "run", side_effect=lambda command: commands.append(command)),
            ):
                artifact = build_desktop.build_pyinstaller(clean=False, appimage=False)
            self.assertEqual(artifact, dist / "LegalFedLLM.exe")
            command = commands[0]
            self.assertIn("--icon", command)
            self.assertIn(str(icon), command)

    def test_client_runtime_bundle_contains_only_runtime_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "desktop/client-runtime"
            template.mkdir(parents=True)
            (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
            (template / "Dockerfile").write_text("FROM python:3.11\n", encoding="utf-8")
            (template / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
            for package in ("client", "shared"):
                package_root = root / package
                package_root.mkdir()
                (package_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
                (package_root / "__pycache__").mkdir()
                (package_root / "__pycache__/module.pyc").write_bytes(b"cache")
            destination = root / "bundle"
            with (
                mock.patch.object(build_desktop, "ROOT", root),
                mock.patch.object(build_desktop, "CLIENT_RUNTIME_TEMPLATE", template),
            ):
                build_desktop.prepare_client_runtime_bundle(destination)
            self.assertTrue((destination / "client/module.py").is_file())
            self.assertTrue((destination / "shared/module.py").is_file())
            self.assertFalse((destination / "client/__pycache__").exists())
            self.assertFalse((destination / "shared/__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
