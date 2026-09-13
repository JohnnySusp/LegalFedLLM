from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop import windows_launcher


class WindowsLauncherTests(unittest.TestCase):
    def _release_layout(self, root: Path) -> None:
        (root / "desktop").mkdir()
        (root / "desktop/app.py").write_text("", encoding="utf-8")
        (root / "scripts").mkdir()
        (root / "scripts/setup_windows_desktop.ps1").write_text("", encoding="utf-8")
        (root / "scripts/verify_windows_desktop.py").write_text("", encoding="utf-8")
        (root / "requirements.txt").write_text("fastapi==0.116.1\n", encoding="utf-8")
        (root / "requirements-desktop.txt").write_text("-r requirements.txt\n", encoding="utf-8")

    def test_release_layout_requires_setup_and_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_layout(root)
            windows_launcher.validate_release_layout(root)
            (root / "scripts/verify_windows_desktop.py").unlink()
            with self.assertRaises(windows_launcher.LauncherError):
                windows_launcher.validate_release_layout(root)

    def test_valid_local_environment_launches_without_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_layout(root)
            with (
                mock.patch.object(windows_launcher, "verify_local_environment", return_value=True),
                mock.patch.object(windows_launcher, "bootstrap_local_environment") as bootstrap,
                mock.patch.object(windows_launcher, "launch_desktop", return_value=17) as launch,
            ):
                result = windows_launcher.run(root)
            self.assertEqual(result, 17)
            bootstrap.assert_not_called()
            launch.assert_called_once_with(root.resolve())

    def test_invalid_local_environment_bootstraps_then_launches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_layout(root)
            with (
                mock.patch.object(
                    windows_launcher,
                    "verify_local_environment",
                    side_effect=[False, True],
                ),
                mock.patch.object(windows_launcher, "bootstrap_local_environment") as bootstrap,
                mock.patch.object(windows_launcher, "launch_desktop", return_value=0) as launch,
            ):
                result = windows_launcher.run(root)
            self.assertEqual(result, 0)
            bootstrap.assert_called_once_with(root.resolve())
            launch.assert_called_once_with(root.resolve())

    def test_failed_verification_after_bootstrap_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_layout(root)
            with (
                mock.patch.object(
                    windows_launcher,
                    "verify_local_environment",
                    side_effect=[False, False],
                ),
                mock.patch.object(windows_launcher, "bootstrap_local_environment") as bootstrap,
                mock.patch.object(windows_launcher, "launch_desktop") as launch,
            ):
                with self.assertRaises(windows_launcher.LauncherError):
                    windows_launcher.run(root)
            bootstrap.assert_called_once_with(root.resolve())
            launch.assert_not_called()

    def test_launch_desktop_uses_repo_local_venv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            with mock.patch.object(windows_launcher.subprocess, "call", return_value=0) as call:
                result = windows_launcher.launch_desktop(root)
            self.assertEqual(result, 0)
            command = call.call_args.args[0]
            self.assertEqual(command, [str(python), "-m", "desktop.app"])
            self.assertEqual(call.call_args.kwargs["cwd"], root)
            self.assertEqual(call.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
