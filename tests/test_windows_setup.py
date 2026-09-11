from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_windows_desktop.ps1"


class WindowsDesktopSetupTests(unittest.TestCase):
    def test_setup_pins_cuda_torch_and_verifies_gpu_contract(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("requirements-desktop.txt", source)
        self.assertIn("torch==2.13.0", source)
        self.assertIn("https://download.pytorch.org/whl/cu132", source)
        self.assertIn("--force-reinstall --no-deps", source)
        self.assertIn('expected_version = "2.13.0+cu132"', source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('torch.version.cuda != "13.2"', source)
        self.assertIn("torch.cuda.is_bf16_supported()", source)
        self.assertIn("$TorchProbe | & $VenvPython -", source)
        self.assertIn("$Verification | & $VenvPython -", source)

    def test_setup_uses_repo_local_venv(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('Join-Path $RepoRoot ".venv"', source)
        self.assertIn('Join-Path $VenvRoot "Scripts\\python.exe"', source)
        self.assertIn("-m venv", source)


if __name__ == "__main__":
    unittest.main()
