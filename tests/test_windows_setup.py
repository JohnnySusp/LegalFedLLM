from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_windows_desktop.ps1"
VERIFIER = ROOT / "scripts" / "verify_windows_desktop.py"


class WindowsDesktopSetupTests(unittest.TestCase):
    def test_setup_pins_cuda_torch_and_calls_runtime_verifier(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("requirements-desktop.txt", source)
        self.assertIn("torch==2.13.0+cu132", source)
        self.assertIn("https://download.pytorch.org/whl/cu132", source)
        self.assertIn("--no-deps", source)
        self.assertNotIn("--force-reinstall", source)
        self.assertIn("2.13.0+cu132", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('torch.version.cuda == "13.2"', source)
        self.assertIn("torch.cuda.is_bf16_supported()", source)
        self.assertIn("verify_windows_desktop.py", source)
        self.assertIn("Invoke-Checked $VenvPython $Verifier", source)

    def test_setup_checks_cuda_torch_before_installing_general_requirements(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        torch_probe = source.index("$TorchProbe | & $VenvPython -")
        torch_install = source.index("torch==2.13.0+cu132 --index-url $TorchIndex")
        requirements_install = source.index("-m pip install -r $Requirements")
        self.assertLess(torch_probe, requirements_install)
        self.assertLess(torch_install, requirements_install)

    def test_setup_uses_repo_local_venv_and_requires_python_314_x64(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('Join-Path $RepoRoot ".venv"', source)
        self.assertIn('Join-Path $VenvRoot "Scripts\\python.exe"', source)
        self.assertIn("-m venv", source)
        self.assertIn("sys.version_info[:2] != (3, 14)", source)
        self.assertIn('struct.calcsize("P") * 8 != 64', source)

    def test_runtime_verifier_checks_pins_pip_and_cuda_contract(self) -> None:
        source = VERIFIER.read_text(encoding="utf-8")
        self.assertIn("requirements-desktop.txt", source)
        self.assertIn("metadata.version", source)
        self.assertIn('"pip", "check"', source)
        self.assertIn('EXPECTED_TORCH = "2.13.0+cu132"', source)
        self.assertIn('EXPECTED_CUDA = "13.2"', source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn("torch.cuda.is_bf16_supported()", source)


if __name__ == "__main__":
    unittest.main()
