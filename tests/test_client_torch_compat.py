from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import client.peft_backend as peft_backend


class ClientTorchCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        peft_backend._TORCH_NATIVE_BMM_COMPAT_CONFIGURED = False

    def tearDown(self) -> None:
        peft_backend._TORCH_NATIVE_BMM_COMPAT_CONFIGURED = False

    def _torch(self, version: str) -> object:
        return types.SimpleNamespace(__version__=version)

    def test_linux_torch_213_without_python_headers_disables_only_bmm_once(self) -> None:
        registry = types.SimpleNamespace(deregister_op_overrides=mock.Mock())
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(peft_backend.sys, "platform", "linux"),
                mock.patch.object(peft_backend.sysconfig, "get_path", return_value=directory),
                mock.patch.object(peft_backend.importlib, "import_module", return_value=registry),
            ):
                self.assertTrue(
                    peft_backend._configure_torch_native_bmm_compat(
                        self._torch("2.13.0+cu130")
                    )
                )
                self.assertTrue(
                    peft_backend._configure_torch_native_bmm_compat(
                        self._torch("2.13.0+cu130")
                    )
                )

        registry.deregister_op_overrides.assert_called_once_with(
            disable_op_symbols="bmm"
        )

    def test_existing_python_headers_leave_native_bmm_unchanged(self) -> None:
        registry_import = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "Python.h").write_text("", encoding="utf-8")
            with (
                mock.patch.object(peft_backend.sys, "platform", "linux"),
                mock.patch.object(peft_backend.sysconfig, "get_path", return_value=directory),
                mock.patch.object(peft_backend.importlib, "import_module", registry_import),
            ):
                self.assertFalse(
                    peft_backend._configure_torch_native_bmm_compat(
                        self._torch("2.13.0+cu130")
                    )
                )
        registry_import.assert_not_called()

    def test_pre_213_torch_leaves_native_bmm_unchanged(self) -> None:
        registry_import = mock.Mock()
        with (
            mock.patch.object(peft_backend.sys, "platform", "linux"),
            mock.patch.object(peft_backend.importlib, "import_module", registry_import),
        ):
            self.assertFalse(
                peft_backend._configure_torch_native_bmm_compat(
                    self._torch("2.12.1+cu130")
                )
            )
        registry_import.assert_not_called()

    def test_missing_compatibility_api_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(peft_backend.sys, "platform", "linux"),
                mock.patch.object(peft_backend.sysconfig, "get_path", return_value=directory),
                mock.patch.object(
                    peft_backend.importlib,
                    "import_module",
                    return_value=types.SimpleNamespace(),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "deregister_op_overrides is unavailable",
                ):
                    peft_backend._configure_torch_native_bmm_compat(
                        self._torch("2.13.0+cu130")
                    )

    def test_non_linux_platform_does_not_apply_linux_compatibility(self) -> None:
        registry_import = mock.Mock()
        with (
            mock.patch.object(peft_backend.sys, "platform", "win32"),
            mock.patch.object(peft_backend.importlib, "import_module", registry_import),
        ):
            self.assertFalse(
                peft_backend._configure_torch_native_bmm_compat(
                    self._torch("2.13.0+cu130")
                )
            )
        registry_import.assert_not_called()


if __name__ == "__main__":
    unittest.main()
