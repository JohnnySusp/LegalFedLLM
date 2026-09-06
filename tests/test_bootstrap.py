from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.env_bootstrap import load_env_file, load_or_create_env, parse_env_text


class EnvironmentBootstrapTests(unittest.TestCase):
    def _template(self, root: Path, role: str) -> Path:
        path = root / f"{role}.env.example"
        if role == "host":
            content = """LEGALFEDLLM_BOOTSTRAP_ROLE=host
ADMIN_TOKEN=
REGISTRATION_TOKEN=
INTERNAL_API_TOKEN=
HOST_DATA_DIR=/tmp/host
"""
        elif role == "client":
            content = """LEGALFEDLLM_BOOTSTRAP_ROLE=client
REGISTRATION_TOKEN=
QWEN_CLIENT_ADMIN_TOKEN=
GRANITE_CLIENT_ADMIN_TOKEN=
"""
        else:
            content = """LEGALFEDLLM_BOOTSTRAP_ROLE=development
ADMIN_TOKEN=placeholder-admin-token
REGISTRATION_TOKEN=placeholder-registration-token
INTERNAL_API_TOKEN=placeholder-internal-token
CLIENT_ADMIN_TOKEN=placeholder-client-admin-token
QWEN_CLIENT_ADMIN_TOKEN=placeholder-qwen-client-admin-token
GRANITE_CLIENT_ADMIN_TOKEN=placeholder-granite-client-admin-token
"""
        path.write_text(content, encoding="utf-8")
        return path

    def test_host_env_is_generated_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            template = self._template(root, "host")

            first = load_or_create_env(
                path=env_path,
                template_path=template,
                role="host",
            )
            first_bytes = env_path.read_bytes()
            second = load_or_create_env(
                path=env_path,
                template_path=template,
                role="host",
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(env_path.read_bytes(), first_bytes)
            for name in ("ADMIN_TOKEN", "REGISTRATION_TOKEN", "INTERNAL_API_TOKEN"):
                self.assertGreaterEqual(len(first.values[name]), 32)
                self.assertNotIn("placeholder-", first.values[name])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o600)

    def test_client_requires_host_registration_token_and_generates_only_local_admins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = self._template(root, "client")
            env_path = root / ".env"

            with self.assertRaisesRegex(ValueError, "REGISTRATION_TOKEN"):
                load_or_create_env(
                    path=env_path,
                    template_path=template,
                    role="client",
                )

            result = load_or_create_env(
                path=env_path,
                template_path=template,
                role="client",
                replacements={"REGISTRATION_TOKEN": "host-issued-registration-secret"},
            )
            self.assertEqual(
                result.values["REGISTRATION_TOKEN"],
                "host-issued-registration-secret",
            )
            self.assertNotIn("ADMIN_TOKEN", result.values)
            self.assertGreaterEqual(len(result.values["QWEN_CLIENT_ADMIN_TOKEN"]), 32)
            self.assertGreaterEqual(len(result.values["GRANITE_CLIENT_ADMIN_TOKEN"]), 32)

    def test_existing_env_with_another_role_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            env_path.write_text(
                "LEGALFEDLLM_BOOTSTRAP_ROLE=host\n"
                "ADMIN_TOKEN=host-admin\n"
                "REGISTRATION_TOKEN=host-registration\n"
                "INTERNAL_API_TOKEN=host-internal\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not 'client'"):
                load_or_create_env(
                    path=env_path,
                    template_path=self._template(root, "client"),
                    role="client",
                    replacements={"REGISTRATION_TOKEN": "registration-secret"},
                )

    def test_load_env_file_preserves_explicit_process_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("A=from-file\nB=file-only\n", encoding="utf-8")
            with patch.dict(os.environ, {"A": "process"}, clear=True):
                values = load_env_file(path)
                self.assertEqual(values, {"A": "from-file", "B": "file-only"})
                self.assertEqual(os.environ["A"], "process")
                self.assertEqual(os.environ["B"], "file-only")

    def test_parser_ignores_comments_and_unquotes_simple_values(self) -> None:
        self.assertEqual(
            parse_env_text("# comment\nA=1\nB='two'\nC=\"three\"\n"),
            {"A": "1", "B": "two", "C": "three"},
        )


if __name__ == "__main__":
    unittest.main()
