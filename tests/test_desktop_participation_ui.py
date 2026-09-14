from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from desktop.app import (
    _desktop_icon_path,
    _participation_allowed,
    _participation_inflight_resolved,
    _low_vram_restart_required,
)


class DesktopParticipationUiTests(unittest.TestCase):
    def _round(self, **overrides):
        payload = {
            "round_id": "round-000003",
            "state": "COLLECTING",
            "selected": True,
            "participated": False,
        }
        payload.update(overrides)
        return payload

    def test_participation_is_available_only_for_selected_collecting_client(self) -> None:
        self.assertTrue(
            _participation_allowed(
                self._round(),
                compatible=True,
                coordinator_connected=True,
            )
        )
        self.assertFalse(
            _participation_allowed(
                self._round(participated=True),
                compatible=True,
                coordinator_connected=True,
            )
        )
        self.assertFalse(
            _participation_allowed(
                self._round(state="DISTILLING"),
                compatible=True,
                coordinator_connected=True,
            )
        )

    def test_inflight_request_stays_guarded_while_same_round_remains_retryable(self) -> None:
        self.assertFalse(
            _participation_inflight_resolved(
                self._round(),
                "round-000003",
            )
        )

    def test_inflight_request_resolves_when_client_participated(self) -> None:
        self.assertTrue(
            _participation_inflight_resolved(
                self._round(participated=True),
                "round-000003",
            )
        )

    def test_inflight_request_resolves_when_round_advances(self) -> None:
        self.assertTrue(
            _participation_inflight_resolved(
                self._round(round_id="round-000004"),
                "round-000003",
            )
        )

    def test_low_vram_restart_requirement_compares_effective_and_requested_state(self) -> None:
        self.assertTrue(_low_vram_restart_required(False, True))
        self.assertTrue(_low_vram_restart_required(True, False))
        self.assertFalse(_low_vram_restart_required(True, True))
        self.assertFalse(_low_vram_restart_required(False, False))

    def test_reset_defaults_confirms_before_persisting_restart_required_low_vram_change(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "desktop" / "app.py").read_text(encoding="utf-8")
        method = source.split("        def _reset_settings_defaults(self) -> None:", 1)[1].split(
            "        def _create_profile(self) -> None:", 1
        )[0]
        confirmation = "if restart_required and not self._confirm_low_vram_restart():"
        reset_call = "settings = self.manager.reset_desktop_settings()"
        self.assertIn(confirmation, method)
        self.assertIn(reset_call, method)
        self.assertLess(method.index(confirmation), method.index(reset_call))

    def test_manual_low_vram_change_and_reset_share_restart_confirmation(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "desktop" / "app.py").read_text(encoding="utf-8")
        self.assertIn("def _confirm_low_vram_restart(self) -> bool:", source)
        self.assertIn("confirmed = self._confirm_low_vram_restart()", source)
        self.assertIn(
            "if restart_required and not self._confirm_low_vram_restart():",
            source,
        )

    def test_low_vram_change_stays_open_and_describes_manual_restart(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "desktop" / "app.py").read_text(encoding="utf-8")
        settings_block = source.split("        def _set_low_vram_mode(self, enabled: bool) -> None:", 1)[1].split(
            "        def _reset_settings_defaults(self) -> None:", 1
        )[0]
        self.assertIn("Restart LegalFedLLM manually", settings_block)
        self.assertIn("LegalFedLLM will remain open", settings_block)
        self.assertIn("the current Client remains", settings_block)
        self.assertNotIn("_quit_for_settings_change", settings_block)
        self.assertNotIn("application.quit()", settings_block)
        self.assertNotIn("QTimer.singleShot", settings_block)

    def test_manual_close_retains_normal_local_ai_shutdown_prompt(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "desktop" / "app.py").read_text(encoding="utf-8")
        self.assertIn("if self.local_ai.is_running():", source)
        self.assertNotIn("_settings_change_quit", source)
        self.assertIn(
            "Settings reset to defaults: Constant Learning on, Debug Mode off, Low VRAM Mode on.",
            source,
        )

    def test_ui_uses_point_fonts_instead_of_pixel_font_styles(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "desktop" / "app.py").read_text(encoding="utf-8")
        self.assertIn("font.setPointSizeF(point_size)", source)
        self.assertNotIn("font-size: 20px", source)
        self.assertNotIn("font-size: 18px", source)
        self.assertNotIn("font-size: 17px", source)

    def test_icon_prefers_ico_on_windows_and_png_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "legalfedllm.ico").write_bytes(b"ico")
            (root / "legalfedllm.png").write_bytes(b"png")

            self.assertEqual(
                _desktop_icon_path(root, platform="win32"),
                root / "legalfedllm.ico",
            )
            self.assertEqual(
                _desktop_icon_path(root, platform="linux"),
                root / "legalfedllm.png",
            )

    def test_icon_path_returns_none_when_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_desktop_icon_path(Path(directory), platform="win32"))


if __name__ == "__main__":
    unittest.main()
