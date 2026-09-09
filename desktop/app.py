from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable

import httpx

from client.model_profiles import (
    GRANITE_3_3_2B_CLIENT_PROFILE_ID,
    QWEN_PROFILE_ID,
    ollama_model_for_profile,
)
from desktop.local_ai import LocalAiStack
from desktop.profiles import DesktopProfile, PortableProfileManager


APP_TITLE = "LegalFedLLM Client"
LOCAL_MODEL = "legalfedllm-local"
HOST_MODEL = "legalfedllm-host"
DIAGNOSTIC_MODES = ("state", "gpu")


class AgentApi:
    def __init__(self, profile: DesktopProfile, admin_token: str):
        self.base_url = f"http://127.0.0.1:{profile.agent_port}"
        self.headers = {"X-Client-Admin-Token": admin_token}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> Any:
        response = httpx.request(
            method,
            self.base_url + path,
            headers=self.headers,
            json=json_body,
            timeout=timeout,
        )
        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", detail))
            except Exception:
                pass
            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
        if response.status_code == 204:
            return None
        return response.json()

    def health(self) -> dict[str, Any]:
        response = httpx.get(self.base_url + "/health", timeout=1.0)
        response.raise_for_status()
        return response.json()

    def register(self) -> dict[str, Any]:
        return self._request("POST", "/v1/register", timeout=30.0)

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/ui/status", timeout=3.0)

    def compatibility(self) -> dict[str, Any]:
        return self._request("GET", "/v1/model-compatibility", timeout=10.0)

    def participate(self) -> dict[str, Any]:
        return self._request("POST", "/v1/participate", timeout=24 * 60 * 60)

    def suggestions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/learning/suggestions", timeout=3.0)

    def resolve_suggestion(self, suggestion_id: str, learn: bool) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/learning/suggestions/{suggestion_id}",
            json_body={"learn": learn},
            timeout=5.0,
        )

    def host_preview(self, round_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/rounds/{round_id}/host-preview",
            timeout=24 * 60 * 60,
        )

    def host_consent(self, round_id: str, consent: bool) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/rounds/{round_id}/host-consent",
            json_body={"consent": consent},
            timeout=24 * 60 * 60,
        )


class AgentProcessController:
    def __init__(self, manager: PortableProfileManager):
        self.manager = manager
        self.process: subprocess.Popen[Any] | None = None
        self.profile: DesktopProfile | None = None

    @staticmethod
    def _self_command() -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable]
        return [sys.executable, "-m", "desktop.app"]

    def start(
        self,
        profile: DesktopProfile,
        *,
        enrollment_token: str | None,
    ) -> None:
        self.stop()
        environment = self.manager.agent_environment(
            profile,
            enrollment_token=enrollment_token,
        )
        environment["PYTHONUNBUFFERED"] = "1"
        command = self._self_command() + ["--agent"]
        self.profile = profile
        self.process = subprocess.Popen(command, env=environment)

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class MonitorLoop:
    def __init__(self, profile: DesktopProfile, admin_token: str, mode: str):
        self.profile = profile
        self.api = AgentApi(profile, admin_token)
        self.mode = mode

    def run(self) -> int:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"LegalFedLLM diagnostics — {self.profile.display_name}")
            print(f"View: {self.mode}\n")
            try:
                if self.mode == "gpu":
                    self._gpu()
                elif self.mode == "health":
                    self._health()
                elif self.mode == "tunnel":
                    self._tunnel()
                else:
                    self._state()
            except KeyboardInterrupt:
                return 0
            except Exception as exc:
                print(f"Unavailable: {exc}")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                return 0

    def _gpu(self) -> None:
        nvidia = shutil.which("nvidia-smi")
        if nvidia is None:
            print("nvidia-smi not found. LegalFedLLM 1.0 supports NVIDIA/CUDA Clients only.")
            return
        subprocess.run([nvidia], check=False)

    def _health(self) -> None:
        health = self.api.health()
        print("HTTP 200 OK")
        print(json.dumps(health, indent=2, sort_keys=True))

    def _tunnel(self) -> None:
        status = self.api._request("GET", "/v1/tunnel", timeout=2.0)
        print(json.dumps(status, indent=2, sort_keys=True))

    def _state(self) -> None:
        status = self.api.status()
        print("HTTP 200 OK")
        print(json.dumps(status, indent=2, sort_keys=True))


def _profile_model_label(profile_id: str) -> str:
    if profile_id == QWEN_PROFILE_ID:
        return "Qwen 3 1.7B"
    if profile_id == GRANITE_3_3_2B_CLIENT_PROFILE_ID:
        return "Granite 3.3 2B"
    return profile_id




def _poll_failure_state(
    *,
    controller_running: bool,
    agent_has_been_healthy: bool,
) -> tuple[str, str]:
    if controller_running and not agent_has_been_healthy:
        return (
            "Waiting for SSH authentication",
            "Enter the Host SSH password in the launch terminal. "
            "LegalFedLLM will start the Client Agent API after the tunnel is established.",
        )
    if controller_running:
        return (
            "Client Agent busy/unresponsive",
            "The Client Agent did not answer this status poll. Do not re-enter the SSH password "
            "unless OpenSSH itself prompts for it in the launch terminal.",
        )
    return "Disconnected", "Client Agent unavailable"


def _enrollment_ready(
    health: dict[str, Any],
    pending_enrollment_token: str | None,
    enrollment_attempted: bool,
) -> bool:
    if not pending_enrollment_token or enrollment_attempted:
        return False
    tunnel = health.get("tunnel") or {}
    return bool(tunnel.get("forward_reachable"))


def _host_preview_should_start(
    round_id: str,
    previewed_rounds: set[str],
    inflight_rounds: set[str],
) -> bool:
    return round_id not in previewed_rounds and round_id not in inflight_rounds


def _host_preview_finished(
    round_id: str,
    previewed_rounds: set[str],
    inflight_rounds: set[str],
    *,
    succeeded: bool,
) -> None:
    inflight_rounds.discard(round_id)
    if succeeded:
        previewed_rounds.add(round_id)


def _diagnostics_ready(health: dict[str, Any]) -> bool:
    tunnel = health.get("tunnel") or {}
    if not tunnel.get("enabled"):
        return True
    return bool(tunnel.get("forward_reachable"))


def _browser_launch_ready(
    *,
    agent_healthy: bool,
    local_ai_ready: bool,
    already_attempted: bool,
) -> bool:
    return agent_healthy and local_ai_ready and not already_attempted


def _local_ai_start_ready(
    *,
    agent_healthy: bool,
    already_attempted: bool,
) -> bool:
    return agent_healthy and not already_attempted


def open_default_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        return False


def _registration_exists(manager: PortableProfileManager, profile: DesktopProfile) -> bool:
    return (
        manager.profile_paths(profile.profile_id).client_data
        / "identity"
        / "registration.json"
    ).is_file()


def _linux_relaunch_in_terminal(argv: list[str]) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if os.getenv("LEGALFEDLLM_CONSOLE_PARENT") == "1" or sys.stdin.isatty():
        return False
    executable = os.getenv("APPIMAGE", "").strip()
    if executable:
        command = [executable, "--console-parent", *argv]
    elif getattr(sys, "frozen", False):
        command = [sys.executable, "--console-parent", *argv]
    else:
        command = [sys.executable, "-m", "desktop.app", "--console-parent", *argv]
    terminals = (
        ("x-terminal-emulator", ["-e"]),
        ("konsole", ["-e"]),
        ("gnome-terminal", ["--"]),
        ("xterm", ["-e"]),
    )
    environment = dict(os.environ)
    environment["LEGALFEDLLM_CONSOLE_PARENT"] = "1"
    tmux = shutil.which("tmux")
    session_name = f"legalfedllm-{os.getpid()}"
    terminal_command = command
    if tmux:
        environment["LEGALFEDLLM_TMUX_SESSION"] = session_name
        terminal_command = [
            tmux,
            "new-session",
            "-s",
            session_name,
            shlex.join(command),
        ]
    for name, prefix in terminals:
        path = shutil.which(name)
        if path:
            subprocess.Popen([path, *prefix, *terminal_command], env=environment)
            return True
    return False


def _desktop_restart_command(data_root: Path) -> list[str]:
    executable = os.getenv("APPIMAGE", "").strip()
    if executable:
        command = [executable]
    elif getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        command = [sys.executable, "-m", "desktop.app"]
    return [*command, "--data-root", str(Path(data_root).resolve())]

def _diagnostic_command(mode: str, profile_id: str, data_root: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [
            sys.executable,
            "--monitor",
            mode,
            "--profile-id",
            profile_id,
            "--data-root",
            str(data_root),
        ]
    return [
        sys.executable,
        "-m",
        "desktop.app",
        "--monitor",
        mode,
        "--profile-id",
        profile_id,
        "--data-root",
        str(data_root),
    ]


def launch_diagnostics(
    manager: PortableProfileManager,
    profile: DesktopProfile,
) -> list[subprocess.Popen[Any]]:
    if os.getenv("LEGALFEDLLM_DISABLE_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}:
        return []
    modes = DIAGNOSTIC_MODES
    processes: list[subprocess.Popen[Any]] = []
    if sys.platform.startswith("linux"):
        tmux = shutil.which("tmux")
        if tmux and os.getenv("TMUX"):
            for mode in modes:
                subprocess.run(
                    [
                        tmux,
                        "new-window",
                        "-d",
                        "-n",
                        f"lf-{mode}",
                        *(_diagnostic_command(mode, profile.profile_id, manager.data_root)),
                    ],
                    check=False,
                )
            return processes
        terminal = shutil.which("x-terminal-emulator") or shutil.which("konsole") or shutil.which("gnome-terminal")
        if terminal:
            for mode in modes:
                command = _diagnostic_command(mode, profile.profile_id, manager.data_root)
                if Path(terminal).name == "gnome-terminal":
                    processes.append(subprocess.Popen([terminal, "--", *command]))
                else:
                    processes.append(subprocess.Popen([terminal, "-e", *command]))
            return processes
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        for mode in modes:
            processes.append(
                subprocess.Popen(
                    _diagnostic_command(mode, profile.profile_id, manager.data_root),
                    creationflags=creationflags,
                )
            )
    return processes


def stop_diagnostics(processes: list[subprocess.Popen[Any]]) -> None:
    while processes:
        process = processes.pop()
        if process.poll() is not None:
            continue
        try:
            process.terminate()
        except OSError:
            continue



def run_gui(data_root: Path | None = None) -> int:
    from PySide6.QtCore import QThread, QTimer, Signal
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    class Worker(QThread):
        success = Signal(object)
        failure = Signal(str)

        def __init__(self, call: Callable[[], Any]):
            super().__init__()
            self.call = call

        def run(self) -> None:
            try:
                self.success.emit(self.call())
            except Exception as exc:
                self.failure.emit(str(exc))

    class CreateProfileDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Create LegalFedLLM profile")
            form = QFormLayout(self)
            self.name = QLineEdit()
            self.name.setPlaceholderText("My LegalFedLLM client")
            self.model = QComboBox()
            for profile_id in (QWEN_PROFILE_ID, GRANITE_3_3_2B_CLIENT_PROFILE_ID):
                self.model.addItem(
                    f"{_profile_model_label(profile_id)} — {ollama_model_for_profile(profile_id)}",
                    profile_id,
                )
            self.ssh_target = QLineEdit()
            self.ssh_target.setPlaceholderText("user@host.example")
            self.ssh_port = QSpinBox()
            self.ssh_port.setRange(1, 65535)
            self.ssh_port.setValue(22)
            self.local_port = QSpinBox()
            self.local_port.setRange(1, 65535)
            self.local_port.setValue(8000)
            self.agent_port = QSpinBox()
            self.agent_port.setRange(1, 65535)
            self.agent_port.setValue(8001)
            self.token = QLineEdit()
            self.token.setEchoMode(QLineEdit.EchoMode.Password)
            self.token.setPlaceholderText("one-time Host enrollment token")
            form.addRow("Profile name", self.name)
            form.addRow("Client model", self.model)
            form.addRow("SSH target", self.ssh_target)
            form.addRow("SSH port", self.ssh_port)
            form.addRow("Coordinator local port", self.local_port)
            form.addRow("Client Agent port", self.agent_port)
            form.addRow("Enrollment token", self.token)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            form.addRow(buttons)

        def values(self) -> dict[str, Any]:
            return {
                "display_name": self.name.text().strip(),
                "model_profile_id": str(self.model.currentData()),
                "ssh_target": self.ssh_target.text().strip(),
                "ssh_port": self.ssh_port.value(),
                "coordinator_local_port": self.local_port.value(),
                "agent_port": self.agent_port.value(),
                "enrollment_token": self.token.text().strip(),
            }

        def accept(self) -> None:
            values = self.values()
            if not values["display_name"] or not values["ssh_target"] or not values["enrollment_token"]:
                QMessageBox.warning(self, APP_TITLE, "Profile name, SSH target and enrollment token are required.")
                return
            super().accept()

    class EnrollmentDialog(QDialog):
        def __init__(self, profile: DesktopProfile, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Enroll Client profile")
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"{profile.display_name} is not enrolled. Enter a fresh one-time Host token."))
            self.token = QLineEdit()
            self.token.setEchoMode(QLineEdit.EchoMode.Password)
            layout.addWidget(self.token)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def accept(self) -> None:
            if not self.token.text().strip():
                QMessageBox.warning(self, APP_TITLE, "Enrollment token is required.")
                return
            super().accept()

    class LearningDialog(QDialog):
        def __init__(self, suggestion: dict[str, Any], parent=None):
            super().__init__(parent)
            self.setWindowTitle("Local learning")
            self.choice: bool | None = None
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Add this LOCAL interaction to the Client's one-use local learning queue?"))
            layout.addWidget(QLabel("Prompt"))
            prompt = QTextEdit()
            prompt.setReadOnly(True)
            prompt.setPlainText(str(suggestion.get("prompt", "")))
            prompt.setMaximumHeight(130)
            layout.addWidget(prompt)
            layout.addWidget(QLabel("Answer"))
            answer = QTextEdit()
            answer.setReadOnly(True)
            answer.setPlainText(str(suggestion.get("answer", "")))
            answer.setMaximumHeight(170)
            layout.addWidget(answer)
            row = QHBoxLayout()
            learn = QPushButton("Learn from this")
            dismiss = QPushButton("Dismiss")
            learn.clicked.connect(lambda: self._finish(True))
            dismiss.clicked.connect(lambda: self._finish(False))
            row.addWidget(learn)
            row.addWidget(dismiss)
            layout.addLayout(row)

        def _finish(self, choice: bool) -> None:
            self.choice = choice
            self.accept()

    class HostLearningDialog(QDialog):
        def __init__(self, sample_count: int, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Host knowledge available")
            self.choice: bool | None = None
            layout = QVBoxLayout(self)
            layout.addWidget(
                QLabel(
                    f"The verified Host package is a better teacher for {sample_count} reference sample(s).\n\n"
                    "Allow this Client to learn from the selected Host knowledge?"
                )
            )
            row = QHBoxLayout()
            accept = QPushButton("Learn from Host")
            decline = QPushButton("Not now")
            accept.clicked.connect(lambda: self._finish(True))
            decline.clicked.connect(lambda: self._finish(False))
            row.addWidget(accept)
            row.addWidget(decline)
            layout.addLayout(row)

        def _finish(self, choice: bool) -> None:
            self.choice = choice
            self.accept()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.manager = PortableProfileManager(data_root)
            self.controller = AgentProcessController(self.manager)
            self.local_ai = LocalAiStack(self.manager.data_root)
            self.profile: DesktopProfile | None = None
            self.api: AgentApi | None = None
            self.pending_enrollment_token: str | None = None
            self.enrollment_attempted = False
            self.agent_has_been_healthy = False
            self.diagnostics_launched = False
            self.diagnostic_processes: list[subprocess.Popen[Any]] = []
            self.compatible = False
            self.last_health: dict[str, Any] = {}
            self.last_status: dict[str, Any] = {}
            self.local_ai_payload: dict[str, Any] | None = None
            self.local_ai_start_attempted = False
            self.anythingllm_browser_attempted = False
            self.workers: set[Worker] = set()
            self.poll_worker: Worker | None = None
            self.suggestion_dialog_open = False
            self.suggestion_resolution_inflight: set[str] = set()
            self.previewed_rounds: set[str] = set()
            self.host_preview_inflight: set[str] = set()
            self._build_ui()
            self._profile_menu()
            self._settings_menu()
            self._resize_for_screen()
            QTimer.singleShot(0, self._bootstrap_profile)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._poll)
            self.timer.start(2000)

        def _build_ui(self) -> None:
            self.setWindowTitle(APP_TITLE)
            central = QWidget()
            layout = QVBoxLayout(central)
            top = QHBoxLayout()
            title = QLabel("LegalFedLLM Client")
            title.setStyleSheet("font-size: 20px; font-weight: 600;")
            top.addWidget(title)
            top.addStretch(1)
            self.settings_button = QToolButton()
            self.settings_button.setText("⚙")
            self.settings_button.setToolTip("Options")
            self.settings_button.setFixedSize(50, 50)
            self.settings_button.setStyleSheet("font-size: 18px;")
            self.settings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            top.addWidget(self.settings_button)
            self.profile_button = QToolButton()
            self.profile_button.setText("☰")
            self.profile_button.setToolTip("Profiles")
            self.profile_button.setFixedSize(50, 50)
            self.profile_button.setStyleSheet("font-size: 18px;")
            self.profile_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            top.addWidget(self.profile_button)
            layout.addLayout(top)

            group = QGroupBox("Current Client")
            form = QFormLayout(group)
            self.profile_label = QLabel("—")
            self.model_label = QLabel("—")
            self.connection_label = QLabel("Starting…")
            self.round_label = QLabel("—")
            self.participants_label = QLabel("—")
            self.my_state_label = QLabel("—")
            self.queue_label = QLabel("0")
            form.addRow("Profile", self.profile_label)
            form.addRow("Model", self.model_label)
            form.addRow("Coordinator", self.connection_label)
            form.addRow("Round", self.round_label)
            form.addRow("Participants", self.participants_label)
            form.addRow("My state", self.my_state_label)
            form.addRow("Local learning queue", self.queue_label)
            layout.addWidget(group)

            self.message = QLabel("Starting Client Agent…")
            self.message.setWordWrap(True)
            layout.addWidget(self.message)
            layout.addStretch(1)

            self.provider_button = QPushButton("AnythingLLM integration details")
            self.provider_button.clicked.connect(self._show_provider_details)
            layout.addWidget(self.provider_button)

            self.participate_button = QPushButton("Participate in the current federated round")
            self.participate_button.setMinimumHeight(90)
            self.participate_button.setStyleSheet("font-size: 17px; font-weight: 600;")
            self.participate_button.clicked.connect(self._participate)
            self.participate_button.setEnabled(False)
            layout.addWidget(self.participate_button)
            self.setCentralWidget(central)

        def _resize_for_screen(self) -> None:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.resize(480, 600)
                return
            available = screen.availableGeometry()
            width = max(420, int(available.width() / 4))
            height = max(540, int(available.height() / 2))
            self.resize(width, height)

        def _bootstrap_profile(self) -> None:
            profile = self.manager.active_profile()
            if profile is None:
                self._create_profile()
            else:
                self._activate_profile(profile)

        def _profile_menu(self) -> None:
            menu = QMenu(self)
            for profile in self.manager.list_profiles():
                action = QAction(profile.display_name, menu)
                action.triggered.connect(lambda checked=False, p=profile: self._activate_profile(p))
                menu.addAction(action)
            if menu.actions():
                menu.addSeparator()
            create = QAction("Create new profile…", menu)
            create.triggered.connect(self._create_profile)
            menu.addAction(create)
            self.profile_button.setMenu(menu)

        def _settings_menu(self) -> None:
            settings = self.manager.desktop_settings()
            menu = QMenu(self)
            self.constant_learning_action = QAction("Constant Learning", menu)
            self.constant_learning_action.setCheckable(True)
            self.constant_learning_action.setChecked(settings["constant_learning"])
            self.constant_learning_action.toggled.connect(self._set_constant_learning)
            menu.addAction(self.constant_learning_action)

            self.debug_mode_action = QAction("Debug Mode", menu)
            self.debug_mode_action.setCheckable(True)
            self.debug_mode_action.setChecked(settings["debug_mode"])
            self.debug_mode_action.toggled.connect(self._set_debug_mode)
            menu.addAction(self.debug_mode_action)

            self.low_vram_mode_action = QAction("Low VRAM Mode", menu)
            self.low_vram_mode_action.setCheckable(True)
            self.low_vram_mode_action.setChecked(settings["low_vram_mode"])
            self.low_vram_mode_action.setToolTip(
                "Reduce peak CUDA memory pressure during Client training and reverse distillation."
            )
            self.low_vram_mode_action.toggled.connect(self._set_low_vram_mode)
            menu.addAction(self.low_vram_mode_action)

            menu.addSeparator()
            self.reset_defaults_action = QAction("Reset Defaults", menu)
            self.reset_defaults_action.triggered.connect(self._reset_settings_defaults)
            menu.addAction(self.reset_defaults_action)
            self.settings_button.setMenu(menu)

        def _set_constant_learning(self, enabled: bool) -> None:
            self.manager.set_desktop_setting("constant_learning", enabled)
            self.message.setText(
                "Constant Learning enabled: new LOCAL interactions will be queued automatically."
                if enabled
                else "Constant Learning disabled: new LOCAL interactions will ask for Learn/Dismiss consent."
            )

        def _set_debug_mode(self, enabled: bool) -> None:
            self.manager.set_desktop_setting("debug_mode", enabled)
            if enabled:
                self.diagnostics_launched = False
                if self.last_health:
                    self._ensure_diagnostics(self.last_health)
                self.message.setText("Debug Mode enabled.")
            else:
                stop_diagnostics(self.diagnostic_processes)
                self.diagnostics_launched = False
                self.message.setText("Debug Mode disabled. Extra diagnostic terminals are closed when possible.")

        def _set_low_vram_mode(self, enabled: bool) -> None:
            previous = self.manager.desktop_settings()["low_vram_mode"]

            if enabled:
                answer = QMessageBox.warning(
                    self,
                    "Low VRAM Mode",
                    "This setting reduces peak GPU-memory pressure during Client training "
                    "and reverse distillation by enabling gradient checkpointing and PyTorch "
                    "expandable CUDA memory segments.\n\n"
                    "Gradient checkpointing trades memory usage for additional computation. "
                    "Training may take longer and keep the GPU under sustained load for longer, "
                    "which can increase GPU temperatures, power usage, and fan noise. "
                    "Enable this mode when you encounter CUDA out-of-memory errors.\n\n"
                    "This will automatically restart the application. Are you sure you want to proceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
            else:
                answer = QMessageBox.question(
                    self,
                    "Low VRAM Mode",
                    "This will automatically restart the application. Are you sure you want to proceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )

            if answer != QMessageBox.StandardButton.Yes:
                self.low_vram_mode_action.blockSignals(True)
                self.low_vram_mode_action.setChecked(previous)
                self.low_vram_mode_action.blockSignals(False)
                return

            self.manager.set_desktop_setting("low_vram_mode", enabled)
            self.message.setText(
                f"Low VRAM Mode {'enabled' if enabled else 'disabled'}. Restarting LegalFedLLM…"
            )
            QTimer.singleShot(0, self._restart_application)

        def _restart_application(self) -> None:
            self.timer.stop()
            stop_diagnostics(self.diagnostic_processes)
            self.diagnostics_launched = False
            self.controller.stop()
            command = _desktop_restart_command(self.manager.data_root)
            os.execvpe(command[0], command, dict(os.environ))

        def _reset_settings_defaults(self) -> None:
            settings = self.manager.reset_desktop_settings()
            for action, key in (
                (self.constant_learning_action, "constant_learning"),
                (self.debug_mode_action, "debug_mode"),
                (self.low_vram_mode_action, "low_vram_mode"),
            ):
                action.blockSignals(True)
                action.setChecked(settings[key])
                action.blockSignals(False)
            stop_diagnostics(self.diagnostic_processes)
            self.diagnostics_launched = False
            self.message.setText(
                "Settings reset to defaults: Constant Learning on, Debug Mode off, Low VRAM Mode off. "
                "Restart the Client Agent/LegalFedLLM if Low VRAM Mode changed."
            )

        def _create_profile(self) -> None:
            dialog = CreateProfileDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                if not self.manager.list_profiles():
                    self.message.setText("Create a profile to start LegalFedLLM.")
                return
            values = dialog.values()
            token = values.pop("enrollment_token")
            try:
                profile = self.manager.create(**values)
            except Exception as exc:
                QMessageBox.critical(self, APP_TITLE, str(exc))
                return
            self.pending_enrollment_token = token
            self._activate_profile(profile, enrollment_token=token)

        def _activate_profile(
            self,
            profile: DesktopProfile,
            enrollment_token: str | None = None,
        ) -> None:
            if self.profile and self.profile.profile_id == profile.profile_id and self.controller.running():
                return
            if not _registration_exists(self.manager, profile) and not enrollment_token:
                dialog = EnrollmentDialog(profile, self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    self.message.setText("This profile requires a one-time enrollment token before it can connect.")
                    return
                enrollment_token = dialog.token.text().strip()
            self.manager.set_active(profile.profile_id)
            self.profile = profile
            self.api = AgentApi(profile, self.manager.admin_token(profile.profile_id))
            self.pending_enrollment_token = enrollment_token
            self.enrollment_attempted = False
            self.agent_has_been_healthy = False
            stop_diagnostics(self.diagnostic_processes)
            self.diagnostics_launched = False
            self.compatible = False
            self.last_health = {}
            self.local_ai_payload = None
            self.local_ai_start_attempted = False
            self.anythingllm_browser_attempted = False
            self.suggestion_resolution_inflight.clear()
            self.previewed_rounds.clear()
            self.host_preview_inflight.clear()
            self.profile_label.setText(profile.display_name)
            self.model_label.setText(
                f"{_profile_model_label(profile.model_profile_id)} ({profile.ollama_model})"
            )
            self.connection_label.setText("Starting…")
            self.round_label.setText("—")
            self.participants_label.setText("—")
            self.my_state_label.setText("Not connected")
            self.message.setText(
                "Connecting to the Host. Enter the SSH password in the launch terminal. "
                "The Client Agent API will start after the SSH tunnel is established."
            )
            try:
                self.controller.start(profile, enrollment_token=enrollment_token)
            except Exception as exc:
                QMessageBox.critical(self, APP_TITLE, f"Could not start Client Agent: {exc}")
            self._profile_menu()

        def _ensure_local_ai(self) -> None:
            if self.profile is None:
                return
            if not _local_ai_start_ready(
                agent_healthy=self.agent_has_been_healthy,
                already_attempted=self.local_ai_start_attempted,
            ):
                return
            self.local_ai_start_attempted = True
            profile = self.profile
            admin_token = self.manager.admin_token(profile.profile_id)
            self.message.setText(
                "SSH connected. Starting local Ollama and AnythingLLM…"
            )
            self._run_worker(
                lambda: self.local_ai.prepare(profile, admin_token),
                self._local_ai_ready,
                self._local_ai_failed,
            )

        def _local_ai_ready(self, payload: Any) -> None:
            self.local_ai_payload = dict(payload) if isinstance(payload, dict) else {}
            self._maybe_open_anythingllm()
            if self.agent_has_been_healthy:
                self.message.setText(
                    f"Ready. AnythingLLM is available at {self.local_ai_payload.get('anythingllm_url', 'http://127.0.0.1:3001')}."
                )

        def _maybe_open_anythingllm(self) -> None:
            if not _browser_launch_ready(
                agent_healthy=self.agent_has_been_healthy,
                local_ai_ready=self.local_ai_payload is not None,
                already_attempted=self.anythingllm_browser_attempted,
            ):
                return
            self.anythingllm_browser_attempted = True
            url = str((self.local_ai_payload or {}).get("anythingllm_url") or "http://127.0.0.1:3001")
            open_default_browser(url.rstrip("/") + "/")

        def _local_ai_failed(self, error: str) -> None:
            self.message.setText(f"Local AnythingLLM/Ollama stack unavailable: {error}")
            QMessageBox.warning(
                self,
                APP_TITLE,
                "LegalFedLLM could not prepare the local Ollama/AnythingLLM stack.\n\n" + error,
            )

        def _run_worker(
            self,
            call: Callable[[], Any],
            success: Callable[[Any], None],
            failure: Callable[[str], None] | None = None,
        ) -> Worker:
            worker = Worker(call)
            self.workers.add(worker)
            worker.success.connect(success)
            worker.failure.connect(failure or self._show_error)
            worker.finished.connect(lambda: self.workers.discard(worker))
            worker.start()
            return worker

        def _poll(self) -> None:
            if self.api is None or self.poll_worker is not None:
                return

            def load() -> dict[str, Any]:
                health = self.api.health()
                compatibility = self.api.compatibility()
                if health.get("enrolled"):
                    status = self.api.status()
                    suggestions = self.api.suggestions()
                else:
                    status = {
                        "enrolled": False,
                        "learning_queue": health.get("learning_queue") or {},
                        "tunnel": health.get("tunnel") or {},
                        "coordinator_connected": False,
                        "round": None,
                    }
                    suggestions = []
                return {
                    "health": health,
                    "status": status,
                    "compatibility": compatibility,
                    "suggestions": suggestions,
                }

            self.poll_worker = Worker(load)
            self.poll_worker.success.connect(self._poll_success)
            self.poll_worker.failure.connect(self._poll_failure)
            self.poll_worker.finished.connect(self._poll_finished)
            self.poll_worker.start()

        def _poll_finished(self) -> None:
            self.poll_worker = None

        def _poll_failure(self, error: str) -> None:
            self.participate_button.setEnabled(False)
            connection, message = _poll_failure_state(
                controller_running=self.controller.running(),
                agent_has_been_healthy=self.agent_has_been_healthy,
            )
            self.connection_label.setText(connection)
            if connection == "Disconnected":
                self.message.setText(f"{message}: {error}")
            else:
                self.message.setText(message)

        def _ensure_diagnostics(self, health: dict[str, Any]) -> None:
            if self.profile is None or self.diagnostics_launched:
                return
            if not self.manager.desktop_settings()["debug_mode"]:
                return
            if not _diagnostics_ready(health):
                return
            self.diagnostic_processes = launch_diagnostics(self.manager, self.profile)
            self.diagnostics_launched = True

        def _poll_success(self, payload: dict[str, Any]) -> None:
            self.agent_has_been_healthy = True
            self._ensure_local_ai()
            self.last_health = payload["health"]
            status = payload["status"]
            compatibility = payload["compatibility"]
            self._ensure_diagnostics(self.last_health)
            self._maybe_open_anythingllm()
            self.last_status = status
            self.compatible = bool(compatibility.get("compatible"))
            if not status.get("enrolled"):
                tunnel = payload["health"].get("tunnel") or {}
                if not tunnel.get("forward_reachable"):
                    self.connection_label.setText("Waiting for SSH authentication")
                    self.message.setText(
                        "Enter the Host SSH password in the launch terminal. "
                        "Enrollment begins only after the SSH tunnel is connected."
                    )
                    return
                self.connection_label.setText("SSH connected / enrollment pending")
                if _enrollment_ready(
                    payload["health"],
                    self.pending_enrollment_token,
                    self.enrollment_attempted,
                ):
                    self.enrollment_attempted = True
                    self.message.setText("Enrolling this profile with its one-time Host token…")
                    self._run_worker(self.api.register, self._registered, self._registration_failed)
                elif not self.pending_enrollment_token:
                    self.message.setText(
                        "This profile is not enrolled. Select it from the profile menu to enter a fresh one-time Host token."
                    )
                return
            self.pending_enrollment_token = None

            if status.get("coordinator_connected"):
                self.connection_label.setText("Connected")
            else:
                self.connection_label.setText("Tunnel/Coordinator unavailable")
            if not self.compatible:
                error = compatibility.get("error") or "Selected Ollama model is not compatible."
                self.message.setText(str(error))
            elif status.get("coordinator_connected"):
                self.message.setText("Ready.")

            learning = status.get("learning_queue") or {}
            self.queue_label.setText(str(learning.get("queued_example_count", 0)))
            round_info = status.get("round")
            enable = False
            if round_info:
                self.round_label.setText(f"{round_info['round_id']} — {round_info['state']}")
                self.participants_label.setText(
                    f"{round_info['accepted_count']} / {round_info['quorum']} required"
                )
                if round_info.get("participated"):
                    self.my_state_label.setText("Participated")
                elif round_info.get("selected"):
                    self.my_state_label.setText("Selected / not participated")
                else:
                    self.my_state_label.setText("Not selected")
                enable = bool(
                    self.compatible
                    and status.get("coordinator_connected")
                    and round_info.get("selected")
                    and not round_info.get("participated")
                    and round_info.get("state") == "COLLECTING"
                )
                self._maybe_preview_host(round_info, status)
            else:
                self.round_label.setText("No current round")
                self.participants_label.setText("—")
                self.my_state_label.setText("Idle")
            self.participate_button.setEnabled(enable)
            self._maybe_prompt_learning(payload.get("suggestions") or [])

        def _registered(self, _payload: Any) -> None:
            self.pending_enrollment_token = None
            self.message.setText("Profile enrolled. The enrollment token has been consumed.")

        def _registration_failed(self, error: str) -> None:
            self.connection_label.setText("Enrollment failed")
            self.message.setText(
                f"Enrollment failed: {error}. "
                "The Client Agent has been stopped; select this profile again to retry with a fresh token if needed."
            )
            self.pending_enrollment_token = None
            self.controller.stop()
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Enrollment failed. LegalFedLLM will not restart the Agent or ask for another token automatically.\n\n"
                "Check the SSH/Coordinator connection, then select this profile again when you are ready to retry.",
            )

        def _participate(self) -> None:
            if self.api is None:
                return
            self.participate_button.setEnabled(False)
            self.message.setText(
                "Participating: applying queued local learning if present, running Dᴾ reference inference, signing and submitting the Client package…"
            )
            self._run_worker(self.api.participate, self._participated)

        def _participated(self, payload: Any) -> None:
            self.message.setText(
                f"Participated successfully in {payload.get('round_id', 'the current round')}."
            )

        def _maybe_prompt_learning(self, suggestions: list[dict[str, Any]]) -> None:
            if self.suggestion_dialog_open or not suggestions or self.api is None:
                return
            suggestion = next(
                (
                    item
                    for item in suggestions
                    if str(item.get("suggestion_id", ""))
                    and str(item.get("suggestion_id", "")) not in self.suggestion_resolution_inflight
                ),
                None,
            )
            if suggestion is None:
                return
            suggestion_id = str(suggestion.get("suggestion_id", ""))
            if self.manager.desktop_settings()["constant_learning"]:
                self.suggestion_resolution_inflight.add(suggestion_id)
                self._run_worker(
                    lambda: self.api.resolve_suggestion(suggestion_id, True),
                    lambda _, sid=suggestion_id: self._suggestion_resolved(sid, True),
                    lambda error, sid=suggestion_id: self._suggestion_failed(sid, error),
                )
                return

            self.suggestion_dialog_open = True
            dialog = LearningDialog(suggestion, self)
            dialog.exec()
            choice = bool(dialog.choice)
            self.suggestion_dialog_open = False
            self.suggestion_resolution_inflight.add(suggestion_id)
            self._run_worker(
                lambda: self.api.resolve_suggestion(suggestion_id, choice),
                lambda _, sid=suggestion_id, learn=choice: self._suggestion_resolved(sid, learn),
                lambda error, sid=suggestion_id: self._suggestion_failed(sid, error),
            )

        def _suggestion_resolved(self, suggestion_id: str, learn: bool) -> None:
            self.suggestion_resolution_inflight.discard(suggestion_id)
            self.message.setText(
                "Interaction added to the local learning queue."
                if learn
                else "Interaction dismissed; nothing was added to local learning."
            )

        def _suggestion_failed(self, suggestion_id: str, error: str) -> None:
            self.suggestion_resolution_inflight.discard(suggestion_id)
            self._show_error(error)

        def _maybe_preview_host(
            self,
            round_info: dict[str, Any],
            status: dict[str, Any],
        ) -> None:
            if self.api is None or not round_info.get("participated"):
                return
            round_id = str(round_info.get("round_id", ""))
            if not round_id or round_info.get("state") != "COMPLETED":
                return
            client_state = status.get("client_state") or {}
            if client_state.get("last_completed_round") == round_id:
                return
            if not _host_preview_should_start(
                round_id,
                self.previewed_rounds,
                self.host_preview_inflight,
            ):
                return
            self.host_preview_inflight.add(round_id)
            self.message.setText("Checking the verified Host package for useful reverse knowledge…")
            self._run_worker(
                lambda: self.api.host_preview(round_id),
                lambda payload, rid=round_id: self._host_preview_ready(rid, payload),
                lambda error, rid=round_id: self._host_preview_failed(rid, error),
            )

        def _host_preview_failed(self, round_id: str, error: str) -> None:
            _host_preview_finished(
                round_id,
                self.previewed_rounds,
                self.host_preview_inflight,
                succeeded=False,
            )
            self.message.setText(
                "Host reverse-knowledge preview failed and remains retryable for this round."
            )
            self._show_error(error)

        def _host_preview_ready(self, round_id: str, payload: dict[str, Any]) -> None:
            _host_preview_finished(
                round_id,
                self.previewed_rounds,
                self.host_preview_inflight,
                succeeded=True,
            )
            if not payload.get("requires_consent"):
                self.message.setText("Host package contained no better teacher samples for this Client; no reverse learning is needed.")
                return
            count = int(payload.get("host_teacher_sample_count", 0))
            dialog = HostLearningDialog(count, self)
            dialog.exec()
            consent = bool(dialog.choice)
            self.message.setText(
                "Applying verified Host knowledge…" if consent else "Declining reverse Host learning for this round…"
            )
            self._run_worker(
                lambda: self.api.host_consent(round_id, consent),
                lambda _: self.message.setText(
                    "Reverse Host learning completed." if consent else "Reverse Host learning declined for this round."
                ),
            )

        def _show_provider_details(self) -> None:
            if self.profile is None:
                return
            text = (
                "AnythingLLM is configured automatically when this profile is activated.\n\n"
                f"AnythingLLM UI:\nhttp://127.0.0.1:3001\n\n"
                f"OpenAI-compatible base URL:\nhttp://127.0.0.1:{self.profile.agent_port}/v1\n\n"
                f"Models:\n- {LOCAL_MODEL}\n- {HOST_MODEL}\n\n"
                f"Runtime files:\n{self.local_ai.runtime_root}\n\n"
                "LOCAL stays on the Client. HOST explicitly forwards the prompt to the Host queue. "
                "AnythingLLM native Generic OpenAI tool calling is disabled for the current 1.0 scope."
            )
            QMessageBox.information(self, "AnythingLLM integration details", text)

        def _show_error(self, error: str) -> None:
            self.message.setText(error)
            QMessageBox.warning(self, APP_TITLE, error)

        def closeEvent(self, event) -> None:  # type: ignore[override]
            if self.local_ai.is_running():
                prompt = QMessageBox(self)
                prompt.setWindowTitle(APP_TITLE)
                prompt.setIcon(QMessageBox.Icon.Question)
                prompt.setText("Stop Ollama and AnythingLLM too?")
                prompt.setInformativeText(
                    "Closing LegalFedLLM can also stop its local Docker services. "
                    "Persistent AnythingLLM workspaces and Ollama models will be kept."
                )
                stop_button = prompt.addButton(
                    "Stop Docker and Close",
                    QMessageBox.ButtonRole.AcceptRole,
                )
                leave_button = prompt.addButton(
                    "Leave Docker Running",
                    QMessageBox.ButtonRole.DestructiveRole,
                )
                cancel_button = prompt.addButton(QMessageBox.StandardButton.Cancel)
                prompt.setDefaultButton(cancel_button)
                prompt.exec()
                clicked = prompt.clickedButton()
                if clicked is cancel_button or clicked is None:
                    event.ignore()
                    return
                if clicked is stop_button:
                    try:
                        self.local_ai.stop()
                    except Exception as exc:
                        QMessageBox.warning(
                            self,
                            APP_TITLE,
                            "Could not stop Ollama/AnythingLLM. LegalFedLLM will remain open.\n\n" + str(exc),
                        )
                        event.ignore()
                        return
                elif clicked is not leave_button:
                    event.ignore()
                    return
            self.timer.stop()
            stop_diagnostics(self.diagnostic_processes)
            self.controller.stop()
            event.accept()

    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LegalFedLLM portable desktop Client")
    parser.add_argument("--agent", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--monitor", choices=("state", "health", "gpu", "tunnel"))
    parser.add_argument("--profile-id")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--console-parent", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.agent:
        from desktop.agent_entry import run_agent

        return run_agent()
    if args.monitor:
        if not args.profile_id:
            parser.error("--monitor requires --profile-id")
        manager = PortableProfileManager(args.data_root)
        profile = manager.load(args.profile_id)
        return MonitorLoop(profile, manager.admin_token(profile.profile_id), args.monitor).run()

    passthrough = []
    if _linux_relaunch_in_terminal(passthrough):
        return 0
    if args.console_parent:
        os.environ["LEGALFEDLLM_CONSOLE_PARENT"] = "1"
    return run_gui(args.data_root)


if __name__ == "__main__":
    raise SystemExit(main())
