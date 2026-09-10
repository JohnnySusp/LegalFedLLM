from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any


class SshTunnelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SshTunnelConfig:
    enabled: bool
    target: str
    ssh_port: int = 22
    local_port: int = 8000
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    reconnect_seconds: float = 3.0
    external: bool = False
    external_forward_host: str = "127.0.0.1"

    @classmethod
    def from_environment(cls) -> "SshTunnelConfig":
        enabled = os.getenv("CLIENT_SSH_TUNNEL_ENABLED", "false").strip().lower()
        external = os.getenv("CLIENT_SSH_TUNNEL_EXTERNAL", "false").strip().lower()
        return cls(
            enabled=enabled in {"1", "true", "yes"},
            target=os.getenv("CLIENT_SSH_TARGET", "").strip(),
            ssh_port=int(os.getenv("CLIENT_SSH_PORT", "22")),
            local_port=int(os.getenv("CLIENT_COORDINATOR_LOCAL_PORT", "8000")),
            remote_host=os.getenv(
                "CLIENT_COORDINATOR_REMOTE_HOST", "127.0.0.1"
            ).strip(),
            remote_port=int(os.getenv("CLIENT_COORDINATOR_REMOTE_PORT", "8000")),
            reconnect_seconds=float(
                os.getenv("CLIENT_SSH_RECONNECT_SECONDS", "3")
            ),
            external=external in {"1", "true", "yes"},
            external_forward_host=os.getenv(
                "CLIENT_SSH_EXTERNAL_FORWARD_HOST", "127.0.0.1"
            ).strip(),
        )

    def validate(self) -> None:
        if not self.enabled:
            if self.external:
                raise SshTunnelError(
                    "CLIENT_SSH_TUNNEL_EXTERNAL requires CLIENT_SSH_TUNNEL_ENABLED=true"
                )
            return
        if not self.external and not self.target:
            raise SshTunnelError("CLIENT_SSH_TARGET is required")
        for name, value in (
            ("ssh_port", self.ssh_port),
            ("local_port", self.local_port),
            ("remote_port", self.remote_port),
        ):
            if value < 1 or value > 65535:
                raise SshTunnelError(f"{name} must be between 1 and 65535")
        if not self.remote_host:
            raise SshTunnelError("remote Coordinator host must not be blank")
        if self.external and not self.external_forward_host:
            raise SshTunnelError("external SSH forward host must not be blank")
        if self.reconnect_seconds <= 0:
            raise SshTunnelError("SSH reconnect interval must be positive")


class SshTunnelManager:
    def __init__(self, config: SshTunnelConfig):
        config.validate()
        self.config = config
        self._process: subprocess.Popen[Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._last_exit_code: int | None = None
        self._restart_count = 0

    def _command(self) -> list[str]:
        executable = shutil.which("ssh")
        if executable is None:
            raise SshTunnelError(
                "OpenSSH client executable 'ssh' was not found in PATH"
            )
        c = self.config
        return [
            executable,
            "-N",
            "-p",
            str(c.ssh_port),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            (
                f"127.0.0.1:{c.local_port}:"
                f"{c.remote_host}:{c.remote_port}"
            ),
            c.target,
        ]

    def start(self) -> None:
        if not self.config.enabled or self.config.external:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._monitor,
                name="legalfedllm-ssh-tunnel",
                daemon=True,
            )
            self._thread.start()

    def _spawn(self) -> subprocess.Popen[Any]:
        print(
            "Opening LegalFedLLM SSH tunnel. Enter the Host SSH password "
            "when prompted. The password is not stored.",
            flush=True,
        )
        return subprocess.Popen(self._command())

    def _monitor(self) -> None:
        while not self._stop.is_set():
            try:
                process = self._spawn()
            except Exception as exc:
                print(f"SSH tunnel start failed: {exc}", flush=True)
                if self._stop.wait(self.config.reconnect_seconds):
                    return
                self._restart_count += 1
                continue
            with self._lock:
                self._process = process
            code = process.wait()
            with self._lock:
                self._last_exit_code = code
                self._process = None
            if self._stop.is_set():
                return
            self._restart_count += 1
            print(
                f"SSH tunnel exited with code {code}; retrying in "
                f"{self.config.reconnect_seconds:g}s.",
                flush=True,
            )
            if self._stop.wait(self.config.reconnect_seconds):
                return

    def wait_until_forward_reachable(self, *, poll_seconds: float = 0.2) -> None:
        if not self.config.enabled:
            return
        if poll_seconds <= 0:
            raise SshTunnelError("SSH forward poll interval must be positive")
        while not self._stop.is_set():
            status = self.status()
            if status["forward_reachable"]:
                print(
                    "LegalFedLLM SSH tunnel established; starting Client Agent API.",
                    flush=True,
                )
                return
            time.sleep(poll_seconds)
        raise SshTunnelError("SSH tunnel stopped before the forward became reachable")

    def stop(self) -> None:
        self._stop.set()
        if self.config.external:
            return
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=6)

    def _forward_reachable(self) -> bool:
        if not self.config.enabled:
            return False
        host = (
            self.config.external_forward_host
            if self.config.external
            else "127.0.0.1"
        )
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, self.config.local_port)) == 0
        finally:
            sock.close()

    def status(self) -> dict[str, Any]:
        forward_reachable = self._forward_reachable()
        with self._lock:
            process = self._process
            process_running = process is not None and process.poll() is None
            pid = process.pid if process_running else None
            last_exit_code = self._last_exit_code
        running = forward_reachable if self.config.external else process_running
        return {
            "enabled": self.config.enabled,
            "external": self.config.external,
            "running": running,
            "pid": pid,
            "forward_reachable": forward_reachable,
            "local_port": self.config.local_port,
            "remote_host": self.config.remote_host,
            "remote_port": self.config.remote_port,
            "ssh_target": self.config.target if self.config.enabled else None,
            "ssh_port": self.config.ssh_port if self.config.enabled else None,
            "restart_count": self._restart_count,
            "last_exit_code": last_exit_code,
        }
