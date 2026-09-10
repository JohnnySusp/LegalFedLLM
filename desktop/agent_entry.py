from __future__ import annotations

import importlib
import os
import signal
from types import FrameType

from client.tunnel import SshTunnelConfig, SshTunnelManager
from desktop.client_docker import ClientDockerStack


def _docker_agent_enabled() -> bool:
    return os.getenv("LEGALFEDLLM_DESKTOP_DOCKER_AGENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _run_inprocess_agent() -> int:
    uvicorn = importlib.import_module("uvicorn")
    client_main = importlib.import_module("client.main")

    host = os.getenv("CLIENT_AGENT_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CLIENT_AGENT_PORT", "8001"))
    tunnel = SshTunnelManager(SshTunnelConfig.from_environment())
    tunnel.start()
    try:
        tunnel.wait_until_forward_reachable()
        app = client_main.create_app(
            client_main.runtime_from_environment(),
            client_main.gateway_from_environment(),
            tunnel_manager=tunnel,
        )
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=os.getenv("CLIENT_AGENT_LOG_LEVEL", "info"),
        )
    finally:
        tunnel.stop()
    return 0


def _run_docker_agent() -> int:
    tunnel = SshTunnelManager(SshTunnelConfig.from_environment())
    stack = ClientDockerStack(os.environ)
    stopping = False

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        stack.stop()

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    tunnel.start()
    try:
        tunnel.wait_until_forward_reachable()
        return stack.run_foreground()
    finally:
        stack.stop()
        tunnel.stop()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def run_agent() -> int:
    if _docker_agent_enabled():
        return _run_docker_agent()
    return _run_inprocess_agent()


if __name__ == "__main__":
    raise SystemExit(run_agent())
