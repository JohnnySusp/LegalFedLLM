from __future__ import annotations

import os

from client.tunnel import SshTunnelConfig, SshTunnelManager


def run_agent() -> int:
    import uvicorn

    from client.main import create_app, gateway_from_environment, runtime_from_environment

    host = os.getenv("CLIENT_AGENT_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CLIENT_AGENT_PORT", "8001"))
    tunnel = SshTunnelManager(SshTunnelConfig.from_environment())
    tunnel.start()
    try:
        tunnel.wait_until_forward_reachable()
        app = create_app(
            runtime_from_environment(),
            gateway_from_environment(),
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


if __name__ == "__main__":
    raise SystemExit(run_agent())
