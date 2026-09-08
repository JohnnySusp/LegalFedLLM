from __future__ import annotations

import unittest
from unittest import mock

from desktop.agent_entry import run_agent


class DesktopAgentEntryTests(unittest.TestCase):
    def test_initial_ssh_forward_is_ready_before_runtime_and_uvicorn_start(self) -> None:
        events: list[str] = []
        tunnel = mock.Mock()
        tunnel.start.side_effect = lambda: events.append("tunnel-start")
        tunnel.wait_until_forward_reachable.side_effect = lambda: events.append("tunnel-ready")
        tunnel.stop.side_effect = lambda: events.append("tunnel-stop")

        with (
            mock.patch("desktop.agent_entry.SshTunnelConfig.from_environment", return_value=object()),
            mock.patch("desktop.agent_entry.SshTunnelManager", return_value=tunnel),
            mock.patch(
                "client.main.runtime_from_environment",
                side_effect=lambda: events.append("runtime") or object(),
            ),
            mock.patch(
                "client.main.gateway_from_environment",
                side_effect=lambda: events.append("gateway") or object(),
            ),
            mock.patch(
                "client.main.create_app",
                side_effect=lambda *args, **kwargs: events.append("create-app") or object(),
            ) as create_app,
            mock.patch(
                "uvicorn.run",
                side_effect=lambda *args, **kwargs: events.append("uvicorn"),
            ),
        ):
            self.assertEqual(run_agent(), 0)

        self.assertEqual(
            events,
            [
                "tunnel-start",
                "tunnel-ready",
                "runtime",
                "gateway",
                "create-app",
                "uvicorn",
                "tunnel-stop",
            ],
        )
        self.assertIs(create_app.call_args.kwargs["tunnel_manager"], tunnel)


if __name__ == "__main__":
    unittest.main()
