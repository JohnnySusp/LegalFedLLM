from __future__ import annotations

import unittest
from unittest import mock

from client.tunnel import SshTunnelConfig, SshTunnelManager


class SshTunnelTests(unittest.TestCase):
    def test_command_uses_native_openssh_and_loopback_forward_without_password_storage(self) -> None:
        config = SshTunnelConfig(
            enabled=True,
            target="iosider@example-host",
            ssh_port=2222,
            local_port=8000,
            remote_host="127.0.0.1",
            remote_port=8000,
        )
        manager = SshTunnelManager(config)
        with mock.patch("client.tunnel.shutil.which", return_value="/usr/bin/ssh"):
            command = manager._command()
        self.assertEqual(command[0], "/usr/bin/ssh")
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("ServerAliveInterval=30", command)
        self.assertIn("127.0.0.1:8000:127.0.0.1:8000", command)
        self.assertEqual(command[-1], "iosider@example-host")
        self.assertNotIn("password", " ".join(command).lower())

    def test_disabled_tunnel_needs_no_ssh_target(self) -> None:
        manager = SshTunnelManager(SshTunnelConfig(enabled=False, target=""))
        manager.start()
        manager.wait_until_forward_reachable()
        status = manager.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["running"])
        self.assertFalse(status["forward_reachable"])

    def test_wait_for_forward_blocks_until_managed_ssh_is_running_and_reachable(self) -> None:
        manager = SshTunnelManager(
            SshTunnelConfig(enabled=True, target="iosider@example-host")
        )
        states = [
            {"running": True, "forward_reachable": False},
            {"running": True, "forward_reachable": True},
        ]
        with (
            mock.patch.object(manager, "status", side_effect=states) as status,
            mock.patch("client.tunnel.time.sleep") as sleep,
        ):
            manager.wait_until_forward_reachable(poll_seconds=0.01)
        self.assertEqual(status.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_wait_for_forward_rejects_nonpositive_poll_interval(self) -> None:
        manager = SshTunnelManager(
            SshTunnelConfig(enabled=True, target="iosider@example-host")
        )
        with self.assertRaisesRegex(Exception, "poll interval must be positive"):
            manager.wait_until_forward_reachable(poll_seconds=0)


if __name__ == "__main__":
    unittest.main()
