from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_split_round_tmux.sh"


class SplitRoundTmuxScriptTests(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_plan_derives_fresh_run_names_without_side_effects(self) -> None:
        result = self.run_script("--label", "r3b", "--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tmux_session=legalfedllm-r3b", result.stdout)
        self.assertIn("compose_project=legalfedllm-split-r3b", result.stdout)
        self.assertIn(
            "remote_host_data=/scratch/legalfedllm-test/artifacts/split-round-host-r3b",
            result.stdout,
        )
        self.assertIn(
            "remote_coordinator_data=/scratch/legalfedllm-test/artifacts/split-round-coordinator-r3b",
            result.stdout,
        )
        self.assertIn("ssh_target=iosider@10.64.82.151", result.stdout)
        self.assertIn("ssh_port=34335", result.stdout)

    def test_plan_can_reuse_existing_huggingface_cache_volume(self) -> None:
        result = self.run_script(
            "--label",
            "r3b",
            "--hf-cache-volume",
            "legalfedllm-split-r3_huggingface_cache",
            "--plan",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "hf_cache_volume=legalfedllm-split-r3_huggingface_cache",
            result.stdout,
        )

    def test_invalid_label_is_rejected_before_any_remote_action(self) -> None:
        result = self.run_script("--label", "R3 bad", "--plan")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--label must use lowercase letters", result.stderr)

    def test_help_requires_no_password_storage(self) -> None:
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Do not put the SSH password in this script or an env file", result.stdout)
        self.assertIn("password", result.stdout.lower())
        self.assertNotIn("--password", result.stdout)


    def test_plan_exposes_bounded_terminal_evidence_observation(self) -> None:
        result = self.run_script("--label", "r4", "--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evidence_observation_timeout_seconds=7200", result.stdout)

    def test_finalizer_observes_coordinator_until_terminal_state(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('FINALIZE_WRAPPER="$RUN_DIR/finalize.sh"', source)
        self.assertIn('COMPLETED|SKIPPED|ABORTED', source)
        self.assertIn('Waiting for Coordinator terminal evidence...', source)
        self.assertIn('Coordinator evidence observation timed out before a terminal state', source)
        self.assertIn("No automatic recovery was attempted.", source)

        runner_wait = source.index('while [[ ! -f "$RUNNER_EXIT" ]]')
        terminal_wait = source.index('COMPLETED|SKIPPED|ABORTED')
        client_snapshot = source.index('http://127.0.0.1:8001/health > "$FINAL_CLIENT"')
        self.assertLess(runner_wait, terminal_wait)
        self.assertLess(terminal_wait, client_snapshot)

    def test_remote_commands_use_argument_safe_wrappers(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('A40_STACK_WRAPPER="$RUN_DIR/a40-stack.sh"', source)
        self.assertIn('A40_CREATE_WRAPPER="$RUN_DIR/a40-create-round.sh"', source)
        self.assertIn('bash -s --', source)
        self.assertIn('cd "$repo"', source)
        self.assertNotIn('bash -lc $(q "cd $(q "$REMOTE_REPO")', source)

    def test_host_health_wait_fails_when_tmux_stack_exits(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if ! tmux has-session -t "$SESSION"', source)
        self.assertIn('A40 stack tmux session exited while waiting for $label health.', source)
        self.assertIn("tail -n 120 '$REMOTE_STACK_LOG'", source)


if __name__ == "__main__":
    unittest.main()
