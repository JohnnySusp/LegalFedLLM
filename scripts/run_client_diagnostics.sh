#!/usr/bin/env bash
set -euo pipefail

PROFILE_ID=${1:?usage: scripts/run_client_diagnostics.sh PROFILE_ID [DATA_ROOT]}
DATA_ROOT=${2:-"${LEGALFEDLLM_DATA_ROOT:-$PWD/LegalFedLLM-data}"}
SESSION=${LEGALFEDLLM_DIAGNOSTICS_SESSION:-"legalfedllm-${PROFILE_ID}"}
PYTHON=${PYTHON:-python}

command_for() {
  printf '%q ' "$PYTHON" -m desktop.app --monitor "$1" --profile-id "$PROFILE_ID" --data-root "$DATA_ROOT"
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the Linux diagnostics helper." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n state "$(command_for state)"
tmux new-window -t "$SESSION" -n gpu "$(command_for gpu)"
tmux select-window -t "$SESSION:state"
exec tmux attach-session -t "$SESSION"
