#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABEL=""
SSH_TARGET="iosider@10.64.82.151"
SSH_PORT="34335"
REMOTE_REPO="/scratch/legalfedllm-test/work/LegalFedLLM"
REMOTE_VENV="/scratch/legalfedllm-test/.venv"
HOST_ENV=".env.host-r3"
CLIENT_ENV=".env.remote-client-r3"
HF_CACHE_VOLUME=""
REMOTE_RUNTIME_ROOT="/scratch/legalfedllm-test"
PLAN_ONLY=false

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_split_round_tmux.sh --label LABEL [options]

Run one split-machine LegalFedLLM round in a local tmux dashboard.
The A40 runs Host + Coordinator; the local machine runs client-1.

Required:
  --label LABEL             Fresh run label, e.g. r3b

Options:
  --ssh-target TARGET       SSH destination (default: iosider@10.64.82.151)
  --ssh-port PORT           SSH port (default: 34335)
  --remote-repo PATH        A40 LegalFedLLM checkout
  --remote-venv PATH        A40 Python virtual environment
  --host-env PATH           Host env path relative to remote repo
  --client-env PATH         Client env path relative to local repo
  --remote-runtime-root P   A40 runtime root (default: /scratch/legalfedllm-test)
  --hf-cache-volume NAME    Reuse an existing Docker HF cache volume
  --plan                    Print the derived run plan without making changes
  -h, --help                Show this help

SSH password handling:
  Do not put the SSH password in this script or an env file. The script opens
  one OpenSSH ControlMaster connection and the terminal asks for the password
  once. Every monitoring/orchestration SSH channel reuses that authenticated
  connection.

The script fails closed. It never repairs an ambiguous submission, commits a
pending Client package, resubmits a package, or calls /sync outside the normal
scripts/run_remote_round.py path.
USAGE
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --label)
      (($# >= 2)) || fail "--label requires a value"
      LABEL="$2"
      shift 2
      ;;
    --ssh-target)
      (($# >= 2)) || fail "--ssh-target requires a value"
      SSH_TARGET="$2"
      shift 2
      ;;
    --ssh-port)
      (($# >= 2)) || fail "--ssh-port requires a value"
      SSH_PORT="$2"
      shift 2
      ;;
    --remote-repo)
      (($# >= 2)) || fail "--remote-repo requires a value"
      REMOTE_REPO="$2"
      shift 2
      ;;
    --remote-venv)
      (($# >= 2)) || fail "--remote-venv requires a value"
      REMOTE_VENV="$2"
      shift 2
      ;;
    --host-env)
      (($# >= 2)) || fail "--host-env requires a value"
      HOST_ENV="$2"
      shift 2
      ;;
    --client-env)
      (($# >= 2)) || fail "--client-env requires a value"
      CLIENT_ENV="$2"
      shift 2
      ;;
    --remote-runtime-root)
      (($# >= 2)) || fail "--remote-runtime-root requires a value"
      REMOTE_RUNTIME_ROOT="$2"
      shift 2
      ;;
    --hf-cache-volume)
      (($# >= 2)) || fail "--hf-cache-volume requires a value"
      HF_CACHE_VOLUME="$2"
      shift 2
      ;;
    --plan)
      PLAN_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "$LABEL" ]] || fail "--label is required"
[[ "$LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || \
  fail "--label must use lowercase letters, digits and hyphens only"
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || fail "--ssh-port must be an integer"
if [[ -n "$HF_CACHE_VOLUME" ]]; then
  [[ "$HF_CACHE_VOLUME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
    fail "--hf-cache-volume contains unsupported characters"
fi

SESSION="legalfedllm-$LABEL"
COMPOSE_PROJECT="legalfedllm-split-$LABEL"
RUN_DIR="$ROOT/artifacts/split-round-$LABEL"
RUNNER_LOG="$RUN_DIR/runner.log"
RUNNER_EXIT="$RUN_DIR/runner.exit"
CREATE_LOG="$RUN_DIR/create-round.log"
ROUND_ID_FILE="$RUN_DIR/round-id.txt"
FINAL_COORDINATOR="$RUN_DIR/coordinator-final-state.json"
FINAL_CLIENT="$RUN_DIR/client-final-health.json"
RESULT_FILE="$RUN_DIR/result.txt"
FINALIZE_WRAPPER="$RUN_DIR/finalize.sh"
EVIDENCE_OBSERVATION_TIMEOUT_SECONDS=7200
EVIDENCE_OBSERVATION_POLL_SECONDS=10
COMPOSE_OVERRIDE="$RUN_DIR/compose-cache.override.yaml"
CLEANUP_SCRIPT="$RUN_DIR/cleanup.sh"
META_FILE="$RUN_DIR/run-info.txt"
A40_STACK_WRAPPER="$RUN_DIR/a40-stack.sh"
A40_CREATE_WRAPPER="$RUN_DIR/a40-create-round.sh"
REMOTE_HOST_DATA="$REMOTE_RUNTIME_ROOT/artifacts/split-round-host-$LABEL"
REMOTE_COORDINATOR_DATA="$REMOTE_RUNTIME_ROOT/artifacts/split-round-coordinator-$LABEL"
REMOTE_STACK_LOG="$REMOTE_RUNTIME_ROOT/logs/split-host-stack-$LABEL.log"
REMOTE_CREATE_LOG="$REMOTE_RUNTIME_ROOT/logs/split-round-create-$LABEL.log"
CONTROL_SOCKET="/tmp/legalfedllm-${UID}-${LABEL}.sock"
CLIENT_ENV_ABS="$ROOT/$CLIENT_ENV"

print_plan() {
  cat <<PLAN
label=$LABEL
tmux_session=$SESSION
compose_project=$COMPOSE_PROJECT
local_repo=$ROOT
client_env=$CLIENT_ENV_ABS
local_run_dir=$RUN_DIR
ssh_target=$SSH_TARGET
ssh_port=$SSH_PORT
ssh_control_socket=$CONTROL_SOCKET
remote_repo=$REMOTE_REPO
remote_venv=$REMOTE_VENV
host_env=$HOST_ENV
remote_host_data=$REMOTE_HOST_DATA
remote_coordinator_data=$REMOTE_COORDINATOR_DATA
remote_stack_log=$REMOTE_STACK_LOG
hf_cache_volume=${HF_CACHE_VOLUME:-<fresh project-scoped cache>}
evidence_observation_timeout_seconds=$EVIDENCE_OBSERVATION_TIMEOUT_SECONDS
PLAN
}

if [[ "$PLAN_ONLY" == true ]]; then
  print_plan
  exit 0
fi

for command in bash tmux ssh docker curl python nvidia-smi; do
  command -v "$command" >/dev/null 2>&1 || fail "required local command not found: $command"
done

docker compose version >/dev/null 2>&1 || fail "docker compose is not available"
[[ -f "$CLIENT_ENV_ABS" ]] || fail "Client env does not exist: $CLIENT_ENV_ABS"
[[ -x "$ROOT/.venv/bin/python" ]] || fail "local virtualenv Python not found: $ROOT/.venv/bin/python"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
! tmux has-session -t "$SESSION" 2>/dev/null || fail "tmux session already exists: $SESSION"

if docker volume ls --format '{{.Name}}' | grep -q "^${COMPOSE_PROJECT}_"; then
  fail "Docker volumes already exist for $COMPOSE_PROJECT; use a fresh label"
fi
if [[ -n "$HF_CACHE_VOLUME" ]] && ! docker volume inspect "$HF_CACHE_VOLUME" >/dev/null 2>&1; then
  fail "requested Hugging Face cache volume does not exist: $HF_CACHE_VOLUME"
fi

python - <<'PY' || fail "local port 8000 or 8001 is already in use"
import socket
for port in (8000, 8001):
    s = socket.socket()
    s.settimeout(0.25)
    try:
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()
    if in_use:
        raise SystemExit(1)
PY

if [[ -e "$CONTROL_SOCKET" ]]; then
  if ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" -O check "$SSH_TARGET" >/dev/null 2>&1; then
    fail "an SSH ControlMaster already exists for label $LABEL"
  fi
  rm -f "$CONTROL_SOCKET"
fi

cleanup_master_on_error=true
session_created=false
client_started=false
cleanup_on_exit() {
  rc=$?
  trap - EXIT
  if [[ $rc -ne 0 ]]; then
    if [[ "$session_created" == true ]]; then
      tmux kill-session -t "$SESSION" 2>/dev/null || true
    fi
    if [[ "$client_started" == true ]]; then
      files=(-f "$ROOT/compose.clients.yaml")
      if [[ -n "$HF_CACHE_VOLUME" && -f "$COMPOSE_OVERRIDE" ]]; then
        files+=(-f "$COMPOSE_OVERRIDE")
      fi
      docker compose -p "$COMPOSE_PROJECT" --env-file "$CLIENT_ENV_ABS" "${files[@]}" --profile qwen down >/dev/null 2>&1 || true
    fi
    if [[ "$cleanup_master_on_error" == true && -S "$CONTROL_SOCKET" ]]; then
      ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" -O exit "$SSH_TARGET" >/dev/null 2>&1 || true
    fi
  fi
  exit "$rc"
}
trap cleanup_on_exit EXIT

echo "Opening the A40 SSH ControlMaster. Enter the SSH password once when prompted."
ssh \
  -M \
  -S "$CONTROL_SOCKET" \
  -p "$SSH_PORT" \
  -o ControlPersist=no \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  -fN \
  "$SSH_TARGET"

ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" -O check "$SSH_TARGET" >/dev/null

remote_preflight=$(cat <<'REMOTE'
set -euo pipefail
repo="$1"
venv="$2"
host_env="$3"
host_data="$4"
coord_data="$5"
stack_log="$6"

cd "$repo"
test -f "$host_env"
test -x "$venv/bin/python"

"$venv/bin/python" - <<'PY'
import socket
for port in (8000, 8002):
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        used = sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()
    if used:
        raise SystemExit(f"remote port {port} is already in use")
PY

for path in "$host_data" "$coord_data"; do
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
    echo "remote runtime directory is not fresh: $path" >&2
    exit 1
  fi
done
if [[ -e "$stack_log" ]]; then
  echo "remote stack log already exists: $stack_log" >&2
  exit 1
fi
REMOTE
)

ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" \
  bash -s -- \
  "$REMOTE_REPO" "$REMOTE_VENV" "$HOST_ENV" \
  "$REMOTE_HOST_DATA" "$REMOTE_COORDINATOR_DATA" "$REMOTE_STACK_LOG" \
  <<<"$remote_preflight"

local_registration_hash="$($ROOT/.venv/bin/python - "$CLIENT_ENV_ABS" <<'PY'
import hashlib
import sys
from pathlib import Path
value = ""
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line.startswith("REGISTRATION_TOKEN="):
        value = line.split("=", 1)[1].strip()
        break
if not value:
    raise SystemExit("REGISTRATION_TOKEN is missing from Client env")
print(hashlib.sha256(value.encode()).hexdigest())
PY
)"

remote_registration_hash="$(ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" \
  "$REMOTE_VENV/bin/python" - "$REMOTE_REPO/$HOST_ENV" <<'PY'
import hashlib
import sys
from pathlib import Path
value = ""
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line.startswith("REGISTRATION_TOKEN="):
        value = line.split("=", 1)[1].strip()
        break
if not value:
    raise SystemExit("REGISTRATION_TOKEN is missing from Host env")
print(hashlib.sha256(value.encode()).hexdigest())
PY
)"

[[ "$local_registration_hash" == "$remote_registration_hash" ]] || \
  fail "Host and Client REGISTRATION_TOKEN values do not match"

mkdir -p "$RUN_DIR"
ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" \
  "mkdir -p '$REMOTE_HOST_DATA' '$REMOTE_COORDINATOR_DATA' '$REMOTE_RUNTIME_ROOT/logs'"

cat > "$META_FILE" <<META
label=$LABEL
tmux_session=$SESSION
compose_project=$COMPOSE_PROJECT
client_env=$CLIENT_ENV_ABS
ssh_target=$SSH_TARGET
ssh_port=$SSH_PORT
ssh_control_socket=$CONTROL_SOCKET
remote_repo=$REMOTE_REPO
remote_venv=$REMOTE_VENV
host_env=$HOST_ENV
remote_host_data=$REMOTE_HOST_DATA
remote_coordinator_data=$REMOTE_COORDINATOR_DATA
remote_stack_log=$REMOTE_STACK_LOG
hf_cache_volume=$HF_CACHE_VOLUME
META

if [[ -n "$HF_CACHE_VOLUME" ]]; then
  cat > "$COMPOSE_OVERRIDE" <<YAML
volumes:
  huggingface_cache:
    external: true
    name: $HF_CACHE_VOLUME
YAML
fi

COMPOSE_FILES=(-f "$ROOT/compose.clients.yaml")
if [[ -n "$HF_CACHE_VOLUME" ]]; then
  COMPOSE_FILES+=(-f "$COMPOSE_OVERRIDE")
fi

q() {
  printf '%q' "$1"
}

SSH_BASE="ssh -S $(q "$CONTROL_SOCKET") -p $(q "$SSH_PORT") $(q "$SSH_TARGET")"

cat > "$A40_STACK_WRAPPER" <<EOF_A40_STACK
#!/usr/bin/env bash
set -euo pipefail
ssh -S $(q "$CONTROL_SOCKET") -p $(q "$SSH_PORT") $(q "$SSH_TARGET") bash -s -- \
  $(q "$REMOTE_REPO") \
  $(q "$REMOTE_VENV") \
  $(q "$HOST_ENV") \
  $(q "$REMOTE_HOST_DATA") \
  $(q "$REMOTE_COORDINATOR_DATA") \
  $(q "$REMOTE_STACK_LOG") <<'REMOTE_A40_STACK'
EOF_A40_STACK
cat >> "$A40_STACK_WRAPPER" <<'REMOTE_A40_STACK_BODY'
set -euo pipefail
repo="$1"
venv="$2"
host_env="$3"
host_data="$4"
coord_data="$5"
stack_log="$6"

cd "$repo"
source "$venv/bin/activate"
export HOST_DATA_DIR="$host_data"
export COORDINATOR_DATA_DIR="$coord_data"
PYTHONUNBUFFERED=1 python scripts/run_host_stack.py --env-file "$host_env" 2>&1 | tee "$stack_log"
REMOTE_A40_STACK
REMOTE_A40_STACK_BODY
chmod +x "$A40_STACK_WRAPPER"

cat > "$A40_CREATE_WRAPPER" <<EOF_A40_CREATE
#!/usr/bin/env bash
set -euo pipefail
ssh -S $(q "$CONTROL_SOCKET") -p $(q "$SSH_PORT") $(q "$SSH_TARGET") bash -s -- \
  $(q "$REMOTE_REPO") \
  $(q "$REMOTE_VENV") \
  $(q "$HOST_ENV") \
  $(q "$REMOTE_CREATE_LOG") <<'REMOTE_A40_CREATE'
EOF_A40_CREATE
cat >> "$A40_CREATE_WRAPPER" <<'REMOTE_A40_CREATE_BODY'
set -euo pipefail
repo="$1"
venv="$2"
host_env="$3"
create_log="$4"

cd "$repo"
source "$venv/bin/activate"
python scripts/create_remote_round.py --env-file "$host_env" 2>&1 | tee "$create_log"
REMOTE_A40_CREATE
REMOTE_A40_CREATE_BODY
chmod +x "$A40_CREATE_WRAPPER"

tmux new-session -d -s "$SESSION" -n a40 "$A40_STACK_WRAPPER"
session_created=true

wait_remote_health() {
  local port="$1"
  local label="$2"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "A40 stack tmux session exited while waiting for $label health." >&2
      ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" \
        "if [ -f '$REMOTE_STACK_LOG' ]; then tail -n 120 '$REMOTE_STACK_LOG'; fi" \
        >&2 2>/dev/null || true
      return 1
    fi
    if ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET" \
      "curl --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:${port}/health >/dev/null" \
      >/dev/null 2>&1; then
      echo "$label is healthy"
      return 0
    fi
    sleep 5
  done
  return 1
}

echo "Waiting for A40 Host..."
wait_remote_health 8002 Host || fail "A40 Host did not become healthy"
echo "Waiting for A40 Coordinator..."
wait_remote_health 8000 Coordinator || fail "A40 Coordinator did not become healthy"

curl --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:8000/health >/dev/null || \
  fail "local SSH-forwarded Coordinator health check failed"

echo "Starting fresh local client-1 project: $COMPOSE_PROJECT"
docker compose \
  -p "$COMPOSE_PROJECT" \
  --env-file "$CLIENT_ENV_ABS" \
  "${COMPOSE_FILES[@]}" \
  --profile qwen \
  up --build -d client-1
client_started=true

client_deadline=$((SECONDS + 600))
until curl --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; do
  (( SECONDS < client_deadline )) || fail "client-1 did not become healthy"
  sleep 5
done
echo "client-1 is healthy"

RUNNER_CMD="set -o pipefail; $(q "$ROOT/.venv/bin/python") $(q "$ROOT/scripts/run_remote_round.py") --env-file $(q "$CLIENT_ENV_ABS") 2>&1 | tee $(q "$RUNNER_LOG"); rc=\${PIPESTATUS[0]}; echo \$rc > $(q "$RUNNER_EXIT"); echo; echo \"run_remote_round.py exited with rc=\$rc\"; echo \"This pane is intentionally left open for inspection.\"; while true; do sleep 3600; done"

tmux new-window -d -t "$SESSION" -n round "bash -lc $(q "$RUNNER_CMD")"

CREATE_CMD="while ! grep -Fq 'Clients are registered. Create the round from the Host/Coordinator side now.' $(q "$RUNNER_LOG") 2>/dev/null; do if [[ -f $(q "$RUNNER_EXIT") ]]; then echo 'Runner exited before registration barrier'; while true; do sleep 3600; done; fi; sleep 2; done; echo 'Registration barrier reached; creating the round on A40.'; set -o pipefail; $(q "$A40_CREATE_WRAPPER") | tee $(q "$CREATE_LOG"); rc=\${PIPESTATUS[0]}; if [[ \$rc -ne 0 ]]; then echo \"create_remote_round.py failed with rc=\$rc\"; while true; do sleep 3600; done; fi; round_id=\$(sed -n 's/.*\"round_id\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p' $(q "$CREATE_LOG") | tail -n 1); if [[ -z \"\$round_id\" ]]; then echo 'Could not parse round_id'; while true; do sleep 3600; done; fi; printf '%s\\n' \"\$round_id\" > $(q "$ROUND_ID_FILE"); echo \"Created round: \$round_id\"; while true; do sleep 3600; done"

tmux split-window -d -v -t "$SESSION:round" "bash -lc $(q "$CREATE_CMD")"
tmux select-layout -t "$SESSION:round" even-vertical

COMPOSE_FILE_ARGS="-f $(q "$ROOT/compose.clients.yaml")"
if [[ -n "$HF_CACHE_VOLUME" ]]; then
  COMPOSE_FILE_ARGS+=" -f $(q "$COMPOSE_OVERRIDE")"
fi
CLIENT_LOG_CMD="docker compose -p $(q "$COMPOSE_PROJECT") --env-file $(q "$CLIENT_ENV_ABS") $COMPOSE_FILE_ARGS --profile qwen logs -f --tail=100 client-1"
tmux new-window -d -t "$SESSION" -n local "$CLIENT_LOG_CMD"
tmux split-window -d -h -t "$SESSION:local" "nvidia-smi -l 5"
tmux select-layout -t "$SESSION:local" even-horizontal

A40_GPU_CMD="$SSH_BASE nvidia-smi -l 5"
tmux split-window -d -h -t "$SESSION:a40" "$A40_GPU_CMD"
tmux select-layout -t "$SESSION:a40" even-horizontal

STATE_CMD="while [[ ! -s $(q "$ROUND_ID_FILE") ]]; do echo 'Waiting for round ID...'; sleep 2; done; round_id=\$(cat $(q "$ROUND_ID_FILE")); while true; do clear; date; echo \"round_id=\$round_id\"; $SSH_BASE python -m json.tool $(q "$REMOTE_COORDINATOR_DATA")/rounds/\$round_id/state.json 2>&1 || true; sleep 10; done"
tmux new-window -d -t "$SESSION" -n state "bash -lc $(q "$STATE_CMD")"

cat > "$FINALIZE_WRAPPER" <<EOF_FINALIZE
#!/usr/bin/env bash
set -euo pipefail
RUNNER_EXIT=$(q "$RUNNER_EXIT")
ROUND_ID_FILE=$(q "$ROUND_ID_FILE")
FINAL_COORDINATOR=$(q "$FINAL_COORDINATOR")
FINAL_CLIENT=$(q "$FINAL_CLIENT")
RESULT_FILE=$(q "$RESULT_FILE")
LOCAL_PYTHON=$(q "$ROOT/.venv/bin/python")
REMOTE_COORDINATOR_DATA=$(q "$REMOTE_COORDINATOR_DATA")
CONTROL_SOCKET=$(q "$CONTROL_SOCKET")
SSH_PORT=$(q "$SSH_PORT")
SSH_TARGET=$(q "$SSH_TARGET")
EVIDENCE_OBSERVATION_TIMEOUT_SECONDS=$(q "$EVIDENCE_OBSERVATION_TIMEOUT_SECONDS")
EVIDENCE_OBSERVATION_POLL_SECONDS=$(q "$EVIDENCE_OBSERVATION_POLL_SECONDS")
EOF_FINALIZE
cat >> "$FINALIZE_WRAPPER" <<'EOF_FINALIZE_BODY'
SSH_BASE=(ssh -S "$CONTROL_SOCKET" -p "$SSH_PORT" "$SSH_TARGET")

while [[ ! -f "$RUNNER_EXIT" ]]; do
  echo 'Waiting for runner completion...'
  sleep 5
done

rc=$(cat "$RUNNER_EXIT")
round_id=''
[[ -s "$ROUND_ID_FILE" ]] && round_id=$(cat "$ROUND_ID_FILE")
evidence_timed_out=false

echo "runner_rc=$rc"
echo "round_id=$round_id"

if [[ -n "$round_id" ]]; then
  evidence_deadline=$((SECONDS + EVIDENCE_OBSERVATION_TIMEOUT_SECONDS))
  while true; do
    tmp_state="${FINAL_COORDINATOR}.tmp"
    if "${SSH_BASE[@]}" cat \
      "$REMOTE_COORDINATOR_DATA/rounds/$round_id/state.json" \
      > "$tmp_state" 2>/dev/null; then
      mv "$tmp_state" "$FINAL_COORDINATOR"
      coordinator_state=$("$LOCAL_PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))' \
        "$FINAL_COORDINATOR" 2>/dev/null || true)
      echo "coordinator_state=${coordinator_state:-unavailable}"
      case "$coordinator_state" in
        COMPLETED|SKIPPED|ABORTED)
          break
          ;;
      esac
    else
      rm -f "$tmp_state"
    fi

    if (( SECONDS >= evidence_deadline )); then
      evidence_timed_out=true
      echo "Coordinator evidence observation timed out after ${EVIDENCE_OBSERVATION_TIMEOUT_SECONDS}s."
      break
    fi
    echo 'Waiting for Coordinator terminal evidence...'
    sleep "$EVIDENCE_OBSERVATION_POLL_SECONDS"
  done
fi

curl --connect-timeout 2 --max-time 5 -fsS \
  http://127.0.0.1:8001/health > "$FINAL_CLIENT" 2>/dev/null || true

"$LOCAL_PYTHON" - "$RESULT_FILE" "$FINAL_COORDINATOR" "$FINAL_CLIENT" \
  "$round_id" "$rc" "$evidence_timed_out" <<'PY_EVAL'
import json
import sys
from pathlib import Path

result_path, coord_path, client_path, round_id, runner_rc, timed_out = sys.argv[1:]
reasons = []
if runner_rc != '0':
    reasons.append(f'runner exited with rc={runner_rc}')
try:
    coord = json.loads(Path(coord_path).read_text())
except Exception as exc:
    coord = {}
    reasons.append(f'final Coordinator state unavailable: {exc}')
try:
    client = json.loads(Path(client_path).read_text())
except Exception as exc:
    client = {}
    reasons.append(f'final Client health unavailable: {exc}')
if timed_out == 'true':
    reasons.append('Coordinator evidence observation timed out before a terminal state')
if coord.get('state') != 'COMPLETED':
    reasons.append(f"Coordinator state is {coord.get('state')!r}, not 'COMPLETED'")
if client.get('last_completed_round') != round_id:
    reasons.append(
        f"Client last_completed_round is {client.get('last_completed_round')!r}, expected {round_id!r}"
    )
status = 'PASS' if not reasons else 'FAIL'
text = status + '\n' + ''.join(f'- {item}\n' for item in reasons)
Path(result_path).write_text(text, encoding='utf-8')
print(text, end='')
PY_EVAL

echo
echo 'Final evidence:'
echo "$FINAL_COORDINATOR"
echo "$FINAL_CLIENT"
echo "$RESULT_FILE"
echo
echo 'No automatic recovery was attempted.'
while true; do sleep 3600; done
EOF_FINALIZE_BODY
chmod +x "$FINALIZE_WRAPPER"

tmux split-window -d -v -t "$SESSION:state" "$FINALIZE_WRAPPER"
tmux select-layout -t "$SESSION:state" even-vertical

SSH_CHECK="ssh -S $(q "$CONTROL_SOCKET") -p $(q "$SSH_PORT") -O check $(q "$SSH_TARGET")"
TUNNEL_CMD="while true; do clear; date; $SSH_CHECK 2>&1 || true; python - <<'PY'
import socket
sock = socket.socket()
sock.settimeout(1)
try:
    ok = sock.connect_ex(('127.0.0.1', 8000)) == 0
finally:
    sock.close()
print('local 127.0.0.1:8000 TCP:', 'reachable' if ok else 'not reachable')
PY
 sleep 5; done"
tmux new-window -d -t "$SESSION" -n tunnel "bash -lc $(q "$TUNNEL_CMD")"

cat > "$CLEANUP_SCRIPT" <<EOF_CLEANUP
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(q "$ROOT")
SESSION=$(q "$SESSION")
COMPOSE_PROJECT=$(q "$COMPOSE_PROJECT")
CLIENT_ENV=$(q "$CLIENT_ENV_ABS")
CONTROL_SOCKET=$(q "$CONTROL_SOCKET")
SSH_PORT=$(q "$SSH_PORT")
SSH_TARGET=$(q "$SSH_TARGET")
COMPOSE_OVERRIDE=$(q "$COMPOSE_OVERRIDE")
HF_CACHE_VOLUME=$(q "$HF_CACHE_VOLUME")

tmux kill-session -t "\$SESSION" 2>/dev/null || true
files=(-f "\$ROOT/compose.clients.yaml")
if [[ -n "\$HF_CACHE_VOLUME" ]]; then files+=(-f "\$COMPOSE_OVERRIDE"); fi
docker compose -p "\$COMPOSE_PROJECT" --env-file "\$CLIENT_ENV" "\${files[@]}" --profile qwen down || true
if [[ -S "\$CONTROL_SOCKET" ]]; then
  ssh -S "\$CONTROL_SOCKET" -p "\$SSH_PORT" -O exit "\$SSH_TARGET" >/dev/null 2>&1 || true
fi
echo "Stopped tmux session, client container, and SSH ControlMaster."
echo "Runtime artifacts and Docker volumes were preserved."
EOF_CLEANUP
chmod +x "$CLEANUP_SCRIPT"

cat <<INFO

LegalFedLLM split round '$LABEL' is running.

The SSH password was not stored. The authenticated OpenSSH ControlMaster at:
  $CONTROL_SOCKET
is being reused by all A40 panes and the localhost:8000 tunnel.

Tmux session:
  $SESSION

Detach without stopping the round:
  Ctrl-b d

Reattach:
  tmux attach -t $SESSION

Useful windows:
  round   runner + automatic round creation
  local   client-1 logs + local GPU
  a40     Host/Coordinator stack + A40 GPU
  state   persisted Coordinator state + final PASS/FAIL evidence
  tunnel  SSH ControlMaster/tunnel status

Run artifacts:
  $RUN_DIR

After inspection, stop the dashboard/services without deleting evidence:
  $CLEANUP_SCRIPT

Important: the dashboard never performs manual acknowledgement recovery.
If the runner returns 503, the final pane will report FAIL and preserve state.
INFO

cleanup_master_on_error=false
trap - EXIT
exec tmux attach-session -t "$SESSION"
