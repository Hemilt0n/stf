#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DEFAULT_HOST="hang@home-pc-ubuntu"
DEFAULT_REMOTE_REPO="/home/hang/repos/stf"

HOST="${DEFAULT_HOST}"
REMOTE_REPO="${DEFAULT_REMOTE_REPO}"
REMOTE_CONFIG_DIR="${DEFAULT_REMOTE_REPO}/configs/remote"
GPU="0"
RETRIES="0"
DELAY_SEC="5"
STARTUP_SECONDS="15"
ALLOW_VERSION_MISMATCH="0"
REMOTE_COMMAND=""
SESSION=""
CONFIG=""
PURPOSE=""
COMMAND="${1:-}"

runtime_root() {
  printf '%s\n' "${REPO_ROOT}/log/remote_runs"
}

records_dir() {
  printf '%s\n' "$(runtime_root)/records"
}

ensure_runtime_dirs() {
  mkdir -p "$(records_dir)"
}

now_ts() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

now_compact() {
  date '+%Y%m%d-%H%M%S'
}

sanitize_token() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/^$/run/'
}

append_ledger() {
  local title="$1"
  shift
  local ledger
  ledger="$(runtime_root)/experiments.md"
  {
    printf '## %s | %s\n\n' "$(now_ts)" "$title"
    for line in "$@"; do
      printf -- '- %s\n' "$line"
    done
    printf '\n'
  } >>"${ledger}"
}

remote_exec() {
  tailscale ssh "${HOST}" "$1"
}

remote_capture() {
  tailscale ssh "${HOST}" "$1" 2>/dev/null || true
}

local_git_branch() {
  git -C "${REPO_ROOT}" branch --show-current
}

local_git_head() {
  git -C "${REPO_ROOT}" rev-parse HEAD
}

local_git_status() {
  git -C "${REPO_ROOT}" status --short --branch
}

remote_git_branch() {
  remote_exec "git -C $(printf '%q' "${REMOTE_REPO}") branch --show-current"
}

remote_git_head() {
  remote_exec "git -C $(printf '%q' "${REMOTE_REPO}") rev-parse HEAD"
}

remote_git_status() {
  remote_exec "git -C $(printf '%q' "${REMOTE_REPO}") status --short --branch"
}

remote_data_link() {
  remote_capture "readlink -f $(printf '%q' "${REMOTE_REPO}/data")"
}

remote_tmux_path() {
  remote_capture "command -v tmux"
}

parse_metadata_json() {
  python3 - "$1" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
metadata = {"task": "", "name": "", "message": ""}
try:
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXPERIMENT" and isinstance(node.value, ast.Call):
                    for kw in node.value.keywords:
                        if kw.arg in metadata and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            metadata[kw.arg] = kw.value.value
except SyntaxError:
    pass

for key in metadata:
    if metadata[key]:
        continue
    match = re.search(rf"{key}\s*=\s*['\"]([^'\"]+)['\"]", text)
    if match:
        metadata[key] = match.group(1)

if not metadata["task"]:
    metadata["task"] = "unknown-task"
if not metadata["name"]:
    metadata["name"] = path.stem

print(json.dumps(metadata, ensure_ascii=True))
PY
}

json_field() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$1"
}

record_path_for_session() {
  printf '%s\n' "$(records_dir)/$1.json"
}

write_record() {
  local record_path="$1"
  local pane_file="$2"
  local state_file="$3"
  local latest_run_dir="${4:-}"
  local final_val_line="${5:-}"
  local gpu_summary_line="${6:-}"
  python3 - "${record_path}" "${pane_file}" "${state_file}" "${latest_run_dir}" "${final_val_line}" "${gpu_summary_line}" <<'PY'
import json
import os
import sys
from pathlib import Path

record_path = Path(sys.argv[1])
pane_file = Path(sys.argv[2])
state_file = Path(sys.argv[3])
latest_run_dir = sys.argv[4]
final_val_line = sys.argv[5]
gpu_summary_line = sys.argv[6]

payload = {
    "created_at": os.environ["CREATED_AT"],
    "updated_at": os.environ["UPDATED_AT"],
    "status": os.environ["STATUS"],
    "host": os.environ["HOST"],
    "remote_repo": os.environ["REMOTE_REPO"],
    "remote_config_dir": os.environ["REMOTE_CONFIG_DIR"],
    "local_branch": os.environ["LOCAL_BRANCH"],
    "local_head": os.environ["LOCAL_HEAD"],
    "local_status": os.environ["LOCAL_STATUS"],
    "remote_branch": os.environ["REMOTE_BRANCH"],
    "remote_head": os.environ["REMOTE_HEAD"],
    "remote_status": os.environ["REMOTE_STATUS"],
    "remote_data_link": os.environ["REMOTE_DATA_LINK"],
    "tmux_path": os.environ["TMUX_PATH"],
    "session": os.environ["SESSION"],
    "purpose": os.environ["PURPOSE"],
    "config_local": os.environ["CONFIG_LOCAL"],
    "config_remote": os.environ["CONFIG_REMOTE"],
    "task": os.environ["TASK"],
    "name": os.environ["NAME"],
    "message": os.environ["MESSAGE"],
    "gpu": os.environ["GPU"],
    "retries": os.environ["RETRIES"],
    "delay_sec": os.environ["DELAY_SEC"],
    "startup_seconds": os.environ["STARTUP_SECONDS"],
    "state_file": os.environ["STATE_TRACK_FILE"],
    "remote_command": os.environ["REMOTE_COMMAND_VALUE"],
    "pane_tail": pane_file.read_text(encoding="utf-8") if pane_file.exists() else "",
    "state_tail": state_file.read_text(encoding="utf-8") if state_file.exists() else "",
    "latest_run_dir": latest_run_dir,
    "final_val_line": final_val_line,
    "gpu_summary_line": gpu_summary_line,
}
record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps({"record_path": str(record_path), **payload}, indent=2, ensure_ascii=True))
PY
}

  load_record_env() {
  python3 - "$1" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

keys = [
    "host", "remote_repo", "remote_config_dir", "session", "purpose",
    "config_local", "config_remote", "task", "name", "message",
    "gpu", "retries", "delay_sec", "startup_seconds", "state_file",
    "remote_command", "created_at", "updated_at", "status",
    "local_branch", "local_head", "local_status",
    "remote_branch", "remote_head", "remote_status",
    "remote_data_link", "tmux_path",
]
for key in keys:
    print(f"{key.upper()}={shlex.quote(str(data.get(key, '')))}")
PY
}

session_exists() {
  remote_exec "tmux has-session -t $(printf '%q' "${SESSION}")" >/dev/null 2>&1
}

capture_pane_to_file() {
  local out_file="$1"
  remote_capture "tmux capture-pane -pt $(printf '%q' "${SESSION}") -S -80" >"${out_file}"
}

capture_state_to_file() {
  local out_file="$1"
  remote_capture "tail -n 20 $(printf '%q' "${STATE_TRACK_FILE}") 2>/dev/null" >"${out_file}"
}

find_latest_run_dir() {
  remote_capture "find -L $(printf '%q' "${REMOTE_REPO}/runs/${TASK}") -maxdepth 1 -mindepth 1 -type d -name $(printf '%q' "${NAME}_*") 2>/dev/null | sort | tail -n 1"
}

extract_last_line() {
  local needle="$1"
  local file="$2"
  python3 - "$needle" "$file" <<'PY'
import sys
needle = sys.argv[1]
path = sys.argv[2]
last = ""
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        if needle in line:
            last = line.rstrip("\n")
print(last)
PY
}

usage() {
  cat <<'EOF'
Usage:
  remote_train.sh inspect [--host HOST] [--remote-repo PATH]
  remote_train.sh launch --config PATH --purpose TEXT [options]
  remote_train.sh status --session NAME

Options for launch:
  --host HOST
  --remote-repo PATH
  --remote-config-dir PATH
  --session NAME
  --gpu ID
  --retries N
  --delay-sec N
  --startup-seconds N
  --allow-version-mismatch
  --remote-command CMD
EOF
}

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --remote-repo)
      REMOTE_REPO="$2"
      shift 2
      ;;
    --remote-config-dir)
      REMOTE_CONFIG_DIR="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --purpose)
      PURPOSE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --delay-sec)
      DELAY_SEC="$2"
      shift 2
      ;;
    --startup-seconds)
      STARTUP_SECONDS="$2"
      shift 2
      ;;
    --allow-version-mismatch)
      ALLOW_VERSION_MISMATCH="1"
      shift
      ;;
    --remote-command)
      REMOTE_COMMAND="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ensure_runtime_dirs

case "${COMMAND}" in
  inspect)
    LOCAL_BRANCH="$(local_git_branch)"
    LOCAL_HEAD="$(local_git_head)"
    LOCAL_STATUS="$(local_git_status)"
    REMOTE_BRANCH="$(remote_git_branch)"
    REMOTE_HEAD="$(remote_git_head)"
    REMOTE_STATUS="$(remote_git_status)"
    REMOTE_DATA_LINK="$(remote_data_link)"
    TMUX_PATH="$(remote_tmux_path)"
    export REPO_ROOT HOST REMOTE_REPO LOCAL_BRANCH LOCAL_HEAD LOCAL_STATUS REMOTE_BRANCH REMOTE_HEAD REMOTE_STATUS REMOTE_DATA_LINK TMUX_PATH
    python3 - <<'PY'
import json
import os

print(json.dumps({
    "checked_at": os.popen("date '+%Y-%m-%d %H:%M:%S %z'").read().strip(),
    "repo_root": os.environ["REPO_ROOT"],
    "host": os.environ["HOST"],
    "remote_repo": os.environ["REMOTE_REPO"],
    "local": {
        "branch": os.environ["LOCAL_BRANCH"],
        "head": os.environ["LOCAL_HEAD"],
        "status": os.environ["LOCAL_STATUS"],
    },
    "remote": {
        "branch": os.environ["REMOTE_BRANCH"],
        "head": os.environ["REMOTE_HEAD"],
        "status": os.environ["REMOTE_STATUS"],
    },
    "remote_env": {
        "data_link": os.environ["REMOTE_DATA_LINK"],
        "tmux_path": os.environ["TMUX_PATH"],
    },
}, indent=2, ensure_ascii=True))
PY
    ;;

  launch)
    [[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 2; }
    [[ -n "${PURPOSE}" ]] || { echo "--purpose is required" >&2; exit 2; }
    [[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 2; }

    LOCAL_BRANCH="$(local_git_branch)"
    LOCAL_HEAD="$(local_git_head)"
    LOCAL_STATUS="$(local_git_status)"
    REMOTE_BRANCH="$(remote_git_branch)"
    REMOTE_HEAD="$(remote_git_head)"
    REMOTE_STATUS="$(remote_git_status)"
    REMOTE_DATA_LINK="$(remote_data_link)"
    TMUX_PATH="$(remote_tmux_path)"

    if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" && "${ALLOW_VERSION_MISMATCH}" != "1" ]]; then
      echo "Local and remote git heads differ." >&2
      echo "local=${LOCAL_HEAD}" >&2
      echo "remote=${REMOTE_HEAD}" >&2
      echo "Use --allow-version-mismatch only for smoke tests or after user confirmation." >&2
      exit 3
    fi

    CONFIG_LOCAL="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
    METADATA_JSON="$(parse_metadata_json "${CONFIG_LOCAL}")"
    TASK="$(json_field task <<<"${METADATA_JSON}")"
    NAME="$(json_field name <<<"${METADATA_JSON}")"
    MESSAGE="$(json_field message <<<"${METADATA_JSON}")"

    if [[ -z "${SESSION}" ]]; then
      SESSION="stf-$(sanitize_token "${TASK}")-$(sanitize_token "${NAME}")-$(now_compact)"
      SESSION="${SESSION:0:80}"
      SESSION="${SESSION%-}"
    fi

    CONFIG_REMOTE="${REMOTE_CONFIG_DIR%/}/$(basename "${CONFIG_LOCAL%.py}")__$(now_compact).py"
    STATE_TRACK_FILE="${REMOTE_REPO}/log/remote_runs/${SESSION}.state"

    if session_exists; then
      echo "tmux session already exists: ${SESSION}" >&2
      exit 4
    fi

    remote_exec "mkdir -p $(printf '%q' "$(dirname "${CONFIG_REMOTE}")") && cat > $(printf '%q' "${CONFIG_REMOTE}")" <"${CONFIG_LOCAL}"

    if [[ -n "${REMOTE_COMMAND}" ]]; then
      REMOTE_COMMAND_VALUE="${REMOTE_COMMAND}"
    else
      REMOTE_COMMAND_VALUE="cd $(printf '%q' "${REMOTE_REPO}") && mkdir -p log/remote_runs && ./tools/train_queue.sh --config $(printf '%q' "${CONFIG_REMOTE}") --gpu $(printf '%q' "${GPU}") --retries $(printf '%q' "${RETRIES}") --delay-sec $(printf '%q' "${DELAY_SEC}") --state-file $(printf '%q' "${STATE_TRACK_FILE}")"
    fi

    printf -v QUOTED_REMOTE_COMMAND '%q' "${REMOTE_COMMAND_VALUE}"
    remote_exec "tmux new-session -d -s $(printf '%q' "${SESSION}") bash -lc ${QUOTED_REMOTE_COMMAND}"

    if [[ "${STARTUP_SECONDS}" =~ ^[0-9]+$ ]] && [[ "${STARTUP_SECONDS}" -gt 0 ]]; then
      sleep "${STARTUP_SECONDS}"
    fi

    STATUS="exited-early"
    PANE_FILE="$(mktemp)"
    STATE_FILE_TMP="$(mktemp)"
    if session_exists; then
      STATUS="running"
      capture_pane_to_file "${PANE_FILE}"
    fi
    capture_state_to_file "${STATE_FILE_TMP}"

    CREATED_AT="$(now_ts)"
    UPDATED_AT="${CREATED_AT}"
    RECORD_PATH="$(record_path_for_session "${SESSION}")"
    export CREATED_AT UPDATED_AT STATUS HOST REMOTE_REPO REMOTE_CONFIG_DIR LOCAL_BRANCH LOCAL_HEAD LOCAL_STATUS REMOTE_BRANCH REMOTE_HEAD REMOTE_STATUS REMOTE_DATA_LINK TMUX_PATH SESSION PURPOSE CONFIG_LOCAL CONFIG_REMOTE TASK NAME MESSAGE GPU RETRIES DELAY_SEC STARTUP_SECONDS STATE_TRACK_FILE REMOTE_COMMAND_VALUE
    write_record "${RECORD_PATH}" "${PANE_FILE}" "${STATE_FILE_TMP}"
    append_ledger \
      "launch | session \`${SESSION}\`" \
      "purpose: ${PURPOSE}" \
      "local_head: ${LOCAL_HEAD}" \
      "remote_head: ${REMOTE_HEAD}" \
      "config_local: ${CONFIG_LOCAL}" \
      "config_remote: ${CONFIG_REMOTE}" \
      "task/name: ${TASK} / ${NAME}" \
      "record: ${RECORD_PATH}" \
      "startup_status: ${STATUS}"
    rm -f "${PANE_FILE}" "${STATE_FILE_TMP}"

    if [[ "${STATUS}" != "running" && -z "${REMOTE_COMMAND}" ]]; then
      exit 1
    fi
    ;;

  status)
    [[ -n "${SESSION}" ]] || { echo "--session is required" >&2; exit 2; }
    RECORD_PATH="$(record_path_for_session "${SESSION}")"
    [[ -f "${RECORD_PATH}" ]] || { echo "Record not found: ${RECORD_PATH}" >&2; exit 2; }
    eval "$(load_record_env "${RECORD_PATH}")"

    HOST="${HOST:-${DEFAULT_HOST}}"
    REMOTE_REPO="${REMOTE_REPO:-${DEFAULT_REMOTE_REPO}}"
    REMOTE_CONFIG_DIR="${REMOTE_CONFIG_DIR:-${DEFAULT_REMOTE_REPO}/configs/remote}"
    STATE_TRACK_FILE="${STATE_FILE:-}"
    REMOTE_COMMAND_VALUE="${REMOTE_COMMAND:-}"

    PANE_FILE="$(mktemp)"
    STATE_FILE_TMP="$(mktemp)"
    LOG_FILE_TMP="$(mktemp)"
    if session_exists; then
      STATUS="running"
      capture_pane_to_file "${PANE_FILE}"
    else
      STATUS="stopped"
    fi
    capture_state_to_file "${STATE_FILE_TMP}"
    LATEST_RUN_DIR="$(find_latest_run_dir)"
    if [[ -n "${LATEST_RUN_DIR}" ]]; then
      remote_capture "tail -n 180 $(printf '%q' "${LATEST_RUN_DIR}/logs/train.log") 2>/dev/null" >"${LOG_FILE_TMP}"
      FINAL_VAL_LINE="$(extract_last_line ' - stf.train - INFO - val epoch=' "${LOG_FILE_TMP}")"
      GPU_SUMMARY_LINE="$(extract_last_line 'gpu memory summary:' "${LOG_FILE_TMP}")"
    else
      FINAL_VAL_LINE=""
      GPU_SUMMARY_LINE=""
    fi

    if grep -q '^OK|' "${STATE_FILE_TMP}" 2>/dev/null || [[ -n "${GPU_SUMMARY_LINE}" ]]; then
      STATUS="completed"
    elif grep -Eq '^(FAIL|MISS)\|' "${STATE_FILE_TMP}" 2>/dev/null; then
      STATUS="failed"
    fi

    UPDATED_AT="$(now_ts)"
    CREATED_AT="${CREATED_AT:-${UPDATED_AT}}"
    STATE_TRACK_FILE="${STATE_FILE:-}"
    REMOTE_COMMAND_VALUE="${REMOTE_COMMAND:-}"
    export CREATED_AT UPDATED_AT STATUS HOST REMOTE_REPO REMOTE_CONFIG_DIR LOCAL_BRANCH LOCAL_HEAD LOCAL_STATUS REMOTE_BRANCH REMOTE_HEAD REMOTE_STATUS REMOTE_DATA_LINK TMUX_PATH SESSION PURPOSE CONFIG_LOCAL CONFIG_REMOTE TASK NAME MESSAGE GPU RETRIES DELAY_SEC STARTUP_SECONDS STATE_TRACK_FILE REMOTE_COMMAND_VALUE
    write_record "${RECORD_PATH}" "${PANE_FILE}" "${STATE_FILE_TMP}" "${LATEST_RUN_DIR}" "${FINAL_VAL_LINE}" "${GPU_SUMMARY_LINE}"
    append_ledger \
      "status | session \`${SESSION}\`" \
      "status: ${STATUS}" \
      "record: ${RECORD_PATH}" \
      "latest_run_dir: ${LATEST_RUN_DIR:-'(not found)'}" \
      "final_val: ${FINAL_VAL_LINE:-'(not found)'}" \
      "gpu_summary: ${GPU_SUMMARY_LINE:-'(not found)'}"
    rm -f "${PANE_FILE}" "${STATE_FILE_TMP}" "${LOG_FILE_TMP}"
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac
