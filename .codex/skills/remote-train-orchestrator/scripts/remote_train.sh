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
CONFIGS=()
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

stage_file_to_remote() {
  local src="$1"
  local dst="$2"
  remote_exec "mkdir -p $(printf '%q' "$(dirname "${dst}")") && cat > $(printf '%q' "${dst}")" <"${src}"
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

json_array_from_args() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1:], ensure_ascii=True))' "$@"
}

record_path_for_session() {
  printf '%s\n' "$(records_dir)/$1.json"
}

build_session_name() {
  local prefix="$1"
  local task="$2"
  local name="$3"
  local stamp="$4"
  local session
  session="${prefix}$(sanitize_token "${task}")-$(sanitize_token "${name}")-${stamp}"
  session="${session:0:80}"
  session="${session%-}"
  printf '%s\n' "${session}"
}

build_queue_session_name() {
  local first_name="$1"
  local last_name="$2"
  local stamp="$3"
  local session
  session="stf-queue-$(sanitize_token "${first_name}")-to-$(sanitize_token "${last_name}")-${stamp}"
  session="${session:0:80}"
  session="${session%-}"
  printf '%s\n' "${session}"
}

session_exists_name() {
  local name="$1"
  remote_exec "tmux has-session -t $(printf '%q' "${name}")" >/dev/null 2>&1
}

session_exists() {
  session_exists_name "${SESSION}"
}

capture_session_pane_to_file() {
  local name="$1"
  local out_file="$2"
  remote_capture "tmux capture-pane -pt $(printf '%q' "${name}") -S -80" >"${out_file}"
}

capture_pane_to_file() {
  local out_file="$1"
  capture_session_pane_to_file "${SESSION}" "${out_file}"
}

capture_state_to_file() {
  local out_file="$1"
  remote_capture "tail -n 20 $(printf '%q' "${STATE_TRACK_FILE}") 2>/dev/null" >"${out_file}"
}

find_latest_run_dir_for_task_name() {
  local task="$1"
  local name="$2"
  remote_capture "find -L $(printf '%q' "${REMOTE_REPO}/runs/${task}") -maxdepth 1 -mindepth 1 -type d -name $(printf '%q' "${name}_*") 2>/dev/null | sort | tail -n 1"
}

find_latest_run_dir() {
  find_latest_run_dir_for_task_name "${TASK}" "${NAME}"
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

load_queue_state_env() {
  python3 - "$1" <<'PY'
import os
import shlex
import sys

path = sys.argv[1]
current = {}
last_done = {}
queue_done = {}

if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            parts = raw.split("|")
            kind = parts[0]
            if kind in {"PENDING", "RUNNING", "DONE"}:
                event = {"kind": kind}
                base_keys = ["idx", "session", "config", "task", "name"]
                for key, value in zip(base_keys, parts[1:6]):
                    event[key] = value
                for item in parts[6:]:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        event[key] = value
                if kind == "RUNNING":
                    current = event
                elif kind == "DONE":
                    last_done = event
                    if current.get("session") == event.get("session"):
                        current = {}
            elif kind == "QUEUE_DONE":
                queue_done = {"kind": kind}
                for item in parts[1:]:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        queue_done[key] = value

def emit(key: str, value: str) -> None:
    print(f"{key}={shlex.quote(str(value))}")

emit("QUEUE_CURRENT_SESSION", current.get("session", ""))
emit("QUEUE_CURRENT_CONFIG", current.get("config", ""))
emit("QUEUE_CURRENT_TASK", current.get("task", ""))
emit("QUEUE_CURRENT_NAME", current.get("name", ""))
emit("QUEUE_CURRENT_STATE_FILE", current.get("state_file", ""))
emit("QUEUE_LAST_SESSION", last_done.get("session", ""))
emit("QUEUE_LAST_CONFIG", last_done.get("config", ""))
emit("QUEUE_LAST_TASK", last_done.get("task", ""))
emit("QUEUE_LAST_NAME", last_done.get("name", ""))
emit("QUEUE_LAST_RUN_DIR", last_done.get("run_dir", ""))
emit("QUEUE_LAST_EXIT", last_done.get("exit", ""))
emit("QUEUE_DONE_EXIT", queue_done.get("exit", ""))
PY
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

def load_json_env(key: str):
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw

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

optional_pairs = {
    "queue_mode": os.environ.get("QUEUE_MODE", "") == "1",
    "controller_session": os.environ.get("CONTROLLER_SESSION", ""),
    "active_session": os.environ.get("ACTIVE_SESSION", ""),
    "queue_state_file": os.environ.get("QUEUE_STATE_FILE", ""),
    "startup_passed": os.environ.get("STARTUP_PASSED", "") == "1",
    "sequential_confirmed": os.environ.get("SEQUENTIAL_CONFIRMED", "") == "1",
    "sequential_confirmation": os.environ.get("SEQUENTIAL_CONFIRMATION", ""),
    "allow_version_mismatch": os.environ.get("ALLOW_VERSION_MISMATCH", "") == "1",
}
for key, value in optional_pairs.items():
    if isinstance(value, bool):
        if value:
            payload[key] = value
    elif value:
        payload[key] = value

for env_key, payload_key in [
    ("CONFIG_LOCAL_LIST_JSON", "config_local_list"),
    ("CONFIG_REMOTE_LIST_JSON", "config_remote_list"),
    ("SESSION_LIST_JSON", "session_list"),
]:
    value = load_json_env(env_key)
    if value not in (None, "", []):
        payload[payload_key] = value

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

string_keys = [
    "host", "remote_repo", "remote_config_dir", "session", "purpose",
    "config_local", "config_remote", "task", "name", "message",
    "gpu", "retries", "delay_sec", "startup_seconds", "state_file",
    "remote_command", "created_at", "updated_at", "status",
    "local_branch", "local_head", "local_status",
    "remote_branch", "remote_head", "remote_status",
    "remote_data_link", "tmux_path", "controller_session",
    "active_session", "queue_state_file", "sequential_confirmation",
]
for key in string_keys:
    print(f"{key.upper()}={shlex.quote(str(data.get(key, '')))}")

bool_keys = [
    "queue_mode", "startup_passed", "sequential_confirmed", "allow_version_mismatch",
]
for key in bool_keys:
    value = "1" if data.get(key, False) else "0"
    print(f"{key.upper()}={shlex.quote(value)}")

list_keys = [
    "config_local_list", "config_remote_list", "session_list",
]
for key in list_keys:
    value = json.dumps(data.get(key, []), ensure_ascii=True)
    print(f"{key.upper()}_JSON={shlex.quote(value)}")
PY
}

usage() {
  cat <<'EOF'
Usage:
  remote_train.sh inspect [--host HOST] [--remote-repo PATH]
  remote_train.sh launch --config PATH [--config PATH ...] --purpose TEXT [options]
  remote_train.sh status --session NAME

Options for launch:
  --host HOST
  --remote-repo PATH
  --remote-config-dir PATH
  --session NAME
  --config PATH
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
      CONFIGS+=("$2")
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
    [[ "${#CONFIGS[@]}" -gt 0 ]] || { echo "At least one --config is required" >&2; exit 2; }
    [[ -n "${PURPOSE}" ]] || { echo "--purpose is required" >&2; exit 2; }
    for cfg in "${CONFIGS[@]}"; do
      [[ -f "${cfg}" ]] || { echo "Config not found: ${cfg}" >&2; exit 2; }
    done

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

    if [[ "${#CONFIGS[@]}" -gt 1 && -n "${REMOTE_COMMAND}" ]]; then
      echo "--remote-command is only supported for single-config launch" >&2
      exit 2
    fi

    QUEUE_STAMP="$(now_compact)"
    CONFIG_LOCAL_LIST=()
    CONFIG_REMOTE_LIST=()
    TASK_LIST=()
    NAME_LIST=()
    MESSAGE_LIST=()
    CHILD_SESSION_LIST=()

    for cfg in "${CONFIGS[@]}"; do
      CONFIG_LOCAL="$(cd "$(dirname "${cfg}")" && pwd)/$(basename "${cfg}")"
      METADATA_JSON="$(parse_metadata_json "${CONFIG_LOCAL}")"
      TASK="$(json_field task <<<"${METADATA_JSON}")"
      NAME="$(json_field name <<<"${METADATA_JSON}")"
      MESSAGE="$(json_field message <<<"${METADATA_JSON}")"
      CONFIG_REMOTE="${REMOTE_CONFIG_DIR%/}/$(basename "${CONFIG_LOCAL%.py}")__${QUEUE_STAMP}.py"
      CHILD_SESSION="$(build_session_name "stf-" "${TASK}" "${NAME}" "${QUEUE_STAMP}")"
      CONFIG_LOCAL_LIST+=("${CONFIG_LOCAL}")
      CONFIG_REMOTE_LIST+=("${CONFIG_REMOTE}")
      TASK_LIST+=("${TASK}")
      NAME_LIST+=("${NAME}")
      MESSAGE_LIST+=("${MESSAGE}")
      CHILD_SESSION_LIST+=("${CHILD_SESSION}")
    done

    CONFIG_LOCAL="${CONFIG_LOCAL_LIST[0]}"
    CONFIG_REMOTE="${CONFIG_REMOTE_LIST[0]}"
    TASK="${TASK_LIST[0]}"
    NAME="${NAME_LIST[0]}"
    MESSAGE="${MESSAGE_LIST[0]}"

    LAST_INDEX="$((${#NAME_LIST[@]} - 1))"
    if [[ -z "${SESSION}" ]]; then
      if [[ "${#CONFIGS[@]}" -eq 1 ]]; then
        SESSION="${CHILD_SESSION_LIST[0]}"
      else
        SESSION="$(build_queue_session_name "${NAME_LIST[0]}" "${NAME_LIST[$LAST_INDEX]}" "${QUEUE_STAMP}")"
      fi
    fi

    STATE_TRACK_FILE="${REMOTE_REPO}/log/remote_runs/${SESSION}.state"
    CONTROLLER_SESSION="${SESSION}"
    QUEUE_MODE="0"
    STARTUP_PASSED="0"
    SEQUENTIAL_CONFIRMED="0"
    SEQUENTIAL_CONFIRMATION=""
    ACTIVE_SESSION=""
    QUEUE_STATE_FILE=""
    CONFIG_LOCAL_LIST_JSON="$(json_array_from_args "${CONFIG_LOCAL_LIST[@]}")"
    CONFIG_REMOTE_LIST_JSON="$(json_array_from_args "${CONFIG_REMOTE_LIST[@]}")"
    SESSION_LIST_JSON="$(json_array_from_args "${CHILD_SESSION_LIST[@]}")"

    if session_exists_name "${SESSION}"; then
      echo "tmux session already exists: ${SESSION}" >&2
      exit 4
    fi
    for child_session in "${CHILD_SESSION_LIST[@]}"; do
      if session_exists_name "${child_session}"; then
        echo "tmux session already exists: ${child_session}" >&2
        exit 4
      fi
    done

    for idx in "${!CONFIG_LOCAL_LIST[@]}"; do
      stage_file_to_remote "${CONFIG_LOCAL_LIST[$idx]}" "${CONFIG_REMOTE_LIST[$idx]}"
    done

    if [[ "${#CONFIGS[@]}" -eq 1 ]]; then
      if [[ -n "${REMOTE_COMMAND}" ]]; then
        REMOTE_COMMAND_VALUE="${REMOTE_COMMAND}"
      else
        REMOTE_COMMAND_VALUE="cd $(printf '%q' "${REMOTE_REPO}") && mkdir -p log/remote_runs && ./tools/train_queue.sh --config $(printf '%q' "${CONFIG_REMOTE}") --gpu $(printf '%q' "${GPU}") --retries $(printf '%q' "${RETRIES}") --delay-sec $(printf '%q' "${DELAY_SEC}") --state-file $(printf '%q' "${STATE_TRACK_FILE}")"
      fi

      printf -v QUOTED_REMOTE_COMMAND '%q' "${REMOTE_COMMAND_VALUE}"
      remote_exec "tmux new-session -d -s $(printf '%q' "${SESSION}") bash -lc ${QUOTED_REMOTE_COMMAND}"
    else
      QUEUE_MODE="1"
      QUEUE_STATE_FILE="${REMOTE_REPO}/log/remote_runs/${SESSION}.queue.state"
      CONTROLLER_SCRIPT_REMOTE="${REMOTE_REPO}/log/remote_runs/${SESSION}.controller.sh"
      CONTROLLER_SCRIPT_LOCAL="$(mktemp)"

      {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n'
        printf 'REMOTE_REPO=%q\n' "${REMOTE_REPO}"
        printf 'GPU=%q\n' "${GPU}"
        printf 'RETRIES=%q\n' "${RETRIES}"
        printf 'DELAY_SEC=%q\n' "${DELAY_SEC}"
        printf 'QUEUE_STATE_FILE=%q\n' "${QUEUE_STATE_FILE}"
        printf 'mkdir -p %q %q\n' "${REMOTE_REPO}/log/remote_runs" "${REMOTE_REPO}/log/train_queue"
        printf ': > %q\n' "${QUEUE_STATE_FILE}"
        printf 'CONFIGS=(\n'
        for item in "${CONFIG_REMOTE_LIST[@]}"; do
          printf '  %q\n' "${item}"
        done
        printf ')\n'
        printf 'TASKS=(\n'
        for item in "${TASK_LIST[@]}"; do
          printf '  %q\n' "${item}"
        done
        printf ')\n'
        printf 'NAMES=(\n'
        for item in "${NAME_LIST[@]}"; do
          printf '  %q\n' "${item}"
        done
        printf ')\n'
        printf 'SESSIONS=(\n'
        for item in "${CHILD_SESSION_LIST[@]}"; do
          printf '  %q\n' "${item}"
        done
        printf ')\n'
        cat <<'EOF'
for idx in "${!CONFIGS[@]}"; do
  printf 'PENDING|%s|%s|%s|%s|%s\n' \
    "$idx" "${SESSIONS[$idx]}" "${CONFIGS[$idx]}" "${TASKS[$idx]}" "${NAMES[$idx]}" >>"${QUEUE_STATE_FILE}"
done

queue_exit=0
for idx in "${!CONFIGS[@]}"; do
  child_config="${CONFIGS[$idx]}"
  child_task="${TASKS[$idx]}"
  child_name="${NAMES[$idx]}"
  child_session="${SESSIONS[$idx]}"
  child_state_file="${REMOTE_REPO}/log/remote_runs/${child_session}.state"
  child_exit_file="${REMOTE_REPO}/log/remote_runs/${child_session}.exit"

  rm -f "${child_exit_file}" "${child_state_file}"
  printf 'RUNNING|%s|%s|%s|%s|%s|state_file=%s\n' \
    "$idx" "${child_session}" "${child_config}" "${child_task}" "${child_name}" "${child_state_file}" >>"${QUEUE_STATE_FILE}"

  child_command="cd ${REMOTE_REPO} && mkdir -p log/remote_runs && ./tools/train_queue.sh --config ${child_config} --gpu ${GPU} --retries ${RETRIES} --delay-sec ${DELAY_SEC} --state-file ${child_state_file}; exit_code=\$?; printf '%s\n' \"\${exit_code}\" > ${child_exit_file}; exit \"\${exit_code}\""
  printf -v quoted_child_command '%q' "${child_command}"
  tmux new-session -d -s "${child_session}" bash -lc "${quoted_child_command}"

  while tmux has-session -t "${child_session}" >/dev/null 2>&1; do
    sleep 5
  done

  child_exit_code="1"
  if [[ -f "${child_exit_file}" ]]; then
    child_exit_code="$(cat "${child_exit_file}")"
  fi
  child_run_dir="$(find -L "${REMOTE_REPO}/runs/${child_task}" -maxdepth 1 -mindepth 1 -type d -name "${child_name}_*" 2>/dev/null | sort | tail -n 1)"
  printf 'DONE|%s|%s|%s|%s|%s|exit=%s|run_dir=%s|state_file=%s\n' \
    "$idx" "${child_session}" "${child_config}" "${child_task}" "${child_name}" "${child_exit_code}" "${child_run_dir}" "${child_state_file}" >>"${QUEUE_STATE_FILE}"
  if [[ "${child_exit_code}" != "0" ]]; then
    queue_exit=1
  fi
done

printf 'QUEUE_DONE|exit=%s\n' "${queue_exit}" >>"${QUEUE_STATE_FILE}"
exit "${queue_exit}"
EOF
      } >"${CONTROLLER_SCRIPT_LOCAL}"
      chmod +x "${CONTROLLER_SCRIPT_LOCAL}"
      stage_file_to_remote "${CONTROLLER_SCRIPT_LOCAL}" "${CONTROLLER_SCRIPT_REMOTE}"
      rm -f "${CONTROLLER_SCRIPT_LOCAL}"

      REMOTE_COMMAND_VALUE="cd $(printf '%q' "${REMOTE_REPO}") && bash $(printf '%q' "${CONTROLLER_SCRIPT_REMOTE}")"
      printf -v QUOTED_REMOTE_COMMAND '%q' "${REMOTE_COMMAND_VALUE}"
      remote_exec "tmux new-session -d -s $(printf '%q' "${SESSION}") bash -lc ${QUOTED_REMOTE_COMMAND}"
    fi

    if [[ "${STARTUP_SECONDS}" =~ ^[0-9]+$ ]] && [[ "${STARTUP_SECONDS}" -gt 0 ]]; then
      sleep "${STARTUP_SECONDS}"
    fi

    STATUS="exited-early"
    PANE_FILE="$(mktemp)"
    STATE_FILE_TMP="$(mktemp)"
    LATEST_RUN_DIR=""

    if [[ "${QUEUE_MODE}" == "1" ]]; then
      remote_capture "cat $(printf '%q' "${QUEUE_STATE_FILE}") 2>/dev/null" >"${STATE_FILE_TMP}"
      eval "$(load_queue_state_env "${STATE_FILE_TMP}")"
      ACTIVE_SESSION="${QUEUE_CURRENT_SESSION:-}"
      if session_exists_name "${SESSION}"; then
        STATUS="running"
        if [[ -n "${ACTIVE_SESSION}" ]] && session_exists_name "${ACTIVE_SESSION}"; then
          capture_session_pane_to_file "${ACTIVE_SESSION}" "${PANE_FILE}"
          STARTUP_PASSED="1"
          SEQUENTIAL_CONFIRMED="1"
          SEQUENTIAL_CONFIRMATION="Active config session ${ACTIVE_SESSION} is running while later configs remain queued behind controller ${SESSION}."
        else
          capture_session_pane_to_file "${SESSION}" "${PANE_FILE}"
        fi
      fi
    else
      if session_exists; then
        STATUS="running"
        capture_pane_to_file "${PANE_FILE}"
        STARTUP_PASSED="1"
      fi
      capture_state_to_file "${STATE_FILE_TMP}"
    fi

    CREATED_AT="$(now_ts)"
    UPDATED_AT="${CREATED_AT}"
    RECORD_PATH="$(record_path_for_session "${SESSION}")"
    export CREATED_AT UPDATED_AT STATUS HOST REMOTE_REPO REMOTE_CONFIG_DIR LOCAL_BRANCH LOCAL_HEAD LOCAL_STATUS REMOTE_BRANCH REMOTE_HEAD REMOTE_STATUS REMOTE_DATA_LINK TMUX_PATH SESSION PURPOSE CONFIG_LOCAL CONFIG_REMOTE TASK NAME MESSAGE GPU RETRIES DELAY_SEC STARTUP_SECONDS STATE_TRACK_FILE REMOTE_COMMAND_VALUE QUEUE_MODE CONTROLLER_SESSION ACTIVE_SESSION QUEUE_STATE_FILE STARTUP_PASSED SEQUENTIAL_CONFIRMED SEQUENTIAL_CONFIRMATION ALLOW_VERSION_MISMATCH CONFIG_LOCAL_LIST_JSON CONFIG_REMOTE_LIST_JSON SESSION_LIST_JSON
    write_record "${RECORD_PATH}" "${PANE_FILE}" "${STATE_FILE_TMP}" "${LATEST_RUN_DIR}"
    if [[ "${QUEUE_MODE}" == "1" ]]; then
      append_ledger \
        "launch | session \`${SESSION}\`" \
        "purpose: ${PURPOSE}" \
        "local_head: ${LOCAL_HEAD}" \
        "remote_head: ${REMOTE_HEAD}" \
        "config_local_list: ${CONFIG_LOCAL_LIST[*]}" \
        "config_remote_list: ${CONFIG_REMOTE_LIST[*]}" \
        "child_sessions: ${CHILD_SESSION_LIST[*]}" \
        "record: ${RECORD_PATH}" \
        "startup_status: ${STATUS}" \
        "startup_passed: ${STARTUP_PASSED}" \
        "sequential_confirmed: ${SEQUENTIAL_CONFIRMED}"
    else
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
    fi
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
    CONTROLLER_SESSION="${CONTROLLER_SESSION:-${SESSION}}"

    PANE_FILE="$(mktemp)"
    STATE_FILE_TMP="$(mktemp)"
    LOG_FILE_TMP="$(mktemp)"
    LATEST_RUN_DIR=""
    FINAL_VAL_LINE=""
    GPU_SUMMARY_LINE=""
    ACTIVE_SESSION="${ACTIVE_SESSION:-}"

    if [[ "${QUEUE_MODE:-0}" == "1" ]]; then
      remote_capture "cat $(printf '%q' "${QUEUE_STATE_FILE}") 2>/dev/null" >"${STATE_FILE_TMP}"
      eval "$(load_queue_state_env "${STATE_FILE_TMP}")"
      ACTIVE_SESSION="${QUEUE_CURRENT_SESSION:-}"

      if [[ -n "${ACTIVE_SESSION}" ]] && session_exists_name "${ACTIVE_SESSION}"; then
        STATUS="running"
        capture_session_pane_to_file "${ACTIVE_SESSION}" "${PANE_FILE}"
      elif session_exists_name "${SESSION}"; then
        STATUS="running"
        capture_session_pane_to_file "${SESSION}" "${PANE_FILE}"
      else
        STATUS="stopped"
      fi

      if [[ -n "${QUEUE_CURRENT_TASK:-}" && -n "${QUEUE_CURRENT_NAME:-}" ]]; then
        LATEST_RUN_DIR="$(find_latest_run_dir_for_task_name "${QUEUE_CURRENT_TASK}" "${QUEUE_CURRENT_NAME}")"
      fi
      if [[ -z "${LATEST_RUN_DIR}" && -n "${QUEUE_LAST_RUN_DIR:-}" ]]; then
        LATEST_RUN_DIR="${QUEUE_LAST_RUN_DIR}"
      fi
      if [[ -z "${LATEST_RUN_DIR}" && -n "${QUEUE_LAST_TASK:-}" && -n "${QUEUE_LAST_NAME:-}" ]]; then
        LATEST_RUN_DIR="$(find_latest_run_dir_for_task_name "${QUEUE_LAST_TASK}" "${QUEUE_LAST_NAME}")"
      fi

      if [[ "${QUEUE_DONE_EXIT:-}" == "0" ]]; then
        STATUS="completed"
      elif [[ -n "${QUEUE_DONE_EXIT:-}" ]]; then
        STATUS="failed"
      elif [[ "${STATUS}" == "running" ]]; then
        STARTUP_PASSED="${STARTUP_PASSED:-1}"
      fi
    else
      if session_exists; then
        STATUS="running"
        capture_pane_to_file "${PANE_FILE}"
      else
        STATUS="stopped"
      fi
      capture_state_to_file "${STATE_FILE_TMP}"
      LATEST_RUN_DIR="$(find_latest_run_dir)"
      if grep -q '^OK|' "${STATE_FILE_TMP}" 2>/dev/null; then
        STATUS="completed"
      elif grep -Eq '^(FAIL|MISS)\|' "${STATE_FILE_TMP}" 2>/dev/null; then
        STATUS="failed"
      fi
    fi

    if [[ -n "${LATEST_RUN_DIR}" ]]; then
      remote_capture "tail -n 180 $(printf '%q' "${LATEST_RUN_DIR}/logs/train.log") 2>/dev/null" >"${LOG_FILE_TMP}"
      FINAL_VAL_LINE="$(extract_last_line ' - stf.train - INFO - val epoch=' "${LOG_FILE_TMP}")"
      GPU_SUMMARY_LINE="$(extract_last_line 'gpu memory summary:' "${LOG_FILE_TMP}")"
      if [[ -n "${GPU_SUMMARY_LINE}" && "${STATUS}" == "stopped" ]]; then
        STATUS="completed"
      fi
    fi

    UPDATED_AT="$(now_ts)"
    CREATED_AT="${CREATED_AT:-${UPDATED_AT}}"
    export CREATED_AT UPDATED_AT STATUS HOST REMOTE_REPO REMOTE_CONFIG_DIR LOCAL_BRANCH LOCAL_HEAD LOCAL_STATUS REMOTE_BRANCH REMOTE_HEAD REMOTE_STATUS REMOTE_DATA_LINK TMUX_PATH SESSION PURPOSE CONFIG_LOCAL CONFIG_REMOTE TASK NAME MESSAGE GPU RETRIES DELAY_SEC STARTUP_SECONDS STATE_TRACK_FILE REMOTE_COMMAND_VALUE QUEUE_MODE CONTROLLER_SESSION ACTIVE_SESSION QUEUE_STATE_FILE STARTUP_PASSED SEQUENTIAL_CONFIRMED SEQUENTIAL_CONFIRMATION ALLOW_VERSION_MISMATCH CONFIG_LOCAL_LIST_JSON CONFIG_REMOTE_LIST_JSON SESSION_LIST_JSON
    write_record "${RECORD_PATH}" "${PANE_FILE}" "${STATE_FILE_TMP}" "${LATEST_RUN_DIR}" "${FINAL_VAL_LINE}" "${GPU_SUMMARY_LINE}"
    append_ledger \
      "status | session \`${SESSION}\`" \
      "status: ${STATUS}" \
      "record: ${RECORD_PATH}" \
      "active_session: ${ACTIVE_SESSION:-'(none)'}" \
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
