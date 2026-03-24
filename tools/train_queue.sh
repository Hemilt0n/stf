#!/usr/bin/env bash
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/train_queue.sh [options] --config <path> [--config <path> ...]
  tools/train_queue.sh [options] --queue-file <file>

Options:
  --config <path>       Add one training config (can be repeated).
  --queue-file <file>   Read config list from file (one path per line, '#' supported).
  --gpu <id>            CUDA device id passed via CUDA_VISIBLE_DEVICES (default: 0).
  --retries <n>         Retry count per config after failure (default: 0).
  --delay-sec <n>       Sleep seconds between retries (default: 5).
  --stop-on-fail        Stop queue immediately if one config fails.
  --no-skip-success     Do not skip configs already marked successful in state file.
  --state-file <path>   Queue state file path (default: log/train_queue/queue.state).
  -h, --help            Show help.
EOF
}

GPU_ID="0"
RETRIES=0
DELAY_SEC=5
STOP_ON_FAIL=0
SKIP_SUCCESS=1
STATE_FILE="log/train_queue/queue.state"
LOG_DIR="log/train_queue"
declare -a CONFIGS=()
QUEUE_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config"; exit 2; }
      CONFIGS+=("$2")
      shift 2
      ;;
    --queue-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --queue-file"; exit 2; }
      QUEUE_FILE="$2"
      shift 2
      ;;
    --gpu)
      [[ $# -ge 2 ]] || { echo "Missing value for --gpu"; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --retries)
      [[ $# -ge 2 ]] || { echo "Missing value for --retries"; exit 2; }
      RETRIES="$2"
      shift 2
      ;;
    --delay-sec)
      [[ $# -ge 2 ]] || { echo "Missing value for --delay-sec"; exit 2; }
      DELAY_SEC="$2"
      shift 2
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=1
      shift
      ;;
    --no-skip-success)
      SKIP_SUCCESS=0
      shift
      ;;
    --state-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --state-file"; exit 2; }
      STATE_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -n "$QUEUE_FILE" ]]; then
  [[ -f "$QUEUE_FILE" ]] || { echo "Queue file not found: $QUEUE_FILE"; exit 2; }
  while IFS= read -r line; do
    trimmed="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$trimmed" ]] && continue
    [[ "${trimmed:0:1}" == "#" ]] && continue
    CONFIGS+=("$trimmed")
  done < "$QUEUE_FILE"
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No configs provided."
  usage
  exit 2
fi

if ! [[ "$RETRIES" =~ ^[0-9]+$ ]]; then
  echo "--retries must be a non-negative integer"
  exit 2
fi

if ! [[ "$DELAY_SEC" =~ ^[0-9]+$ ]]; then
  echo "--delay-sec must be a non-negative integer"
  exit 2
fi

mkdir -p "$LOG_DIR" "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

has_success() {
  local cfg="$1"
  awk -F'|' -v cfg="$cfg" '$1=="OK" && $2==cfg {found=1} END{exit !found}' "$STATE_FILE"
}

record_state() {
  local status="$1"
  local cfg="$2"
  local attempt="$3"
  local code="$4"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "${status}|${cfg}|${ts}|attempt=${attempt}|exit=${code}" >> "$STATE_FILE"
}

RUN_ID="$(date '+%Y%m%d-%H%M%S')"
TOTAL="${#CONFIGS[@]}"
DONE=0
SKIPPED=0
FAILED=0

echo "Queue start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "Total configs: ${TOTAL}"
echo "GPU: ${GPU_ID}"
echo "Retries: ${RETRIES}"
echo "State file: ${STATE_FILE}"
echo

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[MISS] config not found: $cfg"
    record_state "MISS" "$cfg" 0 127
    FAILED=$((FAILED + 1))
    if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
      break
    fi
    continue
  fi

  if [[ "$SKIP_SUCCESS" -eq 1 ]] && has_success "$cfg"; then
    echo "[SKIP] already successful: $cfg"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  max_attempts=$((RETRIES + 1))
  success=0
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    base_name="$(basename "${cfg%.py}")"
    log_file="${LOG_DIR}/${RUN_ID}_${base_name}_attempt${attempt}.log"
    echo "[RUN ] cfg=${cfg} attempt=${attempt}/${max_attempts}"
    echo "       log=${log_file}"

    CUDA_VISIBLE_DEVICES="$GPU_ID" uv run stf train --config "$cfg" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    if [[ "$exit_code" -eq 0 ]]; then
      echo "[ OK ] cfg=${cfg} attempt=${attempt}"
      record_state "OK" "$cfg" "$attempt" "$exit_code"
      DONE=$((DONE + 1))
      success=1
      break
    fi

    echo "[FAIL] cfg=${cfg} attempt=${attempt} exit=${exit_code}"
    record_state "FAIL" "$cfg" "$attempt" "$exit_code"
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      echo "       retry in ${DELAY_SEC}s..."
      sleep "$DELAY_SEC"
    fi
  done

  if [[ "$success" -eq 0 ]]; then
    FAILED=$((FAILED + 1))
    if [[ "$STOP_ON_FAIL" -eq 1 ]]; then
      break
    fi
  fi
done

echo
echo "Queue done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "Summary: done=${DONE}, skipped=${SKIPPED}, failed=${FAILED}, total=${TOTAL}"
[[ "$FAILED" -eq 0 ]]
