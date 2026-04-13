---
name: remote-train-orchestrator
description: Launch, monitor, and recover STF training jobs on the remote GPU server over Tailscale SSH. Use when the user wants a config run remotely in tmux, needs remote preflight checks, wants local-only `configs/remote` staging, or needs experiment tracking and later result follow-up without flooding the main context with live training logs.
---

# Remote Train Orchestrator

Use this skill when STF code is edited locally but real training must run on the remote GPU server.

Current workflow assumptions:
- Remote host is `hang@home-pc-ubuntu`.
- Remote repo is `/home/hang/repos/stf`.
- Long runs should live in a named `tmux` session.
- Runtime state should be recoverable from local files under `log/remote_runs/` plus the remote `tmux` session.

## Core Rules

1. Delegate long remote runs to a worker subagent.
Keep the main agent context small. The worker owns preflight, staging, launch, startup monitoring, and later follow-up. The main agent should retain only the session name, record path, and a short status summary.

2. Do not auto-sync code versions.
Always compare local and remote git heads before launch. If they differ, stop and tell the user. Do not auto-`git pull`, auto-`git push`, or patch the remote repo in place unless the user explicitly asks.

3. Treat remote configs as disposable runtime inputs.
Create or edit the local config under `configs/remote/` when needed, but do not rely on that directory being versioned. Stage the config to the remote repo under `configs/remote/` with a timestamped filename. Do not edit tracked remote configs in place.

4. Prefer `tailscale ssh` for both execution and file staging.
In this environment, `tailscale ssh ... 'cat > remote-file' < local-file` is proven to work. Direct `scp` may fail on host-key setup. Only prefer `scp` if you have already confirmed it works in the current shell.

5. Monitor only the startup window.
After launch, watch the `tmux` pane briefly to catch immediate errors. Once startup is stable, stop streaming logs into context. Re-check later through the record file and `tmux` or run-artifact inspection.

## Standard Workflow

### 1. Preflight

Run the helper first:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh inspect
```

Confirm:
- local branch and head
- remote branch and head
- remote worktree cleanliness
- remote `tmux` availability
- remote `data` symlink target

If the dataset root in the target config is uncertain, inspect existing remote configs and the remote `data/` tree before editing the config.

Useful reference:
- `references/observed-environment.md`

### 2. Prepare the Local Remote-Only Config

Preferred location:

```bash
configs/remote/<experiment>.py
```

Guidelines:
- Start from an existing config or `configs/flow/template_all_options.py`.
- Adapt dataset roots to the remote server, not the local workstation.
- Keep the config local-only unless the user explicitly wants it versioned.
- Set a meaningful `ExperimentConfig.name` and `message`; the helper records both.

### 3. Launch the Remote Run

Use the helper:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch \
  --config configs/remote/<experiment>.py \
  --purpose "<why this run exists>" \
  --startup-seconds 20
```

Default launch behavior:
- compare local and remote git heads
- stage the config to remote `configs/remote/<name>__<timestamp>.py`
- create a meaningful tmux session name from `task`, `name`, and timestamp
- run `tools/train_queue.sh` on the remote host
- write local runtime state under `log/remote_runs/`

Local runtime files:
- `log/remote_runs/records/<session>.json`
- `log/remote_runs/experiments.md`

If the user explicitly wants a custom remote command for smoke testing, use `--remote-command`.

### 4. Startup Monitoring

During the first check window, verify:
- `tmux has-session -t <session>` still succeeds
- pane output does not show import errors, config errors, missing data, or CUDA setup failure
- the queue state file is being written when using the default launch command

If the session exits during startup, inspect the pane tail and report the failure instead of retrying blindly.

### 5. Long-Run Recovery and Follow-Up

Reconstruct state from:
- `log/remote_runs/records/<session>.json`
- `log/remote_runs/experiments.md`
- remote `tmux` session
- remote queue state file
- latest matching run directory under `runs/<task>/<name>_*`

Use:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh status --session <session>
```

When the run is finished, `status` should be enough to recover:
- whether the session is still alive
- the latest pane tail
- queue state tail
- latest matching run directory
- final validation summary line, if present in `logs/train.log`
- GPU memory summary line, if present

## Reporting Expectations

When reporting back to the main agent or the user:
- give the session name
- give the local record path
- give the remote staged config path
- state whether startup passed
- if completed, quote the final validation summary line and GPU memory summary line
- update `log/remote_runs/experiments.md` through the helper-driven record flow instead of pasting long raw logs into context

## Guardrails

- Do not keep tailing full training logs into the main thread.
- Do not patch the remote repo to fix version mismatch unless the user explicitly asks.
- Do not overwrite an existing tmux session of the same name.
- Do not assume `configs/remote/` already exists locally; create it only when needed.
- Do not assume dataset roots from local configs are valid on the remote host.
- Do not assume helper scripts mentioned in old docs still exist; inspect the current repo before depending on them.

## Resources

- Remote environment notes: `references/observed-environment.md`
- Helper CLI: `scripts/remote_train.sh`
