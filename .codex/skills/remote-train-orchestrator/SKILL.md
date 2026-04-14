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

For comparison runs or any remote workflow that can burn substantial GPU time, prefer a stronger worker model (`gpt-5.4` with `high` reasoning effort or above) rather than a mini model. The skill can state this requirement, but the caller must still enforce it explicitly in `spawn_agent`.

2. Do not auto-sync code versions.
Always compare local and remote git heads before launch. If they differ, stop and tell the user. Do not auto-`git pull`, auto-`git push`, or patch the remote repo in place unless the user explicitly asks.

3. Treat remote configs as disposable runtime inputs.
Create or edit the local config under `configs/remote/` when needed, but do not rely on that directory being versioned. Stage the config to the remote repo under `configs/remote/` with a timestamped filename. Do not edit tracked remote configs in place.

When the user wants a review checkpoint before execution, prefer the two-phase helper flow: `prepare` first, then `launch-prepared` only after the user confirms.

For local browsing convenience, the helper keeps lightweight config snapshots under:
- `configs/remote/review/<session>/` for prepared configs waiting for inspection
- `configs/remote/running/<session>/` for configs currently being executed
- `configs/remote/completed/<session>/` for finished configs

4. Prefer `tailscale ssh` for both execution and file staging.
In this environment, `tailscale ssh ... 'cat > remote-file' < local-file` is proven to work. Direct `scp` may fail on host-key setup. Only prefer `scp` if you have already confirmed it works in the current shell.

5. Monitor only the startup window.
After launch, watch the `tmux` pane briefly to catch immediate errors. Once startup is stable, stop streaming logs into context. Re-check later through the record file and `tmux` or run-artifact inspection.

6. Multi-experiment comparisons must be serialized unless the user explicitly requests parallelism.
If the task is an A/B or multi-config comparison on the same remote host, do not launch separate independent runs that target the same GPU by default. Prefer one queue controller and sequential execution order. Each config run should still get its own named `tmux` session, but only one config session should be active at a time unless the user explicitly asks for parallelism and the GPU allocation is unambiguous.

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

Fast path:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch \
  --config configs/remote/<experiment>.py \
  --purpose "<why this run exists>" \
  --startup-seconds 20
```

Two-phase path when the user wants to inspect staged configs first:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh prepare \
  --config configs/remote/<experiment>.py \
  --purpose "<why this run exists>"

bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch-prepared \
  --session <prepared-session>
```

Default launch behavior:
- compare local and remote git heads
- stage the config to remote `configs/remote/<name>__<timestamp>.py`
- for single-config launch: create one meaningful `tmux` session from `task`, `name`, and timestamp
- for multi-config launch: create one queue controller session and let it start one per-config `tmux` session at a time, in order
- run `tools/train_queue.sh` for each config on the remote host
- write local runtime state under `log/remote_runs/`

Local runtime files:
- `log/remote_runs/records/<session>.json`
- `log/remote_runs/experiments.md`

Prepared runs write a record with `status=prepared`; `launch-prepared` reuses that record instead of restaging everything again.
Prepared runs also snapshot the local configs into `configs/remote/review/<session>/` so the user has one obvious place to inspect them.

If the user explicitly wants a custom remote command for smoke testing, use `--remote-command` only for a single-config launch.

For compare runs that should execute sequentially, pass multiple `--config` values to one `launch` call. The helper will stage all configs, create one queue controller session, and then sequentially create one named `tmux` session per config. The default shape is:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch \
  --config configs/remote/<baseline>.py \
  --config configs/remote/<variant>.py \
  --purpose "<compare purpose>" \
  --gpu 0 \
  --startup-seconds 20
```

Use one queue session name for recovery and record the ordered config list plus the per-config child sessions in the summary.

If the user wants the terminal state preserved after completion, pass `--keep-finished-session`. This keeps finished `tmux` sessions available for manual inspection instead of letting them disappear immediately.
When a prepared run is launched, the helper copies the checked configs from `review/` into `running/`; once the run reaches a terminal state, `status` moves that local snapshot into `completed/`.

### 4. Startup Monitoring

During the first check window, verify:
- `tmux has-session -t <session>` still succeeds
- pane output does not show import errors, config errors, missing data, or CUDA setup failure
- the queue/controller state file is being written when using the default launch command
- for compare queues, the first child config session exists and later configs have not started yet

If the session exits during startup, inspect the pane tail and report the failure instead of retrying blindly.

### 5. Long-Run Recovery and Follow-Up

Reconstruct state from:
- `log/remote_runs/records/<session>.json`
- `log/remote_runs/experiments.md`
- remote controller `tmux` session, if still running
- remote child `tmux` session, if one config is currently active
- remote queue/controller state file
- latest matching run directory under `runs/<task>/<name>_*`

Use:

```bash
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh status --session <session>
```

When the run is finished, `status` should be enough to recover:
- whether the controller session is still alive
- which child config session is currently active, if any
- whether the run was only prepared and not launched yet
- the latest pane tail
- queue state tail
- latest matching run directory
- final validation summary line, if present in `logs/train.log`
- GPU memory summary line, if present

## Reporting Expectations

When reporting back to the main agent or the user:
- give the queue session name
- give the local record path
- give the remote staged config path or ordered staged config list
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
- Do not treat `prepare` as launch; nothing should be running on the remote host until `launch-prepared` or direct `launch` is called.
- Do not treat “launch baseline, then launch variant” as sequential execution; if both config sessions are alive at once on the same GPU, that is parallelism.
- Do not open multiple config sessions on the same GPU for a comparison run unless the user explicitly asked for parallel execution.

## Resources

- Remote environment notes: `references/observed-environment.md`
- Helper CLI: `scripts/remote_train.sh`
