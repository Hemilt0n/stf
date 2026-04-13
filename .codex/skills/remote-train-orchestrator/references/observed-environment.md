# Observed STF Remote Environment

This file records the remote facts observed while building the skill. Re-check them with
`scripts/remote_train.py inspect` before relying on them for a real run.

## Stable-ish paths

- Remote host: `hang@home-pc-ubuntu`
- Remote repo: `/home/hang/repos/stf`
- Remote data symlink: `/home/hang/repos/stf/data -> /home/hang/data`
- Remote runs symlink: `/home/hang/repos/stf/runs -> /mnt/d/hang/results/stf`
- Remote tmux binary: `/usr/bin/tmux`

## Dataset layout observed

- `/home/hang/data/CIA/band4_serialized`
- `/home/hang/data/mnt -> /mnt/d/hang/Datasets/STF/Datasets`

Observed remote configs referenced both styles:
- `data/CIA/band4_serialized`
- `data/mnt/CIA/private_data/hh_setting-1-patch_band4/...`

Therefore, inspect the target dataset root before launching a new config. Do not assume the
local workstation path is valid on the server.

## Remote config convention observed

The remote repo already had a non-versioned `configs/remote/` area containing staged configs such as:
- `change_aware_perf_24g.py`
- `change_aware_fine_t1_noise_warmup_200.py`
- `template_all_options.py`

The skill follows that convention but stages timestamped filenames to avoid collisions.

## Transport observations

- `tailscale ssh` worked for remote commands.
- `tailscale ssh 'cat > remote-file' < local-file` worked for config staging.
- direct `scp` to `100.126.246.86` failed in this environment on host-key verification.

Default policy for this skill:
- use `tailscale ssh` for staging and execution
- only prefer `scp` if the current shell has already been prepared and verified

## Result artifact observations

Typical finished run directories contained:
- `configs/<staged-config>.py`
- `logs/train.log`
- `tensorboard/events.out.tfevents.*`
- optional `debug/val_step_epoch_*.csv`

The final `train.log` contained lines like:
- `val epoch=... loss=..., RMSE=..., ...`
- `gpu memory summary: total=..., peak_allocated=..., ...`

The helper script uses those lines for concise completion summaries.
