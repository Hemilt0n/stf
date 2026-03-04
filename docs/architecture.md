# Architecture

## Goals
- Fast iteration for research experiments
- Single CLI and Python API
- Explicit compatibility shims for legacy configs and checkpoints

## Layers
- `stf.data`: datasets, samplers, transforms
- `stf.models`: model definitions (STFDiff + Flow)
- `stf.engine`: train/eval/predict runtime
- `stf.metrics`: metric modules
- `stf.compat`: legacy migration and checkpoint loading
- `stf.cli` / `stf.api`: user entry points

## Runtime Data Flow
1. Config loader builds `ExperimentConfig` from a Python config file.
2. Engine materializes dataloaders/model/optimizer/scheduler.
3. Train/eval/predict loops write outputs under `runs/<task>/<exp>`.
4. Compatibility helpers bridge legacy configs/checkpoints.
