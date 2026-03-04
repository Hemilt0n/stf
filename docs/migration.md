# Migration

Use `stf migrate-config` to generate a new-style Python config from a legacy config file.

## What it does
- Detects task (`flow` or `stfdiff`) from config path/name.
- Loads common legacy fields (`train_dataloader`, `model`, `optimizer`, etc.).
- Emits a new config that wraps these fields into `ExperimentConfig`.

## Command

```bash
uv run stf migrate-config --legacy-config /abs/path/old.py --output configs/flow/new.py
```

## Checkpoint Compatibility
- Legacy checkpoint keys supported: `model`, `optimizer`, `ema`.
- Epoch can be inferred from filename pattern `model_epoch_<n>.pth`.
