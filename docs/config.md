# Config Spec

Each config exports one object:

```python
EXPERIMENT = ExperimentConfig(...)
```

Key sections:
- `task`: `"stfdiff"` or `"flow"`
- `name`: experiment name
- `data`: dataloaders (`train/val/test`)
- `model`: model module
- `optimizer`: callable accepting `params`
- `scheduler`: callable accepting optimizer (optional)
- `metrics`: list of metric modules
- `train`: max epochs and intervals
- `io`: output/log/image behavior
- `resume_from`: optional checkpoint path

## Train Debug Knobs

Validation step diagnostics (optional):

- `train.val_step_log_keys`:
  - `False` by default.
  - When `True`, `val` log lines include `sample_idx` and `key` preview for each iter.
- `train.val_step_log_max_keys`:
  - `8` by default.
  - Limits how many keys are printed per val iter line.
- `train.val_step_save_csv`:
  - `False` by default.
  - When `True`, dumps per-iter validation debug CSV to
    `runs/<task>/<exp>_<timestamp>/debug/val_step_epoch_XXXX.csv`.

Sampler layout offline dump:

```bash
uv run python tools/dump_sampler_layout.py \
  --config configs/flow/change_aware_perf_24g.py \
  --split val \
  --epoch 0
```

Default output:
- `runs/debug/<exp_name>_<split>_sampler_epoch<epoch>_<timestamp>.csv`

## Change-aware knobs (model-level)

The following optional arguments are now supported in core models:

- `condition_dropout_p`: training-time dropout on `fine_t1` conditioning (where applicable)
- `change_loss_weight`: up-weight loss in high-change regions inferred from `|coarse_t2 - coarse_t1|`
- `coarse_consistency_weight`: additional loss to align prediction with `coarse_t2` at coarse resolution
- `coarse_consistency_loss_type`: `\"l1\"` or `\"l2\"`

Default values keep legacy behavior (`0.0` for new weights/dropout).
