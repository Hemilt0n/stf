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

## Change-aware knobs (model-level)

The following optional arguments are now supported in core models:

- `condition_dropout_p`: training-time dropout on `fine_t1` conditioning (`GaussianFlowMatching` / `GaussianDiffusion`)
- `change_loss_weight`: up-weight loss in high-change regions inferred from `|coarse_t2 - coarse_t1|`
- `coarse_consistency_weight`: additional loss to align prediction with `coarse_t2` at coarse resolution
- `coarse_consistency_loss_type`: `\"l1\"` or `\"l2\"`

Default values keep legacy behavior (`0.0` for new weights/dropout).

## Default Flow Architecture

- Default training wrapper for `task="flow"` is `GaussianFlowMatching`.
- `FlowMatching` is retained for ablation/debug only and is not the recommended default path.

## Train-side warmup knobs

The following optional `TrainConfig` arguments control `fine_t1` anti-shortcut warmup:

- `fine_t1_noise_warmup_epochs` / `fine_t1_noise_warmup_steps`: warmup length (`steps` takes priority)
- `fine_t1_noise_power`: decay shape (larger means slower early decay)
- `fine_t1_noise_std`: Gaussian noise scale
- `fine_t1_noise_alpha_tail`: terminal alpha floor after warmup (`0.0` keeps old behavior; `>0` keeps a non-zero noise tail)
