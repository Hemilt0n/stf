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
