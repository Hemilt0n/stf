# AGENT Handoff

本文件用于会话交接与快速恢复上下文。新一轮开发前，请先阅读此文件与 `README.md`。

## 1. 当前状态（2026-03-11）

- 仓库路径: `/home/hang/repos/stf`
- 当前工作分支（常用）: `plan/change-aware-fusion-roadmap`
- 远程仓库: `origin = https://github.com/Hemilt0n/stf.git`
- 已同步的关键提交:
  - `plan/change-aware-fusion-roadmap`: `22a545d`
  - `master`: `5010d28`（与上面改动等价，cherry-pick）

## 2. 路径与运行约定

- 依赖管理: `uv` + `pyproject.toml`
- 数据路径: `data -> /home/hang/repos/stf-tmp/data`（软链接）
- 结果路径: `runs -> /mnt/d/hang/results`（软链接）
- `.gitignore` 已约束:
  - 忽略 `/data`、`/runs`
  - 忽略 `/configs/remote/*`，保留 `/configs/remote/.gitkeep`

## 3. 最近关键改动（必须知道）

1. `PredTrajNet` 接口已统一为与 `PredNoiseNet` 一致（仅保留“非双分支”差异）
   - 文件: `stf/models/pred_resnet.py`
   - 统一签名:
     - `forward(coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, time, x_self_cond=None)`
   - `init_conv` 改为三路融合输入:
     - `Conv2d(init_dim * 3, init_dim, 1)`（fine/coarse/noisy）

2. `FlowMatching` 与 `ResidualGaussianFlowMatching` 已改为调用统一接口
   - 文件: `stf/models/flow.py`
   - 不再使用旧式 `self.model(x_t, t, coarse1, coarse2)` 调用

3. `master` 已同步上述接口统一改动，可在主干直接继续开发
4. Flow family 上新增可选高频约束接口（默认全关闭，兼容旧配置）
   - 文件: `stf/models/flow.py`, `stf/models/hf_losses.py`
   - 新参数（`FlowMatching` / `GaussianFlowMatching` / `ResidualGaussianFlowMatching`）:
     - `grad_loss_weight`
     - `lap_loss_weight`, `lap_num_scales`
     - `ranking_loss_weight`, `ranking_margin`
     - `hf_mask_strategy`, `hf_mask_quantile`, `hf_mask_threshold`, `hf_mask_topk_ratio`

## 4. 当前接口硬约束

- 模型训练主入口（flow/diffusion family）保持:
  - `model(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02) -> loss`
- 采样入口保持:
  - `model.sample(coarse_img_01, coarse_img_02, fine_img_01) -> pred`
- `PredTrajNet` / `PredNoiseNet` 内部 denoiser/velocity 网络统一参数顺序:
  - `(coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02_or_x_t, time, x_self_cond=None)`

## 5. 快速验证命令

```bash
uv run python -m compileall -q stf/models
uv run pytest -q tests/smoke
```

toy 配置快速训练（按需二选一）:

```bash
uv run stf train --config configs/flow/change_aware_toy.py
uv run stf train --config configs/stfdiff/change_aware_toy.py
```

高频约束最小实验矩阵（Flow）:

```bash
uv run stf train --config configs/flow/change_aware_toy.py
uv run stf train --config configs/flow/change_aware_toy_hf_grad.py
uv run stf train --config configs/flow/change_aware_toy_hf_grad_lap.py
uv run stf train --config configs/flow/change_aware_toy_hf_grad_lap_rank.py
```

## 6. 常见告警与说明

- `libgomp: Invalid value for environment variable OMP_NUM_THREADS`
  - 建议设置为正整数，例如 `export OMP_NUM_THREADS=1`
- `torch.cuda.amp.*` FutureWarning
  - 后续可迁移到 `torch.amp.GradScaler('cuda', ...)` 和 `torch.amp.autocast('cuda', ...)`
  - 当前不影响训练执行

## 7. 后续改动建议入口

- 加/改模型: `stf/models/`
- 加/改指标并接入验证日志: `stf/metrics/` + 对应 config 的 `metrics`
- 调参优先在 `configs/*.py` 完成，尽量不动 `stf/engine/*`

## 8. 交接检查清单

- 切分支并确认干净工作区:
  - `git status --short --branch`
- 先跑 smoke:
  - `uv run pytest -q tests/smoke`
- 若改了模型接口，必须同时检查:
  - `stf/models/diffusion.py`
  - `stf/models/flow.py`
  - 对应 config 的模型构造参数
