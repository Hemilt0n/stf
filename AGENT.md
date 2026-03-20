# AGENT Handoff

本文件用于会话交接与快速恢复上下文。新一轮开发前，请先阅读此文件与 `README.md`。

## 1. 当前状态（2026-03-20）

- 仓库路径: `/home/hang/repos/stf`
- 当前工作分支（常用）: `research/perf-24g-flow`
- 远程仓库: `origin = https://github.com/Hemilt0n/stf.git`
- 已同步的关键提交:
  - `research/perf-24g-flow`: `9280db8`
  - `master`: `5eeadef`

## 2. 路径与运行约定

- 依赖管理: `uv` + `pyproject.toml`
- 数据路径: `data -> /home/hang/repos/stf-tmp/data`（软链接）
- 结果路径: `runs -> /mnt/d/hang/results`（软链接）
- 遥感归一化约定（必须）: `RescaleToMinusOneOne(..., data_range=[0, 10000])`
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

3. 遥感归一化默认值已统一:
   - 约定: `RescaleToMinusOneOne(..., data_range=[0, 10000])`
   - `master` 提交: `5eeadef`

4. 新增 Flow 24G 性能优化研究分支（`research/perf-24g-flow`）
   - 提交: `9280db8`
   - 新增训练性能开关（`TrainConfig`）:
     - `precision` (`fp16` / `bf16`)
     - `enable_tf32`
     - `deterministic`, `cudnn_benchmark`
     - `non_blocking_transfer`
     - `train_log_interval`
     - `compile_model`, `compile_mode`, `compile_dynamic`
     - `use_channels_last`
   - 训练引擎能力更新:
     - 迁移到 `torch.amp.autocast` + `torch.amp.GradScaler`
     - 支持 `bf16` 路径（不启用 scaler）
     - 可选 `torch.compile`、channels-last、non_blocking
     - CUDA backend 配置可通过 config 控制（TF32 / deterministic / benchmark）
   - 新增研究配置:
     - `configs/flow/change_aware_perf_24g.py`
     - `configs/flow/change_aware_perf_24g_compile.py`

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

24G 性能研究配置（Flow）:

```bash
uv run stf train --config configs/flow/change_aware_perf_24g.py
uv run stf train --config configs/flow/change_aware_perf_24g_compile.py
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
- 确认分支位置:
  - `git branch --show-current`
- 先跑 smoke:
  - `uv run pytest -q tests/smoke`
- 若改了模型接口，必须同时检查:
  - `stf/models/diffusion.py`
  - `stf/models/flow.py`
  - 对应 config 的模型构造参数

## 9. 本次更新记录

- 更新日期: `2026-03-20 13:27:06 +0800`
- 更新内容:
  - 同步当前分支/提交信息
  - 补充 24G 性能优化研究改动与配置入口
  - 补充训练引擎性能开关能力说明
