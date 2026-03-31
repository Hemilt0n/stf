# STF 项目进展记录

最后更新: 2026-03-31 16:12:12 +0800  
当前分支: `dev/fine-t1-noise-warmup`

## 1. 文档用途

用于持续记录以下内容，便于会话切换后快速恢复上下文：
- 想法与目标
- 已实现内容
- 代码改动范围
- 实验结果（尤其是你跑完后回传给我的结果）
- 阶段总结与后续计划

## 2. 记录规范

- 时间统一使用: `YYYY-MM-DD HH:MM:SS +0800`
- 每条记录至少包含:
  - 背景/目标
  - 实现与代码改动
  - 验证或实验结果
  - 总结与下一步
- 如果是你回传的实验结果，会在“用户实验回传记录”中追加，并在“阶段总结”里同步结论。

## 3. 时间线（已补齐）

### 2026-03-04 20:12:06 +0800 | `6c7b40a`

- 背景/目标:
  - 初始化 STF 仓库，形成可训练/可评估的统一工程骨架。
- 实现与代码改动:
  - 建立 CLI、配置系统、训练/评估/预测引擎、数据管线、模型与指标模块。
  - 增加基础文档与 smoke 测试。
- 结果:
  - 项目从单文件/原型状态进入标准化工程结构。

### 2026-03-04 20:28:46 +0800 | `0ad3cfe`

- 背景/目标:
  - 针对“大变化场景”明确问题与改进路线。
- 实现与代码改动:
  - 新增路线文档: [变化场景改进计划.md](/home/hang/repos/stf/docs/变化场景改进计划.md)。
- 结果:
  - 确立 P1/P2/P3 分层方案与最小实验矩阵方向。

### 2026-03-05 13:54:18 +0800 | `9aab002`

- 背景/目标:
  - 先以最小侵入方式接入 change-aware 训练控制。
- 实现与代码改动:
  - Flow 与 Diffusion 配置加入 change-aware 控制项。
  - 模型侧接入对应参数与逻辑入口。
  - 文档补充配置说明。
- 主要文件:
  - `configs/flow/change_aware_toy.py`
  - `configs/stfdiff/change_aware_toy.py`
  - `stf/models/flow.py`
  - `stf/models/diffusion.py`
- 结果:
  - change-aware 训练已可通过配置开关启用。

### 2026-03-09 13:29:10 +0800 | `c78a2c7`

- 背景/目标:
  - 修正 toy 数据切分，避免 train/val 数据根路径混用。
- 实现与代码改动:
  - 拆分 flow/stfdiff toy 配置中的 train/val root。
- 结果:
  - 验证数据泄漏风险下降，实验可比性更好。

### 2026-03-09 15:36:33 +0800 | `5bf70f3`

- 背景/目标:
  - 增加反映“复制倾向”的指标，便于评估变化区域质量。
- 实现与代码改动:
  - 新增 `TRP` 指标并接入 train/eval 验证日志。
  - 配置默认指标集纳入 `TRP`。
  - 增加对应 smoke 测试。
- 主要文件:
  - `stf/metrics/trp.py`
  - `stf/engine/train.py`
  - `stf/engine/eval.py`
  - `tests/smoke/test_metric_trp.py`
- 结果:
  - 可在验证阶段直接观察复制倾向变化。

### 2026-03-11 10:42:46 +0800 | `22a545d`

- 背景/目标:
  - 统一 diffusion 与 flow 的 `PredTrajNet` 接口，减少分叉维护成本。
- 实现与代码改动:
  - 统一 `PredTrajNet` 前向签名与调用路径。
  - flow 模型改为使用统一接口。
- 主要文件:
  - `stf/models/pred_resnet.py`
  - `stf/models/flow.py`
- 结果:
  - 模型族接口一致性提升，后续扩展高频损失更直接。

### 2026-03-11 18:16:06 +0800 | `1e4755b`

- 背景/目标:
  - 在 Flow 侧引入高频约束，验证对大变化细节恢复的帮助。
- 实现与代码改动:
  - 新增高频损失模块 `hf_losses.py`（梯度/Laplacian/ranking）。
  - 在 `FlowMatching` 家族中接入可选 HF loss 参数（默认关闭，兼容旧配置）。
  - 增加 4 组最小实验矩阵配置（baseline / grad / grad+lap / grad+lap+rank）。
  - 增加配置加载与 HF loss smoke 测试。
- 主要文件:
  - `stf/models/hf_losses.py`
  - `stf/models/flow.py`
  - `configs/flow/change_aware_toy_hf_grad*.py`
  - `tests/smoke/test_flow_hf_losses.py`
- 结果:
  - 高频约束具备可配置、可消融的实验入口。

### 2026-03-11 19:49:48 +0800 | `d57defe`

- 背景/目标:
  - 保证 Gaussian Flow 变体与 HF loss 接口一致。
- 实现与代码改动:
  - 对齐 `GaussianFlowMatching` / `ResidualGaussianFlowMatching` 的 HF 参数与行为。
  - 扩展 smoke 测试覆盖对应变体。
- 结果:
  - Flow 家族 HF loss 接口完成统一。

### 2026-03-12 08:34:26 +0800 | `47cabb4`

- 背景/目标:
  - 统一遥感数据归一化范围，修正默认值不一致风险。
- 实现与代码改动:
  - 将相关配置统一为 `RescaleToMinusOneOne(..., data_range=[0, 10000])`。
  - 同步更新 `README.md` 与 `AGENTS.md` 约定。
- 主要文件:
  - `configs/flow/*.py`
  - `configs/stfdiff/*.py`
  - `README.md`
  - `AGENTS.md`
- 结果:
  - 归一化口径统一，降低实验结果偏差风险。

### 2026-03-12 11:21:47 +0800 | working tree（未提交）

- 背景/目标:
  - 实验结果文件体积较大（`train.log` 与 TensorBoard events），需要可复用的预处理工作流，降低会话上下文成本。
- 实现与代码改动:
  - 新增 `tools/summarize_train_log.py`，将 `train.log` 聚合为 epoch 级摘要。
  - 新增 `tools/export_tb_scalars.py`，导出 TensorBoard scalar 的 tag/step 摘要。
  - 新增流程文档 [programm.md](/home/hang/repos/stf/docs/programm.md)。
  - 更新 `AGENTS.md`，加入“实验结果同步工作流”章节。
- 结果:
  - 后续只需给实验路径映射，即可自动生成小文件并继续做跨实验对比。

### 2026-03-24 14:12:00 +0800 | `cc32fcc`

- 背景/目标:
  - 将 `master` 的性能优化与工程规范同步到 `plan/change-aware-fusion-roadmap`，避免分支能力漂移。
- 实现与代码改动:
  - 执行 `git merge master`，解决冲突文件:
    - `README.md`
    - `stf/models/pred_resnet.py`
    - `tests/smoke/test_config_loader.py`
  - 同步引入：
    - Flow 24G 性能配置与训练性能开关
    - 训练结束显存峰值汇总日志
    - SDPA attention backend 接口
    - `AGENTS.md` 命名规范与 `.gitignore` 整理
    - `setuptools>=68,<81` 及 lock 更新（TensorBoard 兼容）
- 验证:
  - `uv run pytest -q tests/smoke/test_config_loader.py tests/smoke/test_attention_sdpa.py`
  - 结果：`7 passed`
- 结果:
  - `plan` 分支已对齐 `master` 关键优化，后续可直接进入 HF 矩阵实验阶段。

### 2026-03-24 17:23:42 +0800 | `dev/fine-t1-noise-warmup`（进行中）

- 背景/目标:
  - 训练初期模型输出过快贴近 `fine_img_01`，需要强制打断“复制捷径”。
- 实现与代码改动:
  - 在 `TrainConfig` 增加 `fine_t1` 噪声 warmup 参数（epochs/steps/power/std）。
  - 在 `TrainEngine` 训练输入阶段加入 `fine_img_01` 高斯噪声替代并按幂次曲线衰减至 0。
  - 新增两份上机配置：
    - `configs/flow/change_aware_toy_fine_t1_noise_warmup_300.py`
    - `configs/flow/change_aware_toy_fine_t1_noise_warmup_500.py`
  - 新增 smoke 测试覆盖调度与输入替换逻辑。
- 验证:
  - `uv run pytest -q tests/smoke/test_fine_t1_noise_warmup.py tests/smoke/test_config_loader.py tests/smoke/test_train_interval_semantics.py`
  - 结果：`11 passed`
- 下一步:
  - 你上机跑两份配置，确认前期输出是否摆脱贴近 `fine_img_01` 的现象，再决定是否合并主线。

### 2026-03-24 17:45:00 +0800 | `dev/fine-t1-noise-warmup`（进行中）

- 背景/目标:
  - 统一默认 Flow 架构，避免 `FlowMatching` 在主线配置中继续导致过拟合风险。
- 实现与代码改动:
  - 将 `configs/flow/*.py` 的主训练包装器统一切换为 `GaussianFlowMatching`。
  - change-aware 相关配置统一设置 `condition_dropout_p=0.1`；`minimal` 保持 `0.0`。
  - 在 `docs/config.md`、`docs/开发手册.md`、`AGENTS.md` 声明默认架构与历史来源。
- 历史核查（git）:
  - `condition_dropout_p` 功能提交：`9aab002`（`2026-03-05 13:54:18 +0800`）
  - 方案文档记录提交：`0ad3cfe`（`2026-03-04 20:28:46 +0800`）
- 验证:
  - 待本轮改动完成后统一 smoke。

### 2026-03-31 16:12:12 +0800 | `dev/fine-t1-noise-warmup`（进行中）

- 背景/目标:
  - `warmup_200` 仍出现偏参考图与幻觉地物，需要保留少量噪声尾值抑制后期捷径回流。
- 实现与代码改动:
  - 新增 `TrainConfig.fine_t1_noise_alpha_tail`（默认 `0.0`，保持向后兼容）。
  - 训练调度改为从 `alpha=1` 衰减到 `alpha_tail`，warmup 后保持 `alpha_tail` 不再降为 0。
  - 增加参数校验（`alpha_tail` 范围、与 warmup 配置联动）。
  - 补充 smoke 测试覆盖 tail 调度与 warmup 后行为。
- 验证:
  - `uv run pytest -q tests/smoke/test_fine_t1_noise_warmup.py tests/smoke/test_config_loader.py`
  - 结果：`11 passed`

## 4. 已有验证结果（当前可确认）

### 2026-03-12 11:01:48 +0800 | smoke

- 命令:
  - `uv run pytest -q tests/smoke`
- 结果:
  - `13 passed in 3.66s`
- 结论:
  - 当前分支核心 smoke 检查通过，近期改动未破坏基础流程。

### 2026-03-12 11:21:47 +0800 | 日志预处理脚本冒烟验证

- 样例实验:
  - `runs/flow/minimal_20260309-153553`
- 命令:
  - `python tools/summarize_train_log.py runs/flow/minimal_20260309-153553/logs/train.log`
  - `.venv/bin/python tools/export_tb_scalars.py runs/flow/minimal_20260309-153553/tensorboard`
- 结果:
  - 生成 `logs/train.epoch.csv`、`logs/train.epoch.md`
  - 生成 `tensorboard/tb.scalars.step.csv`、`tensorboard/tb.scalars.summary.csv`、`tensorboard/tb.scalars.summary.md`
- 结论:
  - 工作流可直接在实验目录产出紧凑摘要文件，满足后续跨实验对比输入要求。

### 2026-03-12 19:55:04 +0800 | 两组真实实验日志聚合

- 命令:
  - `.venv/bin/python tools/summarize_train_log.py runs/flow/change_aware_cia_gaussianflow_20260311-110448/logs/train.log`
  - `.venv/bin/python tools/summarize_train_log.py runs/flow/change_aware_hf_grad_cia_gaussianflow_20260312-083223/logs/train.log`
- 结果:
  - 两个实验目录均生成 `logs/train.epoch.csv` 与 `logs/train.epoch.md`。
- 结论:
  - 已具备基于 epoch 级摘要做横向定量对比的输入文件。

## 5. 用户实验回传记录（你跑完后我持续补）

> 说明: 该区专门记录你回传给我的实验结果与结论，我会按同一格式持续追加。

### 当前状态

已跑实验（用户回传）:
- `change_aware` -> `runs/flow/change_aware_cia_gaussianflow_20260311-110448`
- `change_aware + hf_grad` -> `runs/flow/change_aware_hf_grad_cia_gaussianflow_20260312-083223`

目视结论（用户）:
- 与 baseline 无明显差别，仍然缺少细节、对变化的精细捕捉。

量化汇总（助手基于 `train.log` 聚合，epoch=999）:

| 指标 | baseline (`change_aware`) | `hf_grad` | 差值 (`hf_grad - baseline`) |
| --- | --- | --- | --- |
| val_loss | 0.0061 | 0.0059 | -0.0002 |
| RMSE | 0.0345 | 0.0342 | -0.0003 |
| MAE | 0.0152 | 0.0142 | -0.0010 |
| PSNR | 30.3219 | 30.3864 | +0.0645 |
| SSIM | 0.8942 | 0.8834 | -0.0108 |
| ERGAS | 3.2920 | 3.1837 | -0.1083 |
| CC | 0.7588 | 0.7431 | -0.0157 |
| SAM | 0.0947 | 0.0748 | -0.0199 |
| UIQI | 0.7458 | 0.7354 | -0.0104 |
| TRP | -0.1533 | -0.1971 | -0.0438 |

阶段判断（当前两组）:
- `hf_grad` 在 `val_loss/RMSE/MAE/PSNR/ERGAS/SAM` 上有改善。
- 但在 `SSIM/CC/UIQI/TRP` 上回退，且用户目视“无明显提升”。
- 结论: 当前 `hf_grad` 默认参数尚不足以作为推荐默认配置，需要继续做 `grad+lap` / `grad+lap+rank` 与权重调参。

### 新增回传（2026-03-31）

实验:
- `warmup_200` -> `runs/flow/change_aware_fine_t1_noise_warmup_200_cia_20260326-230345`

用户目视结论:
- 噪声减小后清晰度恢复较快。
- 细节仍偏向参考图像（`fine_t1`），且出现“本不存在地物块”的幻觉现象。

量化对比（同窗口 500 epoch，`epoch=499`）:
- warmup_200:
  - `val_loss=0.0059`, `RMSE=0.0336`, `MAE=0.0133`, `PSNR=30.6356`
  - `SSIM=0.9006`, `CC=0.7582`, `UIQI=0.7479`, `TRP=-0.1800`
- baseline (`change_aware_cia_gaussianflow`, `epoch=499`):
  - `val_loss=0.0062`, `RMSE=0.0352`, `MAE=0.0155`, `PSNR=30.0734`
  - `SSIM=0.8962`, `CC=0.7485`, `UIQI=0.7351`, `TRP=-0.1792`

趋势判断:
- warmup_200 在前期（~`epoch<=200`）指标明显劣化，后期逐步追回并在多数指标上超过 baseline。
- `TRP` 基本持平略差（`-0.1800` vs `-0.1792`），说明“复制倾向/伪变化”问题未被根治。
- 与用户目视一致：该策略改善了训练路径与后期清晰度，但对语义真实性约束不足，仍可能诱发幻觉块。

后续计划（基于本轮结果）:
1. 继续保留 warmup 思路，但改为“非零尾值”调度（不让 `alpha` 直接降到 0）。
2. 在 warmup 基础上叠加轻量一致性约束（优先 coarse/change-aware 约束），抑制幻觉地物。
3. 开一组短跑 A/B（`warmup200` vs `warmup200+tail` vs `warmup200+tail+consistency`）先看前 300 epoch 视觉与 TRP。

### 回传模板（建议）

- 时间:
- 配置:
- 数据集与切分:
- 关键超参:
- 主要指标（RMSE/MAE/SSIM/SAM/CC/TRP）:
- 现象（尤其 high-change 桶）:
- 与 baseline 对比:
- 结论:
- 下一步建议:

## 6. 阶段总结（截至 2026-03-24）

- 已完成:
  - change-aware 基础改造已落地（flow + diffusion）。
  - 评价体系补充了 `TRP`，支持复制倾向分析。
  - Flow 高频约束接口和 4 组最小实验矩阵已落地。
  - Flow 家族接口（含 gaussian 变体）已统一。
  - 归一化口径统一为 `data_range=[0, 10000]`。
  - 新增实验结果预处理脚本（`train.log` epoch 聚合 + TensorBoard scalar 摘要）。
  - 新增可复用流程文档 `docs/programm.md` 并同步到 `AGENTS.md`。
  - 已完成两组真实实验对比（`change_aware` vs `change_aware + hf_grad`）并沉淀量化结果。
  - `plan` 分支已合入 `master` 性能优化链路（`perf_24g`/SDPA/显存摘要/TensorBoard 兼容修复）。
- 仍待完成:
  - 继续补齐 `hf_grad_lap`、`hf_grad_lap_rank` 两组，形成 4 组完整矩阵对比。
  - 补充 high-change 分桶指标，验证是否缓解“复制 `fine_t1`”问题。
  - Diffusion 侧是否迁移同等 HF 约束，等待 Flow 结论。

## 7. 后续计划（短期）

1. 跑完并汇总 4 组 Flow 高频约束实验（含 TRP 与 high-change 分桶）。
2. 以当前两组结果为基线，优先验证 `grad+lap` 与 `grad+lap+rank` 是否能修复 `SSIM/CC/UIQI/TRP` 回退。
3. 对比 baseline，确认是否达成“high-change 至少 5% 改善、overall 不明显回退”。
4. 固化一版推荐默认配置（含权重与 mask 策略）；若稳定再迁移到 Diffusion 侧。
