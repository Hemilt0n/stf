# STF 项目进展记录

最后更新: 2026-03-12 11:21:47 +0800  
当前分支: `plan/change-aware-fusion-roadmap`

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
  - 同步更新 `README.md` 与 `AGENT.md` 约定。
- 主要文件:
  - `configs/flow/*.py`
  - `configs/stfdiff/*.py`
  - `README.md`
  - `AGENT.md`
- 结果:
  - 归一化口径统一，降低实验结果偏差风险。

### 2026-03-12 11:21:47 +0800 | working tree（未提交）

- 背景/目标:
  - 实验结果文件体积较大（`train.log` 与 TensorBoard events），需要可复用的预处理工作流，降低会话上下文成本。
- 实现与代码改动:
  - 新增 `tools/summarize_train_log.py`，将 `train.log` 聚合为 epoch 级摘要。
  - 新增 `tools/export_tb_scalars.py`，导出 TensorBoard scalar 的 tag/step 摘要。
  - 新增流程文档 [programm.md](/home/hang/repos/stf/docs/programm.md)。
  - 更新 `AGENT.md`，加入“实验结果同步工作流”章节。
- 结果:
  - 后续只需给实验路径映射，即可自动生成小文件并继续做跨实验对比。

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

## 5. 用户实验回传记录（你跑完后我持续补）

> 说明: 该区专门记录你回传给我的实验结果与结论，我会按同一格式持续追加。

### 当前状态

- 暂无你回传的训练/评估实测结果。

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

## 6. 阶段总结（截至 2026-03-12）

- 已完成:
  - change-aware 基础改造已落地（flow + diffusion）。
  - 评价体系补充了 `TRP`，支持复制倾向分析。
  - Flow 高频约束接口和 4 组最小实验矩阵已落地。
  - Flow 家族接口（含 gaussian 变体）已统一。
  - 归一化口径统一为 `data_range=[0, 10000]`。
  - 新增实验结果预处理脚本（`train.log` epoch 聚合 + TensorBoard scalar 摘要）。
  - 新增可复用流程文档 `docs/programm.md` 并同步到 `AGENT.md`。
- 仍待完成:
  - 你回传完整实验结果后，做阶段性对比总结并固化默认推荐配置。
  - Diffusion 侧是否迁移同等 HF 约束，等待 Flow 结论。

## 7. 后续计划（短期）

1. 跑完并汇总 4 组 Flow 高频约束实验（含 TRP 与 high-change 分桶）。
2. 对比 baseline，确认是否达成“high-change 至少 5% 改善、overall 不明显回退”。
3. 固化一版推荐默认配置（含权重与 mask 策略）。
4. 若 Flow 结论稳定，将 HF 约束并行迁移到 Diffusion 侧。
