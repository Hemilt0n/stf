# STF 项目进展记录

最后更新: 2026-04-19 11:29:50 +0800  
当前分支: `feat/geo-edit-residual-flow-stage1`

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

### 2026-04-13 10:20:00 +0800 | `feat/repo-private-remote-run-skill`（进行中）

- 背景/目标:
  - 本机仅用于调研、改代码与轻量测试，真实训练需要在远程 GPU 服务器执行。
  - 需要一套仓库内可恢复、低上下文污染的远程训练工作流。
- 实现与代码改动:
  - 新增仓库私有 skill: `.codex/skills/remote-train-orchestrator/`。
  - 新增 skill 主文档 `SKILL.md`，约定远程预检、远程配置 staging、`tmux` 启动、记录恢复、结果汇报流程。
  - 新增 `references/observed-environment.md`，记录当前远程主机、`data/`、`runs/`、`tmux`、CIA 数据根等已核对事实。
  - 新增 `scripts/remote_train.sh`，提供 `inspect` / `launch` / `status` 三个子命令。
  - 本地运行记录约定为 `log/remote_runs/records/*.json` 与 `log/remote_runs/experiments.md`。
- 验证:
  - `python3 /home/hang/.codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/remote-train-orchestrator`
  - 结果：`Skill is valid!`
  - 使用无害 smoke run 验证 `tailscale ssh` 下发配置、远程 `tmux`、本地 record/ledger、`status` 恢复链路。
- 总结与下一步:
  - 仓库内已具备可复用的远程训练托管流程。
  - 下一步优先补充 ETA 估算与远程通知钩子。

### 2026-04-13 10:57:38 +0800 | `feat/repo-private-remote-run-skill`（远程实训）

- 背景/目标:
  - 用新 skill 实跑一次真实远程训练，验证端到端流程不是只停留在 smoke。
  - 配置参考 `configs/flow/template_all_options.py`，数据集使用 CIA，训练 10 epochs。
- 实现与代码改动:
  - 基于模板配置生成远程专用本地配置 `configs/remote/cia_template_10ep.py`（不纳入 git）。
  - 通过 skill 在远程生成 staged config:
    - `/home/hang/repos/stf/configs/remote/cia_template_10ep__20260413-105738.py`
  - 训练会话:
    - `tmux session = stf-flow-cia-template-10ep-20260413-105738`
  - 本地状态记录:
    - `log/remote_runs/records/stf-flow-cia-template-10ep-20260413-105738.json`
    - `log/remote_runs/experiments.md`
- 预检与问题处理:
  - 发现本地与远程 git head 不一致。
  - 处理方式：远程仓库拉到当前 `origin/master` 并与本地跟踪 head 对齐后再启动。
  - 核对远程 CIA 数据根为 `data/CIA/band4_serialized`。
- 结果:
  - 最终运行目录:
    - `/home/hang/repos/stf/runs/flow/cia_template_10ep_20260413-105742`
  - `epoch=9` 验证结果:
    - `val loss=0.0176`
    - `RMSE=0.0655`
    - `MAE=0.0411`
    - `PSNR=23.7844`
    - `SSIM=0.3856`
    - `ERGAS=6.2441`
    - `CC=0.3182`
    - `SAM=0.2484`
    - `UIQI=0.2648`
    - `TRP=-0.0284`
  - GPU 显存摘要:
    - `peak_allocated=10726.43 MiB (65.50%)`
    - `peak_reserved=11964.00 MiB (73.06%)`
- 总结与下一步:
  - 远程训练 skill 已通过真实 10 epoch CIA 运行验证。
  - 下一步可在 `status` 基础上增加 ETA 估算，以及训练结束通知钩子。

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

### 新增回传（2026-04-01 09:40 +0800）| 严格配对 baseline 复核

实验:
- baseline（严格配对）: `runs/flow/change_aware_fine_t1_baseline_matched_cia_20260331-173119`
- warmup: `runs/flow/change_aware_fine_t1_noise_warmup_200_cia_20260326-230345`

对齐方式:
- 两边均按 `tools/summarize_train_log.py` 聚合为 `train.epoch.csv`。
- 仅比较共同验证点（`epoch=49,99,...,499`），控制变量一致，仅差异为 `fine_t1` 噪声 warmup/tail 相关参数。

量化结论（`epoch=499`）:
- baseline:
  - `val_loss=0.0058`, `RMSE=0.0334`, `MAE=0.0134`, `PSNR=30.7065`
  - `SSIM=0.9010`, `CC=0.7614`, `ERGAS=3.1436`, `SAM=0.0642`, `UIQI=0.7514`, `TRP=-0.1808`
- warmup_200:
  - `val_loss=0.0059`, `RMSE=0.0336`, `MAE=0.0133`, `PSNR=30.6356`
  - `SSIM=0.9006`, `CC=0.7582`, `ERGAS=3.1486`, `SAM=0.0638`, `UIQI=0.7479`, `TRP=-0.1800`

趋势判断:
- warmup_200 在前 200 epoch 明显更慢（前期 `val_loss/RMSE/PSNR` 全部落后），后期逐步追回。
- 到末轮多数指标 baseline 略优（`val_loss/RMSE/PSNR/SSIM/CC/ERGAS/UIQI`），warmup 在 `MAE/SAM/TRP` 略优。
- 说明 warmup 参数对学习动力学“有效”（显著改变前中期收敛路径），但在这组严格配对实验中未转化为整体最终指标优势。

阶段结论（本轮复核）:
- “warmup 有效”应定义为: 能抑制早期捷径依赖并改变收敛轨迹，而非保证最终 scalar 全面提升。
- 若目标是最终指标超越 baseline，下一步应重点验证 `warmup + tail + 一致性约束` 的组合，而非单独 warmup。

补充对比（2026-04-01 10:05 +0800）| `warmup_200` vs `warmup_200_tail`:
- tail 实验: `runs/flow/change_aware_fine_t1_noise_warmup_200_tail_20260331-225043`
- 对齐口径: 两边都在 `epoch=49,99,...,499` 做验证点对齐比较。
- `epoch=499`:
  - `warmup_200` 更优: `val_loss/RMSE/MAE/PSNR/SSIM/ERGAS/CC/SAM/UIQI`
  - `warmup_200_tail` 仅 `TRP` 更优（`-0.1668` vs `-0.1800`）
- 结论:
  - 当前 `fine_t1_noise_alpha_tail=0.1` 对主质量指标是负收益，不能作为默认推荐。
  - tail 机制仍有价值（TRP 改善），但需要下调尾值强度或配合一致性约束后再评估。

### 2026-04-02 10:45:00 +0800 | val step 抖动排查与可观测性增强

- 背景/目标:
  - 用户反馈同一 `val_epoch` 内不同 `iter` 指标波动明显，需要判断是否正常并可定位到具体样本批次。
- 实现与代码改动:
  - `TrainConfig` 新增可选调试项:
    - `val_step_log_keys`（按 iter 记录 `sample_idx/key`）
    - `val_step_log_max_keys`（日志 key 预览上限）
    - `val_step_save_csv`（将每个 val iter 的 loss/metrics/key 落盘到 `runs/.../debug`）
  - `TrainEngine._run_val_epoch` 接入上述开关，按 epoch 生成:
    - `debug/val_step_epoch_XXXX.csv`
  - 新增离线脚本:
    - `tools/dump_sampler_layout.py`
    - 可基于 config + sampler + epoch 导出批次相对位置到 `runs/debug/*.csv`（无需加载图像张量）。
- 结果:
  - 可直接将异常波动的 `val_step` 映射到具体批次位置与样本 key，区分“难样本分布”与“训练数值异常”。

### 新增回传（2026-04-16 16:45:50 +0800）| `ResidualGaussianFlowMatching` 对历史 Gaussian baseline

实验:
- 历史 Gaussian baseline: `runs/flow/change_aware_fine_t1_baseline_matched_cia_20260331-173119`
- Residual: `runs/residual-flow/cia_compare_residualgaussianflow_500ep_cv_20260414-143621`

实验口径:
- 数据根使用 `data/CIA/band4_serialized`。
- Residual 实验配置:
  - `task="residual-flow"`
  - `show_images=True`
  - `show_bands=(0, 1, 2)`
  - `seed=42`
  - `fine_t1` warmup / tail 关闭
- 为避免重复计算，本轮未重跑新的 Gaussian 对照，而是直接复用历史 baseline。

量化对比（`epoch=499`）:
- Gaussian baseline:
  - `val_loss=0.0058`, `RMSE=0.0334`, `MAE=0.0134`, `PSNR=30.7065`
  - `SSIM=0.9010`, `CC=0.7614`, `ERGAS=3.1436`, `SAM=0.0642`, `UIQI=0.7514`, `TRP=-0.1808`
- Residual:
  - `val_loss=0.0058`, `RMSE=0.0334`, `MAE=0.0134`, `PSNR=30.7315`
  - `SSIM=0.9036`, `CC=0.7600`, `ERGAS=3.1385`, `SAM=0.0669`, `UIQI=0.7481`, `TRP=-0.1757`

差异总结:
- Residual 略优:
  - `PSNR` `+0.0250`
  - `SSIM` `+0.0026`
  - `ERGAS` `-0.0051`
  - `TRP` `+0.0051`（更接近 0）
- Gaussian baseline 略优:
  - `CC` `+0.0014`
  - `SAM` `-0.0027`
  - `UIQI` `+0.0033`
- `val_loss/RMSE/MAE` 基本持平。

显存摘要:
- Gaussian baseline:
  - `peak_allocated=10726.43 MiB`, `peak_reserved=11984.00 MiB`
- Residual:
  - `peak_allocated=10726.43 MiB`, `peak_reserved=12194.00 MiB`
- 结论:
  - 两者显存占用同量级，Residual 的 `peak_reserved` 略高。

结论与限制:
- 当前结果可表述为：`ResidualGaussianFlowMatching` 在这轮实际对比中总体持平略优，至少没有输给历史 Gaussian baseline。
- 但这不是严格纯控制变量结论：
  - 历史 Gaussian baseline 含 `condition_dropout_p=0.1`
  - 本轮 Residual 按实验设计为 `dropout=0`
- 因此当前更稳妥的结论是：
  - Residual 具备进入后续实验主线的价值；
  - 若要把优势严格归因到 wrapper 本身，仍需补一组同口径 Gaussian 重跑来封口。

### 2026-04-02 11:20:00 +0800 | 离线序列化数据预处理与自动识别

- 背景/目标:
  - 当前 `is_serialize_data` 仅优化样本元信息访问，无法消除影像文件读取/解码开销。
  - 需要提供离线预处理能力，并在训练侧自动兼容 raw/serialized 数据目录。
- 实现与代码改动:
  - 新增脚本:
    - `tools/serialize_dataset.py`
    - 将 `input-root` 下指定后缀（默认 `.tif`）离线转换为 `.npy/.npz`，并保持目录结构与文件 stem。
    - 在输出根目录写入 `/.stf_serialized.json` 标记（包含 `data_suffix` 等元信息）。
  - `LoadData` 读取扩展:
    - `.tif` -> `tifffile.imread`
    - `.npy/.npz` -> `numpy.load`
  - 数据集自动识别:
    - `SpatioTemporalFusionDataset` / `SpatioTemporalFusionDatasetForSPSTFM` 在数据根存在 marker 时，优先 marker 指定后缀，避免多后缀共存时采样歧义。
- 验证:
  - 新增 `tests/smoke/test_data_serialization_io.py`（格式读取兼容 + marker 后缀优先）。
  - 新增 `tests/smoke/test_serialize_dataset_tool.py`（脚本输出与 marker 冒烟）。

补充（2026-04-02 11:35:00 +0800）| 目录规范化:
- 将脚本默认输出规则改为同级 `_serialized` 目录，便于 dataroot 切换：
  - 单 split: `<root>/train` -> `<root>_serialized/train`
  - 多 split: `<dataset>/{train,val,test}` + `--splits train,val,test`
    -> `<dataset>_serialized/{train,val,test}`
- 该规范已写入 `docs/config.md` 与 `AGENTS.md`。

### 2026-04-02 12:05:00 +0800 | `ExperimentConfig.task` 放开为自定义字符串

- 背景/目标:
  - 用户希望 `ExperimentConfig.task` 不再被固定白名单限制，支持自定义命名。
- 实现与代码改动:
  - 移除 `stf/config/loader.py` 中对 `task in {'flow','stfdiff'}` 的强校验。
  - 保留文档建议约定（推荐继续使用 `flow` / `stfdiff`），但框架层不强制。
  - 补充 smoke 测试 `tests/smoke/test_config_loader.py::test_load_config_with_custom_task`。
- 验证:
  - `test_config_loader` 覆盖默认配置与自定义 `task` 场景。

### 2026-04-02 12:25:00 +0800 | 新增全量模板配置（含性能/debug分类）

- 背景/目标:
  - 避免新分支/新实验遗漏新增配置项，提供统一模板作为复制起点。
- 实现与代码改动:
  - 新增 `configs/flow/template_all_options.py`：
    - 覆盖当前常用 `ExperimentConfig` / `TrainConfig` 选项。
    - 按“基础 / 性能 / debug”分组注释。
    - 性能相关与 debug 相关默认关闭。
  - 补充 `test_config_loader` 冒烟，确保模板可加载且默认开关语义正确。
  - 在 `docs/config.md`、`AGENTS.md` 补充模板入口说明。
- 验证:
  - `tests/smoke/test_config_loader.py` 覆盖模板配置加载。

补充（2026-04-02 12:38:00 +0800）| 模板默认值修正:
- 根据需求，将模板中的性能项默认值改为“开启”，并对齐 `configs/flow/change_aware_perf_24g.py`：
  - dataloader: `num_workers/pin_memory/persistent_workers/prefetch_factor`
  - train: `use_mixed_precision=True`, `precision="bf16"`, `enable_tf32=True`,
    `deterministic=False`, `cudnn_benchmark=True`, `non_blocking_transfer=True`,
    `use_channels_last=True`
- debug 项默认保持关闭。

### 2026-04-02 15:10:00 +0800 | warmup 机制最小移植到主干（默认关闭）

- 背景/目标:
  - 在主干保留 `fine_t1` 噪声 warmup 能力，便于后续按需启用实验；默认不启用，不改变现有训练行为。
- 实现与代码改动:
  - `TrainConfig` 新增 warmup 字段:
    - `fine_t1_noise_warmup_epochs`
    - `fine_t1_noise_warmup_steps`
    - `fine_t1_noise_power`
    - `fine_t1_noise_std`
    - `fine_t1_noise_alpha_tail`
  - `TrainEngine` 接入 warmup 调度与输入替换:
    - 训练输入阶段对 `fine_img_01` 按调度注入高斯噪声
    - 记录 `train/fine_t1_noise_alpha` 到后端日志
    - warmup 相关参数校验与 tail 约束
  - 模板配置 `configs/flow/template_all_options.py` 补齐 warmup 选项（默认关闭）。
  - 新增 smoke: `tests/smoke/test_fine_t1_noise_warmup.py`。
- 验证:
  - `tests/smoke/test_fine_t1_noise_warmup.py`
  - `tests/smoke/test_config_loader.py`

### 2026-04-02 15:40:00 +0800 | 新增梯度累积配置（默认等价关闭）

- 背景/目标:
  - 支持 `micro-batch + grad accumulation` 训练形态，同时保证默认行为不变。
- 实现与代码改动:
  - `TrainConfig` 增加 `grad_accum_steps`（默认 `1`）。
  - 训练循环改为按 `grad_accum_steps` 控制 optimizer step/update 频率。
  - 当 `grad_accum_steps=1` 时行为与原先逐 iter 更新等价。
  - 模板配置 `configs/flow/template_all_options.py` 补充该字段。
- 验证:
  - 新增 `tests/smoke/test_train_grad_accum.py`，覆盖：
    - `grad_accum_steps=1` 时 optimizer.step 次数与 iter 数一致（等价不启用）。
    - `grad_accum_steps>1` 时步数按累积逻辑执行。
  - 兼容回归：`tests/smoke/test_config_loader.py`。

### 2026-04-13 18:30:00 +0800 | 远程 compare 编排改为“每 config 一个 tmux session，helper 串行调度”

- 背景/目标:
  - 先前远程 compare 流程容易退化为“多次 `launch` -> 多个 session 并发抢同一 GPU”。
  - 目标是保留“每个 config 各自独立 tmux session、名称可恢复”的优点，同时从 helper 层保证串行，不再依赖 skill 文本约束或临时自定义 `--remote-command`。
- 实现与代码改动:
  - `.codex/skills/remote-train-orchestrator/scripts/remote_train.sh` 现支持重复 `--config`。
  - 单 config 启动行为保持原样。
  - 多 config 启动时:
    - helper 先统一 stage 所有 config 到远程；
    - 启动一个 queue/controller session；
    - controller 顺序拉起每个 config 自己的命名 `tmux session`；
    - 任意时刻仅允许一个 child config session 运行；
    - 本地 record 增补 queue/session/config list、active child session、sequential confirmation 等字段。
  - `.codex/skills/remote-train-orchestrator/SKILL.md` 与 `AGENTS.md` 同步更新为新的 helper 语义与推荐调用。
- 验证:
  - `bash -n .codex/skills/remote-train-orchestrator/scripts/remote_train.sh`

补充（2026-04-14 20:00:00 +0800）| 远程 helper 增加 `prepare/launch-prepared` 与保留完成 session:
- 背景/目标:
  - 用户希望在真正发车前先停在 staging 阶段，手动检查或微调，再让 agent 执行。
  - 用户还希望训练结束后不要立刻丢失 `tmux` 现场，便于人工复核最后输出。
- 实现与代码改动:
  - `remote_train.sh` 新增:
    - `prepare --config ... --purpose ...`
    - `launch-prepared --session ...`
  - `prepare` 只做 preflight/staging/record，不启动任何远程训练 session。
  - `launch-prepared` 从 `records/<session>.json` 恢复已准备的远程命令与 staged config 列表，再真正启动。
  - 新增 `--keep-finished-session`:
    - 单 config session 可保留；
    - 多 config compare 中的 controller / child session 可保留；
    - helper 等待逻辑从“session 消失”改为“exit file 出现或 session 已退出”，避免 `remain-on-exit` 卡住串行队列。
- 影响:
  - 远程执行更适合“先准备、后确认、再执行”的协作节奏。
  - 结束后的 `tmux` pane 可被用户手动检查，不再必须依赖日志文件回溯。

补充（2026-04-14 20:20:00 +0800）| 本地 `configs/remote` 增加最小状态分类:
- 背景/目标:
  - 当远程配置变多时，仅靠文件名在 `configs/remote/` 根目录中查找，不利于用户快速定位“待检查 / 正在执行 / 已完成”的配置。
- 实现与代码改动:
  - helper 现在自动维护三个本地目录：
    - `configs/remote/review/<session>/`
    - `configs/remote/running/<session>/`
    - `configs/remote/completed/<session>/`
  - `prepare` 将本地配置快照到 `review/`；
  - `launch` / `launch-prepared` 将实际执行版本快照到 `running/`；
  - `status` 在终态时把本地执行快照归档到 `completed/`。
- 约束:
  - 不额外引入更复杂的状态层级，避免目录状态机膨胀。

### 2026-04-16 18:49:25 +0800 | `feat/geo-edit-residual-flow-stage1`（进行中）

- 背景/目标:
  - 基于“地理对齐下的局部编辑 + 全局季节迁移”叙事，先落地 `Geo-Edit Residual Flow` 的 Stage 1。
  - 第一阶段只改 `ResidualGaussianFlowMatching` 的起点分布，不提前重写 backbone。
- 实现与代码改动:
  - `stf/models/flow.py`
    - 新增 `build_soft_change_map(...)`，由 `|coarse_t2 - coarse_t1|` 构造软变化图。
    - `ResidualGaussianFlowMatching` 新增可选 geo-edit 参数：
      - `geo_edit_enabled`
      - `geo_edit_sigma_low`
      - `geo_edit_sigma_high`
      - `geo_edit_mask_power`
      - `geo_edit_mask_smooth_kernel`
    - 新增 `_build_residual_start_distribution(...)`：
      - 默认关闭时保持旧行为：`z_mean = coarse_weight * coarse_delta`，`sigma = noise_std`
      - 开启后改为“空间变化感知”的起点分布：
        - 不变区接近 identity residual，低噪声
        - 变化区接近 `coarse_delta`，高噪声
  - `tests/smoke/test_geo_edit_residual_flow.py`
    - 覆盖无变化软掩码、局部变化高亮、legacy 路径等价、geo-edit 路径空间变 `sigma/z_mean`
- 验证:
  - `python -m compileall -q stf/models`
  - `uv run pytest -q tests/smoke/test_geo_edit_residual_flow.py`
  - 结果：`4 passed`
- 阶段结论:
  - Stage 1 已具备最小可运行实现，且默认行为保持向后兼容。
  - 下一步可在实验配置中显式启用 geo-edit 参数，验证其是否优于统一高斯起点与全局 warmup。

补充（2026-04-19 11:29:50 +0800）| `Geo-Edit Residual Flow` Stage 1 首轮 500 epoch 结果:
- 实验:
  - geo_edit stage1: `runs/geo_edit/cia_compare_residualgaussianflow_geo_edit_stage1_500ep_20260416-190231`
  - residual baseline: `runs/residual-flow/cia_compare_residualgaussianflow_500ep_cv_20260414-143621`
  - 历史 Gaussian baseline: `runs/flow/change_aware_fine_t1_baseline_matched_cia_20260331-173119`
- 对齐口径:
  - 数据根统一为 `data/CIA/band4_serialized`
  - `seed=42`
  - optimizer / dataloader / train switches 与 residual baseline 保持一致
  - 仅额外开启:
    - `geo_edit_enabled=True`
    - `geo_edit_sigma_low=0.1`
    - `geo_edit_sigma_high=1.0`
    - `geo_edit_mask_power=1.0`
    - `geo_edit_mask_smooth_kernel=3`
- `epoch=499` 量化结果:
  - geo_edit stage1:
    - `val_loss=0.0059`, `RMSE=0.0334`, `MAE=0.0133`, `PSNR=30.7446`
    - `SSIM=0.9014`, `CC=0.7542`, `ERGAS=3.1281`, `SAM=0.0651`, `UIQI=0.7428`, `TRP=-0.1597`
  - residual baseline:
    - `val_loss=0.0058`, `RMSE=0.0334`, `MAE=0.0134`, `PSNR=30.7315`
    - `SSIM=0.9036`, `CC=0.7600`, `ERGAS=3.1385`, `SAM=0.0669`, `UIQI=0.7481`, `TRP=-0.1757`
- 差异总结:
  - geo_edit stage1 略优:
    - `MAE`
    - `PSNR`
    - `ERGAS`
    - `SAM`
    - `TRP`（改善最明显，`-0.1757 -> -0.1597`）
  - residual baseline 略优:
    - `val_loss`
    - `SSIM`
    - `CC`
    - `UIQI`
  - `RMSE` 基本持平。
- 阶段判断:
  - 这不是失败实验，说明“局部编辑式 residual flow 起点”方向成立。
  - 最强正信号是 `TRP` 明显改善，说明复制旧时相的倾向被有效削弱。
  - 但 `SSIM/CC/UIQI` 回退，说明当前 edit mask 或 spatial noise 仍然过宽，低变化区域一致性受到影响。
- 下一步计划（Stage 1.1）:
  1. 跑 `Sharper Mask`：
     - `geo_edit_mask_smooth_kernel=1`
     - `geo_edit_mask_power=1.5`
  2. 跑 `Sharper Mask + Stronger Edit`：
     - `geo_edit_mask_smooth_kernel=1`
     - `geo_edit_mask_power=1.5`
     - `geo_edit_sigma_low=0.0`
     - `geo_edit_sigma_high=1.2`
  3. 目标:
     - 尽量保住 `TRP` 改善
     - 同时把 `SSIM/CC/UIQI` 拉回到 baseline 附近
  4. 在 Stage 1.1 没做完之前，不进入 `trust map` / `dual-head`
     - 先把 Stage 1 的 edit 区域控制收窄，再判断是否需要进入 Stage 2

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

1. 完成 `Geo-Edit Residual Flow` Stage 1.1 两组定向参数收缩实验：
   - `Sharper Mask`
   - `Sharper Mask + Stronger Edit`
2. 重点观察 `TRP` 是否保持改善，同时 `SSIM/CC/UIQI` 是否回升。
3. 若 Stage 1.1 成立，再进入 Stage 2 `Condition Trust Map`。
4. 若 Stage 1.1 仍旧表现为“TRP 改善但结构一致性回退”，再考虑把 `Boundary/Topology` 约束提前，而不是立即上双头。
