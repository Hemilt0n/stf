# AGENTS Handoff

本文件用于会话交接与快速恢复上下文。新一轮开发前，请先阅读本文件与 `README.md`。

## 0. 文件命名约定

- 本仓库统一使用 `AGENTS.md`（复数）作为会话上下文入口文档。
- 旧文件名 `AGENT.md` 已废弃，不再维护。

## 1. 当前状态（2026-03-24）

- 仓库路径: `/home/hang/repos/stf`
- 当前分支: `dev/fine-t1-noise-warmup`（基于 `plan/change-aware-fusion-roadmap`）
- 远程仓库: `origin = https://github.com/Hemilt0n/stf.git`
- 已合入 `master` 的关键提交:
  - `677dbde`（合并 `research/perf-24g-flow`）
  - `5eeadef`（统一 `data_range=[0, 10000]`）

## 2. 已落地的性能优化（master）

- 训练侧新增性能开关（`TrainConfig`）:
  - `precision` (`fp16` / `bf16`)
  - `enable_tf32`
  - `deterministic`, `cudnn_benchmark`
  - `non_blocking_transfer`
  - `train_log_interval`
  - `compile_model`, `compile_mode`, `compile_dynamic`
  - `use_channels_last`
  - `fine_t1_noise_warmup_epochs`, `fine_t1_noise_warmup_steps`
  - `fine_t1_noise_power`, `fine_t1_noise_std`, `fine_t1_noise_alpha_tail`
- `TrainEngine` 已支持:
  - `torch.amp.autocast` + `torch.amp.GradScaler`
  - `bf16` 训练路径
  - 训练结束自动输出 `gpu memory summary`（峰值显存/占比/剩余）
- Flow 性能配置:
  - `configs/flow/change_aware_perf_24g.py`
  - `configs/flow/change_aware_perf_24g_compile.py`
- SDPA 接口已接入并保留后端开关:
  - `attention_backend in {'auto','sdpa','classic'}`
  - 代码位于 `stf/models/unet.py` 与 `stf/models/pred_resnet.py`

## 3. 最新实验结论（同步）

- `perf_24g` 实测（32G）:
  - `peak_allocated=21366.58 MiB (66.30%)`
- 结论:
  - 当前配置下 24G 目标已达成（有余量）。
- `perf_24g_compile`:
  - 编译初始化时间过长，当前阶段暂停。
- 当前实测配置:
  - `batch_size=32`
  - 主训练路径（`pred_resnet` 的 `PredTrajNet`）目前仍未真正启用主干 attention 计算块。

## 3.1 默认架构声明（2026-03-24）

- Flow 默认训练包装器：`GaussianFlowMatching`（用于主线实验配置）。
- `FlowMatching` 仅保留用于消融与调试，不作为默认主线。
- 历史记录：
  - `condition_dropout_p` 首次引入提交：`9aab002`（`2026-03-05 13:54:18 +0800`）
  - 对应文档方案记录：`0ad3cfe`（`2026-03-04 20:28:46 +0800`）

## 4. 分支协作与同步规范

- `AGENTS.md` 建议作为“主干基线文档”:
  - 所有稳定结论先更新到 `master`。
  - 特定分支实验细节（临时参数、中间失败记录）保留在分支文档或 commit 信息中。
- 不同分支是否完全一致:
  - 不要求逐字一致。
  - 但必须保持: 当前状态、关键入口命令、硬约束接口、已知风险这四项一致。

### master 优化迁移到 plan 分支（标准做法）

优先级 1（推荐）: 在 `plan/*` 上合并 `master`

```bash
git checkout plan/<branch>
git pull --ff-only origin plan/<branch>
git merge master
```

优先级 2（选择性迁移）: 仅挑 commit `cherry-pick`

```bash
git checkout plan/<branch>
git cherry-pick <commit1> <commit2> ...
```

说明:
- 当 `master` 改动是一组相关能力（本次性能开关 + 显存日志 + SDPA 接口）时，优先 merge，避免漏依赖。
- 仅当 plan 分支需要严格控变更面时，才用 cherry-pick。

## 4.1 文档治理（强制）

- 凡是涉及**架构/接口**的改动（模型包装器切换、`forward/sample` 签名、训练输入语义、配置字段新增/弃用），必须在同一提交链路中同步更新文档。
- 最低要求：
  - 在 `docs/progress.md` 追加一条带时间戳的变更记录（改了什么、为什么、影响范围、验证结果）。
  - 在 `AGENTS.md` 同步“当前默认架构/关键接口约束/分支状态”。
- 未更新文档的架构/接口改动视为不完整交付。

## 4.2 文档分工（master vs 分支）

- `AGENTS.md`（入口/会话恢复，主干基线）:
  - 记录当前默认架构、关键接口约束、跨分支同步规则、关键命令。
  - `master` 上维护“稳定事实”；分支上可临时追加“当前实验偏离点”。
- `docs/progress.md`（时间线与实验账本）:
  - 记录每次实现、验证、用户回传结果和阶段结论。
  - 分支可记录进行中的实验细节；合并后将稳定结论回写到主干时间线。
- `docs/programm.md`（实验结果同步流程）:
  - 只描述“结果如何回传与汇总”的流程规范（`train.log` 聚合为主，TB 可选）。
  - 与具体模型架构解耦，不承载架构决策。
- `docs/config.md` / `docs/开发手册.md`（规范与开发说明）:
  - 前者写配置语义和默认值，后者写开发约束与默认技术路线。
  - 当默认架构发生变化时，两者必须同步更新。

## 5. Git 操作注意事项（重要）

- 在当前 CLI 环境中，`git` 写操作经常需要提权。
- 以下命令建议直接按提权方式运行，避免先失败再重试:
  - `git checkout ...`
  - `git add/commit`
  - `git merge`
  - `git pull`
  - `git push`

## 6. 快速验证命令

```bash
uv run python -m compileall -q stf/models
uv run pytest -q tests/smoke
```

训练入口:

```bash
uv run stf train --config configs/flow/change_aware_toy.py
uv run stf train --config configs/flow/change_aware_perf_24g.py
uv run stf train --config configs/flow/change_aware_toy_fine_t1_noise_warmup_300.py
uv run stf train --config configs/flow/change_aware_toy_fine_t1_noise_warmup_500.py
```
