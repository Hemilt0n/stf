# AGENTS Handoff

本文件用于会话交接与快速恢复上下文。新一轮开发前，请先阅读本文件与 `README.md`。

## 0. 文件命名约定

- 本仓库统一使用 `AGENTS.md`（复数）作为会话上下文入口文档。
- 旧文件名 `AGENT.md` 已废弃，不再维护。

## 1. 当前状态（2026-04-13）

- 仓库路径: `/home/hang/repos/stf`
- 当前主干分支: `master`
- 远程仓库: `origin = https://github.com/Hemilt0n/stf.git`
- 已合入 `master` 的关键提交:
  - `677dbde`（合并 `research/perf-24g-flow`）
  - `5eeadef`（统一 `data_range=[0, 10000]`）

## 2. 已落地的性能优化（master）

- 训练侧新增性能开关（`TrainConfig`）:
  - `grad_accum_steps`
  - `precision` (`fp16` / `bf16`)
  - `enable_tf32`
  - `deterministic`, `cudnn_benchmark`
  - `non_blocking_transfer`
  - `train_log_interval`
  - `compile_model`, `compile_mode`, `compile_dynamic`
  - `use_channels_last`
  - `fine_t1_noise_warmup_epochs`, `fine_t1_noise_warmup_steps`
  - `fine_t1_noise_power`, `fine_t1_noise_std`, `fine_t1_noise_alpha_tail`
  - `val_step_log_keys`, `val_step_log_max_keys`, `val_step_save_csv`（验证步调试开关）
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
- 新增离线 sampler 布局导出脚本:
  - `tools/dump_sampler_layout.py`
  - 输出 `runs/debug/*.csv`，用于定位固定顺序 val 批次中的难样本位置。
- 新增离线数据序列化脚本:
  - `tools/serialize_dataset.py`
  - 支持将 `.tif` 预处理为 `.npy/.npz`，保持目录结构并生成 `/.stf_serialized.json` 标记。
  - 目录规范支持：`<dataset>/{train,val,test}` -> `<dataset>_serialized/{train,val,test}`（`--splits train,val,test`）。
  - `LoadData` 已兼容 raw/serialized 自动读取；数据集存在 marker 时会优先 marker 指定后缀。

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

## 3.3 配置接口更新（2026-04-02）

- `ExperimentConfig.task` 现已允许任意字符串（`stf/config/loader.py` 不再做白名单校验）。
- 建议约定仍使用 `flow` / `stfdiff`，便于目录组织与团队协作；但框架不再强制限制。

## 3.4 远程训练工作流（2026-04-13）

- 新增仓库私有 skill:
  - `.codex/skills/remote-train-orchestrator/`
  - 入口脚本: `.codex/skills/remote-train-orchestrator/scripts/remote_train.sh`
- 设计目标:
  - 在本机无 GPU、服务器有 GPU 的前提下，通过 `tailscale ssh hang@home-pc-ubuntu` 发起远程训练。
  - 训练前强制检查本地/远程 git head 是否一致。
  - 远程运行使用命名 `tmux session`，避免本地休眠导致任务丢失。
  - 多配置对比任务由 helper 串行编排：每个 config 各自拥有命名 `tmux session`，但同一条 compare queue 任意时刻只允许一个 config session 运行。
  - 支持两阶段工作流：`prepare` 只做 staging 与记录，`launch-prepared` 在确认后再真正启动。
  - 支持 `--keep-finished-session`，便于训练结束后保留 tmux 现场供人工检查。
  - 主线程上下文只保留摘要，长日志留在远程和本地记录文件中。
- 本地记录约定:
  - 运行状态记录: `log/remote_runs/records/<session>.json`
  - 实验账本: `log/remote_runs/experiments.md`
- 远程配置约定:
  - 远程专用配置在本地 `configs/remote/*.py` 编辑，然后单独下发到服务器。
  - `configs/remote/*` 为机器本地覆盖区，默认不纳入 git；只保留 `.gitkeep`。
  - 为便于人工检查，helper 会自动维护三个本地状态目录：
    - `configs/remote/review/<session>/`：待检查 / 待确认启动
    - `configs/remote/running/<session>/`：正在执行
    - `configs/remote/completed/<session>/`：已完成
- 当前远程环境已核对事实:
  - 远程主机: `hang@home-pc-ubuntu`
  - 远程仓库: `/home/hang/repos/stf`
  - 远程 `data -> /home/hang/data`
  - 远程 `runs -> /mnt/d/hang/results/stf`
  - `tmux` 可用
- 当前推荐数据根:
  - CIA serialized: `data/CIA/band4_serialized`
- 当前推荐调用:
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh inspect`
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh prepare --config ... --purpose ...`
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch --config ... --purpose ...`
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch --config ... --config ... --purpose ...`（对比任务：顺序执行，每个 config 单独 tmux session）
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh launch-prepared --session ...`
  - `bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh status --session ...`

## 3.2 `fine_t1` warmup 严格配对复核（2026-04-01）

- 对比实验:
  - baseline（严格配对）: `runs/flow/change_aware_fine_t1_baseline_matched_cia_20260331-173119`
  - warmup: `runs/flow/change_aware_fine_t1_noise_warmup_200_cia_20260326-230345`
- 对齐口径:
  - 两边均使用 `tools/summarize_train_log.py` 按 epoch 聚合。
  - 仅比较共同验证点（`49,99,...,499`），除 warmup 参数外其余保持一致。
- `epoch=499` 结论:
  - baseline 在 `val_loss/RMSE/PSNR/SSIM/CC/ERGAS/UIQI` 略优。
  - warmup 在 `MAE/SAM/TRP` 略优。
- `warmup_200` vs `warmup_200_tail`（`runs/flow/change_aware_fine_t1_noise_warmup_200_tail_20260331-225043`）:
  - 在共同验证点 `49,99,...,499` 下，`warmup_200_tail` 仅 `TRP` 更优，其余主指标整体回退。
  - 当前 `fine_t1_noise_alpha_tail=0.1` 不作为默认推荐参数。
- 总结:
  - warmup 参数对训练动力学有效（前期抑制捷径、后期追回），但本轮严格配对下未形成最终整体指标优势。
  - 后续优先验证“更弱 tail（更小 alpha）+ consistency”组合，而非直接复用 `alpha_tail=0.1`。
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
bash .codex/skills/remote-train-orchestrator/scripts/remote_train.sh inspect
```

训练入口:

```bash
uv run stf train --config configs/flow/change_aware_toy.py
uv run stf train --config configs/flow/change_aware_perf_24g.py
uv run stf train --config configs/flow/template_all_options.py
```

模板约定:
- 新开分支时优先复制 `configs/flow/template_all_options.py` 起步，避免遗漏新增配置项。
