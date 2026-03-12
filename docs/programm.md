# 实验结果同步工作流（programm）

最后更新: 2026-03-12 17:50:38 +0800

## 1. 目标

将实验目录中的大文件优先从 `logs/train.log` 预处理为小体积摘要文件，便于：
- 低上下文成本同步给助手
- 跨 session 快速复盘
- 多实验横向对比
- 说明: TensorBoard 仅作为可选补充，不是默认输入

## 2. 输入约定

先提供实验路径映射（实验名 -> 目录），示例：

```text
baseline -> runs/flow/change_aware_toy_20260312-xxxxxx
hf_grad -> runs/flow/change_aware_toy_hf_grad_20260312-xxxxxx
hf_grad_lap -> runs/flow/change_aware_toy_hf_grad_lap_20260312-xxxxxx
hf_grad_lap_rank -> runs/flow/change_aware_toy_hf_grad_lap_rank_20260312-xxxxxx
```

每个实验目录建议至少包含：
- `logs/train.log`
- 可视化图像目录（如 `vis/`、`pred/`、`samples/`）

可选（仅兜底场景）：
- `tensorboard/events.out.tfevents.*`

## 3. train.log 预处理（按 epoch）

脚本：`tools/summarize_train_log.py`

```bash
uv run python tools/summarize_train_log.py <run_dir>/logs/train.log
```

输出（写回原目录）：
- `<run_dir>/logs/train.epoch.csv`
- `<run_dir>/logs/train.epoch.md`

字段说明（核心）：
- `train_loss`: 优先使用 `avg_loss`，否则回退到 iter 均值
- `val_loss`: 优先使用 val 汇总行 `loss`，否则回退到 iter 均值
- 其余指标（如 `RMSE/MAE/SSIM/TRP`）来自 val 汇总行

## 4. 可选：TensorBoard 预处理（按 tag/step）

仅在以下场景使用：
- `train.log` 缺少关键标量
- 需要 step 级曲线而非 epoch 汇总

脚本：`tools/export_tb_scalars.py`

```bash
uv run python tools/export_tb_scalars.py <run_dir>/tensorboard
```

输出（写回原目录）：
- `<run_dir>/tensorboard/tb.scalars.step.csv`
- `<run_dir>/tensorboard/tb.scalars.summary.csv`
- `<run_dir>/tensorboard/tb.scalars.summary.md`

字段说明（核心）：
- `tb.scalars.step.csv`: 每个 `step` 的各 tag 值（宽表）
- `tb.scalars.summary.csv`: 每个 tag 的 `count/min/max/mean/last_value`

## 5. 助手协作流程

1. 你给我实验路径映射。  
2. 我先调用 `train.log` 聚合脚本生成小文件。  
3. 仅在必要时再补充 TensorBoard 导出。  
4. 我基于 `train.epoch.*`（必要时叠加 `tb.scalars.*`）做跨实验对比与阶段总结。  
5. 你补充图像目视结论，我再合并到 `docs/progress.md` 和结论建议中。  

## 6. 图像目视回传模板

每个实验一段：

```text
实验名:
样本/场景ID:
主观结论（变化区域细节、伪影、是否复制 fine_t1）:
与 baseline 对比（更好/持平/更差）:
置信度（高/中/低）:
```
