# stf

面向时空融合研究的重构版工程，当前主线任务：
- `stfdiff`
- `flow`

## 1. 环境准备（uv）

```bash
uv sync --group dev
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 2. 项目运行流程

1. 准备数据路径（可复用 `stf-tmp` 的数据软链）
```bash
ln -s /home/hang/repos/stf-tmp/data ./data
```

2. 训练（示例）
```bash
uv run stf train --config configs/flow/minimal.py
```

3. 评估
```bash
uv run stf eval --config configs/stfdiff/minimal.py --checkpoint /path/to/model_epoch_10.pth
```

4. 推理
```bash
uv run stf predict --config configs/stfdiff/minimal.py --checkpoint /path/to/model_epoch_10.pth
```

5. 迁移旧配置
```bash
uv run stf migrate-config \
  --legacy-config /home/hang/repos/stf-tmp/config/flow/toy_train_config.py \
  --output configs/flow/migrated_toy.py
```

运行产物统一输出到：
- `runs/<task>/<exp_timestamp>/checkpoints`
- `runs/<task>/<exp_timestamp>/logs`
- `runs/<task>/<exp_timestamp>/tensorboard`
- `runs/<task>/<exp_timestamp>/images`

## 3. 训练配置如何按需自定义

每个配置文件导出一个 `EXPERIMENT` 对象，类型为 `ExperimentConfig`，核心位置在 [stf/config/types.py](stf/config/types.py)。

最常改的字段：
- `task`: `flow` 或 `stfdiff`
- `name`: 实验名（决定输出目录前缀）
- `data`: `train_dataloader/val_dataloader/test_dataloader`
- `model`: 你的模型实例
- `optimizer`: 传 `params` 的构造器（一般用 `functools.partial`）
- `scheduler`: 可选
- `metrics`: 指标列表
- `train`: `max_epochs/val_interval/save_interval/use_ema/use_mixed_precision/grad_clip_norm`
- `io`: `output_root/save_images/show_images/show_bands`
- `resume_from`: 断点恢复路径

遥感数据范围约定（必须）：
- `RescaleToMinusOneOne(..., data_range=[0, 10000])`
- 不要使用 `data_range=[0, 100]`

建议优先通过改配置完成实验切换，避免直接改引擎代码。

## 4. 开发时如何“改动最小、复用最强”

优先级（从推荐到不推荐）：
1. 只改 `configs/*.py`（最快、影响面最小）
2. 新增可插拔模块并在配置中替换（高复用）
3. 改 `engine` 训练循环（影响面最大，最后再做）

具体策略：
- 新增模型：在 `stf/models/` 实现并在 `stf/models/__init__.py` 导出，然后在配置里替换 `model=...`。
- 新增指标：在 `stf/metrics/` 实现并导出，然后在配置 `metrics=[...]` 中添加。
- 新增数据变换：在 `stf/data/transforms/` 增加并导出，然后挂到 `transform_func_list`。
- 保持接口兼容以减少引擎改动：
  - 训练阶段 `model(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02) -> loss`
  - 验证/推理阶段 `model.sample(coarse_img_01, coarse_img_02, fine_img_01) -> pred`
  - 数据 batch 至少包含：`coarse_img_01/coarse_img_02/fine_img_01/fine_img_02`
- 如果新模型接口不同，优先写一个“适配器模型”来对齐上述接口，而不是直接改 `stf/engine/*.py`。

## 5. 中文指导手册

更完整的中文开发指南见：
- [docs/开发手册.md](docs/开发手册.md)

## 6. 目录结构

- `stf/`: 包代码
- `configs/`: 可运行实验配置
- `docs/`: 架构、配置、迁移与中文手册
- `tests/smoke/`: 冒烟测试
- `runs/<task>/<exp>`: 训练/评估/推理产物
