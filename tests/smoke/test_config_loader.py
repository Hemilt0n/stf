from stf.config import load_experiment


def test_load_flow_config():
    exp = load_experiment("configs/flow/minimal.py")
    assert exp.task == "flow"
    assert exp.model is not None
    assert exp.data.train_dataloader is not None
    assert exp.train.fine_t1_noise_warmup_epochs == 0
    assert exp.train.fine_t1_noise_warmup_steps == 0
    assert exp.train.fine_t1_noise_alpha_tail == 0.0
    assert exp.train.val_step_log_keys is False
    assert exp.train.val_step_save_csv is False


def test_load_stfdiff_config():
    exp = load_experiment("configs/stfdiff/minimal.py")
    assert exp.task == "stfdiff"
    assert exp.model is not None
    assert exp.data.val_dataloader is not None


def test_load_flow_perf_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g.py")
    assert exp.task == "flow"
    assert exp.train.precision == "bf16"
    assert exp.train.enable_tf32 is True
    assert exp.train.compile_model is False
    assert exp.train.val_step_log_max_keys == 8


def test_load_flow_perf_compile_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g_compile.py")
    assert exp.task == "flow"
    assert exp.train.compile_model is True
    assert exp.train.compile_dynamic is True


def test_load_flow_template_all_options_config():
    exp = load_experiment("configs/flow/template_all_options.py")
    assert exp.task == "flow"
    assert exp.train.use_mixed_precision is True
    assert exp.train.precision == "bf16"
    assert exp.train.enable_tf32 is True
    assert exp.train.cudnn_benchmark is True
    assert exp.train.non_blocking_transfer is True
    assert exp.train.use_channels_last is True
    assert exp.train.compile_model is False
    assert exp.train.fine_t1_noise_warmup_epochs == 0
    assert exp.train.fine_t1_noise_warmup_steps == 0
    assert exp.train.fine_t1_noise_alpha_tail == 0.0
    assert exp.train.val_step_log_keys is False
    assert exp.train.val_step_save_csv is False


def test_load_config_with_custom_task(tmp_path):
    cfg_path = tmp_path / "custom_task_cfg.py"
    cfg_path.write_text(
        "\n".join(
            [
                "from stf.config.types import ExperimentConfig",
                "",
                "EXPERIMENT = ExperimentConfig(",
                '    task=\"my_custom_task\",',
                '    name=\"custom_task_smoke\",',
                "    model=object(),",
                "    optimizer=object(),",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exp = load_experiment(str(cfg_path))
    assert exp.task == "my_custom_task"
