from stf.config import load_experiment


def test_load_flow_config():
    exp = load_experiment("configs/flow/minimal.py")
    assert exp.task == "flow"
    assert exp.model is not None
    assert exp.data.train_dataloader is not None


def test_load_stfdiff_config():
    exp = load_experiment("configs/stfdiff/minimal.py")
    assert exp.task == "stfdiff"
    assert exp.model is not None
    assert exp.data.val_dataloader is not None

def test_load_flow_hf_config():
    exp = load_experiment("configs/flow/change_aware_toy_hf_grad_lap_rank.py")
    assert exp.task == "flow"
    assert exp.model is not None
    assert exp.model.ranking_loss_weight > 0.0


def test_load_flow_perf_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g.py")
    assert exp.task == "flow"
    assert exp.train.precision == "bf16"
    assert exp.train.enable_tf32 is True
    assert exp.train.compile_model is False


def test_load_flow_perf_compile_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g_compile.py")
    assert exp.task == "flow"
    assert exp.train.compile_model is True
    assert exp.train.compile_dynamic is True
