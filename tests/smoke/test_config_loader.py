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
