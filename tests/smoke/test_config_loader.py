from stf.config import load_experiment
from stf.models import GaussianFlowMatching


def test_load_flow_config():
    exp = load_experiment("configs/flow/minimal.py")
    assert exp.task == "flow"
    assert exp.model is not None
    assert isinstance(exp.model, GaussianFlowMatching)
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
    assert isinstance(exp.model, GaussianFlowMatching)
    assert exp.model.ranking_loss_weight > 0.0


def test_load_flow_perf_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g.py")
    assert exp.task == "flow"
    assert isinstance(exp.model, GaussianFlowMatching)
    assert exp.model.condition_dropout_p == 0.1
    assert exp.train.precision == "bf16"
    assert exp.train.enable_tf32 is True
    assert exp.train.compile_model is False


def test_load_flow_perf_compile_config():
    exp = load_experiment("configs/flow/change_aware_perf_24g_compile.py")
    assert exp.task == "flow"
    assert exp.train.compile_model is True
    assert exp.train.compile_dynamic is True


def test_load_flow_fine_t1_noise_warmup_300_config():
    exp = load_experiment("configs/flow/change_aware_toy_fine_t1_noise_warmup_300.py")
    assert exp.task == "flow"
    assert exp.train.fine_t1_noise_warmup_epochs == 300
    assert exp.train.fine_t1_noise_power == 4.0


def test_load_flow_fine_t1_noise_warmup_500_config():
    exp = load_experiment("configs/flow/change_aware_toy_fine_t1_noise_warmup_500.py")
    assert exp.task == "flow"
    assert exp.train.fine_t1_noise_warmup_epochs == 500
    assert exp.train.fine_t1_noise_power == 6.0


def test_load_flow_fine_t1_noise_warmup_200_tail_config():
    exp = load_experiment("configs/flow/change_aware_fine_t1_noise_warmup_200_tail.py")
    assert exp.task == "flow"
    assert isinstance(exp.model, GaussianFlowMatching)
    assert exp.model.condition_dropout_p == 0.1
    assert exp.model.change_loss_weight == 1.0
    assert exp.model.coarse_consistency_weight == 0.2
    assert exp.train.fine_t1_noise_warmup_epochs == 200
    assert exp.train.fine_t1_noise_alpha_tail == 0.1


def test_load_flow_fine_t1_baseline_matched_cia_config():
    exp = load_experiment("configs/flow/change_aware_fine_t1_baseline_matched_cia.py")
    assert exp.task == "flow"
    assert isinstance(exp.model, GaussianFlowMatching)
    assert exp.model.condition_dropout_p == 0.1
    assert exp.model.change_loss_weight == 1.0
    assert exp.model.coarse_consistency_weight == 0.2
    assert exp.train.max_epochs == 500
    assert exp.train.val_interval == 50
    assert exp.train.save_interval == 50
    assert exp.train.fine_t1_noise_warmup_epochs == 0
    assert exp.train.fine_t1_noise_warmup_steps == 0
    assert exp.train.fine_t1_noise_alpha_tail == 0.0
