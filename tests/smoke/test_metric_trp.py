import torch

from stf.metrics import TRP


def test_trp_prefers_target_when_prediction_close_to_target():
    metric = TRP()
    ref = torch.zeros((1, 3, 4, 4))
    tgt = torch.ones((1, 3, 4, 4))
    pred = torch.ones((1, 3, 4, 4)) * 0.9
    value = metric(tgt, pred, ref)
    assert value.item() > 0


def test_trp_prefers_reference_when_prediction_close_to_reference():
    metric = TRP()
    ref = torch.zeros((1, 3, 4, 4))
    tgt = torch.ones((1, 3, 4, 4))
    pred = torch.ones((1, 3, 4, 4)) * 0.1
    value = metric(tgt, pred, ref)
    assert value.item() < 0
