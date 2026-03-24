import pytest
import torch
import torch.nn.functional as F

from stf.models.pred_resnet import Attention as PredResnetAttention
from stf.models.pred_resnet import PredNoiseNet, PredTrajNet
from stf.models.unet import Attention as UnetAttention


def _supports_sdpa() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


def test_attention_backends_run_in_unet_and_pred_resnet():
    x = torch.randn((2, 64, 16, 16))
    backends = ["classic", "auto"]
    if _supports_sdpa():
        backends.append("sdpa")

    for backend in backends:
        for attn_cls in (UnetAttention, PredResnetAttention):
            attn = attn_cls(dim=64, heads=4, dim_head=32, attn_backend=backend)
            out = attn(x)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()


def test_invalid_attention_backend_raises():
    with pytest.raises(ValueError):
        UnetAttention(dim=32, attn_backend="invalid")
    with pytest.raises(ValueError):
        PredResnetAttention(dim=32, attn_backend="invalid")
    with pytest.raises(ValueError):
        PredNoiseNet(dim=32, channels=3, out_dim=3, dim_mults=(1, 2), attention_backend="invalid")
    with pytest.raises(ValueError):
        PredTrajNet(dim=32, channels=3, out_dim=3, dim_mults=(1, 2), attention_backend="invalid")
