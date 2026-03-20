import math

from functools import partial
from collections import namedtuple


import torch
from torch import nn, einsum
import torch.nn.functional as F


from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# small helper modules


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


# class Block(nn.Module):
#     def __init__(self, dim, dim_out, groups=8):
#         super().__init__()
#         self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
#         self.norm = nn.BatachNorm(groups, dim_out)
#         self.act = nn.SiLU()

#     def forward(self, x, scale_shift=None):
#         x = self.proj(x)
#         x = self.norm(x)

#         if exists(scale_shift):
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift

#         x = self.act(x)
#         return x


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.conv_1 = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm_1 = nn.GroupNorm(groups, dim_out)
        self.act_1 = nn.SiLU()

        self.conv_2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm_2 = nn.GroupNorm(groups, dim_out)
        self.act_2 = nn.SiLU()

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.conv_1(x)
        h = self.norm_1(h)
        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        h = self.act_1(h)

        h = self.conv_2(h)
        h = self.norm_2(h)
        h = self.act_2(h)

        return h + self.res_conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, attn_backend="auto"):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        if attn_backend not in {"auto", "sdpa", "classic"}:
            raise ValueError(
                f"Unsupported attention backend: {attn_backend}. "
                "Expected one of ['auto', 'sdpa', 'classic']"
            )
        self.attn_backend = attn_backend
        self.supports_sdpa = hasattr(F, "scaled_dot_product_attention")

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )

        use_sdpa = self.attn_backend == "sdpa" or (
            self.attn_backend == "auto" and self.supports_sdpa
        )
        if use_sdpa:
            if not self.supports_sdpa:
                raise RuntimeError(
                    "scaled_dot_product_attention is not available in this PyTorch build"
                )
            q = rearrange(q, "b h d n -> b h n d")
            k = rearrange(k, "b h d n -> b h n d")
            v = rearrange(v, "b h d n -> b h n d")
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
            return self.to_out(out)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

# model


class PredNoiseNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        attention_backend="auto",
    ):
        super().__init__()
        if attention_backend not in {"auto", "sdpa", "classic"}:
            raise ValueError(
                f"Unsupported attention_backend={attention_backend}. "
                "Expected one of ['auto', 'sdpa', 'classic']"
            )
        self.attention_backend = attention_backend
        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.fine_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)
        self.coarse_init_conv = nn.Conv2d(input_channels * 2, init_dim, 3, 1, 1)

        self.noisy_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)

        self.clean_init_conv = nn.Conv2d(init_dim * 2, init_dim, 1)

        self.clean_downs = nn.ModuleList([])
        self.noisy_downs = nn.ModuleList([])

        self.noise_ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            clean_down = nn.ModuleList(
                [
                    ResBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            noisy_down = nn.ModuleList(
                [
                    ResBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.noisy_downs.append(noisy_down)
            self.clean_downs.append(clean_down)

        mid_dim = dims[-1]
        self.clean_mid_block = ResBlock(mid_dim, mid_dim, groups=resnet_block_groups)
        self.noisy_mid_block = ResBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=8
        )
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            noise_up = nn.ModuleList(
                [
                    ResBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ]
            )
            self.noise_ups.append(noise_up)

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResBlock(
            dim * 2,
            dim,
            time_emb_dim=time_dim,
            groups=resnet_block_groups,
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        x_fine = self.fine_init_conv(fine_img_01)
        x_coarse = self.coarse_init_conv(
            torch.cat((coarse_img_01, coarse_img_02), dim=1)
        )

        x_noisy = self.noisy_init_conv(noisy_fine_img_02)

        x_clean = torch.cat((x_fine, x_coarse), dim=1)
        x_clean = self.clean_init_conv(x_clean)

        noise = x_noisy - x_clean
        # time_factor = rearrange(1 - time, 'b -> b 1 1 1') if t_rescale else 1
        # noise = noise / (time_factor + 1e-8) if t_rescale else noise
        r = noise.clone()

        t = self.time_mlp(time)

        h = []

        down_num = len(self.clean_downs)

        for down_idx in range(down_num):
            clean_res, clean_downsampling = self.clean_downs[down_idx]
            noisy_res, noisy_downsampling = self.noisy_downs[down_idx]

            x_clean = clean_res(x_clean)
            x_noisy = noisy_res(x_noisy, t)
            noise = x_noisy - x_clean
            # time_factor = rearrange(1 - time, 'b -> b 1 1 1') if t_rescale else 1
            # noise = noise / (time_factor + 1e-8) if t_rescale else noise
            h.append(noise)

            x_clean = clean_downsampling(x_clean)
            x_noisy = noisy_downsampling(x_noisy)

        x_clean = self.clean_mid_block(x_clean)
        x_noisy = self.noisy_mid_block(x_noisy, t)

        noise = x_noisy - x_clean
        # time_factor = rearrange(1 - time, 'b -> b 1 1 1') if t_rescale else 1
        # noise = noise / (time_factor + 1e-8) if t_rescale else noise

        up_num = len(self.noise_ups)
        for up_idx in range(up_num):
            noise_res, noise_upsampling = self.noise_ups[up_idx]
            noise = torch.cat((noise, h.pop()), dim=1)
            noise = noise_res(noise, t)
            noise = noise_upsampling(noise)

        noise = torch.cat((noise, r), dim=1)

        noise = self.final_res_block(noise, t)
        return self.final_conv(noise)
    
class PredTrajNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        attention_backend="auto",
    ):
        super().__init__()
        if attention_backend not in {"auto", "sdpa", "classic"}:
            raise ValueError(
                f"Unsupported attention_backend={attention_backend}. "
                "Expected one of ['auto', 'sdpa', 'classic']"
            )
        self.attention_backend = attention_backend
        # Keep argument semantics aligned with PredNoiseNet. The only intended
        # architectural difference is single-branch fusion instead of dual-branch
        # clean/noisy processing.
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.fine_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)
        self.coarse_init_conv = nn.Conv2d(input_channels * 2, init_dim, 3, 1, 1)
        self.noisy_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)
        self.init_conv = nn.Conv2d(init_dim * 3, init_dim, 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_klass = partial(ResBlock, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def _prepare_self_condition_inputs(
        self,
        fine_img_01,
        noisy_fine_img_02,
        coarse_img_01,
        coarse_img_02,
        x_self_cond=None,
    ):
        if not self.self_condition:
            return fine_img_01, noisy_fine_img_02, coarse_img_01, coarse_img_02

        if x_self_cond is None:
            x_self_cond = torch.zeros_like(noisy_fine_img_02)

        fine_in = torch.cat((x_self_cond, fine_img_01), dim=1)
        noisy_in = torch.cat((x_self_cond, noisy_fine_img_02), dim=1)
        coarse_01_in = torch.cat((x_self_cond, coarse_img_01), dim=1)
        coarse_02_in = torch.cat((x_self_cond, coarse_img_02), dim=1)
        return fine_in, noisy_in, coarse_01_in, coarse_02_in

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        fine_in, noisy_in, coarse_01_in, coarse_02_in = self._prepare_self_condition_inputs(
            fine_img_01,
            noisy_fine_img_02,
            coarse_img_01,
            coarse_img_02,
            x_self_cond=x_self_cond,
        )

        x_fine = self.fine_init_conv(fine_in)
        x_coarse = self.coarse_init_conv(torch.cat((coarse_01_in, coarse_02_in), dim=1))
        x_noisy = self.noisy_init_conv(noisy_in)
        x = self.init_conv(torch.cat((x_fine, x_coarse, x_noisy), dim=1))
        r = x.clone()

        t = self.time_mlp(time)
        h = []

        for block, downsample in self.downs:
            x = block(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    model=PredNoiseNet(dim=64, channels=3, out_dim=3, dim_mults=(1, 2, 4))
    writer = SummaryWriter(log_dir='runs/model')
    writer.add_graph(model, (torch.randn(1,3,64,64), torch.randn(1,3,64,64), torch.randn(1,3,64,64), torch.randn(1,3,64,64), torch.randn(1)))
    print(model)
