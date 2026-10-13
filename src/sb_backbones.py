"""Backbone architectures required to load the released BridgeTwist checkpoints."""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistBackboneConfig:
    image_size = 28
    channels = 1
    dim = 64
    dim_mults = (1, 2, 4)


class MnistTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        scale = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class MnistResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class MnistAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        q, k, v = qkv.permute(1, 0, 2, 4, 3)
        attn = (q @ k.transpose(-1, -2)) * (channels // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        return x + self.out(out)


class MnistUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, channel_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        time_channels = model_channels * 4
        self.time_mlp = nn.Sequential(
            MnistTimeEmbedding(model_channels),
            nn.Linear(model_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels
        self.init_conv = nn.Conv2d(in_channels, now_channels, 3, padding=1)

        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(MnistResidualBlock(now_channels, out_channels, time_channels))
                now_channels = out_channels
                channels.append(now_channels)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)

        self.middle_blocks = nn.ModuleList(
            [
                MnistResidualBlock(now_channels, now_channels, time_channels),
                MnistAttentionBlock(now_channels),
                MnistResidualBlock(now_channels, now_channels, time_channels),
            ]
        )
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(MnistResidualBlock(channels.pop() + now_channels, out_channels, time_channels))
                now_channels = out_channels
            if level != 0:
                self.up_blocks.append(nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, MnistResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)
        for block in self.middle_blocks:
            if isinstance(block, MnistResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        for block in self.up_blocks:
            if isinstance(block, MnistResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)
        return self.final_conv(h)


class CifarBackboneConfig:
    def __init__(self):
        self.data = type(
            "data",
            (),
            {
                "dataset": "CIFAR10",
                "image_size": 32,
                "num_channels": 3,
                "centered": True,
                "random_flip": True,
                "uniform_dequantization": False,
            },
        )()
        self.model = type(
            "model",
            (),
            {
                "name": "ncsnpp",
                "scale_by_sigma": False,
                "ema_rate": 0.999999,
                "dropout": 0.15,
                "normalization": "GroupNorm",
                "nonlinearity": "swish",
                "nf": 128,
                "ch_mult": (1, 2, 2, 2),
                "num_res_blocks": 4,
                "attn_resolutions": (16,),
                "resamp_with_conv": True,
                "conditional": True,
                "fir": False,
                "fir_kernel": [1, 3, 3, 1],
                "skip_rescale": True,
                "resblock_type": "biggan",
                "progressive": "none",
                "progressive_input": "none",
                "progressive_combine": "sum",
                "attention_type": "ddpm",
                "init_scale": 0.0,
                "embedding_type": "positional",
                "fourier_scale": 16,
                "conv_size": 3,
                "sigma_min": 0.01,
                "sigma_max": 50,
                "num_scales": 1000,
                "beta_min": 0.1,
                "beta_max": 20.0,
            },
        )()


def get_sigmas(config):
    return np.exp(np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))


def variance_scaling(scale, mode, distribution, dtype=torch.float32, device="cpu"):
    def init(shape, dtype=dtype, device=device):
        fan_in = np.prod(shape) / shape[0] if len(shape) > 1 else 1
        fan_out = shape[0] if len(shape) > 1 else 1
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode: {mode}")
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        if distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)
        raise ValueError(f"invalid distribution: {distribution}")

    return init


def default_init(scale=1.0):
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = torch.einsum("bhwc,cd->bhwd", x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlockpp(nn.Module):
    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(channels) ** -0.5)
        w = torch.reshape(w, (batch, height, width, height * width))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (batch, height, width, height, width))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


def naive_upsample_2d(x, factor=2):
    _batch, channels, height, width = x.shape
    x = torch.reshape(x, (-1, channels, height, 1, width, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, channels, height * factor, width * factor))


def naive_downsample_2d(x, factor=2):
    _batch, channels, height, width = x.shape
    x = torch.reshape(x, (-1, channels, height // factor, factor, width // factor, factor))
    return torch.mean(x, dim=(3, 5))


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        if self.up:
            h = naive_upsample_2d(h, factor=2)
            x = naive_upsample_2d(x, factor=2)
        elif self.down:
            h = naive_downsample_2d(h, factor=2)
            x = naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        return (x + h) / np.sqrt(2.0) if self.skip_rescale else x + h


class CifarNCSNpp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = nn.SiLU()
        self.register_buffer("sigmas", torch.tensor(get_sigmas(config)))
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
        self.conditional = config.model.conditional
        self.skip_rescale = config.model.skip_rescale
        self.resblock_type = config.model.resblock_type.lower()
        self.progressive = config.model.progressive.lower()
        self.progressive_input = config.model.progressive_input.lower()
        self.embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale

        modules = []
        if self.embedding_type != "positional":
            raise ValueError(f"embedding type {self.embedding_type} unknown")

        if self.conditional:
            modules.append(nn.Linear(nf, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = partial(AttnBlockpp, init_scale=init_scale, skip_rescale=self.skip_rescale)
        ResnetBlock = partial(
            ResnetBlockBigGANpp,
            act=self.act,
            dropout=dropout,
            skip_rescale=self.skip_rescale,
            init_scale=init_scale,
            temb_dim=nf * 4,
        )
        channels = config.data.num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            for _ in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        for i_level in reversed(range(num_resolutions)):
            for _ in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True))

        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type != "positional":
            raise ValueError(f"embedding type {self.embedding_type} unknown")
        used_sigmas = self.sigmas[time_cond.long()]
        temb = get_timestep_embedding(time_cond, self.nf)
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            x = 2 * x - 1.0

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
        return h
