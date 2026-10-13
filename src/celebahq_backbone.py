"""CelebA-HQ NCSN++ architecture used by the released SB checkpoint.

Extracted from the CelebA-HQ backbone in TFTF; see licenses/celebahq_NOTICE.md.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]
    
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    
    out = out[:, ::down_y, ::down_x, :]
    
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    
    return out.view(-1, channel, out_h, out_w)

def upsample_2d(x, k, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    k = torch.tensor(k, dtype=torch.float32, device=x.device)
    return upfirdn2d_native(x, k, factor, factor, 1, 1, (p + 1) // 2 + factor - 1, p // 2, (p + 1) // 2 + factor - 1, p // 2)

def downsample_2d(x, k, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = setup_kernel(k) * gain
    p = k.shape[0] - factor
    k = torch.tensor(k, dtype=torch.float32, device=x.device)
    return upfirdn2d_native(x, k, 1, 1, factor, factor, (p + 1) // 2, p // 2, (p + 1) // 2, p // 2)

def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))

def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))

def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device='cpu'):
    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out
    
    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")
    
    return init

def default_init(scale=1.):
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = torch.einsum('bhwc,cd->bhwd', x, self.W) + self.b
        return y.permute(0, 3, 1, 2)

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Combine(nn.Module):
    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = ddpm_conv1x1(dim1, dim2)
        self.method = method
    
    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')

class AttnBlockpp(nn.Module):
    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)
        
        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch
    
    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), 'nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = upsample_2d(x, self.fir_kernel, factor=2)
                h = self.Conv_0(h)
        return h

class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = ddpm_conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch
    
    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = downsample_2d(x, self.fir_kernel, factor=2)
                x = self.Conv_0(x)
        return x

class ResnetBlockDDPMpp(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        
        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
    
    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class ResnetBlockBigGANpp(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0.):
        super().__init__()
        
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = ddpm_conv1x1(in_ch, out_ch)
        
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch
    
    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        
        if self.up:
            if self.fir:
                h = upsample_2d(h, self.fir_kernel, factor=2)
                x = upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = downsample_2d(h, self.fir_kernel, factor=2)
                x = downsample_2d(x, self.fir_kernel, factor=2)
            else:
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
        
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class NCSNpp(nn.Module):
    def __init__(self, image_size=256, num_channels=3, nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2),
                 num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True,
                 fir=True, fir_kernel=[1, 3, 3, 1], skip_rescale=True, 
                 resblock_type='biggan', progressive='output_skip', 
                 progressive_input='input_skip', progressive_combine='sum',
                 embedding_type='fourier', fourier_scale=16, init_scale=0.,
                 centered=True, scale_by_sigma=True):
        super().__init__()
        
        self.act = nn.SiLU()
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = [image_size // (2 ** i) for i in range(self.num_resolutions)]
        self.conditional = True
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type.lower()
        self.progressive = progressive.lower()
        self.progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type.lower()
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        
        modules = []
        
        if embedding_type == 'fourier':
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')
        
        if self.conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        
        AttnBlock = lambda channels: AttnBlockpp(channels, init_scale=init_scale, skip_rescale=skip_rescale)
        Upsample_func = lambda in_ch: Upsample(in_ch=in_ch, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        
        if progressive == 'output_skip':
            self.pyramid_upsample = Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        
        Downsample_func = lambda in_ch: Downsample(in_ch=in_ch, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        
        if progressive_input == 'input_skip':
            self.pyramid_downsample = Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        
        if resblock_type == 'ddpm':
            ResnetBlock = lambda in_ch, out_ch: ResnetBlockDDPMpp(
                self.act, in_ch, out_ch, temb_dim=nf * 4, dropout=dropout,
                init_scale=init_scale, skip_rescale=skip_rescale)
        elif resblock_type == 'biggan':
            ResnetBlock = lambda in_ch, out_ch, down=False, up=False: ResnetBlockBigGANpp(
                self.act, in_ch, out_ch, temb_dim=nf * 4, dropout=dropout,
                fir=fir, fir_kernel=fir_kernel, init_scale=init_scale,
                skip_rescale=skip_rescale, down=down, up=up)
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')
        
        if progressive_input != 'none':
            input_pyramid_ch = num_channels
        
        modules.append(ddpm_conv3x3(num_channels, nf))
        hs_c = [nf]
        
        in_ch = nf
        for i_level in range(self.num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                if resblock_type == 'ddpm':
                    modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                else:
                    modules.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
                
                if self.all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            
            if i_level != self.num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample_func(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch, in_ch, down=True))
                
                if progressive_input == 'input_skip':
                    modules.append(Combine(dim1=input_pyramid_ch, dim2=in_ch, method=progressive_combine))
                    if progressive_combine == 'cat':
                        in_ch *= 2
                
                hs_c.append(in_ch)
        
        in_ch = hs_c[-1]
        if resblock_type == 'ddpm':
            modules.append(ResnetBlock(in_ch=in_ch, out_ch=in_ch))
        else:
            modules.append(ResnetBlock(in_ch, in_ch))
        modules.append(AttnBlock(channels=in_ch))
        if resblock_type == 'ddpm':
            modules.append(ResnetBlock(in_ch=in_ch, out_ch=in_ch))
        else:
            modules.append(ResnetBlock(in_ch, in_ch))
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                if resblock_type == 'ddpm':
                    modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                else:
                    modules.append(ResnetBlock(in_ch + hs_c.pop(), out_ch))
                in_ch = out_ch
            
            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            
            if progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(ddpm_conv3x3(in_ch, num_channels, init_scale=init_scale))
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(ddpm_conv3x3(in_ch, num_channels, bias=True, init_scale=init_scale))
            
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample_func(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch, in_ch, up=True))
        
        assert not hs_c
        
        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(ddpm_conv3x3(in_ch, num_channels, init_scale=init_scale))
        
        self.all_modules = nn.ModuleList(modules)
        self.hs_c_len = len(hs_c)
    
    def forward(self, x, time_cond):
        modules = self.all_modules
        m_idx = 0
        
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')
        
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        
        if not self.centered:
            x = 2 * x - 1.
        
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x
        
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                
                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                
                hs.append(h)
        
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        
        pyramid = None
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            
            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
            
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1
        
        assert not hs
        
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
        
        return h
