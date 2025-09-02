# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:58:41 2025

@author: zhouy
"""

import math

import torch
import torch.nn as nn
from torch.nn import init as init

from timm.layers import DropPath, LayerScale2d, LayerNorm2d
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout2d(drop, inplace=False) if drop > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1,
                 bias=True, 
                 dropout=0, 
                 norm=nn.BatchNorm2d, 
                 act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
    
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, 
                 shape, 
                 base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 qkv_bias=True, 
                 **kwargs):

        super(LinearAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

class MLLA(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 qkv_bias=True, 
                 **kwargs):
        super(MLLA, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        
        self.in_proj = nn.Conv2d(dim, dim, 1, 1)
        self.act_proj = nn.Conv2d(dim, dim, 1, 1)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, 
                                    input_resolution=input_resolution, 
                                    num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Conv2d(dim, dim, 1, 1)
        
        return
        
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        act_res = self.act(self.act_proj(x))
        
        x = self.in_proj(x)
        x = self.act(self.dwc(x)).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x).permute(0, 2, 1).view(B, C, H, W)
        
        x = self.out_proj(x * act_res)
        
        return x

class FFTModule(nn.Module):
    def __init__(self, 
                 dim, 
                 dw=1, 
                 norm='backward', 
                 act_method=nn.GELU, 
                 bias=False):
        super(FFTModule, self).__init__()
        
        self.act_fft = act_method()
        
        hid_dim = dim * dw
        
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        
        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, 1, dim)), requires_grad=True)
            
        self.bias = bias
        self.norm = norm
        
        return
    
    def forward(self, x):
        _, _, H, W = x.shape
        
        # fft
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            b2 = torch.complex(self.b2_real, self.b2_imag)
            
        # conv
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        if self.bias:
            y = y + b1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        
        # conv
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            y = y + b2
        y = rearrange(y, 'b h w c -> b c h w')
        
        # ifft
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        
        return y

class SGFB(nn.Module):
    def __init__(self, dim):
        super(SGFB, self).__init__()
        
        self.conv = nn.Conv2d(2 * dim, 1, 1, 1)
        self.gate = nn.Sigmoid()
        
        return
    
    def forward(self, x1, x2) -> torch.Tensor:
        m = self.gate(self.conv(torch.cat([x1, x2], dim=1)))
        x = m * x1 + (1.0 - m) * x2
        
        return x
    
class Former(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=LayerNorm2d, 
                 **kwargs):
        super(Former, self).__init__()
        
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, drop=drop)
        
        return    

class MLLAFormer(Former):
    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=LayerNorm2d, 
                 **kwargs):
        super(MLLAFormer, self).__init__(
            dim, mlp_ratio, drop, drop_path, act_layer, norm_layer, **kwargs
            )
        
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.mlla = MLLA(dim, input_resolution, num_heads, qkv_bias, **kwargs)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        return
    
    def forward(self, x):
        x = x + self.cpe1(x)
        x = x + self.drop_path(self.mlla(self.norm1(x)))    # MLLA Token mixer
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # FFN
        
        return x

class ConvFormer(Former):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=LayerNorm2d, 
                 **kwargs):
        super(ConvFormer, self).__init__(
            dim, mlp_ratio, drop, drop_path, act_layer, norm_layer, **kwargs
            )
        
        self.conv_blcok = nn.Sequential(
            nn.Conv2d(dim, dim*4, kernel_size=1, stride=1, padding=0, bias=True),
            act_layer(),
            nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, padding=1, bias=True, groups=dim*4),
            act_layer(),
            nn.Conv2d(dim*4, dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
        
        return
    
    def forward(self, x):
        x = x + self.drop_path(self.conv_blcok(self.norm1(x)))      # Conv Token mixer
        x = x + self.drop_path(self.mlp(self.norm2(x)))             # FFN
        
        return x

class FFTFormer(Former):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=LayerNorm2d, 
                 **kwargs):
        super(FFTFormer, self).__init__(
            dim, mlp_ratio, drop, drop_path, act_layer, norm_layer, **kwargs
            )
        
        self.fft = FFTModule(dim)
        
        return
    
    def forward(self, x):
        x = x + self.drop_path(self.fft(self.norm1(x)))    # FFT Token mixer
        x = x + self.drop_path(self.mlp(self.norm2(x)))    # FFN
        
        return x
    
class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()
    
    def forward(self, x1, x2):
        return x1 + x2
    
class MFFBlock(nn.Module):
    def __init__(self,
                 dim, 
                 input_resolution, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer = LayerNorm2d, 
                 **kwargs):
        super(MFFBlock, self).__init__()
        
        mlla_enable = True
        fft_enable = True
        gate_enable = True
        
        if 'mlla_enable' in kwargs:
            mlla_enable = kwargs['mlla_enable']
        if 'fft_enable' in kwargs:
            fft_enable = kwargs['fft_enable']
        if 'gate_enable' in kwargs:
            gate_enable = kwargs['gate_enable']
        
        self.ls1 = LayerScale2d(dim, 1.0)
        if mlla_enable:
            self.mlla = MLLAFormer(dim, input_resolution, num_heads,
                                   mlp_ratio, qkv_bias, drop, drop_path,
                                   act_layer, norm_layer, **kwargs)
        else:
            self.mlla = ConvFormer(dim, mlp_ratio, drop, drop_path,
                                   act_layer, norm_layer, **kwargs)
        if fft_enable:
            self.fft1 = FFTFormer(dim, mlp_ratio, drop, drop_path,
                                  act_layer, norm_layer, **kwargs)
        else:
            self.fft1 = ConvFormer(dim, mlp_ratio, drop, drop_path,
                                   act_layer, norm_layer, **kwargs)
        if gate_enable:
            self.gate1 = SGFB(dim)
        else:
            self.gate1 = Sum()
        
        self.ls2 = LayerScale2d(dim, 1.0)
        self.convb = ConvFormer(dim, mlp_ratio, drop, drop_path,
                                act_layer, norm_layer, **kwargs)
        if fft_enable:
            self.fft2 = FFTFormer(dim, mlp_ratio, drop, drop_path,
                                  act_layer, norm_layer, **kwargs)
        else:
            self.fft2 = ConvFormer(dim, mlp_ratio, drop, drop_path,
                                   act_layer, norm_layer, **kwargs)
        if gate_enable:
            self.gate2 = SGFB(dim)
        else:
            self.gate2 = Sum()
            
        return
    
    def forward(self, x):
        g1 = self.ls1(x)
        g2 = self.mlla(x)
        g3 = self.fft1(x)
        gf = self.gate1(g2, g3)
        x = g1 + gf
        
        l1 = self.ls2(x)
        l2 = self.convb(x)
        l3 = self.fft2(x)
        lf = self.gate2(l2, l3)
        x = l1 + lf
        
        return x
    
    
if __name__ == '__main__':
    img = torch.rand((2,64,16,16),dtype=torch.float32)
    model = MFFBlock(64, (16,16), 4)
    model = model.cuda()
    result = model(img)
    print(result.shape)
    