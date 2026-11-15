"""A single Restormer-inspired channel-attention residual adapter."""
import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm2d(nn.Module):
    """Layer-normalize channels independently at every spatial location."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()


class MDTAAdapter(nn.Module):
    """Channel correlation over spatially normalized Q/K at each attention head."""

    def __init__(self, channels=512, heads=4, residual_scale=0.05):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.channels = channels
        self.heads = heads
        self.norm = ChannelLayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3,
            bias=False)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f'Expected {self.channels} channels, got {channels}')
        q, k, v = self.qkv_depthwise(self.qkv(self.norm(x))).chunk(3, dim=1)
        q = q.reshape(batch, self.heads, channels // self.heads, height * width)
        k = k.reshape(batch, self.heads, channels // self.heads, height * width)
        v = v.reshape(batch, self.heads, channels // self.heads, height * width)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (attention * self.temperature.float()).softmax(dim=-1)
        output = torch.matmul(attention, v.float()).reshape(batch, channels, height, width)
        output = self.project(output.to(x.dtype))
        return x + self.residual_scale.to(x.dtype) * output
