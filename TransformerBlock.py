import numbers

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from deform_conv import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from SoftPool.idea import *
from efficientnet_pytorch.model import MemoryEfficientSwish


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


NEG_INF = -1000000


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # 这是一个条件语句，用于检查 normalized_shape 是否是一个整数（Integral 类型）。
            normalized_shape = (normalized_shape,)
            # 如果 normalized_shape 是整数，将其转化为包含一个元素的元组。
            # 这是为了确保 normalized_shape 最终是一个可迭代对象，即使只有一个参数。
        normalized_shape = torch.Size(normalized_shape)  # torch.Size([256])
        # 最后，将 normalized_shape 转换为 PyTorch 的 torch.Size 对象。
        # torch.Size 是一个包含张量形状维度的特殊类，通常用于确保维度匹配。

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],device='cuda:0', requires_grad=True)
        # torch.Size([512])
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],device='cuda:0', requires_grad=True)
        # torch.Size([512])
        self.normalized_shape = normalized_shape  # torch.Size([512])

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)  # Bottleneck的输入：torch.Size([6, 256, 1])
        # keepdim=True 是指定在计算均值后，保持结果的维度和输入张量一样。
        # 如果 keepdim 为 True，那么输出将保持与输入张量相同的维度数，只是沿着指定维度的尺寸会变成 1。
        sigma = x.var(-1, keepdim=True, unbiased=False)  # Bottleneck的输入：torch.Size([6, 256, 1])
        # var 函数用于计算张量中元素的方差。
        # unbiased=False 意味着方差计算采用不纠正（biased）的方式。
        # 其中 n 是样本数量。如果将 unbiased 设置为 False，则方差计算中的分母会变为 n - 1，这被称为纠正的方差计算，通常用于样本统计。
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        # 这段代码表示了一种对输入张量 x 进行归一化操作的计算


class LayerNorm2(nn.Module):
    def __init__(self, channel, LayerNorm_type):
        super(LayerNorm2, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(channel)
        else:
            self.body = WithBias_LayerNorm(channel)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
        # Bottleneck的输入：torch.Size([6, 512, 16, 16])


###########################################################################
"此处进行改进"


class ChannelInteraction(nn.Module):
    def __init__(self, channel, input_resolution, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.softpool = SoftPool2d(input_resolution)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # nn.ReLU(),
            MemoryEfficientSwish(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        soft_result = self.softpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        soft_out = self.se(soft_result)
        output = self.sigmoid(max_out + avg_out + soft_out)
        return output


class SpatialInteraction(nn.Module):
    def __init__(self, kernel_size1=3, kernel_size2=5, kernel_size3=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size1, padding=kernel_size1 // 2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=kernel_size2, padding=kernel_size2 // 2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=kernel_size3, padding=kernel_size3 // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output1 = self.conv1(result)
        output2 = self.conv2(result)
        output3 = self.conv3(result)
        output = output1 + output2 + output3
        output = self.sigmoid(output)
        return output


###########################################################################

# Axis-based Multi-head Self-Attention
class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, input_resolution, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims  # 512
        self.num_heads = num_heads  # 16
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)

        self.proj = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout(0.)
        #########################################################
        "此处进行改进"
        self.dwConv2d = nn.Sequential(
            nn.Conv2d(num_dims, num_dims, kernel_size=3, stride=1, padding=1, groups=num_dims),
            LayerNorm2(num_dims, 'WithBias'),
            nn.GELU()
        )
        self.channel_interaction = ChannelInteraction(channel=num_dims, input_resolution=input_resolution)
        self.spatial_interaction = SpatialInteraction()
        #########################################################

    def forward(self, x):
        # x: [n, c, h, w]  # torch.Size([1, 32, 256, 256])
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        # torch.Size([1024, 96, 8, 8])
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        # torch.Size([1024, 8, 256])
        k = F.normalize(k, dim=-1)
        # torch.Size([1024, 8, 256])

        res = k.transpose(-2, -1)
        # torch.Size([1024, 256, 8])
        res = torch.matmul(q, res) * self.fac
        # torch.Size([1024, 8, 8])
        res = torch.softmax(res, dim=-1)
        res = torch.matmul(res, v)
        # torch.Size([1024, 8, 256])
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        # torch.Size([1024, 32, 8, 8])
        # res = self.fin(res)
        # # torch.Size([1024, 32, 8, 8])
        ###########################################################
        "此处改进"
        # convolution output
        v_ = einops.rearrange(v, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        conv_x = self.dwConv2d(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(res)

        # C-I
        attened_x = res * torch.sigmoid(channel_map)
        # S-I
        conv_x = conv_x * torch.sigmoid(spatial_map)

        result = attened_x + conv_x

        result = self.proj(result)
        result = self.proj_drop(result)
        ###########################################################
        return result


# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, input_resolution, num_heads, no_mask=False,
                 bias=True) -> None:  # num_dims=512  num_heads=16
        # -> None: - 这部分是函数的注释（函数注解），它指定了函数的返回类型。
        # 在这里，-> None 表示该构造函数不返回任何值（返回类型为None），通常构造函数用于初始化类的实例，而不返回特定的值。
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.row_att = NextAttentionImplZ(num_dims, input_resolution, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, input_resolution, num_heads, bias)
        self.group_size = 3

    def forward(self, x: torch.Tensor):
        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)
        return x


# Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # .chunk(2, dim=1) 被调用。这是一个 PyTorch 的张量方法，用于将张量分成多个块，并返回这些块作为一个元组。
        # 2 是指要将输入张量分成两个块。
        # dim=1 指定了分块操作应该沿着第一个维度（维度索引为 1）进行。
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        # F.gelu(x2) 和 F.gelu(x1)：这两部分代码分别将输入张量 x2 和 x1 分别应用GELU激活函数。
        x = self.project_out(x)
        return x


#  Axis-based Transformer Block
class TransformerBlockShort(nn.Module):
    def __init__(self, channel, input_resolution, num_heads, ffn_expansion_factor=2.66, bias=True,
                 LayerNorm_type='WithBias'):
        super(TransformerBlockShort, self).__init__()

        self.norm1 = LayerNorm2(channel, LayerNorm_type)  # 归一化
        self.attn = NextAttentionZ(channel, input_resolution, num_heads)
        self.norm2 = LayerNorm2(channel, LayerNorm_type)
        self.ffn = FeedForward(channel, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # Bottleneck的输入：torch.Size([6, 512, 16, 16])
        x = x + self.ffn(self.norm2(x))
        # Bottleneck的输入：torch.Size([6, 512, 16, 16])
        return x
