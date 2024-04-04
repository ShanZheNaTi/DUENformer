import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import einops
from einops import rearrange
from deform_conv import *
from TransformerBlock import *
from block import *


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * 3)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout(0.)

        ####################################################################################
        "此处进行改进"
        self.deformConv2d = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            LayerNorm2(dim, 'WithBias'),
            nn.GELU()
        )
        self.channel_interaction = ChannelInteraction(channel=dim, input_resolution=input_resolution)
        self.spatial_interaction = SpatialInteraction()
        ####################################################################################

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        # torch.Size([4, 108, 256, 256])
        q, k, v = qkv.chunk(3, dim=1)
        # q/k/v=torch.Size([4, 36, 256, 256])
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # torch.Size([4, 2, 18, 65536])
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # torch.Size([4, 2, 18, 65536])
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # torch.Size([4, 2, 18, 65536])

        q = torch.nn.functional.normalize(q, dim=-1)
        # torch.Size([4, 2, 18, 65536])
        k = torch.nn.functional.normalize(k, dim=-1)
        # torch.Size([4, 2, 18, 65536])

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(int(c / self.num_heads))
        # torch.Size([4, 2, 18, 18])
        attn = attn.softmax(dim=-1)
        # torch.Size([4, 2, 18, 18])
        out = (attn @ v)
        # torch.Size([4, 2, 18, 65536])
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # torch.Size([4, 36, 256, 256])
        # out = self.project_out(out)
        # # torch.Size([4, 36, 256, 256])
        #########################################################
        "此处进行改进"
        # convolution output
        v_ = einops.rearrange(v, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        conv_x = self.deformConv2d(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(out)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x)

        # S-I
        attened_x = out * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)

        result = attened_x + conv_x

        result = self.proj(result)
        result = self.proj_drop(result)
        #########################################################
        return result


class CATransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, bias):
        super(CATransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = ChannelAttention(dim, input_resolution, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # torch.Size([4, 36, 256, 256])
        x = x + self.ffn(self.norm2(x))
        # torch.Size([4, 36, 256, 256])
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, bias):
        super(TransformerBlock, self).__init__()

        self.pixel = TransformerBlockShort(channel=dim, input_resolution=input_resolution, num_heads=num_heads,
                                           bias=bias)
        # torch.Size([4, 36, 256, 256])

    def forward(self, x):
        x = self.pixel(x)
        return x


class TransformerBlock1(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, bias):
        super(TransformerBlock1, self).__init__()

        self.pixel = TransformerBlockShort(channel=dim, input_resolution=input_resolution, num_heads=num_heads,
                                           bias=bias)
        # torch.Size([4, 36, 256, 256])
        self.glob = CATransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, bias=bias)

    def forward(self, x):
        x = self.glob(self.pixel(x))
        return x


class DeformConv2dBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.deformConv2d = DeformConv2d(dim, dim)

    def forward(self, x):
        x = self.deformConv2d(x) + x
        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        # _, _, h, w = x.shape: 这一行代码从输入张量 x 的形状中提取了两个维度的值，分别是高度（h）和宽度（w）。
        # 前面的两个下划线 _ 表示在此处我们不关心的维度。
        if h % 2 != 0:
            # if h % 2 != 0:: 这个条件语句检查高度 h 是否为奇数。
            # 如果是奇数，下一行的操作会在高度的底部（dim=2）进行零填充（F.pad(x, [0, 0, 1, 0])）
            x = F.pad(x, [0, 0, 1, 0])
            # 在这里 [0, 0, 1, 0] 表示在高度的底部填充一行零。
        if w % 2 != 0:
            # if w % 2 != 0:: 同样，这个条件语句检查宽度 w 是否为奇数。
            # 如果是奇数，下一行的操作会在宽度的右侧（dim=3）进行零填充（F.pad(x, [1, 0, 0, 0])）
            x = F.pad(x, [1, 0, 0, 0])
            # [1, 0, 0, 0] 表示在宽度的右侧填充一列零。
            # 总体来说，这段代码的目的似乎是确保输入张量的高度和宽度都是偶数
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x


##########################################################################
class Generator(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=36,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=2,
                 heads=[2, 2, 2, 2],
                 bias=False):
        super(Generator, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_spatial = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], input_resolution=256, bias=bias) for i in
            range(num_blocks[0])])
        self.deformConv2d1 = DeformConv2dBlock(36)
        self.encoder_level1_channel = nn.Sequential(*[
            CATransformerBlock(dim=dim * 4, input_resolution=128, num_heads=heads[0], bias=bias) for i in
            range(num_blocks[0])])
        self.conv1 = nn.Conv2d(dim * 4, dim * 2, kernel_size=3, padding=1, bias=bias)

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2_spatial = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), input_resolution=128, num_heads=heads[1], bias=bias) for i in
            range(num_blocks[1])])
        self.deformConv2d2 = DeformConv2dBlock(72)
        self.encoder_level2_channel = nn.Sequential(*[
            CATransformerBlock(dim=(dim * 2 ** 1) * 4, input_resolution=64, num_heads=heads[1], bias=bias) for i in
            range(num_blocks[1])])
        self.conv2 = nn.Conv2d((dim * 2 ** 1) * 4, (dim * 2 ** 1) * 2, kernel_size=3, padding=1, bias=bias)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3_spatial = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), input_resolution=64, num_heads=heads[2], bias=bias) for i in
            range(num_blocks[2])])
        self.deformConv2d3 = DeformConv2dBlock(144)
        self.encoder_level3_channel = nn.Sequential(*[
            CATransformerBlock(dim=(dim * 2 ** 2) * 4, input_resolution=32, num_heads=heads[2], bias=bias) for i in
            range(num_blocks[2])])
        self.conv3 = nn.Conv2d((dim * 2 ** 2) * 4, (dim * 2 ** 2) * 2, kernel_size=3, padding=1, bias=bias)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock1(dim=int(dim * 2 ** 3), input_resolution=32, num_heads=heads[3], bias=bias) for i in
            range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), input_resolution=64, num_heads=heads[2], bias=bias) for i in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), input_resolution=128, num_heads=heads[1], bias=bias) for i in
            range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], input_resolution=256, bias=bias) for i in
            range(num_blocks[0])])

        self.reduce_chan_ref = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.refinement = nn.Sequential(*[
            CATransformerBlock(dim=int(dim), num_heads=heads[0], input_resolution=256, bias=bias) for i in
            range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        # torch.size([4, 36, 256, 256])
        ##########################################
        "此处改进"
        out_enc_level1_0 = self.encoder_level1_spatial(inp_enc_level1)
        # torch.size([4, 36, 256, 256])
        out_enc_level1 = self.down1_2(out_enc_level1_0)
        # torch.Size([4, 72, 128, 128])
        deform1_0 = self.deformConv2d1(inp_enc_level1)
        # torch.size([4, 36, 256, 256])
        deform1 = self.down1_2(deform1_0)
        # torch.Size([4, 72, 128, 128])
        stage1 = cat(out_enc_level1, deform1)
        # torch.size([4, 144, 128, 128])
        stage1 = self.encoder_level1_channel(stage1)
        # torch.size([4, 144, 128, 128])
        inp_enc_level2 = self.conv1(stage1)
        # torch.size([4, 72, 128, 128])
        ##########################################
        "此处改进"
        out_enc_level2_0 = self.encoder_level2_spatial(inp_enc_level2)
        # torch.Size([4, 72, 128, 128])
        out_enc_level2 = self.down2_3(out_enc_level2_0)
        # torch.Size([4, 144, 64, 64])
        deform2_0 = self.deformConv2d2(deform1)
        # torch.Size([4, 72, 128, 128])
        deform2 = self.down2_3(deform2_0)
        # torch.Size([4, 144, 64, 64])
        stage2 = cat(out_enc_level2, deform2)
        # torch.Size([4, 288, 64, 64])
        stage2 = self.encoder_level2_channel(stage2)
        # torch.Size([4, 288, 64, 64])
        inp_enc_level3 = self.conv2(stage2)
        # torch.Size([4, 144, 64, 64])
        ###########################################
        "此处改进"
        out_enc_level3_0 = self.encoder_level3_spatial(inp_enc_level3)
        # torch.Size([4, 144, 64, 64])
        out_enc_level3 = self.down3_4(out_enc_level3_0)
        # torch.Size([4, 288, 32, 32])
        deform3_0 = self.deformConv2d3(deform2)
        # torch.Size([4, 144, 64, 64])
        deform3 = self.down3_4(deform3_0)
        # torch.Size([4, 288, 32, 32])
        stage3 = cat(out_enc_level3, deform3)
        # torch.Size([4, 576, 32, 32])
        stage3 = self.encoder_level3_channel(stage3)
        # torch.Size([4, 576, 32, 32])
        inp_enc_level4 = self.conv3(stage3)
        # torch.Size([4, 288, 32, 32])
        ###########################################

        latent = self.latent(inp_enc_level4)
        # torch.Size([4, 288, 32, 32])
        #######################################################

        inp_dec_level3 = self.up4_3(latent)
        # torch.Size([4, 144, 64, 64])
        inp_dec_level3 = cat(inp_dec_level3, out_enc_level3_0 + deform3_0)
        # torch.Size([4, 288, 64, 64])
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # torch.Size([4, 144, 64, 64])
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        # torch.Size([4, 144, 64, 64])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        # torch.Size([4, 72, 128, 128])
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2_0 + deform2_0)
        # torch.Size([4, 144, 128, 128])
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # torch.Size([4, 72, 128, 128])
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # torch.Size([4, 72, 128, 128])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        # torch.Size([4, 36, 256, 256])
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1_0 + deform1_0)
        # torch.Size([4, 72, 256, 256])
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        # torch.Size([4, 36, 256, 256])
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # torch.Size([4, 36, 256, 256])

        ref_out = self.refinement(out_dec_level1)
        # torch.Size([4, 36, 256, 256])
        out = self.output(ref_out) + inp_img
        # torch.Size([4, 3, 256, 256])
        return out
