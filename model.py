# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math
import cv2
# from TransformerBlock import TransformerBlock


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=bias)
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x

        input_x = input_x.view(batch, channel, height * width)
        # [N, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, C, H * W]
        ##########################################
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        result = self.conv_3(result)
        context_mask = result + self.conv_mask(x)
        # [N, 1, H, W]
        ##########################################
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, H * W, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, 1, C, 1]
        context = context.view(batch, channel, 1, 1)
        # [N, C, 1, 1]
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x


##########################################################################
# --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


class CAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(CAM, self).__init__()
        act = nn.LeakyReLU(0.2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            act,
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(in_channels, int(out_channels / 8), kernel_size, padding=0, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(int(out_channels / 8), out_channels, kernel_size, padding=0, bias=False))

        self.conv2 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                   nn.Conv2d(in_channels, int(out_channels / 8), kernel_size, padding=0, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(int(out_channels / 8), out_channels, kernel_size, padding=0, bias=False))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out1 = self.body(x)
        out2 = self.softmax(self.conv1(out1) + self.conv2(out1))
        out = out2 * out1
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()

        # 1x1 Convolution branch
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 1x1 Convolution followed by 3x3 Convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 1x1 Convolution followed by 5x5 Convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # 3x3 MaxPooling followed by 1x1 Convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class WB_CLAHE(nn.Module):
    def __init__(self):
        super(WB_CLAHE, self).__init__()

    def apply_white_balance(self, image):
        # 将图像从RGB转换为Lab颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(lab)

        # 确保 L 通道为 uint8 类型
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        # 对L通道进行直方图均衡化
        l = cv2.equalizeHist(l)

        # 确保 a 和 b 通道与 L 通道类型一致
        a = a.astype('uint8')
        b = b.astype('uint8')

        # 合并Lab通道，并转换回RGB颜色空间
        lab = cv2.merge((l, a, b))
        balanced_image = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        return balanced_image

    def apply_clahe(self, image):
        # 将图像从RGB转换为Lab颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(lab)

        # 确保 L 通道为 uint8 类型
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 应用CLAHE到L通道
        l = clahe.apply(l)

        # 确保 a 和 b 通道与 L 通道类型一致
        a = a.astype('uint8')
        b = b.astype('uint8')

        # 合并Lab通道，并转换回RGB颜色空间
        lab = cv2.merge((l, a, b))
        clahe_image = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

        return clahe_image

    def forward(self, x):
        # 假设输入的图像为 (N, 3, H, W) 的 tensor
        N, C, H, W = x.shape
        # 初始化输出张量
        processed_images_wb = torch.zeros_like(x)
        processed_images_clahe = torch.zeros_like(x)
        for i in range(N):
            # 将图像从 tensor 转为 numpy 格式，并将维度从 (3, H, W) 转为 (H, W, 3)
            image = x[i].detach().cpu().numpy().transpose(1, 2, 0)
            # 应用白平衡和CLAHE
            balanced_image = self.apply_white_balance(image)
            enhanced_image = self.apply_clahe(image)
            # 将结果图像转换回 (3, H, W) 格式，并存入结果张量
            processed_images_wb[i] = torch.from_numpy(balanced_image.transpose(2, 0, 1)).to(x.device)
            processed_images_clahe[i] = torch.from_numpy(enhanced_image.transpose(2, 0, 1)).to(x.device)
        return processed_images_wb, processed_images_clahe


class ColorCorrect(nn.Module):
    def __init__(self, out1=16, out2=16, out3=24, out4=16, out5=24, out6=16):
        super(ColorCorrect, self).__init__()
        self.bot = InceptionModule(in_channels=1, out_1x1=out1, red_3x3=out2, out_3x3=out3, red_5x5=out4, out_5x5=out5,
                                   out_pool=out6)
        self.cam = CAM(in_channels=out1 + out3 + out5 + out6, out_channels=out1 + out3 + out5 + out6, kernel_size=1)
        self.wb_clahe = WB_CLAHE()

    def rgb_to_lab_batch(self, img_tensor):
        # 假设输入的img_tensor形状为 (N, C, H, W)
        N, C, H, W = img_tensor.shape

        # 获取输入张量的设备以及数据类型
        device = img_tensor.device
        dtype = img_tensor.dtype

        # 将张量从 (N, C, H, W) 转换为 (N, H, W, C)
        img_tensor = img_tensor.permute(0, 2, 3, 1)

        # 将张量从 [0, 1] 转换为 [0, 255] 范围
        img_uint8 = (img_tensor * 255).byte().cpu().numpy()

        # 准备输出张量
        L_batch = []
        A_batch = []
        B_batch = []

        for i in range(N):
            # 对每一张图片进行RGB到LAB转换
            img_lab = cv2.cvtColor(img_uint8[i], cv2.COLOR_RGB2LAB)

            # 提取L、A、B分量
            L, A, B = cv2.split(img_lab)

            # 如果需要将L、A、B分量转换回 [0, 1] 范围的浮点类型
            L = L / 255.0
            A = (A - 128) / 255.0
            B = (B - 128) / 255.0

            # 将L、A、B转换回tensor并添加到输出列表
            L_batch.append(torch.from_numpy(L).unsqueeze(0).to(dtype))
            A_batch.append(torch.from_numpy(A).unsqueeze(0).to(dtype))
            B_batch.append(torch.from_numpy(B).unsqueeze(0).to(dtype))

        # 将列表堆叠成batch
        L_batch = torch.stack(L_batch).to(device)
        A_batch = torch.stack(A_batch).to(device)
        B_batch = torch.stack(B_batch).to(device)

        return L_batch, A_batch, B_batch

    def forward(self, x):
        wb, clahe = self.wb_clahe(x)
        L_batch, A_batch, B_batch = self.rgb_to_lab_batch(x + wb + clahe)

        L_batch = self.bot(L_batch)
        A_batch = self.bot(A_batch)
        B_batch = self.bot(B_batch)

        L_batch = self.cam(L_batch)
        A_batch = self.cam(A_batch)
        B_batch = self.cam(B_batch)

        out = L_batch + A_batch + B_batch

        return out


class Down_branch1(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down_branch1, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample_branch1(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2):
        super(DownSample_branch1, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down_branch1(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up_branch1(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up_branch1, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample_branch1(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2):
        super(UpSample_branch1, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up_branch1(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


# ---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, channel_red):
        super(UpSample, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        if channel_red:
            self.post = nn.Conv2d(channels, channels // 2, 1, 1, 0)

        else:
            self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape
        fft_x = torch.fft.fft2(x)
        # 这段代码使用 PyTorch 中的 fft.fft2 函数对输入的二维张量 x 执行二维离散傅里叶变换（2D FFT）。
        mag_x = torch.abs(fft_x)
        # 这段代码使用 PyTorch 中的 torch.abs 函数计算输入张量 fft_x 的绝对值，然后将结果保存在 mag_x 中。
        # 通常，对傅里叶变换的结果取绝对值是为了获取频域中的幅度信息，即振幅谱。
        pha_x = torch.angle(fft_x)
        # 这段代码使用 PyTorch 中的 torch.angle 函数计算输入张量 fft_x 中每个元素的相位角，然后将结果保存在 pha_x 中。
        # 相位角表示了复数在极坐标中的相对位置，通常在傅里叶变换的上下文中用于获取频域中的相位信息。
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        amp_fuse = torch.tile(Mag, (2, 2))
        # 这段代码使用 PyTorch 中的 torch.tile 函数创建一个新的张量 amp_fuse，该张量通过在维度上复制另一个张量 Mag 的值。
        # 在这里，Mag 应该是一个二维张量，而 amp_fuse 是通过在每个维度上重复 Mag 的值而得到的。
        # (2, 2) 是 tile 函数的参数，指定了在每个维度上进行的复制次数。
        # 在这里，Mag 的每个维度上都被复制了两次，因此得到的 amp_fuse 的大小是原始 Mag 大小的两倍。
        pha_fuse = torch.tile(Pha, (2, 2))
        real = amp_fuse * torch.cos(pha_fuse)
        # amp_fuse 是包含某种振幅信息的张量，
        # 逐元素乘法操作 amp_fuse * torch.cos(pha_fuse) 表示对 amp_fuse 中的每个元素分别与相应的 torch.cos(pha_fuse) 中的元素相乘。
        # 这是一种按元素的操作，它将振幅和相位信息结合在一起。
        # 结果张量 real 包含了每个元素的振幅信息乘以相位信息的余弦值，这通常表示复数在实轴上的分量。
        # 这种操作通常用于傅里叶变换的频域到时域的逆变换，以还原原始的时域信号。
        imag = amp_fuse * torch.sin(pha_fuse)
        # amp_fuse 是包含某种振幅信息的张量
        # torch.sin(pha_fuse) 计算 pha_fuse 中每个元素的正弦值。pha_fuse 应该是一个张量，包含相位信息。
        # 逐元素乘法操作 amp_fuse * torch.sin(pha_fuse) 表示对 amp_fuse 中的每个元素分别与相应的 torch.sin(pha_fuse) 中的元素相乘。
        # 这是一种按元素的操作，它将振幅和相位信息结合在一起。
        # 结果张量 imag 包含了每个元素的振幅信息乘以相位信息的正弦值，这通常表示复数在虚轴上的分量。
        # 这种操作通常用于傅里叶变换的频域到时域的逆变换，以还原原始的时域信号。
        out = torch.complex(real, imag)
        # 这段代码使用 PyTorch 中的 torch.complex 函数，通过给定的实部 (real) 和虚部 (imag) 创建一个复数张量 out。
        # 在频域分析或傅里叶变换的上下文中，这种复数张量通常表示一个复数信号，其中实部对应于在实轴上的分量，虚部对应于在虚轴上的分量。
        # torch.complex 函数接受两个张量作为参数，分别表示实部和虚部，并返回一个复数张量。
        # 这种复数张量通常用于傅里叶变换或频域分析的结果，其中复数表示信号在频域的表示。
        # 在时域和频域之间的转换中，复数张量的实部和虚部包含了信号在不同维度上的信息。
        output = torch.fft.ifft2(out)
        # 这段代码使用 PyTorch 中的 torch.fft.ifft2 函数对输入的复数张量 out 进行二维逆离散傅里叶变换（2D IFFT）。
        # 逆离散傅里叶变换用于将频域表示转换回时域表示。
        # 这一行代码的目的是将复数张量 out 进行二维逆离散傅里叶变换，得到信号在时域的表示，结果保存在 output 中。
        # 这种操作通常用于还原原始信号，将其从频域转换回时域。
        output = torch.abs(output)
        # 这段代码使用 PyTorch 中的 torch.abs 函数计算逆傅里叶变换的结果 output 中每个元素的绝对值。
        # 在信号处理的上下文中，这通常用于获取还原信号的振幅信息。
        # torch.abs 函数用于计算输入张量的元素绝对值。对于复数张量，这将返回每个复数的振幅，即实部和虚部的平方和的平方根。
        # 总之，这一行代码的目的是从逆傅里叶变换的结果中提取振幅信息。这种操作通常用于还原原始的时域信号的振幅谱，可能是频域到时域的转换后的一个步骤。
        return self.post(output)


class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class UpS(nn.Module):
    def __init__(self, channels):
        super(UpS, self).__init__()
        self.Fups = UpSample(channels, True)
        self.Sups = UpSample1(channels)
        self.reduce = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)

    def forward(self, x):
        out = torch.cat([self.Fups(x), self.Sups(x)], dim=1)
        # print(out.shape)
        return self.reduce(out)


# ---------- Multi-Scale Resiudal Block (MRB) ----------
class color_context_Net(nn.Module):
    def __init__(self, inp_channels, n_feat1, n_feat2, chan_factor1, chan_factor2, bias):
        super(color_context_Net, self).__init__()

        self.dau_top = RCB(int(n_feat1 * chan_factor1 ** 0), bias=bias, groups=1)
        self.dau_mid = RCB(int(n_feat1 * chan_factor1 ** 1), bias=bias, groups=2)
        self.dau_bot = RCB(int(n_feat1 * chan_factor1 ** 2), bias=bias, groups=4)

        self.ColorCorrect_top = ColorCorrect(out1=16, out2=16, out3=24, out4=16, out5=24, out6=16)
        self.ColorCorrect_middle = ColorCorrect(out1=24, out2=24, out3=36, out4=24, out5=36, out6=24)
        self.ColorCorrect_bottom = ColorCorrect(out1=36, out2=36, out3=54, out4=36, out5=54, out6=36)

        self.conv_in_branch1 = nn.Conv2d(inp_channels, n_feat1, kernel_size=3, padding=1, bias=bias)

        self.down1 = DownSample_branch1(int((chan_factor1 ** 0) * n_feat1), 2, chan_factor1)
        self.down2 = DownSample_branch1(int((chan_factor1 ** 1) * n_feat1), 2, chan_factor1)
        self.up21_1 = UpSample_branch1(int((chan_factor1 ** 1) * n_feat1), 2, chan_factor1)
        self.up32_1 = UpSample_branch1(int((chan_factor1 ** 2) * n_feat1), 2, chan_factor1)

        self.skff_top = SKFF(int(n_feat1 * chan_factor1 ** 0), 2)
        self.skff_mid = SKFF(int(n_feat1 * chan_factor1 ** 1), 2)

        self.conv_out_branch1 = nn.Conv2d(n_feat1, 3, kernel_size=1, padding=0, bias=bias)

        self.conv_in_branch2 = nn.Conv2d(inp_channels, n_feat2, kernel_size=3, padding=1, bias=bias)

        self.TransformerBlock_top = TransformerBlock(channels=int(n_feat2 * chan_factor2 ** 0), num_heads=1,
                                                     expansion_factor=2.66, input_resolution=256)
        self.TransformerBlock_mid = TransformerBlock(channels=int(n_feat2 * chan_factor2 ** 1), num_heads=2,
                                                     expansion_factor=2.66, input_resolution=128)
        self.TransformerBlock_bot = TransformerBlock(channels=int(n_feat2 * chan_factor2 ** 2), num_heads=4,
                                                     expansion_factor=2.66, input_resolution=64)
        self.latent = TransformerBlock(channels=int(n_feat2 * chan_factor2 ** 3), num_heads=8, expansion_factor=2.66,
                                       input_resolution=32)
        self.refinement = TransformerBlock(channels=int(n_feat2 * chan_factor2 ** 0), num_heads=1,
                                           expansion_factor=2.66, input_resolution=256)

        self.up5 = UpS(128)
        self.up4 = UpS(64)
        self.up3 = UpS(32)

        self.down3 = DownSample(16)
        self.down4 = DownSample(32)
        self.down5 = DownSample(64)

        self.conv_out_branch2 = nn.Conv2d(n_feat2, 3, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x_top_color = x.clone()  # 256
        # [N, 3, 256, 256]
        x_mid_color = F.interpolate(x_top_color, scale_factor=0.5)  # 128
        # [N, 3, 128, 128]
        x_bot_color = F.interpolate(x_top_color, scale_factor=0.25)  # 64
        # [N, 3, 64, 64]

        x_top_color = self.ColorCorrect_top(x_top_color)
        # [N, 80, 256, 256]
        x_mid_color = self.ColorCorrect_middle(x_mid_color)
        # [N, 120, 128, 128]
        x_bot_color = self.ColorCorrect_bottom(x_bot_color)
        # [N, 180, 64, 64]

        shallow_feats1 = self.conv_in_branch1(x)
        # [N, 80, 256, 256]
        x_top_context = shallow_feats1.clone()
        # [N, 80, 256, 256]
        x_mid_context = self.down1(x_top_context)
        # [N, 120, 128, 128]
        x_bot_context = self.down2(x_mid_context)
        # [N, 180, 64, 64]

        x_top_context = self.dau_top(x_top_context + x_top_color)
        # [N, 80, 256, 256]
        x_mid_context = self.dau_mid(x_mid_context + x_mid_color)
        # [N, 120, 128, 128]
        x_bot_context = self.dau_bot(x_bot_context + x_bot_color)
        # [N, 180, 64, 64]

        x_mid_context = self.skff_mid([x_mid_context, self.up32_1(x_bot_context)])
        x_top_context = self.skff_top([x_top_context, self.up21_1(x_mid_context)])
        # [N, 80, 256, 256]

        x_top_context = self.dau_top(x_top_context + x_top_color)
        # [N, 80, 256, 256]
        x_mid_context = self.dau_mid(x_mid_context + x_mid_color)
        # [N, 120, 128, 128]
        x_bot_context = self.dau_bot(x_bot_context + x_bot_color)
        # [N, 180, 64, 64]

        x_mid_context = self.skff_mid([x_mid_context, self.up32_1(x_bot_context)])
        x_top_context = self.skff_top([x_top_context, self.up21_1(x_mid_context)])
        # [N, 80, 256, 256]
        out1 = self.conv_out_branch1(x_top_context)
        ###################################################################
        # Transformer部分
        shallow_feats2 = self.conv_in_branch2(x)
        # [N, 16, 256, 256]
        encoder1 = self.TransformerBlock_top(shallow_feats2)
        encoder_down = self.down3(encoder1)
        encoder2 = self.TransformerBlock_mid(encoder_down)
        encoder2_down = self.down4(encoder2)
        encoder3 = self.TransformerBlock_bot(encoder2_down)
        encoder3_down = self.down5(encoder3)

        latent = self.latent(encoder3_down)

        decoder1 = self.TransformerBlock_bot(self.up5(latent) + encoder3)
        decoder2 = self.TransformerBlock_mid(self.up4(decoder1) + encoder2)
        decoder3 = self.TransformerBlock_top(self.up3(decoder2) + encoder1)
        refinement = self.refinement(self.refinement(decoder3))

        out2 = self.conv_out_branch2(refinement)
        ###################################################################

        out = out1 + out2
        out = out + x
        return out


# ---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat1=80, n_feat2=16, chan_factor1=1.5, chan_factor2=2, bias=False):
        super(RRG, self).__init__()
        modules_body = [color_context_Net(inp_channels=3, n_feat1=n_feat1, chan_factor1=chan_factor1, n_feat2=n_feat2,
                                          chan_factor2=chan_factor2, bias=bias) for _ in range(1)]
        modules_body.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
