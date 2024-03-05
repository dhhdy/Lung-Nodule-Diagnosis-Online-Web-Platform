import math
import sys
from typing import Dict, Tuple, Any
from torch.nn.functional import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.eva02 import eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE as transformer
from lib.convnext import convnextv2_tiny as convnext
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Dense_DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dense_DoubleConv, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)
        self.conv2 = DoubleConv(in_channels + out_channels, out_channels)
        self.conv3 = DoubleConv(in_channels + out_channels + out_channels, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        add1 = torch.cat([x1, x], dim=1)
        x2 = self.conv2(add1)
        add2 = torch.cat([x2, x1, x], dim=1)
        x3 = self.conv3(add2)
        return x3

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

class CABlock(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CABlock, self).__init__()
        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # b,c,h,w
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        # b,c,1,h
        x_w = self.avg_pool_y(x)
        # b,c,1,w
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))  # 在第三维度上进行拼接后，卷积
        # b,c,1,h+w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)  # 拆分卷积后的维度
        # b,c,1,h    b,c,1,w
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))  # 分别进行sigmoid激活操作
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # b,c,h,1    b,c,1,w
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2, bias=1):
        super(ECAlayer, self).__init__()
        # x: input features with shape [b, c, h, w]
        self.channel = channel
        self.gamma = gamma
        self.bias = bias

        k_size = int(
            abs((math.log(self.channel, 2) + self.bias) / self.gamma))  # (log底数2,c + 1) / 2 ---->(log 2,512 + 1)/2 = 5
        k_size = k_size if k_size % 2 else k_size + 1  # 按照输入通道数自适应的计算卷积核大小

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # 基于全局空间信息的特征描述符
        # b,c,1,1
        # 变换维度，使张量可以进入卷积层
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # 压缩一个维度后，在转换维度
        # b,c,1,1 ----》 b,c,1 ----》 b,1,c      可以理解为输入卷积的batch，只有一个通道所以维度是1，c理解为序列卷积的特征个数
        y = y.transpose(-1, -2).unsqueeze(-1)
        # b,1,c ----》 b,c,1 ----》 b,c,1,1
        # 多尺度信息融合
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 不要加逗号。。
class Channel_Branch(nn.Module):
    def __init__(self, channel, rate):
        super(Channel_Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, groups=channel, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel // rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channel // rate)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        return x2

# 分支结构横向融合
class Ddfusion(nn.Module):
    def __init__(self, in1, in2, h, w, rate):
        super(Ddfusion, self).__init__()
        # // 2
        self.branch1_x = Channel_Branch(in1, 2)
        # // 3
        self.branch2_x = Channel_Branch(in1, 3)
        # // 6
        self.branch3_x = Channel_Branch(in1, 6)
        # // 2
        self.branch1_y = Channel_Branch(in2, 2)
        # // 3
        self.branch2_y = Channel_Branch(in2, 3)
        # // 6
        self.branch3_y = Channel_Branch(in2, 6)

        self.branch_end_x = Channel_Branch(in1*2, 2)
        self.branch_end_y = Channel_Branch(in2*2, 2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.W = nn.Parameter(torch.ones(2))
        # self.wd = nn.Parameter(torch.ones(1))
        self.eca1 = ECAlayer(channel=in1)
        self.cab2 = CABlock(channel=in2, h=h, w=w)

    # conv trans
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # softmax归一化权重
        w1 = torch.exp(self.W[0]) / torch.sum(torch.exp(self.W))
        w2 = torch.exp(self.W[1]) / torch.sum(torch.exp(self.W))
        # # w3 = self.wd
        # 动态加权attention
        x_pool = self.maxpool1(x)
        x11 = self.branch1_x(x)
        x22 = self.branch2_x(x)
        x33 = self.branch3_x(x)
        #
        y_pool = self.maxpool2(y)
        y11 = self.branch1_y(y)
        y22 = self.branch2_y(y)
        y33 = self.branch3_y(y)
        #
        x_c = torch.cat([x11, x22, x33], dim=1) + x
        x_c = torch.cat([x_c, y_pool], dim=1)
        x_c = self.branch_end_x(x_c)
        x_eca = self.eca1(x_c) + x_c
        y_c = torch.cat([y11, y22, y33], dim=1) + y
        y_c = torch.cat([y_c, x_pool], dim=1)
        y_c = self.branch_end_y(y_c)
        y_cab = self.cab2(y_c) + y_c
        #
        # # cos_simi = cosine_similarity(x_c.unsqueeze(1), y_c.unsqueeze(0), dim=-1).sum()
        #
        # # concat
        z = torch.cat([w1 * x_eca, w2 * y_cab], dim=1)
        return z
        # return torch.cat([x, y], dim=1)


class Cyfusion(nn.Module):
    def __init__(self, in1, in2):
        super(Cyfusion, self).__init__()
        self.attn = Attention(in1, in2, in2)

    # conv trans
    def forward(self,x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x2 = self.attn(x1, x2)
        # z = torch.cat([x1, x2], dim=1)
        z = x1
        return z



class Attention(nn.Module):
    def __init__(self, in1, in2, out):
        super(Attention, self).__init__()

class Conv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class TCDNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_classes: int = 1,
                 dim: int = 768,
                 size: int = 224,
                 pretrained: bool = False,
                 drop_rate: float = 0.2
                    ):
        super(TCDNet, self).__init__()
        self.W = nn.Parameter(torch.ones(3))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.transformer1 = transformer(dim=dim, size=size, pretrained=pretrained, is_pe=False)
        # self.transformer2 = transformer(pretrained=pretrained, is_pe=True)
        self.convnext = convnext(pretrained=pretrained)
        self.up11 = Up(dim, dim//2)
        self.up22 = Up(dim//2, dim//4)
        self.up33 = Up(dim//4, dim//8)
        # 分支融合
        self.ddfs1 = Ddfusion(in1=dim, in2=dim, h=size//16, w=size//16, rate=2)
        self.ddfs2 = Ddfusion(in1=dim//2, in2=dim//2, h=size//8, w=size//8, rate=2)
        self.ddfs3 = Ddfusion(in1=dim//4, in2=dim//4, h=size//4, w=size//4, rate=2)
        self.ddfs4 = Ddfusion(in1=dim//8, in2=dim//8, h=size//2, w=size//2, rate=2)

        # skip connection
        self.out_conv = Conv(in_channels=dim//16, num_classes=self.out_classes)

        self.uf = Up(dim * 2, dim)
        self.u1 = Up(dim, dim // 2)
        self.u2 = Up(dim // 2, dim // 4)
        self.u3 = Up(dim // 4, dim // 8)
        self.ou = Conv(dim // 8, self.out_classes)

        self.u2_ = Up(dim, dim // 2)
        self.u3_ = Up(dim // 2, dim //4)
        self.u4_ = Up(dim // 4, dim // 8)
        self.ou_ = Conv(dim // 8,self.out_classes)

        self.u33_ = Up(dim // 2, dim // 4)
        self.u44_ = Up(dim // 4,  dim // 8)
        self.ouu_ = Conv(dim // 8, self.out_classes)

        self.u444_ = Up(dim // 4, dim // 8)
        self.ouuu_ = Conv(dim // 8, self.out_classes)

        self.upsample_f = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.c1 = ChannelAttention(dim // 8)
        self.c2 = ChannelAttention(dim // 8)
        self.c3 = ChannelAttention(dim // 8)
        self.c4 = ChannelAttention(dim // 8)

    def forward(self, x: torch.Tensor) -> Tuple[Any, Any, Any, Any]:

        # # softmax归一化权重
        # w1 = torch.exp(self.W[0]) / torch.sum(torch.exp(self.W))
        # w2 = torch.exp(self.W[1]) / torch.sum(torch.exp(self.W))
        # w3 = torch.exp(self.W[2]) / torch.sum(torch.exp(self.W))
        # tranformer_upsample
        x0 = self.transformer1(x)
        x0 = torch.transpose(x0, 1, 2)
        x1 = x0.reshape(x0.shape[0], -1, 14, 14)  #[768,14,14]
        x2 = self.up11(x1)  #[384,28,28]
        x3 = self.up22(x2)  #[192,56,56]
        x4 = self.up33(x3)  #[96,112,112]

        #convnext_downsample
        x_ = self.upsample(x)
        y3 = self.convnext.downsample_layers[0](x_)
        y3 = self.convnext.stages[0](y3)  #[96,112,112]
        y2 = self.convnext.downsample_layers[1](y3)
        y2 = self.convnext.stages[1](y2)  #[192,56,56]
        y1 = self.convnext.downsample_layers[2](y2)
        y1 = self.convnext.stages[2](y1)  #[384,28,28]
        y0 = self.convnext.downsample_layers[3](y1)
        y0 = self.convnext.stages[3](y0)  #[768,14,14]

        #234
        #fusion
        z1 = self.ddfs1(y0, x1)  #[1436]
        z2 = self.ddfs2(y1, x2)  #[768,28,28]
        z3 = self.ddfs3(y2, x3)   #[384,,56]
        z4 = self.ddfs4(y3, x4)  #[192,112,112]


        d0_0 = self.upsample_f(y3 + x4)

        d1_0 = z4
        d0_1 = self.u444_(d1_0)
        d0_1 = d0_1 + d0_0
        q1 = self.c1(d0_1)
        d0_1 = q1 * d0_1
        out1 = self.ouuu_(d0_1)

        d2_0 = z3
        d1_1 = self.u33_(d2_0)
        d1_1 = d1_1 + d1_0
        d0_2 = self.u44_(d1_1)
        d0_2 = d0_2 + d0_1 + d0_0
        q2 = self.c2(d0_2)
        d0_2 = q2 * d0_2
        out2 = self.ouu_(d0_2)

        d3_0 = z2
        d2_1 = self.u2_(d3_0)
        d2_1 = d2_1 + d2_0
        d1_2 = self.u3_(d2_1)
        d1_2 = d1_2 + d1_1 + d1_0
        d0_3 = self.u4_(d1_2)
        d0_3 = d0_3 + d0_2 + d0_1 +d0_0
        q3 = self.c3(d0_3)
        d0_3 = q3 * d0_3
        out3 = self.ou_(d0_3)

        d4_0 = z1
        d3_1 = self.uf(d4_0)
        d3_1 = d3_1 + d3_0
        d2_2 = self.u1(d3_1)   #[384,28,28]
        d2_2 = d2_2 + d2_1 + d2_0
        d1_3 = self.u2(d2_2)
        d1_3 = d1_3 + d1_2 + d1_1 + d1_0
        d0_4 = self.u3(d1_3)
        d0_4 = d0_4 + d0_3 + d0_2 + d0_1 + d0_0
        q4 = self.c4(d0_4)
        d0_4 = q4 * d0_4
        out4 = self.ou(d0_4)

        out2 = out2 + out1
        out3 = out3 + out2 +out1
        out4 = out4 + out3 + out1

        return out1, out2, out3, out4

#     def init_weights(self):
#         self.ddfs1.apply(init_weights)
#         self.ddfs2.apply(init_weights)
#         self.ddfs3.apply(init_weights)
#         self.cyfs1.apply(init_weights)
#         self.cyfs2.apply(init_weights)
#         self.cyfs3.apply(init_weights)
#         self.out_conv.apply(init_weights)
#
# def init_weights(m):
#     """
#     Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
#     "nn.Module"
#     :param m: Layer to initialize
#     :return: None
#     """
#     if isinstance(m, nn.Conv2d):
#         '''
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#         trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#         '''
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(m.bias, -bound, bound)
#
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

