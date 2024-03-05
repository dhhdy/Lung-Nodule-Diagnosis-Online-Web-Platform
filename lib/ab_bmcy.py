import math
from typing import  Tuple, Any
import torch
import torch.nn as nn
from lib.eva02 import eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE as transformer
from lib.convnext import convnextv2_femto as convnext

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        # b,c,1,h+w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        # b,c,1,h    b,c,1,w
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # b,c,h,1    b,c,1,w
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(in_planes, in_planes, 1, bias=True, groups=in_planes)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.ones(2))

    def forward(self, x):
        w1 = torch.exp(self.W[0]) / torch.sum(torch.exp(self.W))
        w2 = torch.exp(self.W[1]) / torch.sum(torch.exp(self.W))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(self.fc(x)))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(self.fc(x)))))
        out = w1 * avg_out + w2 * max_out
        return self.sigmoid(out)


class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2, bias=1):
        super(ECAlayer, self).__init__()
        # x: input features with shape [b, c, h, w]
        self.channel = channel
        self.gamma = gamma
        self.bias = bias

        k_size = int(
            abs((math.log(self.channel, 2) + self.bias) / self.gamma))
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #size down(1, 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)  # element-wise dot

class Channel_Branch(nn.Module):
    def __init__(self, channel, rate):
        super(Channel_Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, groups=channel, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel // rate, kernel_size=1, bias=True)
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
        self.eca1 = ECAlayer(channel=in1)
        self.cab2 = CABlock(channel=in2, h=h, w=w)

    # conv trans
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # softmax
        w1 = 2. * torch.exp(self.W[0]) / torch.sum(torch.exp(self.W))
        w2 = 2. * torch.exp(self.W[1]) / torch.sum(torch.exp(self.W))
        # dmic attention
        x_pool = self.maxpool1(x)
        x11 = self.branch1_x(x)
        x22 = self.branch2_x(x)
        x33 = self.branch3_x(x)

        y_pool = self.maxpool2(y)
        y11 = self.branch1_y(y)
        y22 = self.branch2_y(y)
        y33 = self.branch3_y(y)

        x_c = torch.cat([x11, x22, x33], dim=1) + x
        x_ = torch.cat([x_c, x_pool], dim=1)
        x_c = self.branch_end_x(x_) + x_c
        x_eca = x_c * self.eca1(x_c)
        y_c = torch.cat([y11, y22, y33], dim=1) + y
        y_ = torch.cat([y_c, y_pool], dim=1)
        y_c = self.branch_end_y(y_) + y_c
        y_cab = self.cab2(y_c)
        #
        # # concat
        z = torch.cat([w1 * x_eca, w2 * y_cab], dim=1)
        z = torch.cat([x, y], dim=1)
        return z


class Ddtention(nn.Module):
    def __init__(self, in1, in2, h, w, rate):
        super(Ddtention, self).__init__()
        # // 2
        self.branch1_x = Channel_Branch(in1, 2)
        # // 3
        self.branch2_x = Channel_Branch(in1, 3)
        # // 6
        self.branch3_x = Channel_Branch(in1, 6)

        self.branch_end1_x = Channel_Branch(in1*2, 2)
        self.branch_end2_x = Channel_Branch(in2*2, 2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.W = nn.Parameter(torch.ones(2))
        self.eca1 = ECAlayer(channel=in1)

    # conv trans
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # softmax
        w1 = 2. * (torch.exp(self.W[0]) / torch.sum(torch.exp(self.W)))
        w2 = 2. * (torch.exp(self.W[1]) / torch.sum(torch.exp(self.W)))
        # dmic attention
        x_pool = self.maxpool1(x)
        x11 = self.branch1_x(x)
        x22 = self.branch2_x(x)
        x33 = self.branch3_x(x)
        #
        x_c = torch.cat([x11, x22, x33, x_pool], dim=1)
        x_c = self.branch_end1_x(x_c)
        x_c = torch.cat([x_c, y], dim=1)
        x_c = self.branch_end2_x(x_c)
        x_eca = self.eca1(x_c)
        x_eca = x * x_eca
        # # concat
        z = torch.cat([w1 * x, w2 * x_eca], dim=1)
        return z

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
                    ):
        super(TCDNet, self).__init__()
        self.transformer1 = transformer(dim=dim, size=size, pretrained=pretrained, is_pe=False)
        self.convnext = convnext(pretrained=pretrained)
        self.up1 = Up(dim, dim // 2)
        self.up2 = Up(dim // 2, dim // 4)
        self.up3 = Up(dim // 4, dim // 8)
        self.up4 = Up(dim // 8, dim // 16)
        # 分支融合
        self.sf0 = Ddfusion(in1=dim // 2, in2=dim // 2, h=size // 8, w=size // 8, rate=2)
        self.sf1 = Ddfusion(in1=dim // 4, in2=dim // 4, h=size // 4, w=size // 4, rate=2)
        self.sf2 = Ddfusion(in1=dim // 8, in2=dim // 8, h=size // 2, w=size // 2, rate=2)
        self.sf3 = Ddfusion(in1=dim // 16, in2=dim // 16, h=size, w=size, rate=2)

        self.u21 = Up(dim, dim // 2)
        self.u12 = Up(dim // 2, dim // 4)
        self.u03 = Up(dim // 4, dim // 8)
        self.out3 = Conv(dim // 8, out_classes)

        self.u11 = Up(dim // 2, dim // 4)
        self.u02 = Up(dim // 4, dim // 8)
        self.out2 = Conv(dim // 8, out_classes)

        self.u01 = Up(dim // 4, dim // 8)
        self.out1 = Conv(dim // 8, out_classes)

        self.c1 = ChannelAttention(dim // 8)
        self.c2 = ChannelAttention(dim // 8)
        self.c3 = ChannelAttention(dim // 8)


    def forward(self, x: torch.Tensor) -> Tuple[Any, Any, Any, Any]:

        x0 = self.transformer1(x)
        x0 = torch.transpose(x0, 1, 2)
        x1 = self.up1(x0.reshape(x0.shape[0], -1, 14, 14))  #[384,28,28]
        x2 = self.up2(x1)  #[192,56,56]
        x3 = self.up3(x2)  #[96,112,112]
        x4 = self.up4(x3)  #[48,224,224]
        #convnext_downsample
        y4 = self.convnext.downsample_layers[0](x)
        y4 = self.convnext.stages[0](y4)  #[48,224,224]

        y3 = self.convnext.downsample_layers[1](y4)
        y3 = self.convnext.stages[1](y3)  #[96,112,112]
        y2 = self.convnext.downsample_layers[2](y3)
        y2 = self.convnext.stages[2](y2)  #[192,56,56]
        y1 = self.convnext.downsample_layers[3](y2)
        y1 = self.convnext.stages[3](y1)  #[384,28,28]

        #fusion
        z1 = self.sf0(y1, x1)  #[768，28]
        z2 = self.sf1(y2, x2)  #[384,56]
        z3 = self.sf2(y3, x3)   #[192,112]
        z4 = self.sf3(y4, x4)  #[96,224]

        

        d0_0 = z4
        d1_0 = z3
        d0_1 = self.u01(d1_0)
        d0_1 = d0_1 + d0_0
        q1 = self.c1(d0_1)
        d0_1 = q1 * d0_1
        out1 = self.out1(d0_1)

        d2_0 = z2
        d1_1 = self.u11(d2_0)
        d1_1 = d1_1 + d1_0
        d0_2 = self.u02(d1_1)
        d0_2 = d0_2 + d0_1 + d0_0
        q2 = self.c2(d0_2)
        d0_2 = q2 * d0_2
        out2 = self.out2(d0_2)
        out2 = 0.7 * out2 + 0.3 * out1

        d3_0 = z1
        d2_1 = self.u21(d3_0)
        d2_1 = d2_1 + d2_0
        d1_2 = self.u12(d2_1)
        d1_2 = d1_2 + d1_1 + d1_0
        d0_3 = self.u03(d1_2)
        d0_3 = d0_3 + d0_2 + d0_1 + d0_0
        q3 = self.c3(d0_3)
        d0_3 = q3 * d0_3
        out3 = self.out3(d0_3)
        out3 = 0.7 * out3 + 0.2 * out2 + 0.1 * out1
        # out3 = 0.7 * out3 + 0.3 * out2
        return out1, out2, out3

