import sys
sys.path.append('../input/lbunet')
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.eva02 import eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE as transformer
from lib.eva02 import eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE as transformer
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.transformer = transformer(pretrained=pretrained)
        self.drop = nn.Dropout2d(0.5)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels=768, out_channels=512, mid_channels=512)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels=512, out_channels=256, mid_channels=256)
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels=256, out_channels=128, mid_channels=128)
        )
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels=128, out_channels=64, mid_channels=64)
        )
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.transformer(x)
        x1 = torch.transpose(x1, 1, 2)
        x2 = x1.reshape(x1.shape[0], -1, 14, 14)
        x2 = self.drop(x2)

        x3 = self.up3(x2)
        # x3 = self.drop(x3)

        x4 = self.up4(x3)
        # x4 = self.drop(x4)

        x5 = self.up5(x4)
        # x5 = self.drop(x5)

        x6 = self.up6(x5)
        # x6 = self.drop(x6)

        x7 = self.out_conv(x6)

        return x7

# if __name__ == '__main__':
#     model = UNet(3, 1, False)
#     model = model.cuda()
#     # a = torch.ones(3, 3, 224, 224).cuda()
#     a = model(a)
#     print('qaq: ', a.shape)
