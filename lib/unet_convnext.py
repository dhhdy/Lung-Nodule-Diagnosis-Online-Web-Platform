from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),  # 看看效果
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 pretrained: bool = False,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 32, base_c * 16 // factor, bilinear)
        self.up2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up3 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up4 = Up(base_c * 4, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        self.convnext = convnext_pico(pretrained=pretrained)
        self.convnext.head = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            DoubleConv(512, 512)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.convnext.downsample_layers[0](x)
        x1 = self.convnext.stages[0](x1)
        x2 = self.convnext.downsample_layers[1](x1)
        x2 = self.convnext.stages[1](x2)
        x3 = self.convnext.downsample_layers[2](x2)
        x3 = self.convnext.stages[2](x3)
        x4 = self.convnext.downsample_layers[3](x3)
        x4 = self.convnext.stages[3](x4)
        x5 = self.conv2(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

if __name__ == '__main__':
    a = torch.ones(3,1,512,512)
    model = UNet(1,1, False, 32)
    b = model(a)
    print(b.shape)


