import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ResNet34Encoder(nn.Module):
    """
    ResNet34 encoder implemented from scratch.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.inplanes = 64

        # stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # in refference it says "pool, /2" so i think we need padding to make sure the image size /2

        # ResNet34 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # input
        x0 = self.conv1(x)      # 1/2
        x0 = self.bn1(x0)
        x0 = self.relu(x0)      # stem feature

        x1 = self.maxpool(x0)   # 1/4
        x1 = self.layer1(x1)    # 64

        x2 = self.layer2(x1)    # 128
        x3 = self.layer3(x2)    # 256
        x4 = self.layer4(x3)    # 512

        return x1, x2, x3, x4


class DecoderBlock(nn.Module):
    """
    Fig. 2 style:
    upsample -> concat -> conv + relu + bn -> cbam
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_cbam: bool = True):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)

        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

    def forward(self, x, skip):
        # x = self.upsample(x)

        # if x.shape[-2:] != skip.shape[-2:]:
        #     print(x.shape)
        #     x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        return x


class ResNet34_UNet(nn.Module):
    """
    More faithful to the paper figure:
    - ResNet34 encoder
    - UNet decoder
    - decoder uses BN/ReLU + CBAM
    - default output 1 channel, matching the figure

    If your current train.py still uses CrossEntropyLoss, set out_channels=2.
    If you switch to BCEWithLogitsLoss, keep out_channels=1.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, use_cbam: bool = True):
        super().__init__()

        self.encoder = ResNet34Encoder(in_channels=in_channels)

        self.center = nn.Sequential(
            ConvBNReLU(512, 256, kernel_size=3, stride=1),
            ConvBNReLU(256, 256, kernel_size=3, stride=1)
        )

        # skip channels: x4=512, x3=256, x2=128, x1=64
        self.dec4 = DecoderBlock(256, 512, 32, use_cbam=use_cbam)
        self.dec3 = DecoderBlock(32, 256, 32, use_cbam=use_cbam)
        self.dec2 = DecoderBlock(32, 128, 32, use_cbam=use_cbam)
        self.dec1 = DecoderBlock(32, 64, 32, use_cbam=use_cbam)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        # print('main:2',x1.size())
        # print('main:3',x2.size())
        # print('main:4',x3.size())
        # print('main:5',x4.size())

        x = self.center(x4) # 256*7*7
        # print('main:6',x.size())
        x = self.dec4(x, x4)# 256+512*7*7 -> 32*14*14
        # print('main:7',x.size())
        x = self.dec3(x, x3) # 32+256*14*14 -> 32*28*28
        # print('main:8',x.size())
        x = self.dec2(x, x2)# 32+128*28*28 -> 32*56*56
        # print('main:9',x.size())
        x = self.dec1(x, x1)# 32+64*56*56 -> 32*112*112
        # print('main:10',x.size())
        x = self.final_up(x)# 32*112*112 -> 32*224*224
        # print('main:11',x.size())
        x = self.final_conv(x)# 32*224*224 -> 1*224*224
        # print('main:12',x.size())
        return x


if __name__ == "__main__":
    model = ResNet34_UNet(in_channels=3, out_channels=2, use_cbam=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(model)
    print("input shape :", x.shape)
    print("output shape:", y.shape)