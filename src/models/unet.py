import torch
import torch.nn as nn


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    將 feature map 做中心裁切，裁成 target_h x target_w
    """
    _, _, h, w = x.shape
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return x[:, :, start_y:start_y + target_h, start_x:start_x + target_w]


class TwoConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    2x2 max pool + double conv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = TwoConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        # print(x.size())
        x = self.conv(x)
        # print(x.size())
        return x


class Up(nn.Module):
    """
    2x2 up-conv + crop skip + concat + double conv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = TwoConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # print(x.size())

        # 原論文做法：對 skip feature 做 crop 後再 concat
        skip = center_crop(skip, x.shape[2], x.shape[3])

        x = torch.cat([skip, x], dim=1)
        # print(x.size())

        x = self.conv(x)
        # print(x.size())

        return x


class UNet(nn.Module):

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()

        # contracting path
        self.inc = TwoConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 論文提到 contracting path 末端可加 dropout 做 implicit augmentation
        # self.dropout = nn.Dropout(p=0.1)

        # expansive path
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        # print('main1:',x.size())
        x1 = self.inc(x)      # 572 -> 568
        # print('main2:',x1.size())
        x2 = self.down1(x1)   # 568 -> 280
        # print('main3:',x2.size())

        x3 = self.down2(x2)   # 280 -> 136
        # print('main4:',x3.size())
        x4 = self.down3(x3)   # 136 -> 64
        # print('main5:',x4.size())
        x5 = self.down4(x4)   # 64  -> 28
        # print('main6:',x5.size())

        # x5 = self.dropout(x5)
        # print('main7:',x5.size())

        # decoder
        x = self.up1(x5, x4)  # 28 -> 52
        # print('main8:',x.size())
        x = self.up2(x, x3)   # 52 -> 100
        # print('main9:',x.size())
        x = self.up3(x, x2)   # 100 -> 196
        # print('main10:',x.size())
        x = self.up4(x, x1)   # 196 -> 388
        # print('main11:',x.size())

        logits = self.outc(x) # 388 x 388 if input is 572 x 572
        # print('main12:',x.size())
        return logits


if __name__ == "__main__":
    model = UNet(in_channels=3, n_classes=2)
    print(model)

    x = torch.randn(1, 3, 572, 572)
    y = model(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)