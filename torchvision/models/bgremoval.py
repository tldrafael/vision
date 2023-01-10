import torch
import torch.nn as nn


__all__ = ['BGRemovalDecoder']


def conv_block(in_channels, out_channels, ks=3, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, ks, padding=padding, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class ExpandBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = conv_block(in_channels + skip_channels, out_channels)
        self.conv2 = conv_block(out_channels, out_channels)

    def forward(self, x_bottom, x_skip):
        x_bottom = self.upsample(x_bottom)
        x = torch.cat([x_skip, x_bottom], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BGRemovalDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList()
        exp2_factor = [5, 4, 3, 2, 0]
        for i in range(len(exp2_factor) - 1):
            in_channels = 64 * 2 ** exp2_factor[i]
            out_channels = 64 * 2 ** exp2_factor[i + 1]
            self.blocks.append(
                ExpandBlock(in_channels, out_channels, out_channels)
            )

        self.conv2 = conv_block(64, 16, 3)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, list_features):
        x = list_features[-1]
        for i, b in enumerate(self.blocks, 2):
            x = b(x, list_features[-i])

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample3(x)
        x = self.sigmoid(x)
        return x


class BGRemoval(nn.Module):
    def __init__(self):
        super().__init__()

