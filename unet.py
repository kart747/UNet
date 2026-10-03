"""A small U-Net for semantic segmentation, written in PyTorch.

Encoder:    DoubleConv -> MaxPool, repeated once per entry in `features`.
Bottleneck: one DoubleConv at the lowest resolution.
Decoder:    ConvTranspose2d (2x upsample) -> concat with skip -> DoubleConv.
Head:       1x1 conv that maps to the number of output classes.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) twice. Keeps height and width unchanged."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        # ups alternates [upsample, DoubleConv, upsample, DoubleConv, ...]
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: channels grow, e.g. 3 -> 64 -> 128 -> 256 -> 512
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder: mirror of the encoder, deepest level first
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            # Input is 2 * feature channels: upsampled map + skip map, concatenated
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder: save each level's output before pooling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # Deepest skip is needed first on the way up
        skip_connections = skip_connections[::-1]

        # Decoder: step through ups two at a time (upsample, then DoubleConv)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Odd input sizes round down when pooling, so upsampling can come
            # back one pixel short; resize to match the skip before concatenating.
            if x.shape[2:] != skip_connection.shape[2:]:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)  # channels: 2 * feature
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
    print("OK")
