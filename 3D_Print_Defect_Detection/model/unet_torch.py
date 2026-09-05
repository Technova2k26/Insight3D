"""
InSight3D - PyTorch U-Net Architecture Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Constructs U-Net architecture from scratch using PyTorch:
- Encoder (Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU -> MaxPool2d)
- Bottleneck (Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU)
- Decoder (ConvTranspose2d -> Skip Concatenation -> DoubleConv)
- Final Output Layer: 1x1 Conv2d with Sigmoid activation for binary defect mask prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNetTorch(nn.Module):
    """
    PyTorch U-Net Model Architecture for Binary Defect Segmentation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 32,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        f = init_filters

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, f * 1, dropout_rate=dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 1, f * 2, dropout_rate=dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4, dropout_rate=dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8, dropout_rate=dropout_rate))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16, dropout_rate=dropout_rate * 2.0))

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8, dropout_rate=dropout_rate)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 8, f * 4, dropout_rate=dropout_rate)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2, dropout_rate=dropout_rate)

        self.up4 = nn.ConvTranspose2d(f * 2, f * 1, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f * 1, dropout_rate=dropout_rate)

        # Final Layer (1x1 Conv with Sigmoid)
        self.outc = nn.Conv2d(f * 1, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder & Skip Connections
        u1 = self.up1(b)
        x4_cat = torch.cat([u1, x4], dim=1)
        c1 = self.conv_up1(x4_cat)

        u2 = self.up2(c1)
        x3_cat = torch.cat([u2, x3], dim=1)
        c2 = self.conv_up2(x3_cat)

        u3 = self.up3(c2)
        x2_cat = torch.cat([u3, x2], dim=1)
        c3 = self.conv_up3(x2_cat)

        u4 = self.up4(c3)
        x1_cat = torch.cat([u4, x1], dim=1)
        c4 = self.conv_up4(x1_cat)

        logits = self.outc(c4)
        probs = torch.sigmoid(logits)
        return probs


if __name__ == "__main__":
    print("InSight3D PyTorch U-Net Module Test")
    model = UNetTorch(in_channels=1, out_channels=1, init_filters=32)
    dummy_input = torch.randn(2, 1, 256, 256)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
    print(f"Output min={out.min().item():.4f}, max={out.max().item():.4f}")
