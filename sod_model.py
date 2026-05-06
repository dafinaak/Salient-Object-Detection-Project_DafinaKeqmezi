"""
sod_model.py
============
CNN architectures for Salient Object Detection, implemented from scratch.

This file defines two models:

    SODNet           -- Baseline encoder-decoder CNN.
                        4 x (Conv -> ReLU -> MaxPool) for the encoder,
                        4 x (ConvTranspose -> ReLU) for the decoder,
                        1-channel sigmoid output the same H x W as input.

    SODNetImproved   -- Improved version with two architectural changes:
                          (1) BatchNorm after every Conv,
                          (2) U-Net-style skip connections between matching
                              encoder and decoder levels.

Input:  RGB tensor of shape (B, 3, 128, 128), values in [0, 1]
Output: Saliency map of shape (B, 1, 128, 128), values in [0, 1]
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------
class SODNet(nn.Module):
    """Baseline CNN for Salient Object Detection.

    Encoder: 4 x (Conv2d -> ReLU -> MaxPool2d)
    Decoder: 4 x (ConvTranspose2d -> ReLU)
    Output:  1-channel saliency map with sigmoid activation
    """

    def __init__(self):
        super().__init__()

        # ---------- Encoder ---------- (input: 3 x 128 x 128)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )                                                 # 32 x 128 x 128
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )                                                 # 64 x 64 x 64
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )                                                 # 128 x 32 x 32
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )                                                 # 256 x 16 x 16

        self.pool = nn.MaxPool2d(2, 2)

        # ---------- Decoder ---------- (input: 256 x 8 x 8)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )                                                 # 128 x 16 x 16
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )                                                 # 64 x 32 x 32
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )                                                 # 32 x 64 x 64
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )                                                 # 16 x 128 x 128

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)   # 1 x 128 x 128

    def forward(self, x):
        # encoder
        x = self.pool(self.enc1(x))   # 32 x 64 x 64
        x = self.pool(self.enc2(x))   # 64 x 32 x 32
        x = self.pool(self.enc3(x))   # 128 x 16 x 16
        x = self.pool(self.enc4(x))   # 256 x 8 x 8
        # decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        # output
        return torch.sigmoid(self.out_conv(x))


# ---------------------------------------------------------------------------
# Improved model: BatchNorm + U-Net skip connections
# ---------------------------------------------------------------------------
class SODNetImproved(nn.Module):
    """Improved CNN with BatchNorm and U-Net-style skip connections.

    Modifications over baseline:
        1. BatchNorm2d after every convolution (faster convergence,
           better gradient flow).
        2. Encoder feature maps are concatenated into the matching
           decoder level (preserves fine spatial detail lost in compression).
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # ---------- Encoder ----------
        self.enc1 = conv_block(3, 32)        # 32 x 128 x 128
        self.enc2 = conv_block(32, 64)       # 64 x 64 x 64
        self.enc3 = conv_block(64, 128)      # 128 x 32 x 32
        self.enc4 = conv_block(128, 256)     # 256 x 16 x 16

        self.pool = nn.MaxPool2d(2, 2)

        # ---------- Bottleneck ----------
        self.bottleneck = conv_block(256, 512)   # 512 x 8 x 8

        # ---------- Decoder (with skip connections) ----------
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = conv_block(512, 256)     # 256 (upsampled) + 256 (skip from enc4)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)     # 128 + 128 (skip from enc3)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)      # 64 + 64 (skip from enc2)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = conv_block(64, 32)       # 32 + 32 (skip from enc1)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # encoder - save outputs BEFORE pooling for skip connections
        e1 = self.enc1(x)                 # 32 x 128 x 128
        e2 = self.enc2(self.pool(e1))     # 64 x 64 x 64
        e3 = self.enc3(self.pool(e2))     # 128 x 32 x 32
        e4 = self.enc4(self.pool(e3))     # 256 x 16 x 16

        # bottleneck
        b = self.bottleneck(self.pool(e4))   # 512 x 8 x 8

        # decoder with skip connections
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e4], 1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e3], 1))

        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e2], 1))

        d4 = self.up4(d3)
        d4 = self.dec4(torch.cat([d4, e1], 1))

        return torch.sigmoid(self.out_conv(d4))


if __name__ == "__main__":
    # sanity check - run as: python sod_model.py
    for cls in (SODNet, SODNetImproved):
        m = cls()
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            y = m(x)
        print(f"{cls.__name__:18s}  params={n_params:>10,}  in={tuple(x.shape)}  out={tuple(y.shape)}")
