import torch
import torch.nn as nn


class Shared3DPatchEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_dim=128):
        super().__init__()

        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv3d(64, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, feature_dim),
            nn.GELU()
        )

    def forward(self, patches, chunk_size=128):
        if patches.ndim != 5:
            raise ValueError("Expected patches with shape [N, C, D, H, W].")

        if patches.shape[1] != 1:
            raise ValueError("Expected single-channel MRI patches.")

        if patches.shape[2:] != (16, 16, 16):
            raise ValueError("Expected patch dimensions of 16x16x16.")

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        outputs = []

        for start in range(0, patches.shape[0], chunk_size):
            chunk = patches[start:start + chunk_size]
            outputs.append(self.encoder(chunk))

        return torch.cat(outputs, dim=0)
