import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMultiScaleFusion(nn.Module):
    def __init__(
        self,
        adaptive_channels=128,
        bottleneck_channels=32,
        swin_channels=(48, 96, 192, 384, 768)
    ):
        super().__init__()

        self.swin_channels = tuple(swin_channels)

        self.shared_projection = nn.Sequential(
            nn.Conv3d(
                adaptive_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(8, bottleneck_channels),
            nn.GELU()
        )

        self.scale_projections = nn.ModuleList([
            nn.Conv3d(
                bottleneck_channels,
                channels,
                kernel_size=1,
                bias=False
            )
            for channels in self.swin_channels
        ])

        self.fusion_gates = nn.Parameter(
            torch.full((len(self.swin_channels),), 0.1)
        )

    def forward(self, swin_features, adaptive_volume):
        if len(swin_features) != len(self.swin_channels):
            raise ValueError("Unexpected number of Swin feature scales.")

        if adaptive_volume.ndim != 5:
            raise ValueError("Expected adaptive volume [B, C, D, H, W].")

        if adaptive_volume.shape[1] != self.shared_projection[0].in_channels:
            raise ValueError("Unexpected adaptive feature dimension.")

        shared = self.shared_projection(adaptive_volume)

        fused_features = []
        adaptive_features = []

        for index, (feature, projection, expected_channels) in enumerate(
            zip(swin_features, self.scale_projections, self.swin_channels)
        ):
            if feature.ndim != 5:
                raise ValueError("Expected Swin features [B, C, D, H, W].")

            if feature.shape[1] != expected_channels:
                raise ValueError("Unexpected Swin feature channel dimension.")

            if feature.shape[0] != adaptive_volume.shape[0]:
                raise ValueError("Batch sizes do not match.")

            aligned = F.interpolate(
                shared,
                size=feature.shape[2:],
                mode="trilinear",
                align_corners=False
            )

            adaptive_feature = projection(aligned)

            gate = torch.tanh(self.fusion_gates[index])

            fused = feature + gate * adaptive_feature

            fused_features.append(fused)
            adaptive_features.append(adaptive_feature)

        return {
            "fused_features": tuple(fused_features),
            "adaptive_features": tuple(adaptive_features),
            "fusion_gates": torch.tanh(self.fusion_gates)
        }
