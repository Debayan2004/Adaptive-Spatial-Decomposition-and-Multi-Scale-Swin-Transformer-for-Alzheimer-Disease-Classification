import torch
import torch.nn as nn


class MultiScaleClassificationHead(nn.Module):
    def __init__(
        self,
        swin_channels=(48, 96, 192, 384, 768),
        num_classes=3,
        hidden_dim=512,
        dropout=0.3
    ):
        super().__init__()

        total_dim = sum(swin_channels)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, fused_features):
        if len(fused_features) != 5:
            raise ValueError("Expected five fused feature scales.")

        pooled = []

        for feature in fused_features:
            pooled.append(
                self.pool(feature).flatten(1)
            )

        representation = torch.cat(pooled, dim=1)
        logits = self.classifier(representation)

        return {
            "logits": logits,
            "representation": representation
        }
