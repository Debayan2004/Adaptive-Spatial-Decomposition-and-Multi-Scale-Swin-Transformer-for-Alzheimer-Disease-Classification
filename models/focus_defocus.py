import torch
import torch.nn as nn


class LearnableFocusDefocus(nn.Module):
    def __init__(self, feature_dim=128, global_dim=64, hidden_dim=128):
        super().__init__()

        input_dim = feature_dim + global_dim + 3

        self.grouping_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features, coordinates, global_descriptor):
        if features.ndim != 2:
            raise ValueError("Expected features [N, C].")

        if features.shape[1] != 128:
            raise ValueError("Expected 128-dimensional patch features.")

        if coordinates.shape != (features.shape[0], 3):
            raise ValueError("Expected coordinates [N, 3].")

        if global_descriptor.ndim == 2:
            global_descriptor = global_descriptor.squeeze(0)

        if global_descriptor.shape != (64,):
            raise ValueError("Expected global descriptor [64].")

        global_context = global_descriptor.unsqueeze(0).expand(
            features.shape[0], -1
        )

        inputs = torch.cat(
            [features, global_context, coordinates], dim=1
        )

        focus_logits = self.grouping_network(inputs).squeeze(-1)
        focus_weights = torch.sigmoid(focus_logits)
        defocus_weights = 1.0 - focus_weights

        return {
            "focus_logits": focus_logits,
            "focus_weights": focus_weights,
            "defocus_weights": defocus_weights
        }
