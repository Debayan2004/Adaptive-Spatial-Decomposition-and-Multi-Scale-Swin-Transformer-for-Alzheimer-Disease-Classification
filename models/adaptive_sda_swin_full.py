import torch
import torch.nn as nn

from models.adaptive_patch_extractor import AdaptivePatchExtractor
from models.shared_patch_encoder import Shared3DPatchEncoder
from models.focus_defocus import LearnableFocusDefocus
from models.coordinate_reconstruction import CoordinateAwareReconstruction
from models.multiscale_fusion import AdaptiveMultiScaleFusion
from models.classification_head import MultiScaleClassificationHead


class FocusDefocusFeatureMixer(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.focus_projection = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU()
        )

        self.defocus_projection = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU()
        )

    def forward(self, features, focus_weights, defocus_weights):
        focus = self.focus_projection(features)
        defocus = self.defocus_projection(features)

        focus = focus * focus_weights[:, None, None, None, None]
        defocus = defocus * defocus_weights[:, None, None, None, None]

        return focus + defocus


class AdaptiveSDASwinFull(nn.Module):
    def __init__(
        self,
        swin_encoder,
        num_classes=3,
        patch_chunk_size=16
    ):
        super().__init__()

        self.swin_encoder = swin_encoder
        self.patch_chunk_size = patch_chunk_size

        self.patch_extractor = AdaptivePatchExtractor()
        self.patch_encoder = Shared3DPatchEncoder(feature_dim=128)

        self.grouping = LearnableFocusDefocus(
            feature_dim=128,
            global_dim=64
        )

        self.feature_mixer = FocusDefocusFeatureMixer(128)

        self.reconstruction = CoordinateAwareReconstruction(
            feature_dim=128,
            grid_size=32
        )

        self.fusion = AdaptiveMultiScaleFusion(
            adaptive_channels=128,
            bottleneck_channels=32
        )

        self.classification_head = MultiScaleClassificationHead(
            num_classes=num_classes
        )

    def forward(self, volume):
        partition = self.patch_extractor(volume)

        patches = partition["patches"]

        patch_features = self.patch_encoder(
            patches,
            chunk_size=self.patch_chunk_size
        )

        pooled_features = patch_features.mean(dim=(2, 3, 4))

        grouping = self.grouping(
            pooled_features,
            partition["coordinates"],
            partition["global_descriptor"]
        )

        grouped_features = self.feature_mixer(
            patch_features,
            grouping["focus_weights"],
            grouping["defocus_weights"]
        )

        reconstructed = self.reconstruction(
            grouped_features,
            partition["coordinates"],
            partition["extents"],
            partition["levels"],
            partition["gates"]
        )

        swin_features = self.swin_encoder(volume)

        fusion_result = self.fusion(
            swin_features,
            reconstructed["feature_volume"]
        )

        classification = self.classification_head(
            fusion_result["fused_features"]
        )

        return {
            "logits": classification["logits"],
            "representation": classification["representation"],
            "feature_volume": reconstructed["feature_volume"],
            "focus_weights": grouping["focus_weights"],
            "defocus_weights": grouping["defocus_weights"],
            "subdivision_scores": partition["subdivision_scores"],
            "expected_patch_count": partition["expected_patch_count"],
            "active_patch_count": partition["num_patches"],
            "coverage": reconstructed["coverage"],
            "partition_error": reconstructed["partition_error"],
            "fusion_gates": fusion_result["fusion_gates"]
        }
