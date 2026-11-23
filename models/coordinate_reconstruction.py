import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAwareReconstruction(nn.Module):
    def __init__(self, feature_dim=128, grid_size=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size

    def forward(self, patch_features, coordinates, extents, levels, gates):
        if patch_features.ndim != 5:
            raise ValueError("Expected patch features [N, C, D, H, W].")

        n, channels = patch_features.shape[:2]

        if channels != self.feature_dim:
            raise ValueError("Unexpected feature dimension.")

        if coordinates.shape != (n, 3):
            raise ValueError("Expected coordinates [N, 3].")

        if extents.shape != (n,) or levels.shape != (n,) or gates.shape != (n,):
            raise ValueError("Invalid spatial metadata shapes.")

        grid_size = self.grid_size
        device = patch_features.device
        dtype = patch_features.dtype

        feature_sources = []
        index_sources = []
        occupancy_sources = []

        for level in torch.unique(levels, sorted=True).tolist():
            selected = torch.where(levels == level)[0]

            level_extents = extents[selected]
            cell_sizes = torch.round(level_extents * grid_size).long()

            if not torch.all(cell_sizes == cell_sizes[0]):
                raise ValueError("Inconsistent region extents within a level.")

            cell_size = int(cell_sizes[0].item())

            if cell_size < 1 or cell_size > grid_size:
                raise ValueError("Invalid reconstructed region size.")

            features = F.interpolate(
                patch_features[selected],
                size=(cell_size, cell_size, cell_size),
                mode="trilinear",
                align_corners=False
            )

            centers = coordinates[selected].detach()

            starts = torch.round(
                (centers - level_extents.unsqueeze(1) / 2) * grid_size
            ).long()

            offsets = torch.arange(cell_size, device=device)

            zz, yy, xx = torch.meshgrid(
                offsets, offsets, offsets, indexing="ij"
            )

            spatial_offsets = torch.stack(
                [zz, yy, xx], dim=-1
            ).reshape(1, -1, 3)

            positions = starts.unsqueeze(1) + spatial_offsets

            if torch.any(positions < 0) or torch.any(positions >= grid_size):
                raise ValueError("A region lies outside the reconstruction grid.")

            linear_indices = (
                positions[..., 0] * grid_size * grid_size
                + positions[..., 1] * grid_size
                + positions[..., 2]
            ).reshape(-1)

            region_gates = gates[selected].to(dtype)

            weighted_features = (
                features * region_gates.reshape(-1, 1, 1, 1, 1)
            )

            feature_sources.append(
                weighted_features.permute(1, 0, 2, 3, 4).reshape(
                    channels, -1
                )
            )

            index_sources.append(linear_indices)

            hard_occupancy = (
                gates[selected].detach() > 0.5
            ).to(dtype)

            occupancy_sources.append(
                hard_occupancy.unsqueeze(1).expand(
                    -1, cell_size ** 3
                ).reshape(-1)
            )

        all_features = torch.cat(feature_sources, dim=1)
        all_indices = torch.cat(index_sources)
        all_occupancy = torch.cat(occupancy_sources)

        flat_features = torch.zeros(
            channels, grid_size ** 3,
            device=device, dtype=dtype
        ).index_add(
            1, all_indices, all_features
        )

        flat_occupancy = torch.zeros(
            grid_size ** 3, device=device, dtype=dtype
        ).index_add(
            0, all_indices, all_occupancy
        )

        reconstructed = flat_features / flat_occupancy.clamp_min(1.0).unsqueeze(0)

        reconstructed = reconstructed.reshape(
            1, channels, grid_size, grid_size, grid_size
        )

        coverage = (flat_occupancy > 0).float().mean()

        partition_error = (
            flat_occupancy - 1.0
        ).abs().max()

        return {
            "feature_volume": reconstructed,
            "coverage": coverage,
            "weight_volume": flat_occupancy.reshape(
                1, 1, grid_size, grid_size, grid_size
            ),
            "partition_error": partition_error
        }
