import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePatchExtractor(nn.Module):
    def __init__(
        self,
        initial_patch_size=32,
        output_patch_size=16,
        max_depth=2,
        subdivision_ratio=0.25
    ):
        super().__init__()

        self.initial_patch_size = initial_patch_size
        self.output_patch_size = output_patch_size
        self.max_depth = max_depth
        self.subdivision_ratio = subdivision_ratio

        self.global_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.local_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )

        self.subdivision_head = nn.Sequential(
            nn.Linear(131, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1)
        )

        nn.init.constant_(self.subdivision_head[-1].bias, -1.5)

    def describe(self, regions):
        return F.adaptive_avg_pool3d(
            regions, (4, 4, 4)
        ).flatten(1)

    def forward(self, volume):
        if volume.ndim == 4:
            volume = volume.unsqueeze(1)

        if volume.ndim != 5 or volume.shape[0] != 1 or volume.shape[1] != 1:
            raise ValueError("Expected input shape [1, 1, H, W, D].")

        _, _, height, width, depth_size = volume.shape
        patch_size = self.initial_patch_size

        if height != width or width != depth_size or height % patch_size != 0:
            raise ValueError("Volume must be cubic and divisible by initial_patch_size.")

        if patch_size % (2 ** self.max_depth) != 0:
            raise ValueError("Initial patch size must be divisible by 2**max_depth.")

        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")

        if not 0 < self.subdivision_ratio <= 1:
            raise ValueError("subdivision_ratio must be in (0, 1].")

        global_descriptor = self.global_encoder(
            self.describe(volume)
        )

        nodes = []

        for z in range(0, height, patch_size):
            for y in range(0, width, patch_size):
                for x in range(0, depth_size, patch_size):
                    region = volume[
                        :, :, z:z + patch_size,
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    nodes.append({
                        "region": region,
                        "start": (z, y, x),
                        "size": patch_size,
                        "path_gate": volume.new_tensor(1.0)
                    })

        candidates = []
        leaf_gates = []
        coordinates = []
        extents = []
        levels = []
        all_scores = []
        expected_extra_terms = []

        for level in range(self.max_depth + 1):
            if not nodes:
                break

            regions = torch.cat(
                [node["region"] for node in nodes],
                dim=0
            )

            local_descriptors = self.local_encoder(
                self.describe(regions)
            )

            node_coordinates = volume.new_tensor([
                [
                    (node["start"][0] + node["size"] / 2) / height,
                    (node["start"][1] + node["size"] / 2) / width,
                    (node["start"][2] + node["size"] / 2) / depth_size
                ]
                for node in nodes
            ])

            global_context = global_descriptor.expand(len(nodes), -1)

            contexts = torch.cat(
                [local_descriptors, global_context, node_coordinates],
                dim=1
            )

            scores = torch.sigmoid(
                self.subdivision_head(contexts)
            ).squeeze(-1)

            if level < self.max_depth:
                split_count = max(
                    1,
                    math.ceil(self.subdivision_ratio * len(nodes))
                )

                selected_indices = torch.topk(
                    scores.detach(),
                    k=split_count,
                    largest=True
                ).indices

                selected_mask = torch.zeros(
                    len(nodes),
                    device=volume.device,
                    dtype=volume.dtype
                )

                selected_mask[selected_indices] = 1.0

                split_gates = (
                    selected_mask - scores.detach() + scores
                )

                all_scores.append(scores)

                path_gates = torch.stack([
                    node["path_gate"] for node in nodes
                ])

                expected_extra_terms.append(
                    path_gates * split_gates
                )
            else:
                selected_mask = torch.zeros(
                    len(nodes),
                    device=volume.device,
                    dtype=torch.bool
                )
                split_gates = torch.zeros_like(scores)

            next_nodes = []

            for index, node in enumerate(nodes):
                path_gate = node["path_gate"]

                leaf_gate = path_gate * (1.0 - split_gates[index])

                resized = F.interpolate(
                    node["region"],
                    size=(
                        self.output_patch_size,
                        self.output_patch_size,
                        self.output_patch_size
                    ),
                    mode="trilinear",
                    align_corners=False
                )

                candidates.append(resized.squeeze(0))
                leaf_gates.append(leaf_gate)
                coordinates.append(node_coordinates[index])
                extents.append(node["size"] / height)
                levels.append(level)

                if level < self.max_depth and selected_mask[index].item():
                    half = node["size"] // 2

                    for a in range(2):
                        for b in range(2):
                            for c in range(2):
                                z0 = a * half
                                y0 = b * half
                                x0 = c * half

                                child = node["region"][
                                    :,
                                    :,
                                    z0:z0 + half,
                                    y0:y0 + half,
                                    x0:x0 + half
                                ]

                                next_nodes.append({
                                    "region": child,
                                    "start": (
                                        node["start"][0] + z0,
                                        node["start"][1] + y0,
                                        node["start"][2] + x0
                                    ),
                                    "size": half,
                                    "path_gate": path_gate * split_gates[index]
                                })

            nodes = next_nodes

        patches = torch.stack(candidates)
        gates = torch.stack(leaf_gates)

        initial_count = sum(level == 0 for level in levels)

        if expected_extra_terms:
            expected_count = (
                patches.new_tensor(float(initial_count))
                + 7.0 * sum(term.sum() for term in expected_extra_terms)
            )
        else:
            expected_count = patches.new_tensor(float(initial_count))

        hard_active = gates.detach() > 0.5

        return {
            "patches": patches,
            "gates": gates,
            "coordinates": torch.stack(coordinates),
            "extents": volume.new_tensor(extents),
            "levels": torch.tensor(
                levels,
                device=volume.device,
                dtype=torch.long
            ),
            "subdivision_scores": (
                torch.cat(all_scores)
                if all_scores else volume.new_empty((0,))
            ),
            "global_descriptor": global_descriptor,
            "num_patches": int(hard_active.sum().item()),
            "candidate_count": len(candidates),
            "expected_patch_count": expected_count
        }
