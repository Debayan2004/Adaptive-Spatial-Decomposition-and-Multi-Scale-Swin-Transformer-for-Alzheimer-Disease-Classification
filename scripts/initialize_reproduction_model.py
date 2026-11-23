import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from monai.networks.nets import SwinUNETR

from models.adaptive_sda_swin_full import AdaptiveSDASwinFull

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = SwinUNETR(
    in_channels=1,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True
)

checkpoint = torch.load(
    "pretrained/model_swinvit.pt",
    map_location="cpu",
    weights_only=False
)

mapped = {}

for key, value in checkpoint["state_dict"].items():
    key = key.removeprefix("module.")
    key = key.replace(".mlp.fc1.", ".mlp.linear1.")
    key = key.replace(".mlp.fc2.", ".mlp.linear2.")
    mapped[key] = value

target = base_model.swinViT.state_dict()

compatible = {
    key: value
    for key, value in mapped.items()
    if key in target and target[key].shape == value.shape
}

if len(compatible) != len(target):
    raise RuntimeError(
        f"Expected {len(target)} encoder tensors, found {len(compatible)}."
    )

base_model.swinViT.load_state_dict(compatible, strict=True)

model = AdaptiveSDASwinFull(
    swin_encoder=base_model.swinViT,
    num_classes=3,
    patch_chunk_size=8
).to(device)

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

trainable_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "pretrained_checkpoint": "model_swinvit.pt",
        "loaded_swin_tensors": len(compatible),
        "num_classes": 3,
        "class_mapping": {"CN": 0, "MCI": 1, "AD": 2},
        "feature_channels": [48, 96, 192, 384, 768],
        "patch_chunk_size": 8
    },
    "checkpoints/adaptive_sda_swin_initialized.pt"
)

print("\nFULL MODEL INITIALIZATION PASSED")
print("Device:", device)
print("Loaded pretrained Swin tensors:", len(compatible))
print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_count)
print("Checkpoint: checkpoints/adaptive_sda_swin_initialized.pt")

