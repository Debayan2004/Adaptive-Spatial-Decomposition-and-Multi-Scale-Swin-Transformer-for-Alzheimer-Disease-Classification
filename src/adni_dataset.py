import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class ADNIMRIDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.volume_dir = self.root / "volumes"
        self.split_dir = self.root / "splits"

        split_files = {
            "train": "train_subjects.csv",
            "val": "val_subjects.csv",
            "test": "test_subjects.csv"
        }

        if split not in split_files:
            raise ValueError(f"Unsupported split: {split}")

        self.metadata = pd.read_csv(
            self.split_dir / split_files[split]
        )

        self.label_map = {
            0: 0,
            1: 1,
            2: 2
        }

        self.samples = []

        for row in self.metadata.itertuples():
            subject = str(row.Subject)
            label = int(row.Label)

            volume_path = self.volume_dir / f"{subject}.npz"

            if not volume_path.exists():
                raise FileNotFoundError(volume_path)

            self.samples.append((volume_path, label, subject))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        volume_path, label, subject = self.samples[index]

        with np.load(volume_path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32, copy=True)

        if image.shape != (128, 128, 128):
            raise ValueError(
                f"Invalid volume shape for {subject}: {image.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Nonfinite image values found for {subject}"
            )

        image = torch.from_numpy(image).unsqueeze(0)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "subject": subject
        }


def create_dataloader(root, split, batch_size=1, shuffle=None, num_workers=0):
    dataset = ADNIMRIDataset(root, split)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
