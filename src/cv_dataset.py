import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset


class ADNIFoldDataset(Dataset):
    def __init__(self, root, csv_path):
        self.root = Path(root)
        self.volume_dir = self.root / "volumes"

        self.metadata = pd.read_csv(csv_path)

        required_columns = {"Subject", "Label"}

        if not required_columns.issubset(self.metadata.columns):
            raise ValueError("Required CSV columns are missing.")

        if self.metadata["Subject"].duplicated().any():
            raise ValueError("Duplicate subjects found in fold.")

        if not self.metadata["Label"].isin([0, 1, 2]).all():
            raise ValueError("Invalid diagnostic labels found.")

        self.samples = []

        for row in self.metadata.itertuples():
            subject = str(row.Subject)
            label = int(row.Label)

            volume_path = self.volume_dir / f"{subject}.npz"

            if not volume_path.exists():
                raise FileNotFoundError(volume_path)

            self.samples.append(
                (volume_path, label, subject)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        volume_path, label, subject = self.samples[index]

        with np.load(volume_path, allow_pickle=False) as data:
            image = data["image"].astype(
                np.float32,
                copy=True
            )

        if image.shape != (128, 128, 128):
            raise ValueError(
                f"Invalid volume shape: {subject}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Nonfinite image values: {subject}"
            )

        image = torch.from_numpy(image).unsqueeze(0)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "subject": subject
        }
