import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "data" / "cv_5x5"

class_names = {
    0: "CN",
    1: "MCI",
    2: "AD"
}

results = []

for repeat in range(1, 6):
    repeat_dir = CV_DIR / f"repeat_{repeat:02d}"

    for fold in range(1, 6):
        train_path = repeat_dir / f"fold_{fold:02d}_train.csv"
        val_path = repeat_dir / f"fold_{fold:02d}_val.csv"

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        counts = train_df["Label"].value_counts().to_dict()
        total = len(train_df)

        weights = {
            class_names[c]: total / (3 * counts[c])
            for c in range(3)
        }

        assert abs(sum(
            counts[c] * weights[class_names[c]]
            for c in range(3)
        ) / total - 1.0) < 1e-10

        results.append({
            "repeat": repeat,
            "fold": fold,
            "train_subjects": total,
            "validation_subjects": len(val_df),
            "class_counts": {
                class_names[c]: int(counts[c])
                for c in range(3)
            },
            "class_weights": weights
        })

        print(
            f"Repeat {repeat} Fold {fold} | "
            f"Samples={total} | "
            f"Weights={weights}"
        )

output_path = CV_DIR / "fold_training_configuration.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nTotal fold configurations:", len(results))
print("Saved:", output_path)
print("\nFOLD-SPECIFIC CONFIGURATION VALIDATION PASSED")
