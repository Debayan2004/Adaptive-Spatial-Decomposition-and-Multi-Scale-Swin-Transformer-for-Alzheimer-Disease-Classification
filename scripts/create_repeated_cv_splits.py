import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "data" / "cv_5x5"

SEEDS = [42, 123, 2024, 3407, 9999]
N_SPLITS = 5

train_df = pd.read_csv(SPLIT_DIR / "train_subjects.csv")
val_df = pd.read_csv(SPLIT_DIR / "val_subjects.csv")
test_df = pd.read_csv(SPLIT_DIR / "test_subjects.csv")

development_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

assert development_df["Subject"].is_unique
assert test_df["Subject"].is_unique

development_subjects = set(development_df["Subject"])
test_subjects = set(test_df["Subject"])

assert development_subjects.isdisjoint(test_subjects)

assert len(development_df) == 542
assert len(test_df) == 96

assert development_df["Label"].value_counts().to_dict() == {
    1: 264,
    0: 165,
    2: 113
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

development_df.to_csv(
    OUTPUT_DIR / "development_subjects.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "independent_test_subjects.csv",
    index=False
)

summary = {
    "protocol": "Repeated Stratified 5-Fold Cross-Validation",
    "repetitions": len(SEEDS),
    "folds_per_repetition": N_SPLITS,
    "total_fold_evaluations": len(SEEDS) * N_SPLITS,
    "seeds": SEEDS,
    "development_subjects": len(development_df),
    "independent_test_subjects": len(test_df),
    "class_mapping": {
        "CN": 0,
        "MCI": 1,
        "AD": 2
    },
    "repetitions_detail": []
}

for repeat_index, seed in enumerate(SEEDS, start=1):
    repeat_dir = OUTPUT_DIR / f"repeat_{repeat_index:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    repeat_validation_subjects = []

    for fold_index, (train_indices, val_indices) in enumerate(
        splitter.split(
            development_df["Subject"],
            development_df["Label"]
        ),
        start=1
    ):
        fold_train = development_df.iloc[
            train_indices
        ].copy()

        fold_val = development_df.iloc[
            val_indices
        ].copy()

        train_subjects = set(fold_train["Subject"])
        val_subjects = set(fold_val["Subject"])

        assert train_subjects.isdisjoint(val_subjects)
        assert len(train_subjects) + len(val_subjects) == 542
        assert len(train_subjects | val_subjects) == 542

        fold_train.to_csv(
            repeat_dir / f"fold_{fold_index:02d}_train.csv",
            index=False
        )

        fold_val.to_csv(
            repeat_dir / f"fold_{fold_index:02d}_val.csv",
            index=False
        )

        repeat_validation_subjects.extend(
            fold_val["Subject"].tolist()
        )

        summary["repetitions_detail"].append({
            "repeat": repeat_index,
            "fold": fold_index,
            "seed": seed,
            "train_subjects": len(fold_train),
            "validation_subjects": len(fold_val),
            "train_class_counts": {
                str(k): int(v)
                for k, v in fold_train["Label"].value_counts().sort_index().items()
            },
            "validation_class_counts": {
                str(k): int(v)
                for k, v in fold_val["Label"].value_counts().sort_index().items()
            }
        })

    assert len(repeat_validation_subjects) == 542
    assert len(set(repeat_validation_subjects)) == 542
    assert set(repeat_validation_subjects) == development_subjects

    print(
        f"Repeat {repeat_index}: "
        f"seed={seed}, "
        f"subjects={len(repeat_validation_subjects)}, "
        f"unique={len(set(repeat_validation_subjects))}, "
        "PASSED"
    )

with open(
    OUTPUT_DIR / "cv_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nCROSS-VALIDATION PARTITION SUMMARY")
print("Development subjects:", len(development_df))
print("Independent test subjects:", len(test_df))
print("Development class distribution:")
print(development_df["Label"].value_counts().sort_index().to_dict())
print("Independent test class distribution:")
print(test_df["Label"].value_counts().sort_index().to_dict())
print("Repetitions:", len(SEEDS))
print("Folds per repetition:", N_SPLITS)
print("Total fold evaluations:", len(SEEDS) * N_SPLITS)
print("Output directory:", OUTPUT_DIR)
print("\nALL 25 SUBJECT-LEVEL FOLDS VALIDATED")
