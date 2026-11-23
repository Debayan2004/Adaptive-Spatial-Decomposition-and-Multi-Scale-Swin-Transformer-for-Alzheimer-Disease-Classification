import sys
import json
import csv
import random
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from monai.networks.nets import SwinUNETR
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score
)

from src.cv_dataset import ADNIFoldDataset
from models.adaptive_sda_swin_full import AdaptiveSDASwinFull


ROOT = Path(__file__).resolve().parents[1]

with open(
    ROOT / "configs" / "training_config.json",
    "r",
    encoding="utf-8-sig"
) as file:
    CONFIG = json.load(file)

CV_DIR = ROOT / "data" / "cv_5x5"
INITIALIZED_CHECKPOINT = ROOT / CONFIG["initialized_checkpoint"]

CLASS_NAMES = ["CN", "MCI", "AD"]
LABELS = [0, 1, 2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None
    )


def calculate_metrics(targets, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=LABELS,
        zero_division=0
    )

    return {
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=LABELS
        ).tolist()
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    amp_dtype,
    optimizer=None,
    scaler=None
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    targets = []
    predictions = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled
            ):
                output = model(images)
                logits = output["logits"]
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite loss encountered."
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CONFIG["gradient_clip_norm"]
                )

                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        "Nonfinite gradient norm encountered."
                    )

                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        targets.extend(labels.detach().cpu().tolist())

        predictions.extend(
            logits.detach().argmax(dim=1).cpu().tolist()
        )

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / total_samples

    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    repeat,
    fold,
    seed,
    class_weights
):
    torch.save(
        {
            "epoch": epoch,
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_macro_f1": best_score,
            "class_weights": class_weights.cpu(),
            "config": CONFIG,
            "class_mapping": {
                "CN": 0,
                "MCI": 1,
                "AD": 2
            }
        },
        path
    )


def evaluate_best_checkpoint(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
    amp_dtype,
    checkpoint_path,
    prediction_path
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    records = []
    targets = []
    predictions = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(device)

            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled
            ):
                logits = model(images)["logits"]

            probabilities = torch.softmax(
                logits.float(),
                dim=1
            )

            predicted = probabilities.argmax(dim=1)

            for index in range(labels.size(0)):
                subject = batch["subject"][index]
                target = int(labels[index].item())
                prediction = int(predicted[index].item())

                record = {
                    "Subject": subject,
                    "True_Label": target,
                    "Predicted_Label": prediction
                }

                for class_index, class_name in enumerate(CLASS_NAMES):
                    record[f"Probability_{class_name}"] = float(
                        probabilities[index, class_index].item()
                    )

                records.append(record)
                targets.append(target)
                predictions.append(prediction)

    pd.DataFrame(records).to_csv(
        prediction_path,
        index=False
    )

    return calculate_metrics(targets, predictions)


def run_fold(repeat, fold, epochs_override=None):
    seed = (
        CONFIG["seed"]
        + repeat * 1000
        + fold
    )

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp_enabled = (
        CONFIG["use_amp"] and device.type == "cuda"
    )

    amp_dtype = (
        torch.bfloat16
        if CONFIG["amp_dtype"].lower() == "bfloat16"
        else torch.float16
    )

    epochs = (
        epochs_override
        if epochs_override is not None
        else CONFIG["epochs"]
    )

    fold_dir = (
        ROOT
        / "outputs"
        / "cross_validation"
        / f"repeat_{repeat:02d}"
        / f"fold_{fold:02d}"
    )

    fold_dir.mkdir(parents=True, exist_ok=True)

    repeat_dir = CV_DIR / f"repeat_{repeat:02d}"

    train_dataset = ADNIFoldDataset(
        ROOT / "data",
        repeat_dir / f"fold_{fold:02d}_train.csv"
    )

    val_dataset = ADNIFoldDataset(
        ROOT / "data",
        repeat_dir / f"fold_{fold:02d}_val.csv"
    )

    train_loader = create_loader(
        train_dataset,
        True,
        seed
    )

    val_loader = create_loader(
        val_dataset,
        False,
        seed
    )

    train_labels = [
        sample[1] for sample in train_dataset.samples
    ]

    counts = np.bincount(
        train_labels,
        minlength=3
    )

    if np.any(counts == 0):
        raise RuntimeError(
            "A training fold is missing a diagnostic class."
        )

    class_weights = (
        len(train_dataset)
        / (3.0 * counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    base_model = SwinUNETR(
        in_channels=1,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True
    )

    model = AdaptiveSDASwinFull(
        swin_encoder=base_model.swinViT,
        num_classes=3,
        patch_chunk_size=CONFIG["patch_chunk_size"]
    ).to(device)

    initialized = torch.load(
        INITIALIZED_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        initialized["model_state_dict"],
        strict=True
    )

    swin_parameters = list(
        model.swin_encoder.parameters()
    )

    swin_ids = {
        id(parameter)
        for parameter in swin_parameters
    }

    new_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in swin_ids
        and parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": swin_parameters,
                "lr": CONFIG["swin_learning_rate"]
            },
            {
                "params": new_parameters,
                "lr": CONFIG["new_modules_learning_rate"]
            }
        ],
        weight_decay=CONFIG["weight_decay"]
    )

    warmup_epochs = CONFIG["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (
            (epoch - warmup_epochs)
            / max(1, epochs - warmup_epochs)
        )

        return 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )

    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    best_path = fold_dir / "best_model.pt"
    last_path = fold_dir / "last_model.pt"
    history_path = fold_dir / "history.csv"
    metrics_path = fold_dir / "best_validation_metrics.json"
    predictions_path = fold_dir / "validation_predictions.csv"

    best_score = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 65)
    print(f"REPEAT {repeat} | FOLD {fold}")
    print("=" * 65)
    print("Device:", device)
    print("AMP:", amp_enabled)
    print("Seed:", seed)
    print("Training subjects:", len(train_dataset))
    print("Validation subjects:", len(val_dataset))
    print("Training class counts:", counts.tolist())
    print("Class weights:", class_weights.cpu().tolist())

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
            "swin_lr",
            "new_modules_lr",
            "epoch_seconds"
        ])

        for epoch in range(epochs):
            start_time = time.time()

            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                amp_enabled,
                amp_dtype,
                optimizer,
                scaler
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
                amp_dtype
            )

            score = val_metrics["macro_f1"]

            improved = (
                score
                > best_score + CONFIG["minimum_improvement"]
            )

            if improved:
                best_score = score
                best_epoch = epoch + 1
                patience_counter = 0

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    best_score,
                    repeat,
                    fold,
                    seed,
                    class_weights
                )

                with open(
                    metrics_path,
                    "w",
                    encoding="utf-8"
                ) as metrics_file:
                    json.dump(
                        val_metrics,
                        metrics_file,
                        indent=4
                    )
            else:
                patience_counter += 1

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_score,
                repeat,
                fold,
                seed,
                class_weights
            )

            epoch_seconds = time.time() - start_time

            current_lrs = [
                group["lr"]
                for group in optimizer.param_groups
            ]

            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["balanced_accuracy"],
                current_lrs[0],
                current_lrs[1],
                epoch_seconds
            ])

            file.flush()

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(
                f"Train Loss: {train_metrics['loss']:.5f} | "
                f"Accuracy: {train_metrics['accuracy']:.5f} | "
                f"Macro-F1: {train_metrics['macro_f1']:.5f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.5f} | "
                f"Accuracy: {val_metrics['accuracy']:.5f} | "
                f"Macro-F1: {val_metrics['macro_f1']:.5f} | "
                f"Balanced Acc: {val_metrics['balanced_accuracy']:.5f}"
            )

            print(
                f"Best Macro-F1: {best_score:.5f} "
                f"(Epoch {best_epoch})"
            )

            print(
                f"Learning rates: "
                f"Swin={current_lrs[0]:.8f}, "
                f"New={current_lrs[1]:.8f}"
            )

            print(f"Epoch time: {epoch_seconds:.2f}s")

            scheduler.step()

            if (
                patience_counter
                >= CONFIG["early_stopping_patience"]
            ):
                print("\nEARLY STOPPING")
                break

    if not best_path.exists():
        raise RuntimeError(
            "Best checkpoint was not generated."
        )

    best_metrics = evaluate_best_checkpoint(
        model,
        val_loader,
        criterion,
        device,
        amp_enabled,
        amp_dtype,
        best_path,
        predictions_path
    )

    result = {
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "training_subjects": len(train_dataset),
        "validation_subjects": len(val_dataset),
        "training_class_counts": counts.tolist(),
        "class_weights": class_weights.cpu().tolist(),
        "checkpoint": str(best_path.relative_to(ROOT))
    }

    with open(
        fold_dir / "fold_result.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=4)

    print("\nFOLD COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", best_metrics["macro_f1"])
    print("Validation predictions:", predictions_path)

    del model
    del base_model
    del optimizer
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repeat", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        if args.repeat is not None or args.fold is not None:
            parser.error(
                "Do not combine --all with --repeat or --fold."
            )

        jobs = [
            (repeat, fold)
            for repeat in range(1, 6)
            for fold in range(1, 6)
        ]

    else:
        if args.repeat is None or args.fold is None:
            parser.error(
                "Specify --repeat and --fold, or use --all."
            )

        if args.repeat not in range(1, 6):
            parser.error("Repeat must be between 1 and 5.")

        if args.fold not in range(1, 6):
            parser.error("Fold must be between 1 and 5.")

        jobs = [(args.repeat, args.fold)]

    if args.epochs is not None and args.epochs < 1:
        parser.error("Epochs must be positive.")

    results = []

    for repeat, fold in jobs:
        result = run_fold(
            repeat,
            fold,
            args.epochs
        )

        results.append(result)

        summary_path = (
            ROOT
            / "outputs"
            / "cross_validation"
            / "completed_folds.json"
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        existing = []

        if summary_path.exists():
            with open(
                summary_path,
                "r",
                encoding="utf-8"
            ) as file:
                existing = json.load(file)

        existing = [
            item
            for item in existing
            if not (
                item["repeat"] == repeat
                and item["fold"] == fold
            )
        ]

        existing.append(result)

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(existing, file, indent=4)

    print("\nREQUESTED CROSS-VALIDATION JOBS COMPLETED")


if __name__ == "__main__":
    main()
