import sys
import json
import csv
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from monai.networks.nets import SwinUNETR
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    balanced_accuracy_score
)

from src.adni_dataset import ADNIMRIDataset
from models.adaptive_sda_swin_full import AdaptiveSDASwinFull


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "training_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8-sig") as file:
    config = json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(split, shuffle):
    dataset = ADNIMRIDataset(ROOT / "data", split)

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None
    )


def calculate_metrics(targets, predictions):
    labels = [0, 1, 2]

    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0
    )

    return {
        "macro_f1": float(f1_score(
            targets,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0
        )),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=labels
        ).tolist()
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False
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
                dtype=torch.float16,
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

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["gradient_clip_norm"]
                )

                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predicted = logits.detach().argmax(dim=1)

        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(predicted.cpu().tolist())

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
    config
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_macro_f1": best_score,
            "config": config,
            "class_mapping": {
                "CN": 0,
                "MCI": 1,
                "AD": 2
            }
        },
        path
    )


def main():
    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp_enabled = (
        config["use_amp"] and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    log_dir = ROOT / config["log_directory"]
    log_dir.mkdir(parents=True, exist_ok=True)

    best_path = ROOT / config["best_checkpoint"]
    last_path = ROOT / config["last_checkpoint"]
    initialized_path = ROOT / config["initialized_checkpoint"]

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)

    class_weights = torch.tensor(
        config["class_weights"],
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
        patch_chunk_size=config["patch_chunk_size"]
    ).to(device)

    initialized_checkpoint = torch.load(
        initialized_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        initialized_checkpoint["model_state_dict"],
        strict=True
    )

    swin_parameters = list(model.swin_encoder.parameters())
    swin_parameter_ids = {
        id(parameter) for parameter in swin_parameters
    }

    new_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in swin_parameter_ids
        and parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": swin_parameters,
                "lr": config["swin_learning_rate"]
            },
            {
                "params": new_parameters,
                "lr": config["new_modules_learning_rate"]
            }
        ],
        weight_decay=config["weight_decay"]
    )

    epochs = config["epochs"]
    warmup_epochs = config["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = (
            (epoch - warmup_epochs)
            / max(1, epochs - warmup_epochs)
        )

        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    scaler = GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    history_path = log_dir / "training_history.csv"
    metrics_path = log_dir / "best_validation_metrics.json"

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nSDA-SWIN TRAINING")
    print("Device:", device)
    print("AMP enabled:", amp_enabled)
    print("Training subjects:", len(train_loader.dataset))
    print("Validation subjects:", len(val_loader.dataset))
    print("Total parameters:", sum(
        parameter.numel()
        for parameter in model.parameters()
    ))
    print("Swin learning rate:", config["swin_learning_rate"])
    print("New modules learning rate:", config["new_modules_learning_rate"])
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
            "train_macro_f1",
            "train_balanced_accuracy",
            "val_loss",
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
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled
            )

            current_score = val_metrics["macro_f1"]

            improved = (
                current_score
                > best_score + config["minimum_improvement"]
            )

            if improved:
                best_score = current_score
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    best_score,
                    config
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
                epochs_without_improvement += 1

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_score,
                config
            )

            epoch_seconds = time.time() - start_time

            current_lrs = [
                group["lr"]
                for group in optimizer.param_groups
            ]

            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["macro_f1"],
                train_metrics["balanced_accuracy"],
                val_metrics["loss"],
                val_metrics["macro_f1"],
                val_metrics["balanced_accuracy"],
                current_lrs[0],
                current_lrs[1],
                epoch_seconds
            ])

            file.flush()

            print(
                f"\nEpoch {epoch + 1}/{epochs}"
            )

            print(
                f"Train Loss: {train_metrics['loss']:.5f} | "
                f"Train Macro-F1: {train_metrics['macro_f1']:.5f} | "
                f"Train Balanced Acc: {train_metrics['balanced_accuracy']:.5f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.5f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.5f} | "
                f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.5f}"
            )

            print(
                f"LR Swin: {current_lrs[0]:.8f} | "
                f"LR New Modules: {current_lrs[1]:.8f}"
            )

            print(
                f"Best Macro-F1: {best_score:.5f} "
                f"(Epoch {best_epoch})"
            )

            print(
                f"Epoch time: {epoch_seconds:.2f}s"
            )

            scheduler.step()

            if (
                epochs_without_improvement
                >= config["early_stopping_patience"]
            ):
                print("\nEARLY STOPPING")
                break

    print("\nTRAINING COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best validation macro-F1:", best_score)
    print("Best checkpoint:", best_path)
    print("Latest checkpoint:", last_path)
    print("Training history:", history_path)
    print("Best validation metrics:", metrics_path)


if __name__ == "__main__":
    main()
