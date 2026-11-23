# SDA-Swin

Official implementation of SDA-Swin for cross-sectional classification of Alzheimer's disease using structural MRI.

## Architecture

SDA-Swin consists of two parallel branches:

1. Hierarchical Swin Transformer branch for multi-scale MRI feature extraction.
2. Adaptive structural-detail branch for adaptive 3D patch extraction, shared patch encoding, learnable focus-defocus grouping, and coordinate-aware reconstruction.

Both branches are integrated using adaptive multi-scale feature fusion, followed by a classification head.

## Repository Structure

- configs/: Model and training configuration
- models/: Model architecture and modules
- scripts/: Training and cross-validation scripts
- src/: Dataset implementation

## Installation

Python 3.11 is recommended.

Install dependencies:

pip install -r requirements.txt

## Training

Standard training:

python scripts/train.py

Repeated cross-validation:

python scripts/train_repeated_cv.py

Update dataset paths and configuration before running.

## Dataset

This implementation is designed for structural MRI data from ADNI. The dataset is not included. Users must obtain authorized access to ADNI separately.

## Pretrained Weights

Pretrained weights and experimental checkpoints are not included.

## License

Please contact the authors regarding reuse and redistribution.
