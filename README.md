# Coca-Cola Product Instance Segmentation using YOLOv11

This project develops an instance segmentation model to detect and recognize Coca-Cola products in images, specifically trained on Algerian Coca-Cola products.

## Key Features
- **Model**: YOLOv11 for instance segmentation.
- **Dataset**: Custom dataset of ~180 images, manually collected and annotated using Label Studio.
- **Augmentation**: Rotation, flipping, scaling, color jittering, and noise added for robustness.
- **Training**: Trained on NVIDIA RTX 3060 GPU for 86 epochs.

## Repository Structure
Project/
├── dataset/ # Original and augmented images/labels
├── runs/ # Training and validation results
└── Training, validation, and live prediction scripts

## Usage
- **Train**: `python train.py`
- **Validate**: `python test.py`
- **Prediction**: `python prediction.py`

## Limitations
- Small dataset size.
- Struggles with dense product arrangements.
