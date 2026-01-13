# Tumor Detect 1

## Overview
First of all, this notebook is for prototyping, to check if the algorithms and other techniques are realistic for brain tumor classification and segmentation (Multitask). There will be another project for it later.

This notebook implements a deep learning pipeline for tumor detection and segmentation using PyTorch. It focuses on building, training, and evaluating a neural network model for medical image segmentation, with support for Dice loss and preliminary work toward Grad-CAM–based model interpretability.

The notebook is implemented and ran on Google Colab, so there will be modification needed when run it in other environments.

---

## Features
- PyTorch-based training pipeline
- Custom neural network architecture for image segmentation
- Dice Loss for segmentation performance
- Model checkpoint loading/saving
- Dataset loading and preprocessing
- Visualization utilities using Matplotlib
- Initial (work-in-progress) Grad-CAM implementation for explainability

---

## Requirements
The notebook installs and uses the following major dependencies:
- Python 3.8+
- PyTorch
- torchvision
- transformers
- datasets (Hugging Face)
- numpy
- matplotlib
- scikit-image
- opencv-python
- pillow (PIL)
- torchinfo
- accelerate
- tqdm

Most dependencies are installed automatically at the top of the notebook.

---

## Notebook Structure
### 1. Environment Setup
Installs all required libraries using pip.

### 2. Imports
Loads essential libraries for:
Deep learning (PyTorch)
Image processing
Dataset handling
Visualization

### 3. Model Definition
Defines a custom neural network architecture for tumor segmentation, using:
Convolutional layers
Nonlinear activations
Feature extraction suitable for medical imaging

### 4. Loss Function
Implements Dice Loss, which is well-suited for image segmentation tasks with class imbalance.

### 5. Training Setup
Optimizer configuration
Training loop with progress tracking
Model summary using torchinfo

### 6. Model Checkpointing
Automatically loads best_model.pth if available
Saves the best-performing model during training

### 7. Grad-CAM (In Progress)
Early-stage implementation for visualizing model attention
Intended to highlight tumor regions influencing predictions

---

## Notes
- Grad-CAM functionality is marked as work in progress and may require additional refinement.
- Training performance depends heavily on dataset quality and hardware (GPU recommended).
