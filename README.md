# Alzheimer's Disease Detection using 3D Vision Transformer (ViT)

This project implements a 3D Vision Transformer (ViT) model to classify MRI scans for detecting Alzheimer's disease. 
The ViT architecture is leveraged to capture spatial patterns in 3D medical imaging, making it suitable for MRI-based classification tasks.

## Table of Contents

- Overview
- Project Structure
- Installation
- Usage
- Model Architecture
- Experiments and Results
- Contributing
- License

## Overview

Alzheimer's disease affects millions globally, and early detection through imaging techniques such as MRI is crucial for 
effective treatment. This project applies state-of-the-art deep learning techniques using a Vision Transformer adapted for 3D medical imaging.

Key features:

- Utilizes 3D patches from MRI scans to classify Alzheimer's disease stages.
- Designed for both binary classification (Alzheimer's vs. Healthy) and multi-class classification (MCI, AD, Healthy).
- Integrated experiment logging and evaluation tools to track model performance.

## Project Structure

```plaintext
alzheimer_vit/
├── experiments/           # Logs, checkpoints, and experiment results
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── src/                   # Source code for the project
│   ├── data/              # Data loading, augmentation, and preprocessing scripts
│   │   ├── augmentation.py
│   │   ├── data_loader.py
│   │   └── preprocess.py
│   ├── models/            # Model definitions, training, and evaluation scripts
│   │   ├── architectures/ # Model architectures
│   │   │   ├── cnn/
│   │   │   │   └── cnn3d.py
│   │   │   └── vit/
│   │   │       ├── vit3d_b16.py
│   │   │       ├── vit3d_l32.py
│   │   │       └── vit3d_m8.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── utils/             # Utility functions
│   │   ├── logger.py
│   │   ├── train_utils.py
│   │   └── main.py        # Main script to run the model
├── tests/                 # Unit tests for various project components
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_train.py
├── config.yaml            # Configuration file for experiments
├── requirements.txt       # Python dependencies
└── README.md              # Project README


