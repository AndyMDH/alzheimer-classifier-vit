# Alzheimer's Detection using Vision Transformers

This project implements an Alzheimer's disease detection system using Vision Transformers (ViT) and 3D Convolutional Neural Networks. It leverages transfer learning techniques to adapt pre-trained models for MRI scan analysis.

Trying the 🤗Hugging Face Demo: Alzheimer Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to detect Alzheimer's disease using MRI scans. It implements and compares three different deep learning architectures:
1. 2D Vision Transformer (ViT)
2. 3D Vision Transformer
3. 3D Convolutional Neural Network (CNN)

The project utilizes transfer learning techniques, starting with pre-trained models and adapting them for the specific task of Alzheimer's detection.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/alzheimers-detection-vit.git
   cd alzheimers-detection-vit
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
alzheimer_vit/
├── data/
│   ├── test/
│   └── train/
├── experiments/
├── notebooks/
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── architectures/
│   │   │   ├── vit2d.py
│   │   │   ├── vit3d.py
│   │   │   └── cnn3d.py
│   │   ├── evaluate.py
│   │   └── train.py
│   └── utils/
│       └── logger.py
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_train.py
├── .env
├── .gitignore
├── config.yaml
├── main.py
├── README.md
└── requirements.txt
```
## Project Visualisation

## Classifier Performance (table)


## Usage

To run the main script:

```
python main.py
```

This will load the dataset, prepare the data, create the specified model, train it, and evaluate its performance.

## Models

The project implements three types of models:

1. 2D Vision Transformer (8/16/32 Patch Embeddings) (`src/models/architectures/vit2d.py`)
2. 3D Vision Transformer (8/16/32 Patch Embeddings) (`src/models/architectures/vit3d.py`)
3. 3D CNN (ResNet101)(`src/models/architectures/cnn3d.py`)

You can specify which model to use in the `config.yaml` file.

## Dataset

The project uses the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. Data is loaded from Hugging Face datasets and preprocessed using MONAI transforms.

To start training, run:

```
python main.py
```

Training parameters can be adjusted in the `config.yaml` file.

## Evaluation

Model evaluation is performed using the `evaluate_model` function in `src/models/evaluate.py`. It uses MONAI's `SupervisedEvaluator` and calculates the ROC AUC metric.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
