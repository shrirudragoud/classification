# Cat vs Dog Image Classification

A Convolutional Neural Network (CNN) based image classification project to distinguish between cats and dogs.

## Project Structure
```
classification/
├── data/                     # Data directory
│   ├── train/               # Training images
│   │   ├── cats/           
│   │   └── dogs/           
│   └── test/                # Test images
├── src/                     # Source code
│   ├── config.py           # Configuration settings
│   ├── setup_data.py       # Dataset setup script
│   ├── data_loader.py      # Data loading/preprocessing
│   ├── model.py            # CNN model architecture
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── models/                  # Saved models
├── submissions/            # Prediction outputs
└── requirements.txt        # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup the dataset:
```bash
python src/setup_data.py
```
This will:
- Download the dataset using kagglehub
- Organize images into train and test directories
- Split data into cats and dogs categories

## Usage

1. **Training the Model**
```bash
python src/train.py
```
This will:
- Load and preprocess the training data
- Train the CNN model
- Save the best model to `models/best_model.h5`
- Generate training history plots

2. **Making Predictions**
```bash
python src/predict.py
```
This will:
- Load the trained model
- Make predictions on test images
- Generate a submission file in `submissions/predictions.csv`
- Display prediction statistics

## Model Architecture

- Input: 80x80 grayscale images
- Two Conv2D layers (64 filters, 3×3 kernel)
- Two MaxPooling layers (2×2)
- Flatten layer
- Dense layer (64 units)
- Output layer with sigmoid activation

## Training Parameters

- Optimizer: Adam
- Loss: Binary Cross-entropy
- Batch size: 32
- Epochs: 10
- Validation split: 20%

## Files Description

- `config.py`: Contains all configuration parameters
- `setup_data.py`: Handles dataset download and organization
- `data_loader.py`: Handles dataset loading and preprocessing
- `model.py`: Defines the CNN architecture
- `train.py`: Contains the training pipeline
- `predict.py`: Handles prediction and submission file generation

## Requirements

- tensorflow>=2.6.0
- opencv-python>=4.5.0
- numpy>=1.19.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- kagglehub>=0.1.0

## Performance

The model aims to achieve:
- Training accuracy: >90%
- Validation accuracy: >85%

Training metrics and plots are saved in the `models` directory.