
# SleepNet Research

Research and implementation of the novel SleepNet method for both text and image classification tasks.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Features

- **Vision Transformer (ViT) Integration**: Uses the ViT model to enhance feature extraction capabilities.
- **ResNet18 Architecture**: Leverages the power of ResNet18 for accurate image classification.
- **Data Processing**: Incorporates data augmentation and normalization techniques for better generalization across datasets.
- **GPU Memory Optimization**: Periodically clears GPU memory, ensuring efficient resource utilization during long training sessions.
- **Adaptive Learning**: Implements a learning rate scheduler for improved model convergence.

## Requirements

- Python 3.8
- PyTorch
- torchvision
- transformers library (from HuggingFace)
- CUDA-compatible GPU (optional but recommended for faster training)

## Usage

1. **Setup & Installation**:

   Install the necessary packages:
   ```
   pip install -r requirements.txt
   ```

2. **Training text classifier**:

   To train the SleepNet model using default parameters:
   ```bash
   python train_NLP.py
   ```

   For customized training settings:
   ```bash
   python train_NLP.py --dataset 'ag_news' --epochs 50 --num_classes 10
   ```

3. **Training vision classifier**:

   To train the SleepNet model using default parameters:
   ```bash
   python train_vision.py
   ```

   For customized training settings:
   ```bash
   python train_vision.py --dataset 'cifar100' --epochs 50 --num_classes 10
   ```

---
