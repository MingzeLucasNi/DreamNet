# DreamNet Research

A comprehensive research and implementation of the novel SleepNet method, designed for both text and image classification tasks.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Features

- **Vision Transformer (ViT) Integration**: Enhances feature extraction capabilities by leveraging the Vision Transformer (ViT) model, providing state-of-the-art performance in image classification.
- **ResNet18 Architecture**: Utilizes the proven ResNet18 architecture to achieve high accuracy in image classification tasks.
- **Advanced Data Processing**: Employs data augmentation and normalization techniques to improve generalization and robustness across diverse datasets.
- **Efficient GPU Memory Management**: Periodically clears GPU memory to optimize resource utilization, especially during extended training sessions.
- **Adaptive Learning Rate Scheduling**: Includes a dynamic learning rate scheduler to facilitate smoother and more effective model convergence.

## Requirements

- Python 3.8 or later
- PyTorch
- torchvision
- Hugging Face's `transformers` library
- CUDA-compatible GPU (recommended for faster training)

## Usage

1. **Setup & Installation**:

   Install all necessary dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Text Classifier**:

   To train the SleepNet model with default parameters:
   ```bash
   python trainer.py
   ```

   For customized training settings, use:
   ```bash
   python trainer.py --dataset 'ag_news' --epochs 50 --num_classes 10
   ```

3. **Training the Vision Classifier**:

   To train the SleepNet model for image classification:
   ```bash
   python train_vision.py --dataset 'cifar100' --epochs 50 --num_classes 10
   ```
