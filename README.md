# Handwritten-Digit-Recognition

This project implements and compares three neural network models for handwritten digit recognition on the MNIST dataset using TensorFlow/Keras:

- **Model 1 – Baseline Dense Network**  
- **Model 2 – Dense Network with Dropout**  
- **Model 3 – Convolutional Neural Network (CNN)**  

The code is contained in the Jupyter notebook **`Digit_Recognition.ipynb`**, which walks through data loading, preprocessing, model definition, training, evaluation, and comparison.

***

## 1. Project Overview

The goal is to classify 28×28 grayscale images of handwritten digits (0–9) from the MNIST dataset. The project explores how different architectures and regularization choices affect:

- Classification accuracy  
- Overfitting / generalization  
- Model size (number of parameters)  
- Training time  

All experiments use the standard MNIST split: **60,000 training images** and **10,000 test images**.

***

## 2. Models Implemented

All models are implemented in Keras and trained with the Adam optimizer and categorical cross-entropy loss.

### Model 1: Baseline Dense Network

- Input: 784-dimensional vector (flattened 28×28 image)  
- Hidden Layer 1: Dense(128, ReLU)  
- Hidden Layer 2: Dense(64, ReLU)  
- Output: Dense(10, Softmax)  
- Parameters: **109,386**  

### Model 2: Dense Network with Dropout

- Same structure as Model 1, plus:  
  - Dropout(0.2) after each hidden layer  
- Parameters: **109,386**  

### Model 3: Convolutional Neural Network (CNN)

- Input: 28×28×1 grayscale image  
- Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)  
- Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)  
- Flatten  
- Dense(128, ReLU) → Dropout(0.5)  
- Output: Dense(10, Softmax)  
- Parameters: **225,034**  

***

## 3. Results Summary

The notebook computes a comparison table of the three models:

| Model   | Architecture        | Training Accuracy | Test Accuracy | Training Time (approx.) | Parameters |
|--------|---------------------|-------------------|---------------|-------------------------|-----------|
| Model 1 | Baseline Dense      | 0.9945            | 0.9743        | ~15 s                   | 109,386   |
| Model 2 | Dense + Dropout     | 0.9859            | 0.9790        | ~20 s                   | 109,386   |
| Model 3 | CNN (Conv + Pool)   | 0.9918            | 0.9916        | ~400 s                  | 225,034   |

Key observations:

- All models perform well on MNIST, but the **CNN** achieves the **best test accuracy (~99.16%)**.  
- Dropout slightly reduces training accuracy but improves generalization for the dense model.  
- The CNN is more accurate but also **larger** and **much slower** to train than the dense models.

The notebook also includes:

- Training/validation **accuracy and loss curves** for each model  
- A **confusion matrix** for the CNN to visualize per-class performance  

***
