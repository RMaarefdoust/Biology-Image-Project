---

# Neuron Images Project

## Overview
This project focuses on analyzing neuron images using a variety of machine learning techniques. The dataset and the corresponding codes used in this project are outlined below.

## Dataset
- **Filename:** `Dataset-test-original-image.rar`
- **Description:** This file contains the original images of neuron structures that are used for analysis and model training/testing.

## Codes
1. **CBAM_CNN-NerounImage.py**
    - This script implements a Convolutional Block Attention Module (CBAM) based Convolutional Neural Network (CNN) to process and analyze neuron images.
  
2. **SAM.py**
    - This script uses the Sharpness-Aware Minimization (SAM) optimizer for training the neural network to enhance its performance on neuron image data.
  
3. **GradCAM.py**
    - This script applies Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual explanations for decisions made by the CNN model on neuron images.

## Getting Started
To get started with this project, follow the steps below:

### Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Extract the dataset:
    ```bash
    tar -xvf Dataset-test-original-image.rar
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. To train and evaluate the CNN model with CBAM:
    ```bash
    python CBAM_CNN-NerounImage.py
    ```

2. To train the model using the SAM optimizer:
    ```bash
    python SAM.py
    ```

3. To generate Grad-CAM visualizations:
    ```bash
    python GradCAM.py
    ```

