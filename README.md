---

# Biology Image Classification Project

## Project Overview

This project aims to classify biological images using Convolutional Neural Network (CNN) models enhanced with Convolutional Block Attention Module (CBAM).

### Data Preparation

**Data Splitting**: The dataset should be split into training, testing, and validation sets. Place the images in the corresponding folders under the `data` directory.
   - `OriginalImages-split/train`: Training data
   - `OriginalImages-split/test`: Testing data
   - `OriginalImages-split/val`: Validation data

Images dataset: https://drive.google.com/file/d/1Mp7vSnrGLwVAxNP5nosrRD3eWgt4eWaj/view?usp=sharing

### Training the Models

Each model has a separate script for training. You can run the training scripts as follows:

1.**Simple CNN**:
    ```bash
    python CNN.py
    python CNN_Test.py
    ```
    
2. **CNN + CBAM**:
    ```bash
    python CNN+CBAM.py
    python CNN+CBAM_Test.py
    ```

3. **ResNet50**:
    ```bash
    python ResNet50.py
    python ResNet50_Test.py
    ```

4. **ResNet50 + CBAM**:
    ```bash
    python Res+CBAM.py
    python Res+CBAM_Test.py
    ```



## Model Descriptions

### CNN + CBAM

The CNN + CBAM model consists of a standard convolutional neural network architecture enhanced with the Convolutional Block Attention Module to improve feature extraction and focus on important parts of the input image.

### ResNet50 + CBAM

ResNet50 is a well-known deep residual network with 50 layers. By incorporating CBAM, the model enhances its ability to focus on relevant features in the image.

### ResNet110 + CBAM

ResNet110 is a deeper version of the ResNet architecture with 110 layers, providing greater depth and capacity. The addition of CBAM further improves its performance by emphasizing significant features.

