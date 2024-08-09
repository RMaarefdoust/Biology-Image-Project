---

# Biology Image Classification Project

## Project Overview

This project aims to classify biology images using three different Convolutional Neural Network (CNN) models enhanced with Convolutional Block Attention Module (CBAM). The models used are:
1. Generate Dataset Arsenic
2. Augmentation-Train
3. Simple CNN (Train, Test)
4. CNN + CBAM  (Train, Test)
5. ResNet50(Train, Test)
6. ResNet50 + CBAM (Train,Test)
7. ResNet110 + CBAM  (Train,Test)


### Data Preparation

1. **Data Splitting**: The dataset should be split into training, testing, and validation sets. Place the images in the corresponding folders under the `data` directory.
   - `OriginalImages-split/train`: Training data
   - `OriginalImages-split/test`: Testing data
   - `OriginalImages-split/val`: Validation data

2. **Preprocessing**: Run the preprocessing script to prepare the data for training.

```bash
python  generate-dataset-Arsenic.py
```

Original Images: https://drive.google.com/drive/folders/1f64Xn68NcBPcwBku6k71m5OOS_cP_1RE?usp=drive_link

Split Images: https://drive.google.com/drive/folders/1L1BTBZXdasal3ptrQtCFTtBpytGvXbry?usp=drive_link



### Training the Models

Each model has a separate script for training. You can run the training scripts as follows:

1.**Simple CNN**:
    ```bash
    python /Train-Simple-CNN.py
    python /Test-Simple-CNN.py
    ```
2. **CNN + CBAM**:
    ```bash
    python /Train-CNN+CBAM.py
    python /Test-CNN+CBAM.py
    ```

3. **ResNet50**:
    ```bash
    python Train-ResNet50.py
    python Test-ResNet50.py
    ```

4. **ResNet50 + CBAM**:
    ```bash
    python REZNET5-CBAM.py
    ```
5. **ResNet110 + CBAM**:
    ```bash
    python /REZNET110-CBAM.py
    ```


## Model Descriptions

### CNN + CBAM

The CNN + CBAM model consists of a standard convolutional neural network architecture enhanced with the Convolutional Block Attention Module to improve feature extraction and focus on important parts of the input image.

### ResNet50 + CBAM

ResNet50 is a well-known deep residual network with 50 layers. By incorporating CBAM, the model enhances its ability to focus on relevant features in the image.

### ResNet110 + CBAM

ResNet110 is a deeper version of the ResNet architecture with 110 layers, providing greater depth and capacity. The addition of CBAM further improves its performance by emphasizing significant features.

