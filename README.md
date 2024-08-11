# Skin Condition Classification with Data Augmentation and InceptionResNetV2

This project focuses on classifying skin conditions using a deep learning model (InceptionResNetV2) trained on an augmented dataset of skin condition images.

## Table of Contents
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Dataset
The dataset consists of images belonging to six categories of skin conditions:
- Acne
- Carcinoma
- Eczema
- Keratosis
- Milia
- Rosacea

### Original Dataset Structure
- The dataset is stored in a zip file: `Skin_Conditions.zip`.
- The zip file contains folders for each skin condition category, each containing several images.

### Augmented Dataset Structure
- The augmented dataset is generated and stored in `Augmented_Skin_Conditions`.
- Each category contains the original images along with newly generated images through data augmentation.

## Data Augmentation
To enhance the diversity of the training dataset and avoid overfitting, the following augmentation techniques were applied:
- **Rotation**: Images were rotated by a random angle between -30° and 30°.
- **Flipping**: Images were randomly flipped horizontally, vertically, or both.
- **Scaling**: Images were scaled by a random factor between 0.7 and 1.3.
- **Cropping**: A random crop was applied to each image.

The goal was to ensure that each category contains at least 400 images.

## Model Architecture
The model is based on **InceptionResNetV2**, a pre-trained convolutional neural network, with the following modifications:
- **Global Average Pooling**: Reduces the spatial dimensions to a single vector.
- **Fully Connected Layer**: 1024 neurons with ReLU activation.
- **Dropout Layer**: 20% dropout to reduce overfitting.
- **Output Layer**: Softmax activation for multi-class classification.

## Training
The model was fine-tuned on the augmented dataset. The final layers of the pre-trained model were unfrozen and trained along with the custom layers.

### Training Parameters
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.
- **Batch Size**: 32.
- **Epochs**: 15.

### Callbacks
- **Model Checkpoint**: Saves the best model based on validation loss.
- **Early Stopping**: Stops training if the validation loss doesn't improve for 5 epochs.
- **Learning Rate Reduction**: Reduces the learning rate by a factor of 0.2 if validation loss doesn't improve for 3 epochs.

## Evaluation
The model was evaluated on a validation set, and the following metrics were reported:
- **Validation Loss**
- **Validation Accuracy**
- **Confusion Matrix**
- **Classification Report**: Precision, recall, F1-score for each class.

## Dependencies
This project requires the following libraries:
- `os`
- `cv2`
- `random`
- `numpy`
- `zipfile`
- `PIL`
- `tensorflow`
- `sklearn`
- `shutil`
- `google.colab`

## Acknowledgments
This project was developed using TensorFlow and Keras, with the InceptionResNetV2 model architecture provided by the TensorFlow library.
