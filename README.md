# Flower Classification with Transfer Learning using MobileNet

This repository contains a transfer learning model using **MobileNet** for classifying flower species from the popular **Flower Classification Dataset**. The model leverages pre-trained weights from MobileNet, a lightweight neural network architecture, and fine-tunes it for flower classification.

### Key Features:
- **Transfer Learning**: Uses MobileNet as the base model and fine-tunes it to classify flower species.
- **Data Augmentation**: Implements various image augmentation techniques to improve model generalization.
- **Evaluation Metrics**: The model is evaluated using key metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- **Model Prediction**: Includes code to predict flower species on new images and visualize the results.

## Dataset
This model is trained using the **Flower Classification Dataset**, which contains five categories of flowers:
- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

The dataset is available for download through [TensorFlow Datasets](https://www.tensorflow.org/datasets/community_catalog/huggingface/flower_photos).

## Requirements
To run this notebook, ensure you have the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

You can install the dependencies using the following command:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
```

## Instructions to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/flower-classification-transfer-learning.git
   ```
2. Open the Jupyter Notebook file `flower_classification_transfer_learning.ipynb`.
3. Execute the notebook cells sequentially.
4. Review the model's performance, including training and validation accuracy, loss, confusion matrix, and predictions.

## Model Architecture
The model uses **MobileNetV2** as the base, with the top layers frozen and a custom fully connected layer added for classification:
- **MobileNetV2**: A lightweight architecture suitable for mobile devices, pretrained on ImageNet.
- **Data Augmentation**: Random transformations (e.g., rotations, shifts, flips) applied to training data to improve model robustness.
- **Fine-tuning**: Unfreezing the last layers of MobileNet to adapt the model to the flower dataset.

## Evaluation
The model performance is evaluated on:
- **Validation Loss**: The loss value on the validation set during training.
- **Validation Accuracy**: The accuracy achieved on the validation set.
- **Confusion Matrix**: A matrix showing the true vs predicted classes.
- **Classification Report**: Precision, recall, and F1-score for each flower class.
- **Random Image Predictions**: Displays predictions on randomly selected validation images.

## Model Results
The model achieves an accuracy of approximately **87%** on the validation set, though further fine-tuning and adjustments (e.g., learning rate, data augmentation) may improve the performance.

## Future Improvements
- Implement hyperparameter optimization.
- Use more advanced architectures such as **EfficientNet** or **ResNet**.
- Address any class imbalance by incorporating class weights during training.
