# Potato Disease Classification using CNN

This project focuses on the classification of potato diseases using deep learning techniques. Accurate identification of potato diseases is crucial for farmers to manage crops effectively, minimize losses, and ensure high-quality produce. Early detection and classification of diseases enable timely intervention, reducing the spread of infections and enhancing crop yield and quality.

## Dataset
- **Source**: The dataset consists of images of healthy and diseased potato leaves.
- **Classes**: Various classes represent different diseases and healthy leaves.

## Steps in the Notebook

### 1. Data Import and Exploration
- **Loading Data**: Imported the dataset and explored the structure and distribution of images.
- **Visualization**: Visualized sample images from each class to understand the data.

### 2. Data Preprocessing
- **Image Resizing**: Resized all images to a consistent size suitable for the model.
- **Normalization**: Normalized image pixel values to improve model performance.

### 3. Data Augmentation
- **Techniques Used**: Applied augmentation techniques such as rotation, flipping, and zooming to increase the diversity of the training data.
- **Implementation**: Used the Keras `ImageDataGenerator` for augmentation.

### 4. Model Building
- **Architecture**: Constructed a Convolutional Neural Network (CNN) using Keras.
- **Layers**: Included convolutional layers, pooling layers, and dense layers for feature extraction and classification.

### 5. Model Compilation
- **Optimizer**: Used the Adam optimizer for training the model.
- **Loss Function**: Employed categorical cross-entropy as the loss function.
- **Metrics**: Monitored accuracy as the performance metric.

### 6. Model Training
- **Training Process**: Trained the model using the augmented training data.
- **Validation**: Evaluated the model's performance on a validation dataset.

### 7. Model Evaluation
- **Metrics**: Assessed the model using accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Visualized the confusion matrix to understand misclassifications.

### 8. Model Saving
- **Saving the Model**: Saved the trained model to disk for future use.
- **Serialization**: Serialized the model architecture and weights.

### 9. Model Prediction
- **New Data**: Used the trained model to make predictions on new, unseen images.
- **Visualization**: Displayed the predictions alongside the actual images.
