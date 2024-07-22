![FaceRecognition](https://github.com/user-attachments/assets/dbbcca59-a78e-469b-8695-e6610d26b4f0)
# A user-friendly web interface Face Recognition

This project focuses on building a face recognition model. Accurate identification of these faces is essential for professional networking websites, media applications, and business intelligence platforms. The project includes steps for data collection, preprocessing, model training, and deployment, along with a user-friendly web interface for image uploads and face recognition.

## Steps in the Notebook

### 1. Data Import and Exploration
- **Loading Data**: Imported the dataset and explored the structure and distribution of images.
- **Visualization**: Visualized sample images from each class to understand the data.

### 2. Data Preprocessing
- **Image Resizing**: Resized all images to a consistent size suitable for the model.
- **Normalization**: Normalized image pixel values to improve model performance.

### 3. Data Augmentation
- **Techniques Used**: Applied augmentation techniques such as rotation and flipping to increase the diversity of the training data.

### 6. Model Training
- **Training Process**: Trained the model using the augmented training data on 3 different models.
- **Validation**: Evaluated the model's performance on a test dataset.

### 7. Model Evaluation
- **Metrics**: Assessed the model using accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Visualized the confusion matrix to understand misclassifications.

### 8. Model Saving
- **Saving the Model**: Saved the trained model to disk for future use.
- **Serialization**: Serialized the model architecture and weights.

### 9. Model Prediction
- **New Data**: Used the trained model to make predictions on new, unseen images.
- **Visualization**: Displayed the predictions alongside the actual images.

  Finally, the selected model is used to create a user friendly web inference to make predictions.
