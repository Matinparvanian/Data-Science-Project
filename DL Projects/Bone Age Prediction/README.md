<div align="center">
  <img src="https://github.com/user-attachments/assets/92edc8f6-2172-487e-bbe3-b2dd8b893107" alt="boneage1">
</div>


# Bone Age Prediction

## Deep CNN Models for Predicting Bone Age from Hand Radiographs

Bone age prediction from hand radiographs is a task that involves using medical imaging to determine the maturity of a patient’s bones. This information is useful for various medical applications, including growth assessment, diagnosis of endocrine disorders, and monitoring treatment efficacy. Deep Convolutional Neural Networks (CNNs) have proven to be a powerful tool for image classification, image regression, machine vision, and feature extraction tasks. This project explains how we built deep convolutional neural networks to predict bone age in months using hand X-ray image data as input. We used Inception V4 and ResNet-50 architectures to achieve reasonable results.


## Project Objectives
1. **Data Preprocessing**: Implement data augmentation and image resizing to represent essential information accurately and increase the diversity of the dataset.
2. **Model Development**: Evaluate different neural network structures to identify the best model for high-precision bone age prediction.
3. **Comparison of Results**: Compare results using the same features and models for test set labels.
4. **Model Comparison**: Analyze the performance of different deep CNN architectures, such as ResNet-50 and Inception V4.

## Related Work
Deep Convolutional Neural Networks (DCNNs) have been widely used in medical image analysis, including bone age prediction from hand radiographs. Significant contributions include:
- **Growth and Development Assessment**: Rajpurkar et al. (2017) developed a CNN for bone age prediction using data augmentation and transfer learning techniques.
- **Transfer Learning**: Studies have fine-tuned pre-trained DCNNs on bone age prediction tasks, achieving good performance with limited training data.
- **Multi-Modal Approaches**: Combining hand radiographs with other imaging modalities, such as MRI, to improve prediction accuracy.
- **Attention Mechanisms**: Highlighting informative regions of hand radiographs to enhance prediction accuracy.

## Processing Pipeline
1. **Data Collection**: Obtain hand X-ray images from hospitals.
2. **Data Preprocessing**: Resize images, apply data augmentation and normalize pixel values.
3. **Model Training**: Use ResNet-50 and Inception V4 architectures for regression tasks.
4. **Prediction**: Predict bone age in months using trained models.

## Images and Features
- Hand X-ray images are divided into training (12,611 images), validation (1,425 images), and test sets (200 images).
- Each image is resized to 256x256 pixels, and pixel values are normalized.

## Learning Framework
### ResNet-50
- **Architecture**: Consists of identity and convolutional blocks, with layers including convolutional, batch normalization, and GELU activation layers.
- **Processing**: Input is a 3D tensor of shape 256x256x3, with multiple stages of convolutional blocks.

### Inception V4
- **Architecture**: Includes stem, Inception-A, reduction A, Inception-B, reduction B, Inception-C blocks, and global average pooling.
- **Processing**: Input is a 3D tensor of shape 256x256x3, with a combination of convolutional layers and pooling layers.

### Tips
1. **Activation Function**: GELU is used for all hidden layers to prevent the vanishing gradient problem.
2. **Loss Function**: Mean Squared Logarithmic Error (MSLE) is used to handle target values spanning several orders of magnitude.
3. **Optimizer**: Adam optimizer that provides a balance between convergence speed and optimization stability.

## Results
- **Evaluation Metrics**: RMSE, MAPE, and R-squared are used to assess model performance.

## Concluding Remarks
Bone age prediction with low error rates remains challenging in medical applications. This project explored two robust CNN architectures, ResNet-50 and Inception V4, for bone age prediction. Both models showed promising results, with Inception V4 performing slightly better. Future improvements could include using attention mechanisms, increasing image resolution, and employing more powerful computational resources.

## References
1. Rajpurkar, P., et al. Automated bone age assessment using deep convolutional neural networks. Medical Image Computing and Computer-Assisted Intervention, 2017.
2. Loeff, F., et al. Transfer learning for bone age assessment. Medical Image Analysis, 2019.
3. Wang, Z., et al. Multi-Modal Bone Age Assessment using Deep Convolutional Neural Networks. Medical Image Analysis, 2021.
4. Liu, Y., et al. Attention-Based Deep Convolutional Neural Network for Bone Age Assessment. Medical Image Analysis, 2021.
5. Pettersen et al. Hand radiographs for skeletal age assessment in pediatric populations. Clinical Radiology and Radiotherapy, 2015.
6. Lippe et al. Bone age in endocrine disorders. Pediatric Endocrinology and Metabolism, 2016.
7. Bonse et al. Bone age in puberty. Journal of Adolescent Health, 2013.
8. Tassone et al. Bone age in genetic disorders. Medical Genetics Part C: Seminars in Medical Genetics, 2013.
