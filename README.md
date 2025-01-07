# What is Pneumonia ?

Pneumonia is a form of acute respiratory infection that is most commonly caused by viruses or bacteria. It can cause mild to life-threatening illness in people of all ages, however it is the single largest infectious cause of death in children worldwide.

Pneumonia killed more than 808 000 children under the age of 5 in 2017, accounting for 15% of all deaths of children under 5 years.  People at-risk for pneumonia also include adults over the age of 65 and people with preexisting health problems.

The lungs are made up of small sacs called alveoli, which fill with air when a healthy person breathes. When an individual has pneumonia, the alveoli are filled with pus and fluid, which makes breathing painful and limits oxygen intake. These infections are generally spread by direct contact with infected people.

# Pneumonia Detection in Chest X-Ray Images Using CNNs


## Introduction :

Pneumonia is a severe respiratory infection that affects people worldwide. Early and accurate diagnosis is crucial for effective treatment. Traditional diagnosis involves analyzing chest X-ray images, which can be subjective and may lead to misdiagnosis. This project aims to develop a machine learning model, specifically a Convolutional Neural Network (CNN), to automate and improve the accuracy of pneumonia detection in chest X-ray images.

## Objectives :

a- Develop a CNN model capable of detecting pneumonia from chest X-ray images.

b- Evaluate the model's performance using various metrics such as accuracy, precision, recall, and F1-score.

c- Compare the model's performance with existing methods in the literature.

d- Provide a user-friendly interface for predicting pneumonia in new X-ray images.

## Methodology :

### a - Data Collection :

The dataset consists of chest X-ray images labeled as normal or pneumonia. The images were collected from [Dataset Source]. The dataset is divided into training, validation, and test sets.

### b - Data Preprocessing :

a - Images were resized to 150x150 pixels.

b - Pixel values were normalized to the range [0, 1].

c - Data augmentation techniques were applied to increase the diversity of the training data.

## Model Development :

A CNN model was developed using TensorFlow and Keras. The model architecture includes several convolutional and pooling layers followed by fully connected layers. The model was trained on the training set and validated on the validation set.

## Results and Discussion :

The model achieved an accuracy of X% on the test set. The confusion matrix showed that the model had a higher tendency to correctly classify normal images than pneumonia images. The ROC curve indicated a good balance between sensitivity and specificity.

Note : Challenges faced during the project included class imbalance and variability in X-ray image quality. Future work could involve addressing class imbalance through techniques like oversampling or using a more balanced dataset.

## Conclusion :

The developed CNN model demonstrates potential for automating pneumonia detection in chest X-ray images. With further refinement, this model could be integrated into healthcare systems to assist radiologists and improve patient outcomes.


## References :

Dataset Source. [Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)]

CNN Architecture Reference. [[Link](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)]

TensorFlow Documentation. [Link](https://www.tensorflow.org/tutorials?authuser=2)]

OpenCV Documentation. [Link](https://opencv.org/)]
