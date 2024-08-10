# Project Abstract
This objective of this project is to create a detection system using a Convolutional Neural Network (CNN) along with Histogram of Oriented Gradients (HOG) feature extraction. The model uses Python's in-built libraries along with OpenCV, TensorFlow, Keras.

The system is built using datasets containing both positive (pedestrian) and negative (non-pedestrian) samples. 
The data is then augmented by flipping and rotating them. Images are preprocessed by resizing and normalizing pixel values, followed by extracting HOG features to capture essential patterns.
A CNN model is trained on this dataset, with hyperparameter optimization to enhance performance.
The model is then evaluated using peformance metrics such as precision, recall and F1 score.
