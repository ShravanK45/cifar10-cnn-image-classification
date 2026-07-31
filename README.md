# 🚀 CIFAR-10 Image Classification using Convolutional Neural Networks (CNN) | TensorFlow & Keras

## 📌 Project Overview

This project demonstrates the implementation of a Deep Convolutional Neural Network (CNN) for multi-class image classification using the **CIFAR-10** dataset. The objective was to build a robust CNN architecture while gaining practical experience with modern deep learning techniques such as **Batch Normalization, Data Augmentation, Dropout, Early Stopping, Learning Rate Scheduling, and Model Checkpointing**.

This project marks my transition from Classical Machine Learning to Deep Learning and serves as the foundation for my upcoming Computer Vision projects.

---

## 📂 Dataset

The CIFAR-10 dataset consists of **60,000 RGB images** of size **32×32 pixels**, divided into **10 different classes**.

### Classes

* ✈️ Airplane

* 🚗 Automobile

* 🐦 Bird

* 🐱 Cat

* 🦌 Deer

* 🐶 Dog

* 🐸 Frog

* 🐴 Horse

* 🚢 Ship

* 🚚 Truck

* **Training Images:** 50,000

* **Testing Images:** 10,000

---

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 🧠 Deep Learning Concepts Implemented

* Convolutional Neural Networks (CNN)
* Batch Normalization
* Max Pooling
* Dropout Regularization
* Data Augmentation
* Adam Optimizer
* EarlyStopping Callback
* ReduceLROnPlateau Callback
* ModelCheckpoint Callback

---

## 🏗️ Model Architecture

Input (32×32×3)

⬇️

Conv2D (32 Filters)

⬇️

Batch Normalization

⬇️

Conv2D (32 Filters)

⬇️

Batch Normalization

⬇️

MaxPooling2D

⬇️

Dropout (0.3)

⬇️

Conv2D (64 Filters)

⬇️

Batch Normalization

⬇️

Conv2D (64 Filters)

⬇️

Batch Normalization

⬇️

MaxPooling2D

⬇️

Dropout (0.3)

⬇️

Flatten

⬇️

Dense (512)

⬇️

Batch Normalization

⬇️

Dropout (0.5)

⬇️

Output Layer (Softmax - 10 Classes)

---

## 🚀 Training Techniques

To improve model performance and reduce overfitting, the following techniques were implemented:

* Image Data Augmentation
* Batch Normalization
* Dropout Regularization
* Early Stopping
* Learning Rate Reduction on Plateau
* Best Model Checkpoint Saving

---

## 📈 Results

> **Test Accuracy:** **(Add Your Final Accuracy Here)**

The model demonstrates effective learning while maintaining good generalization through regularization and data augmentation techniques.

---

## 📊 Training Curves

Include the following images in the repository:

* Training vs Validation Accuracy
* Training vs Validation Loss

---

## 📚 Key Learnings

Through this project, I gained hands-on experience with:

* Building CNN architectures from scratch
* Understanding convolution and feature extraction
* Preventing overfitting using Dropout and Batch Normalization
* Applying Image Data Augmentation
* Using TensorFlow callbacks for efficient model training
* Monitoring model performance using training and validation metrics

---

## 🔮 Future Improvements

* Confusion Matrix
* Classification Report
* Grad-CAM Visualization
* Transfer Learning using ResNet50
* MobileNetV2
* EfficientNet
* Streamlit Web Application for Image Classification

---

## 🎯 What's Next?

This project is the starting point of my Deep Learning journey.

My upcoming Computer Vision roadmap includes:

* 🩺 Pneumonia Detection using TensorFlow
* 🌍 Intel Image Classification using PyTorch
* 🌿 Plant Disease Detection using Transfer Learning
* 🎯 Object Detection using YOLO
* 🦺 Industrial Safety Monitoring System

---

## 👨‍💻 Author

**Shravan Kundap**

Electronics & Telecommunication Engineering Undergraduate

Aspiring AI/ML Engineer passionate about Machine Learning, Deep Learning, Computer Vision, NLP, Generative AI, and Agentic AI.

If you found this project helpful, feel free to ⭐ the repository.

