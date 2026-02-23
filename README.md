# 🚦 Traffic Sign Recognition System (AI-Based)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Overview

This project implements a **Traffic Sign Recognition (TSR) system** using Deep Learning techniques.  
The model is trained to classify road signs such as speed limits, warnings, and directional signs.

Traffic Sign Recognition is a critical component of **Advanced Driver Assistance Systems (ADAS)** used in modern autonomous and semi-autonomous vehicles.

---

## 🎯 Objectives

- Detect and classify traffic signs from images
- Improve road safety using AI-based automation
- Build a scalable deep learning model for real-world applications

---

## 🧠 Dataset

- **Dataset Used:** GTSRB (German Traffic Sign Recognition Benchmark)
- Contains **43 classes** of traffic signs
- Includes thousands of labeled images under different lighting and conditions

---

## ⚙️ Tech Stack

- **Programming Language:** Python 3.9
- **Libraries:**
  - TensorFlow / Keras
  - OpenCV
  - NumPy
  - Matplotlib
  - Scikit-learn

---

## 🏗️ Model Architecture

- Convolutional Neural Network (CNN)
- Layers include:
  - Conv2D
  - MaxPooling
  - Flatten
  - Dense layers
- Activation: ReLU & Softmax
- Loss Function: Categorical Crossentropy

---

## 📊 Training Details

```python
history = model.fit(
    X_train, 
    y_train, 
    batch_size=32, 
    epochs=5, 
    validation_data=(X_test, y_test)
)
