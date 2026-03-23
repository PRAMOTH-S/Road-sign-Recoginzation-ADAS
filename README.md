# 🚦 Traffic Sign Recognition System (ADAS-Enhanced)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overvie

This project implements a **Traffic Sign Recognition (TSR) system** using Deep Learning.

It serves as a **core module of Advanced Driver Assistance Systems (ADAS)**, enabling vehicles to interpret road signs such as speed limits, warnings, and navigation indicators.

---

## 🚗 ADAS Integration

This system supports ADAS functionalities such as:

- 🛑 **Speed Limit Detection** – Assists in maintaining safe speeds  
- ⚠️ **Warning Sign Recognition** – Detects hazards and alerts drivers  
- 🔄 **Navigation Assistance** – Identifies directional signs  
- 🔔 **Driver Alerts** – Displays real-time predictions  
- 🤖 **Decision Support** – Provides input for autonomous systems  

> ⚠️ Note: This project currently performs **classification only**. Full ADAS requires detection, tracking, and sensor fusion.

---

## 🎯 Objectives

- Classify traffic signs accurately  
- Support intelligent driving systems  
- Build a scalable deep learning pipeline  

---

## 🧠 Dataset

- **GTSRB (German Traffic Sign Recognition Benchmark)**  
- 43 traffic sign classes  
- Real-world variations (lighting, angles, occlusions)

---

## ⚙️ Tech Stack

- **Python 3.9**
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## 🏗️ Model Architecture

- CNN (Convolutional Neural Network)
- Conv2D → MaxPooling → Flatten → Dense
- Activation: ReLU, Softmax
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
