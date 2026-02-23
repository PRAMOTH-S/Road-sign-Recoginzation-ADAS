# 🚦 Traffic Sign Recognition System (ADAS-Enhanced)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project implements a **Traffic Sign Recognition (TSR) system** using Deep Learning.

It is designed as a **core module of Advanced Driver Assistance Systems (ADAS)**, enabling vehicles to interpret road signs such as speed limits, warnings, and directional indicators.

---

## 🚗 ADAS Integration (Important)

This system contributes to ADAS by enabling:

- 🛑 **Speed Limit Detection** → Helps enforce safe driving speeds  
- ⚠️ **Warning Sign Recognition** → Alerts for curves, pedestrians, hazards  
- 🔄 **Navigation Assistance** → Detects turn signs  
- 🔔 **Driver Alerts** → Real-time visual warnings  
- 🤖 **Autonomous Decision Support** → Input for self-driving logic  

> ⚠️ Note: This project currently performs **classification only**. Full ADAS requires real-time detection, tracking, and sensor fusion.

---

## 🎯 Objectives

- Detect and classify traffic signs
- Support ADAS-based automation
- Build a scalable AI model for real-world systems

---

## 🧠 Dataset

- **GTSRB (German Traffic Sign Recognition Benchmark)**
- 43 traffic sign classes
- Real-world variations (lighting, blur, angles)

---

## ⚙️ Tech Stack

- Python 3.9  
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
- Loss: Categorical Crossentropy

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
