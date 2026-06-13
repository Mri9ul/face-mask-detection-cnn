Here is the comprehensive, production-ready `README.md` for your **Face Mask Detection** project. This version includes detailed sections on the fixed architecture (`softmax`), preprocessing workflows, and step-by-step local configuration instructions designed to look highly professional to recruiters and developers alike.

---

```markdown
# 😷 Real-Time Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)

An end-to-end deep learning and computer vision pipeline that detects face masks in images. This project features a custom-built Convolutional Neural Network (CNN) trained with TensorFlow/Keras and deployed as a lightweight, interactive web interface via Streamlit.

---

## 📌 Project Overview

This repository bridges the gap between model training in a research environment (Jupyter/Colab) and practical production deployment. The system handles raw image processing pipelines, accounts for color-space discrepancies native to computer vision libraries, and delivers rapid binary classifications with clear confidence intervals.

### Key Enhancements Implemented
* **Color Space Synchronization:** Fully corrects the classic OpenCV `BGR` to Matplotlib/PIL `RGB` color inversion bug to safeguard inference accuracy.
* **Categorical Probability Distribution:** Replaced baseline independent sigmoid nodes with a multi-class `softmax` output layer paired with `sparse_categorical_crossentropy` loss for stable, bounded predictions.
* **Streamlit Production Layer:** Removed environment-specific dependencies (such as Google Colab's patches) in favor of a universal web-native inference engine.

---


---

## 🚀 Local Installation & Deployment

Follow these instructions to configure and execute the application locally, specifically optimized for handling native virtual environments.

### 1. Environment Initialization

Clone the repository and isolate dependencies within a clean virtual environment:

```bash
git clone [https://github.com/mridulkarar/face-mask-detection-cnn.git](https://github.com/mridulkarar/face-mask-detection-cnn.git)
cd face-mask-detection-cnn

# Create and engage the environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 2. Dependency Resolution

Install the exact required versions of the computation and presentation libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Application Execution

Boot up the local Streamlit engine:

```bash
streamlit run app.py

```

The server will initialize and establish a local loopback interface at: `http://localhost:8501`

---

## 📊 Pipeline & Architecture Specifications

### Model Architecture

The underlying model uses a progressive downsampling design built sequentially using the Keras API:

1. **Feature Extraction:** 2D Convolutional layers running 3x3 kernels coupled with `ReLU` activations.
2. **Spatial Reduction:** 2D Max Pooling layers running standard 2x2 structural windows.
3. **Regularization:** Strategic `Dropout` layers acting as a penalization matrix to avoid training overfitting.
4. **Classification Head:** Dense flattening leading into a 2-unit `softmax` output layer mapping perfectly to class configurations:
* **`Index 0`**: Without Mask
* **`Index 1`**: Wearing a Mask



### Preprocessing Execution Sequence

Before passing any array data into the model weights, inputs follow an identical matrix transformation path:


$$\text{Input Image (BGR)} \longrightarrow \text{RGB Conversion} \longrightarrow \text{Resize (128x128x3)} \longrightarrow \text{Normalization } (\div 255.0) \longrightarrow \text{Batch Expansion (1, 128, 128, 3)}$$

---

## 🛠️ Technology Stack

* **Core Engine:** Python 3.9+
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Computer Vision Suite:** OpenCV (`opencv-python-headless`)
* **Mathematical Operations:** NumPy
* **Application Framework:** Streamlit

```

```
