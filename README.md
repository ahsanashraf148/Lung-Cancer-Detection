# 🧠 Deep Learning for Perception: Lung Cancer Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

## 📌 Project Overview

Lung cancer is among the most lethal cancers globally. This project aims to automate and enhance the **classification of lung cancer subtypes** from histopathological images using **deep learning**. We explored and compared two approaches:

- ✅ **Part 1**: Custom-built **Convolutional Neural Network (CNN)**
- ✅ **Part 2**: **Pretrained Swin Transformer** using transfer learning

---

## 🎯 Objective

Develop a highly accurate, deep learning-based diagnostic tool for early detection of lung cancer subtypes:

- **Adenocarcinoma**
- **Squamous Cell Carcinoma**
- **Benign Tissue**

> 🔍 Early detection = better treatment outcomes.

---

## 🧩 Problem Statement

Traditional histopathological diagnosis of lung cancer:

- ❌ Time-consuming
- ❌ Relies on expert pathologists
- ❌ Subject to human error

**Our solution** leverages deep learning to automate classification, reduce diagnostic time, and improve accuracy.

---

## 🛠️ Methodology

### 📍 Part 1: Custom CNN (TensorFlow/Keras)

Features:

- Residual connections
- Batch normalization
- Dropout regularization
- Multi-stage convolutional pipeline
- Fully connected classifier

**Training Details:**

- Optimizer: `Adamax`
- Loss: `Categorical Crossentropy`
- Augmentations: `ImageDataGenerator`
- Split: Train / Validation / Test

---

### 📍 Part 2: Swin Transformer (PyTorch)

Model: `swin_tiny_patch4_window7_224` from [timm](https://github.com/rwightman/pytorch-image-models)

Key Techniques:

- Fine-tuning on lung cancer dataset
- Pretrained on ImageNet
- Custom classification head
- Training tracked with `tqdm`
- Optimizer: `AdamW`
- Loss: `CrossEntropyLoss`

---

## 📊 Results

### ✅ Custom CNN

| Metric              | Value      |
| ------------------- | ---------- |
| Train Accuracy      | 99.67%     |
| Validation Accuracy | 98.27%     |
| Test Accuracy       | 98.93%     |
| ROC-AUC Score       | **0.9997** |

### ✅ Swin Transformer

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | 98.42%     |
| ROC-AUC Score  | **1.0000** |

---

## 📁 Dataset

- **Kaggle Dataset**: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---

## 📚 References

- 🔗 **Swin Transformer Paper**: [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11325325/)
- 🔗 **timm Library**: [GitHub](https://github.com/rwightman/pytorch-image-models)
- 📘 **Keras and TensorFlow** Documentation
- 📘 **PyTorch** Documentation
- 📘 **Scikit-learn**: Classification reports, metrics, and ROC-AUC

---

## 📌 How to Run

<details>
<summary>📦 Requirements</summary>

- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x
- scikit-learn
- timm
- matplotlib
- tqdm

</details>

```bash
# Clone the repository
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run CNN Model
python train_cnn.py

# Run Swin Transformer
python train_swin.py
```

---

## 📈 Future Work

- Expand to multi-organ cancer detection
- Deploy as a web-based diagnostic tool
- Optimize for edge devices (e.g., Raspberry Pi, Jetson Nano)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
