# Handwritten Digit Classification using Classical Machine Learning (MNIST)

## Project Overview

This project implements an end-to-end machine learning pipeline to classify grayscale images of handwritten digits (0–9) using the MNIST dataset.
The objective is to explore and compare classical machine learning techniques for image classification, focusing on preprocessing, model implementation, hyperparameter tuning, evaluation, and error analysis.

The project emphasizes understanding the strengths and limitations of classical ML models when applied to image data represented as flattened pixel vectors.

The following models are implemented and evaluated:

- K-Nearest Neighbors (KNN) implemented from scratch
- Support Vector Machine (SVM) with and without PCA
- Decision Tree Classifier

---

## Dataset

The MNIST dataset is provided in CSV format, where:

- Each row corresponds to a handwritten digit image
- The first 784 columns represent pixel intensities of a 28×28 grayscale image
- The final column represents the digit label (0–9)

⚠️ **Note:**
The dataset is not included in this repository due to GitHub’s file size limitations.

Please download the MNIST CSV file separately and place it in the following directory:

The dataset can be obtained from sources such as Kaggle or OpenML.

---

## Setup and Run Instructions

### 1. Clone the repository

```bash
git clone https://github.com/utkarshere/MNIST_classification.git
cd MNIST_classification
```

### 2. (Optional) Create and activate a virtual environment

```python
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install required dependencies

```
pip install numpy pandas matplotlib seaborn scikit-learn

```

### 4. Run notebook

MLpipeline.ipynb

### Key Features

#### Data Preprocessing

- Pixel values normalized to the range [0, 1]
- Dataset split into training and testing sets (80/20)
- Dataset statistics explored, including class distribution and sample visualization
- PCA applied optionally to reduce dimensionality for SVM

#### Models Implemented

##### 1. KNN

- Custom implementation using Euclidean distance.
- Hyperparameter `k` tuned using accuracy vs k analysis
- Subsampling applied during tuning and evaluation to manage computational cost.

##### 2. SVM

- RBF kernel used for non-linear classification
- Hyperparameters `C` and `gamma` tuned using GridSearchCV
- PCA applied to reduce dimensionality and improve computational efficiency

##### 3. Decision Tree

- Hyperparameters `max_depth` and `min_samples_split` tuned using cross-validation
- Demonstrates interpretability but lower performance due to overfitting tendencies in high-dimensional data


#### Key Observations

* SVM with PCA achieved the best performance, highlighting the benefit of dimensionality reduction
* KNN showed strong accuracy but was computationally expensive
* Most misclassifications occurred between visually similar digits (e.g., 4 vs 9, 3 vs 8)

### Tools and Libraries Used

- Numpy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
