# ViT-GA2026
Medical Image Classification Using Genetic Algorithm and Vision Transformer Features

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a hybrid machine learning framework designed for medical image classification. The pipeline leverages a pre-trained **Vision Transformer (ViT-Base)** to extract high-dimensional deep features, followed by a multi-core **Parallel Genetic Algorithm (GA)** to perform optimal feature subset selection. The selected feature subset is evaluated using a **Gaussian Naive Bayes** classifier.

---

## 📌 Key Features

- **ViT Feature Extraction**: Extracts a 768-dimensional feature vector per image scan using `vit_base_patch16_224`.
- **Parallelized GA Selection**: Accelerates feature subset search using multi-core parallel processing via `joblib.Parallel`.
- **Fitness Caching**: Employs a cache mechanism to prevent re-evaluating duplicate chromosomes across generations.
- **Flexible Evaluation Strategies**:
  1. Stratified $K$-Fold Cross-Validation ($K=5$).
  2. Fast evaluation via fixed Train/Validation split.

---
## 📁 Repository Structure

```text
├── features_MRIbrain.npz         # ViT feature extraction
├── ViT-GA2026.ipynb              # Jupyter Notebook
├── Results                        # Output plots
├── requirements.txt              # Required dependencies
└── README.md                     # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python $\ge 3.10$
- CUDA-enabled GPU (recommended for fast feature extraction)

### 2. Environment Setup
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/vit-ga2026.git](https://github.com/your-username/vit-ga2026.git)
cd vit-ga-mri-classification
pip install -r requirements.txt
```

### 3. Dataset Structure
Organize your Brain MRI dataset inside the `Kaggle/DataMRI` directory as follows:
This dataset consists of 7023 images divided into four classes. The four classes in the dataset are brain pituitary, notumor, meningioma, and glioma. The dataset is publicly available at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset.
```text
DataMRI/
├── Training/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── NoTumor/
│   └── Pituitary/
└── Testing/
    ├── Glioma/
    └── ...
```
---
##  Usage Guide

This project supports two operational workflows: extracting features directly from raw image files or loading pre-extracted feature caches for rapid genetic algorithm experimentation.

###  Case 1: End-to-End Extraction & GA Selection
Extract 768-dimensional deep feature representations from raw MRI images using pre-trained Vision Transformer (`vit_base_patch16_224`) and optimize feature subsets via parallel Genetic Algorithm.

1. **Run Cell 1 (Feature Extraction)**:
   - Traverses `Training` and `Testing` directories.
   - Converts images to tensors and extracts ViT feature embeddings.
   - Automatically saves features to `features_MRIbrain.npz` for caching.

2. **Run Cell 3: GA (Parallel) with K-Fold Cross-Validation**:
   - Selects the optimal subset of extracted features using parallel fitness evaluations with K-Fold Cross-Validation.
   - Fits a Gaussian Naive Bayes classifier on the selected training features.
3. **Run Cell 4: GA (Parallel) without K-Fold Cross-Validation**:
   - Selects the optimal subset of extracted features using parallel fitness evaluations without K-Fold Cross-Validation.
   - Fits a Gaussian Naive Bayes classifier on the selected training features.
---

### 📌 Case 2: Fast Experimentation via Feature Cache
Skip the time-consuming ViT extraction phase by re-loading pre-calculated feature tensors directly into memory.

1. **Run Cell 2 (Cache Loading)**:
   - Detects the existing `features_MRIbrain.npz` cache file.
   - Loads `features_train`, `features_test`, `y_train`, and `y_test` arrays in seconds.
2. **Run Cell 3: GA (Parallel) with K-Fold Cross-Validation**:
   - Selects the optimal subset of extracted features using parallel fitness evaluations with K-Fold Cross-Validation.
   - Fits a Gaussian Naive Bayes classifier on the selected training features.
3. **Run Cell 4: GA (Parallel) without K-Fold Cross-Validation**:
   - Selects the optimal subset of extracted features using parallel fitness evaluations without K-Fold Cross-Validation.
   - Fits a Gaussian Naive Bayes classifier on the selected training features.
---
### ⚙️ Evaluation Strategies: K-Fold CV vs. Train/Val Split

You can configure the evaluation strategy inside **Cell 2** by setting the `USE_KFOLD` parameter:

* **Single Split Mode (`USE_KFOLD = False`)**:
  - Splits training data into an internal 80/20 Train/Validation set once.
  - Recommended for fast GA optimization across high generation counts ($1000+$ generations).

* **Stratified K-Fold CV (`USE_KFOLD = True`)**:
  - Evaluates each individual's fitness using Stratified $K$-Fold Cross-Validation (default: $K=5$).
  - Recommended for robust validation preventing sample split bias.

```python
# Configure in Cell 2:
USE_KFOLD = False  # Set to True to enable 5-Fold Stratified CV
---

## 📊 Experimental Results

Experimental performance evaluation on the Brain MRI dataset[cite: 1]:


### 1. Performance Comparison

*Performance (mean ± std) and total time comparison of all methods in Application 1 using 5-fold stratified cross-validation.*

| Method | ACC | Precision | Recall | F1-score | Time (s) | Rank |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Proposed | 0.989 ± 0.005 | 0.989 ± 0.006 | 0.989 ± 0.006 | 0.989 ± 0.006 | 283 | 1 |
| FG - ANN | 0.933 ± 0.006 | 0.931 ± 0.007 | 0.930 ± 0.006 | 0.930 ± 0.006 | 130 | 2 |
| FG - SVM | 0.923 ± 0.003 | 0.922 ± 0.003 | 0.920 ± 0.003 | 0.921 ± 0.003 | 128 | 3 |
| FG - Random Forest | 0.916 ± 0.009 | 0.914 ± 0.009 | 0.912 ± 0.010 | 0.912 ± 0.010 | 146 | 4 |
| FG - Naive Bayes | 0.849 ± 0.010 | 0.844 ± 0.011 | 0.843 ± 0.011 | 0.843 ± 0.011 | 120 | 7 |
| FG - AdaBoost | 0.814 ± 0.007 | 0.809 ± 0.007 | 0.809 ± 0.007 | 0.808 ± 0.007 | 305 | 9 |
| FG - Decision Tree | 0.772 ± 0.008 | 0.767 ± 0.009 | 0.764 ± 0.009 | 0.765 ± 0.009 | 127 | 12 |
| AlexNet | 0.850 ± 0.022 | 0.850 ± 0.018 | 0.850 ± 0.018 | 0.850 ± 0.019 | 2133 | 5 |
| InceptionV3 | 0.850 ± 0.021 | 0.850 ± 0.023 | 0.850 ± 0.023 | 0.850 ± 0.021 | 2172 | 6 |
| Inception ResNet V2 | 0.815 ± 0.001 | 0.840 ± 0.003 | 0.815 ± 0.003 | 0.815 ± 0.002 | 1789 | 8 |
| ResNet50 | 0.801 ± 0.022 | 0.810 ± 0.020 | 0.810 ± 0.017 | 0.810 ± 0.021 | 1639 | 10 |
| VGG16 | 0.800 ± 0.012 | 0.800 ± 0.010 | 0.800 ± 0.013 | 0.800 ± 0.011 | 2720 | 11 |

### 2. Convergence Plots

Convergence curves and selected feature count trends are automatically generated and saved under `results/figures/`[cite: 1]:

| Convergence Curve | Selected Features per Generation |
| :---: | :---: |
| ![GA Convergence](results/figures/GA_convergence_curve_MRI.png) | ![Selected Features](results/figures/GA_selected_features_MRI.png) |

---

## 📜 Citation

If you find this code or research useful in your work, please cite it as follows:

```bibtex
@article{Pham2026ViTGA,
  author    = {Toan Dinh Pham},
  title     = {Medical Image Classification Using Genetic Algorithm and Vision Transformer Features},
  journal   = {Neural Computing & Applications (Revsied)},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
