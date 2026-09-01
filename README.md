# ViT-GA2026
Medical Image Classification Using Genetic Algorithm and Vision Transformer Features

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a hybrid machine learning framework designed for medical image classification[cite: 1]. The pipeline leverages a pre-trained **Vision Transformer (ViT-Base)** to extract high-dimensional deep features[cite: 1], followed by a multi-core **Parallel Genetic Algorithm (GA)** to perform optimal feature subset selection[cite: 1]. The selected feature subset is evaluated using a **Gaussian Naive Bayes** classifier[cite: 1].

---

## 📌 Key Features

- **ViT Feature Extraction**: Extracts a 768-dimensional feature vector per image scan using `vit_base_patch16_224`[cite: 1].
- **Parallelized GA Selection**: Accelerates feature subset search using multi-core parallel processing via `joblib.Parallel`[cite: 1].
- **Fitness Caching**: Employs a cache mechanism to prevent re-evaluating duplicate chromosomes across generations[cite: 1].
- **Flexible Evaluation Strategies**:
  1. Stratified $K$-Fold Cross-Validation ($K=5$)[cite: 1].
  2. Fast evaluation via fixed Train/Validation split[cite: 1].

---

## 📁 Repository Structure

```text
├── data/                         # Input dataset directory
├── src/
│   ├── feature_extraction.py     # ViT feature extraction module
│   └── ga_selection.py           # Parallel Genetic Algorithm engine
├── results/                      # Output plots and saved features
├── main.py                       # Main execution entry point
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
git clone [https://github.com/your-username/vit-ga-mri-classification.git](https://github.com/your-username/vit-ga-mri-classification.git)
cd vit-ga-mri-classification
pip install -r requirements.txt
```

### 3. Dataset Structure
Organize your Brain MRI dataset inside the `data/` directory as follows:

```text
data/
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

## 🚀 Usage

Execute the complete end-to-end pipeline (Feature Extraction $\rightarrow$ Parallel GA $\rightarrow$ Final Evaluation):

```bash
python main.py
```

*Extracted features are automatically saved to `ViT_features_MRIbrain.npz` for fast re-execution[cite: 1].*

---

## 📊 Experimental Results

Experimental performance evaluation on the Brain MRI dataset[cite: 1]:

### 1. Performance Comparison

| Method | Initial Features | Selected Features | Validation Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Full ViT + Naive Bayes | 768 | 768 | - | ~82.5% |
| **Proposed ViT + GA + Naive Bayes** | **768** | **~380–420** | **94.2%** | **92.8%** |

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
