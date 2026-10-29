# 🔬 Liver Cirrhosis Stage Prediction Using Deep Learning

A deep learning project focused on predicting histological stages (Stage 1 to Stage 4) of liver cirrhosis using demographic data and clinical blood serum biomarkers.

This repository features comparative benchmarking, architectural optimization, and clinical interpretability across foundational deep learning paradigms for tabular healthcare data.

---

## 🎯 Clinical Objective

Liver cirrhosis is a progressive condition characterized by the replacement of healthy hepatic tissue with non-functional fibrotic scar tissue, leading to hepatic failure and portal hypertension. Histological staging determines prognosis, therapeutic intervention pathways, and liver transplantation prioritization.

This project implements and systematically compares deep learning architectures to predict liver cirrhosis progression stages from routinely collected clinical biomarkers.

---

## 🧠 Evaluated Neural Network Architectures

1. **VanillaANN (Multi-Layer Perceptron)**: Standard feed-forward baseline with fully connected layers, batch normalization, GELU non-linearities, and dropout.
2. **DeepResidualANN (Residual Neural Network)**: Deep neural network with residual shortcut connections ($y = x + \mathcal{F}(x)$) allowing unimpeded gradient flow and stable deep feature representations.
3. **AttentionANN (Feature Self-Attention Network)**: Transformer-inspired architecture projecting clinical biomarkers into learned dense tokens and using Multi-Head Self-Attention (MHSA) to model high-order feature interactions.
4. **CNN1D (1D Convolutional Neural Network)**: Sequential representation of standardized biomarker vectors using multi-channel 1D convolutions with spatial pooling.

---

## 📁 Repository Structure

```
├── data/
│   └── liver_cirrhosis.csv                   # Clinical dataset
├── liver_cirrhosis.csv                       # Dataset copy (root level)
├── Liver_Cirrhosis_Stage_Prediction_Optimized.ipynb # Main evaluated & optimized notebook
├── HALI_Net_Liver_Cirrhosis_Complete.ipynb   # Comprehensive experimental pipeline notebook
├── build_and_execute_notebook.py             # Script to build & execute optimized notebook
├── generate_and_run_notebook.py              # Script to generate & run research notebook
├── update_cell27.py                          # Utility script for notebook updates
├── requirements.txt                          # Project Python dependencies
└── .gitignore                                # Git ignore rules
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Gowdham-B/Liver_Cirrhosis_Stage_Prediction_Using_DeepLearning.git
cd Liver_Cirrhosis_Stage_Prediction_Using_DeepLearning
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebooks

Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
```

Open `Liver_Cirrhosis_Stage_Prediction_Optimized.ipynb` to explore data preprocessing, model architectures, training loops, evaluation metrics (Accuracy, F1-Score, ROC-AUC), and visual analysis.

---

## 📊 Dataset & Features

The dataset comprises demographic and clinical parameters including:
- **Demographics**: Age, Sex
- **Symptoms & Complications**: Ascites, Hepatomegaly, Spiders, Edema
- **Laboratory Biomarkers**: Bilirubin, Cholesterol, Albumin, Copper, Alkaline Phosphatase, SGOT, Triglycerides, Platelets, Prothrombin time
- **Target**: Cirrhosis Histological Stage (Stages 1, 2, 3, 4)

---

## 📜 License

This project is licensed under the MIT License - see repository details for more information.
