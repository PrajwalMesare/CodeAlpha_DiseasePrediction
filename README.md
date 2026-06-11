# 🩺 Disease Prediction from Medical Data
**Disease Prediction from Medical Data**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

**Personal ML Project**

**Developer:** Prajwal Mesare | TGPCET Nagpur | B.Tech CSE (Data Science) 2027

[![GitHub](https://img.shields.io/badge/GitHub-PrajwalMesare-181717?logo=github)](https://github.com/PrajwalMesare)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/prajwal-mesare-700678263)

</div>

---

## 📌 Objective

Predict the **presence of heart disease** from clinical patient data using the **UCI Heart Disease** dataset from Kaggle.

> ⚠️ *This project is for educational purposes only and is not a substitute for professional medical diagnosis.*

---

## 📊 Dataset

| Field | Detail |
|-------|--------|
| **Source** | [Kaggle — Heart Disease UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) |
| **Rows** | 1,025 patients |
| **Features** | 13 clinical features |
| **Target** | `target` — 1 = Heart Disease, 0 = No Disease |
| **Disease Rate** | ~40.2% |

---

## 🗂️ Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── disease_prediction_model.ipynb   ← Full ML pipeline notebook
├── model.py                         ← Standalone inference module
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── heart.csv                    ← Kaggle UCI Heart Disease dataset
│
├── models/
│   ├── disease_model.pkl            ← Best trained model (Gradient Boosting)
│   ├── scaler.pkl                   ← StandardScaler
│   └── feature_names.pkl            ← Feature column names
│
└── outputs/
    ├── eda_plots.png                ← EDA visualizations
    └── model_evaluation.png         ← ROC, confusion matrix, feature importance
```

---

## 🩺 Features

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia type (0–3) |

---

## 🔄 ML Pipeline

```
Load CSV (1,025 patients)
    ↓
EDA & Visualization
    ↓
Train / Test Split (80/20, stratified)
    ↓
StandardScaler
    ↓
Train 5 Models
  → Logistic Regression
  → Random Forest
  → Gradient Boosting  ✅ Best
  → SVM (RBF kernel)
  → XGBoost
    ↓
Evaluate (ROC-AUC, F1, Accuracy, Precision, Recall)
    ↓
Export Best Model
```

---

## 📈 Results

| Model | ROC-AUC | F1-Score | Accuracy |
|-------|:-------:|:--------:|:--------:|
| Logistic Regression | 0.7764 | 0.6144 | 0.6829 |
| Random Forest | 0.8304 | 0.6753 | 0.7512 |
| **Gradient Boosting ✅** | **0.8366** | **0.6624** | **0.7561** |
| SVM | 0.7604 | 0.5974 | 0.6780 |
| XGBoost | 0.8279 | 0.6538 | 0.7415 |

> **Best Model: Gradient Boosting** — ROC-AUC = **0.8366**

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook disease_prediction_model.ipynb
```

### 3. Use the Inference Module
```python
from model import DiseasePredictionModel

m = DiseasePredictionModel()
result = m.predict({
    "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 155,
    "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
})

print(result)
# {'disease_probability': 0.41, 'risk_level': 'Medium',
#  'risk_label': '⚠️  MEDIUM RISK — Further Tests Advised'}
```

---

## 🛠️ Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `joblib` · `Jupyter`

---

## 📜 About

Built as a personal ML project for portfolio and learning purposes.  
