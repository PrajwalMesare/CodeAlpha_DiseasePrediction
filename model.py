"""
disease_prediction — model.py
CodeAlpha Internship Task 4
Author : Prajwal Mesare
GitHub : github.com/PrajwalMesare

Standalone inference module for heart disease prediction.

Usage
-----
    from model import DiseasePredictionModel
    m = DiseasePredictionModel()
    result = m.predict({
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 155,
        "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2
    })
    print(result)
"""

import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_DESCRIPTIONS = {
    "age"      : "Age in years",
    "sex"      : "Sex (1=male, 0=female)",
    "cp"       : "Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)",
    "trestbps" : "Resting blood pressure (mmHg)",
    "chol"     : "Serum cholesterol (mg/dl)",
    "fbs"      : "Fasting blood sugar > 120 mg/dl (1=True, 0=False)",
    "restecg"  : "Resting ECG (0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy)",
    "thalach"  : "Maximum heart rate achieved",
    "exang"    : "Exercise induced angina (1=Yes, 0=No)",
    "oldpeak"  : "ST depression induced by exercise",
    "slope"    : "Slope of peak exercise ST (0=upsloping, 1=flat, 2=downsloping)",
    "ca"       : "Number of major vessels colored by fluoroscopy (0-4)",
    "thal"     : "Thalassemia (0=normal, 1=fixed defect, 2=reversable defect, 3=other)",
}


class DiseasePredictionModel:
    """Wrapper for the trained heart disease prediction model."""

    def __init__(self):
        self.model         = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
        self.scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    def predict(self, input_data: dict) -> dict:
        """
        Predict heart disease risk for one patient.

        Returns
        -------
        dict with keys:
            disease_probability : float
            prediction          : int (0 or 1)
            risk_label          : str
            risk_level          : str ('Low' / 'Medium' / 'High')
        """
        df     = pd.DataFrame([input_data])[self.feature_names]
        df_sc  = self.scaler.transform(df)
        proba  = self.model.predict_proba(df_sc)[0][1]
        pred   = int(proba > 0.5)

        if proba < 0.30:
            risk_level = "Low"
            risk_label = "✅ LOW RISK — Heart Disease Unlikely"
        elif proba < 0.60:
            risk_level = "Medium"
            risk_label = "⚠️  MEDIUM RISK — Further Tests Advised"
        else:
            risk_level = "High"
            risk_label = "❌ HIGH RISK — Heart Disease Likely"

        return {
            "disease_probability": round(float(proba), 4),
            "prediction"         : pred,
            "risk_label"         : risk_label,
            "risk_level"         : risk_level,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sc  = self.scaler.transform(df[self.feature_names])
        probas = self.model.predict_proba(df_sc)[:, 1]
        df = df.copy()
        df["disease_probability"] = probas.round(4)
        df["prediction"]          = (probas > 0.5).astype(int)
        df["risk_level"]          = pd.cut(
            probas, bins=[0, 0.30, 0.60, 1.0],
            labels=["Low", "Medium", "High"]
        )
        return df


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = DiseasePredictionModel()

    patients = [
        {   # Healthy
            "age":55,"sex":0,"cp":0,"trestbps":120,"chol":215,
            "fbs":0,"restecg":0,"thalach":170,"exang":0,
            "oldpeak":0.5,"slope":0,"ca":0,"thal":1
        },
        {   # High risk
            "age":63,"sex":1,"cp":3,"trestbps":160,"chol":340,
            "fbs":1,"restecg":2,"thalach":95,"exang":1,
            "oldpeak":4.2,"slope":2,"ca":3,"thal":3
        },
    ]

    for i, p in enumerate(patients, 1):
        r = m.predict(p)
        print(f"\nPatient {i}")
        print(f"  Disease Probability : {r['disease_probability']*100:.2f}%")
        print(f"  Risk Level          : {r['risk_level']}")
        print(f"  Clinical Decision   : {r['risk_label']}")
