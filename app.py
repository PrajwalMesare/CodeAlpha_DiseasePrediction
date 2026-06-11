"""
app.py — Task 4: Disease Prediction from Medical Data
Personal ML Project
Author: Prajwal Mesare | github.com/PrajwalMesare
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #200122, #6f0000, #a31515);
    padding: 2.2rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    color: white; text-align: center;
}
.header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.header p  { color: #ffb3b3; font-size: 1rem; margin-top: 0.4rem; }

.card {
    background: #f8f9fc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
}
.metric-box {
    background: linear-gradient(135deg, #a31515, #6f0000);
    border-radius: 12px; padding: 1.2rem; color: white; text-align: center;
}
.metric-box .val { font-size: 1.9rem; font-weight: 700; }
.metric-box .lbl { font-size: 0.82rem; opacity: 0.85; }

.risk-low    { background: linear-gradient(135deg,#11998e,#38ef7d);
               padding:1.3rem; border-radius:12px; color:white;
               text-align:center; font-size:1.2rem; font-weight:700; }
.risk-medium { background: linear-gradient(135deg,#f7971e,#ffd200);
               padding:1.3rem; border-radius:12px; color:#1a1a2e;
               text-align:center; font-size:1.2rem; font-weight:700; }
.risk-high   { background: linear-gradient(135deg,#eb3349,#f45c43);
               padding:1.3rem; border-radius:12px; color:white;
               text-align:center; font-size:1.2rem; font-weight:700; }

.warning-box { background:#fff5f5; border-left:4px solid #e53e3e;
               padding:0.8rem 1rem; border-radius:0 8px 8px 0;
               font-size:0.88rem; color:#742a2a; margin-top:1rem; }
.info { background:#fff5f5; border-left:4px solid #c53030;
        padding:0.8rem 1rem; border-radius:0 8px 8px 0;
        font-size:0.9rem; color:#742a2a; margin-bottom:1rem; }

.stButton>button {
    background: linear-gradient(135deg,#a31515,#eb3349);
    color:white; border:none; border-radius:8px;
    padding:0.65rem 2rem; font-weight:600; font-size:1rem;
    width:100%; cursor:pointer;
}
.stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("models/disease_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

def predict(inputs, model, scaler, feature_names):
    df    = pd.DataFrame([inputs])[feature_names]
    df_sc = scaler.transform(df)
    proba = model.predict_proba(df_sc)[0][1]
    if proba < 0.30:
        level, label = "low",    "✅ LOW RISK — Heart Disease Unlikely"
    elif proba < 0.60:
        level, label = "medium", "⚠️ MEDIUM RISK — Further Tests Advised"
    else:
        level, label = "high",   "❌ HIGH RISK — Heart Disease Likely"
    return proba, level, label

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Disease Prediction")
    st.markdown("**Personal ML Project**")
    st.markdown("**Developer:** Prajwal Mesare")
    st.markdown("**Model:** Gradient Boosting")
    st.markdown("**ROC-AUC:** 0.8366")
    st.markdown("---")
    st.markdown("**Dataset:** Kaggle — UCI Heart Disease")
    st.markdown("**Patients:** 1,025")
    st.markdown("**Features:** 13 clinical")
    st.markdown("---")
    page = st.radio("Navigate", ["🩺 Predict", "📊 Model Performance", "📈 EDA Plots"])
    st.markdown("---")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-PrajwalMesare-181717?logo=github)](https://github.com/PrajwalMesare)")
    st.markdown('<div class="warning-box">⚠️ For educational purposes only. Not a substitute for medical advice.</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>Disease Prediction from Medical Data</h1>
    <p>Kaggle — UCI Heart Disease Dataset | Clinical Risk Assessment | Personal ML Project</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-box"><div class="val">1,025</div><div class="lbl">Patients</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box"><div class="val">0.837</div><div class="lbl">Best ROC-AUC</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box"><div class="val">5</div><div class="lbl">Models Trained</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box"><div class="val">GBoost</div><div class="lbl">Best Model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Predict":
    st.markdown("### Enter Patient Clinical Data")
    st.markdown('<div class="info">Fill in the patient clinical values based on the Kaggle UCI Heart Disease dataset features.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Patient Info**")
        age      = st.slider("Age (years)", 20, 80, 55)
        sex      = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp       = st.selectbox("Chest Pain Type",
                                options=[0,1,2,3],
                                format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                                        2:"Non-Anginal Pain",3:"Asymptomatic"}[x])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)

    with col2:
        st.markdown("**🫀 Cardiac Measurements**")
        thalach  = st.slider("Max Heart Rate Achieved", 60, 210, 150)
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope    = st.selectbox("ST Slope",
                                options=[0,1,2],
                                format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        restecg  = st.selectbox("Resting ECG",
                                options=[0,1,2],
                                format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])

    with col3:
        st.markdown("**🔬 Lab Results**")
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
        exang    = st.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
        ca       = st.slider("Major Vessels (CA, 0–4)", 0, 4, 0)
        thal     = st.selectbox("Thalassemia",
                                options=[0,1,2,3],
                                format_func=lambda x: {0:"Normal",1:"Fixed Defect",
                                                        2:"Reversable Defect",3:"Other"}[x])

    st.markdown("---")
    if st.button("🩺 Predict Heart Disease Risk"):
        try:
            model, scaler, feature_names = load_model()
            inp = {
                "age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
                "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
                "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal
            }
            proba, level, label = predict(inp, model, scaler, feature_names)
            prob_pct = proba * 100

            st.markdown("### 🩺 Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="risk-{level}">{label}</div>', unsafe_allow_html=True)
            with r2:
                st.metric("Disease Probability", f"{prob_pct:.1f}%")
            with r3:
                st.metric("Risk Level", level.upper())

            # Gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 2))
            color = "#11998e" if level == "low" else ("#f7971e" if level == "medium" else "#eb3349")
            ax.barh([""], [prob_pct],        color=color,    height=0.5)
            ax.barh([""], [100 - prob_pct],  left=[prob_pct], color="#ffe0e0", height=0.5)
            ax.set_xlim(0, 100)
            ax.axvline(30, color="green",  linestyle="--", linewidth=1, alpha=0.6)
            ax.axvline(60, color="orange", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlabel("Disease Probability (%)")
            ax.set_title(f"Heart Disease Probability: {prob_pct:.1f}%  |  Risk: {level.upper()}", fontsize=13, fontweight="bold")
            ax.spines[['top','right','left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Key risk factors
            st.markdown("#### Key Risk Indicators")
            risks = {
                "Chest Pain Type": ("High" if cp >= 2 else "Low"),
                "Max Heart Rate":  ("Low"  if thalach > 140 else "High"),
                "ST Depression":   ("High" if oldpeak > 2 else "Low"),
                "Major Vessels":   ("High" if ca >= 2 else "Low"),
                "Exercise Angina": ("High" if exang == 1 else "Low"),
            }
            ri_df = pd.DataFrame(list(risks.items()), columns=["Factor","Risk"])
            ri_df["Status"] = ri_df["Risk"].map({"High":"⚠️ Concerning","Low":"✅ Normal"})
            st.dataframe(ri_df[["Factor","Status"]], use_container_width=True, hide_index=True)

            st.markdown('<div class="warning-box">⚠️ This prediction is for educational purposes only. Please consult a qualified cardiologist for medical decisions.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Model not found. Run the notebook first. Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Comparison — All 5 Algorithms")

    perf = pd.DataFrame({
        "Model":     ["Logistic Regression","Random Forest","Gradient Boosting ✅","SVM","XGBoost"],
        "ROC-AUC":   [0.7764, 0.8304, 0.8366, 0.7604, 0.8279],
        "F1-Score":  [0.6144, 0.6753, 0.6624, 0.5974, 0.6538],
        "Accuracy":  [0.6829, 0.7512, 0.7561, 0.6780, 0.7415],
        "Precision": [0.6200, 0.7100, 0.7000, 0.6300, 0.6800],
        "Recall":    [0.6090, 0.6420, 0.6270, 0.5660, 0.6300],
    })
    st.dataframe(
        perf.style.highlight_max(subset=["ROC-AUC","Accuracy","F1-Score"], color="#d4edda")
                  .format({c:"{:.4f}" for c in ["ROC-AUC","F1-Score","Accuracy","Precision","Recall"]}),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(perf["Model"])); w = 0.25
    ax.bar(x-w,   perf["ROC-AUC"],  w, label="ROC-AUC",  color="#a31515", edgecolor="white")
    ax.bar(x,     perf["Accuracy"], w, label="Accuracy",  color="#e74c3c", edgecolor="white")
    ax.bar(x+w,   perf["F1-Score"], w, label="F1-Score",  color="#f1948a", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(perf["Model"], rotation=15, ha="right")
    ax.set_ylim(0.4, 1.0); ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="card">
    <b>Best Model: Gradient Boosting</b> — ROC-AUC: 0.8366 | Accuracy: 75.61%<br><br>
    All models trained with <b>StandardScaler</b> on 13 clinical features.<br>
    Train/Test split: 80/20 stratified.<br>
    Dataset: <b>UCI Heart Disease</b> from Kaggle — 1,025 patients, 40.2% disease rate.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Plots":
    st.markdown("### 📈 Exploratory Data Analysis")

    for path, caption in [
        ("outputs/eda_plots.png",        "Heart Disease — EDA (Target Distribution, Age, Max Heart Rate, Chest Pain, Correlations, Cholesterol)"),
        ("outputs/model_evaluation.png", "Model Evaluation — ROC Curves, Confusion Matrix, Comparison, Feature Importance"),
    ]:
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"{path} not found. Run the notebook first.")
