# ============================================================
# 🚭 Tobacco Use & Mortality — Unified Smart Prediction Dashboard (v7.0)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.compose._column_transformer as ct
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Compatibility Patch for sklearn internals ---
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list): pass
    ct._RemainderColsList = _RemainderColsList

# ------------------- App Setup -------------------
st.set_page_config(page_title="🚭 Tobacco Mortality Predictor", layout="wide")
st.title("🚭 Tobacco Use & Mortality — Unified Smart Prediction Dashboard (v7.0)")
st.caption("Handles missing values, non-numeric columns, and column mismatches automatically.")

# ------------------- Upload Section -------------------
model_file = st.file_uploader("📦 Upload your trained model (.pkl)", type=["pkl"])
data_file = st.file_uploader("📊 Upload dataset (.csv)", type=["csv"])

if not model_file or not data_file:
    st.warning("⚠️ Please upload both the model and dataset files to continue.")
    st.stop()

# ------------------- Load Model -------------------
try:
    model = joblib.load(model_file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ------------------- Load Dataset -------------------
try:
    data = pd.read_csv(data_file)
    st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ------------------- Clean Data -------------------
def clean_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].astype(str).str.contains(r"^\d{4}/\d{2}$").any():
            def convert_year(x):
                if isinstance(x, str) and re.match(r"^\d{4}/\d{2}$", x):
                    y1, y2 = x.split('/')
                    return (int(y1) + int(y1[:2] + y2)) / 2
                try: return float(x)
                except: return np.nan
            df[col] = df[col].apply(convert_year)
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

data = clean_data(data)
st.info("✅ Data cleaned successfully!")

# ------------------- Year Handling -------------------
if 'year_numeric' not in data.columns:
    year_cols = [c for c in data.columns if 'year' in c.lower()]
    if year_cols:
        data['year_numeric'] = pd.to_numeric(data[year_cols[0]], errors='coerce')
        data['year_numeric'].fillna(data['year_numeric'].median(), inplace=True)
        st.success("✅ 'year_numeric' created from available year column.")
    else:
        data['year_numeric'] = 0
        st.warning("⚠️ No year column found; created dummy 'year_numeric' = 0.")

# ------------------- Column Alignment -------------------
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
else:
    expected_cols = list(data.columns)

missing_cols = [c for c in expected_cols if c not in data.columns]
extra_cols = [c for c in data.columns if c not in expected_cols]

if missing_cols:
    st.warning(f"⚠️ Missing columns added with default 0: {missing_cols}")
    for c in missing_cols: data[c] = 0
if extra_cols:
    st.info(f"ℹ️ Ignoring extra columns not used by model: {extra_cols}")
    data = data[expected_cols]

# ------------------- Dataset Overview -------------------
st.markdown("### 📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Columns", len(data.columns))
c3.metric("Missing Values", data.isnull().sum().sum())
st.dataframe(data.head())

# ------------------- Correlation Heatmap -------------------
st.markdown("### 🔗 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# ------------------- Outlier Detection -------------------
st.markdown("### 🚨 Outlier Detection (Z-Score Method)")
numeric_data = data.select_dtypes(include=np.number)
z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
outlier_counts = (z_scores > 3).sum().sum()
st.info(f"Found **{outlier_counts}** potential outliers in numeric columns.")

# ------------------- Prediction Section -------------------
st.markdown("### 🔮 Mortality Risk Prediction")
mode = st.radio("Select Input Method:", ["Select from Dataset", "Manual Entry"])

if mode == "Select from Dataset":
    idx = st.slider("Select Row Index", 0, len(data)-1, 0)
    input_data = data.iloc[[idx]]
    st.dataframe(input_data)
else:
    user_inputs = {col: st.number_input(col, value=float(data[col].mean())) for col in expected_cols}
    input_data = pd.DataFrame([user_inputs])

if st.button("🧩 Predict Now"):
    try:
        pred = model.predict(input_data)[0]
        risk_label = "🟢 Low Risk" if pred == 0 else "🔴 High Risk"
        st.success(f"✅ Predicted Mortality Risk: **{pred} → {risk_label}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            st.write(f"**Confidence:** {proba[pred]*100:.2f}% for class {pred}")
            st.progress(float(proba[pred]))

        st.metric("Predicted Category", risk_label)

        # ------------------- Interpretation -------------------
        if pred == 1:
            st.error("""
            ### 🔴 **High Risk Detected**
            Indicates a strong likelihood of tobacco-related mortality.

            **Possible Causes:**
            - High smoking prevalence or weak tobacco control
            - Older population or poor cessation support
            - Higher hospital admissions & chronic illness patterns

            **🧩 Recommendations:**
            - Strengthen tobacco cessation programs and awareness
            - Increase early screening and local health policies
            - Encourage data-driven policy interventions
            """)
        else:
            st.success("""
            ### 🟢 **Low Risk Detected**
            Indicates stable health and effective tobacco control measures.

            **Positive Indicators:**
            - Low smoking rate and strong policy implementation
            - Effective public awareness and preventive healthcare
            - Better access to medical and screening services

            **✅ Recommendations:**
            - Continue public health awareness efforts
            - Sustain cessation and taxation policies
            - Monitor risk data for early trend detection
            """)

        # ------------------- Feature Distribution -------------------
        st.markdown("### 🧩 Feature Distribution Explorer")
        feature = st.selectbox("Select Feature:", data.columns)
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(fig)

        # ------------------- Feature Impact vs Prediction -------------------
        st.markdown("### 📈 Feature Impact Explorer")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=expected_cols)
            top_feats = importances.sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax)
            ax.set_title("Top 10 Features Influencing Predictions")
            st.pyplot(fig)
        elif hasattr(model, "coef_"):
            coef = pd.Series(model.coef_[0], index=expected_cols)
            top_feats = coef.abs().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=top_feats.values, y=top_feats.index, ax=ax)
            ax.set_title("Top 10 Influential Features (Coefficients)")
            st.pyplot(fig)
        else:
            st.info("Model does not provide feature importances.")

        # ------------------- Model Evaluation -------------------
        st.markdown("### ⚙️ Model Performance (if evaluation data available)")
        if 'target' in data.columns:
            y_true = data['target']
            y_pred = model.predict(data.drop(columns=['target'], errors='ignore'))
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            st.write(f"**Accuracy:** {acc*100:.2f}%")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.caption("Target column not found; skipping performance evaluation.")

        # ------------------- Conclusion -------------------
        st.markdown("### 🧾 Final Conclusion & Next Steps")
        st.write("""
        **✅ Dashboard Summary**
        - Successfully analyzed dataset and predicted mortality risk.
        - Generated insights into key features, trends, and correlations.
        - Provided interpretation and policy recommendations.

        **📌 Next Steps**
        1. Use this model for real-time health surveillance and mortality forecasting.
        2. Integrate more recent health & behavior data (2016–2025) for higher accuracy.
        3. Collaborate with public health agencies to guide tobacco policy decisions.
        4. Deploy predictive dashboards in hospitals, government offices, or research setups.
        """)

        st.success("🎯 Project Complete — Predict, Analyze, and Act for a Tobacco-Free Future!")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

st.markdown("---")
st.caption("© 2025 Tobacco Mortality Predictor | Built with ❤️ using Streamlit + scikit-learn")
