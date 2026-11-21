import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Prediksi Klasifikasi ML - Bab 14", layout="wide")
st.title("Aplikasi Prediksi Machine Learning (Bab 14)")
st.write(
    "Masukkan nilai fitur di bawah ini lalu klik **Prediksi** untuk melihat"
    " hasil model."
)

bundle = joblib.load("model_terbaik.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]

raw_data = load_breast_cancer()
defaults = raw_data.data.mean(axis=0)
target_names = raw_data.target_names

inputs = []
cols = st.columns(3)
for idx, feat in enumerate(feature_names):
  with cols[idx % 3]:
    val = st.number_input(
        label=feat,
        value=float(round(defaults[idx], 4)),
        format="%.4f",
        key=f"feat_{idx}",
    )
    inputs.append(val)

if st.button("Prediksi", type="primary"):
  X_input = np.array([inputs])
  pred = model.predict(X_input)[0]
  label = target_names[pred] if pred < len(target_names) else str(pred)
  st.success(f"Hasil Prediksi: **{label.upper()}** (Kelas: {pred})")

  if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_input)[0]
    st.caption(
        f"Probabilitas -> {target_names[0]}: {proba[0]:.2%} | "
        f"{target_names[1]}: {proba[1]:.2%}"
    )