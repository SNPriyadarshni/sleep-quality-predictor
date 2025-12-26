import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1) Paths / constants
# -----------------------------
DATA_FILE = "sleep.csv"
MODEL_FILE = "sleep_model.pkl"
RANDOM_STATE = 42

# -----------------------------
# 2) Map numeric sleep quality to labels
# -----------------------------
def map_sleep_quality(value):
    if value >= 8:
        return "Good"
    elif value >= 6:
        return "Average"
    else:
        return "Poor"

# -----------------------------
# 3) Preprocessing
# -----------------------------
def preprocess_df(df, scaler=None, fit_scaler=False):
    feature_cols = [
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Age"
    ]
    X_proc = df[feature_cols].astype(float)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_proc)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_proc)
        return X_scaled

# -----------------------------
# 4) Train / save model
# -----------------------------
def train_and_save_model(df, model_path=MODEL_FILE):
    # Convert numeric scores to labels
    df["Sleep_Label"] = df["Quality of Sleep"].apply(map_sleep_quality)

    le = LabelEncoder()
    y = le.fit_transform(df["Sleep_Label"])
    X_scaled, scaler = preprocess_df(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=le.classes_)

    joblib.dump({"model": clf, "scaler": scaler, "le": le}, model_path)
    return {"accuracy": acc, "report": rep, "model": clf, "scaler": scaler, "le": le}

def load_model_or_train(model_path=MODEL_FILE, data_path=DATA_FILE):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    df = pd.read_csv(data_path)
    info = train_and_save_model(df, model_path=model_path)
    return {"model": info["model"], "scaler": info["scaler"], "le": info["le"]}

# -----------------------------
# 5) Prediction + tips
# -----------------------------
def predict_from_inputs(inputs_dict, artefact):
    tmp = pd.DataFrame([inputs_dict])
    X_scaled = preprocess_df(tmp, scaler=artefact["scaler"], fit_scaler=False)
    pred_idx = artefact["model"].predict(X_scaled)[0]
    probs = artefact["model"].predict_proba(X_scaled)[0]
    pred_label = artefact["le"].inverse_transform([pred_idx])[0]
    prob_map = {artefact["le"].inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_map

def generate_personalized_tips(inputs):
    tips = []
    if inputs["Physical Activity Level"] < 30:
        tips.append("Add 20–30 minutes of exercise earlier in the day.")
    if inputs["Stress Level"] >= 7:
        tips.append("Try relaxation or breathing exercises before bed.")
    if inputs["Sleep Duration"] < 7:
        tips.append("Aim for 7–9 hours of sleep per night.")
    if inputs["Daily Steps"] < 5000:
        tips.append("Increase daily walking or movement for better sleep.")
    if not tips:
        tips.append("Your habits look healthy! Maintain consistency.")
    return tips

# -----------------------------
# 6) Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")
st.title("😴 Sleep Quality Predictor")
st.write("Enter your daily habits to predict sleep quality and receive personalized tips.")

# Sidebar: retrain model
if st.sidebar.button("(Re)train model using your CSV"):
    with st.spinner("Training model..."):
        df_real = pd.read_csv(DATA_FILE)
        info = train_and_save_model(df_real)
        st.success(f"Model retrained successfully (Accuracy: {info['accuracy']:.2%})")
        st.text("Classification report:")
        st.text(info["report"])

# Load model
artefact = load_model_or_train()

st.header("Daily Input Data")
col1, col2 = st.columns(2)
with col1:
    sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 300, 30)
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
with col2:
    heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)
    daily_steps = st.slider("Daily Steps", 0, 20000, 5000)
    age = st.slider("Age", 0, 100, 25)

if st.button("🔮 Predict Sleep Quality"):
    inputs = {
        "Sleep Duration": float(sleep_duration),
        "Physical Activity Level": float(physical_activity),
        "Stress Level": float(stress_level),
        "Heart Rate": float(heart_rate),
        "Daily Steps": float(daily_steps),
        "Age": float(age)
    }
    pred_label, prob_map = predict_from_inputs(inputs, artefact)

    st.markdown("### Prediction Result")
    if pred_label == "Good":
        st.success(f"🌙 Sleep Quality: **{pred_label}** ({prob_map[pred_label]*100:.1f}% confidence)")
    elif pred_label == "Average":
        st.info(f"😐 Sleep Quality: **{pred_label}** ({prob_map[pred_label]*100:.1f}% confidence)")
    else:
        st.error(f"⚠️ Sleep Quality: **{pred_label}** ({prob_map[pred_label]*100:.1f}% confidence)")

    st.write("**Prediction probabilities:**")
    st.write(pd.Series(prob_map).sort_values(ascending=False))

    st.markdown("### Personalized Tips")
    for tip in generate_personalized_tips(inputs):
        st.write("- " + tip)

    st.markdown("### Feature Importance")
    feat_names = [
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Age"
    ]
    importances = artefact["model"].feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance (RandomForest)")
    st.pyplot(fig)

st.markdown("---")
st.caption("💡 Tip: Replace `sleep.csv` with your own data and retrain for personalized accuracy.")
