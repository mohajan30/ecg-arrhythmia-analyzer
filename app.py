import streamlit as st
import wfdb
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import io
import os

# Streamlit app title and description
st.title("ECG Arrhythmia Analyzer")
st.write("Upload MIT-BIH .dat, .atr, and optionally .hea files to analyze ECG arrhythmias, compute burden, assess disease risks, and view ML model explanations.")

# Parameters
WINDOW = 256
RANDOM = 42
AAMI10 = ["N", "L", "R", "A", "V", "F", "/", "E", "a", "Q"]
sym2cls = {
    'N': 'N', 'L': 'L', 'R': 'R', 'A': 'A', 'V': 'V', 'F': 'F', '/': '/', 'E': 'E', 'a': 'a', 'Q': 'Q',
    '?': 'Q', 'x': 'Q', 'S': 'Q', '~': 'Q', 'J': 'A', 'j': 'A'
}

# Complication and disease mappings
complication = {
    "N": "Low % normal → structural disease work-up",
    "L": "Persistent LBBB → dyssynchrony; assess for CRT",
    "R": "Persistent RBBB often benign; NEW RBBB + chest pain = MI rule-out",
    "A": "Frequent PACs raise AF & stroke risk",
    "a": "Same AF risk as PAC",
    "V": "PVC burden may cause cardiomyopathy if >10 %",
    "F": "Fusion beats mark high PVC load",
    "/": ">90 % paced = device dependency",
    "E": "Escape beats point to high-grade AV block",
    "Q": "Noise / unclassifiable – repeat test"
}
disease_mappings = {
    "Coronary Artery Disease (Stable CAD, Silent Ischemia)": {
        "markers": ["V", "R"], "yellow": "PVC >10/h or pV>1%; RBBB>50%", "red": "pV>10% or persistent RBBB + pain",
        "comps": ">30 PVC/h: 2.2× mortality (Bhatt 2010); new RBBB triples MI mortality (Mehta 2007)"
    },
    "Acute Myocardial Infarction (Heart Attack)": {
        "markers": ["L", "R", "V"], "yellow": "Any new LBBB/RBBB", "red": "LBBB + ST↑ or ≥6 PVC/min",
        "comps": "New LBBB: 11% mortality (Sgarbossa); ≥6 PVC/min: 4.5× VT/VF (Bigger 1984)"
    },
    "Heart Failure (HFrEF & Pacing-Induced)": {
        "markers": ["L", "/", "V"], "yellow": "LBBB>50% or pacing>40%", "red": "pV>10% ≥3 months",
        "comps": "LBBB doubles mortality if EF≤30%; PVC-CMP: EF drop 11%, 82% recover post-ablation (Latchamsetty 2014)"
    },
    "Arrhythmia (Future Atrial Fibrillation)": {
        "markers": ["A", "a"], "yellow": ">100 PAC/24h or pA>0.5%", "red": "pA>5% (≈3000/day)",
        "comps": "≥100 PAC/day: 2× AF HR; >5%: 10× AF, 2.5× stroke (ARIC 2020)"
    },
    "Stroke (Cardio-Embolic & Cryptogenic)": {
        "markers": ["A", "a", "N"], "yellow": "pA>1% + CHA2DS2-VASc≥2", "red": "pA>5% or pN<80%",
        "comps": "PAC>1%: 2× stroke; >5%: 4× cryptogenic stroke (Perez 2019); low pN: ↑ silent infarcts (Ibrahim 2021)"
    }
}

# Helper Functions
def bandpass(sig, fs=360, lo=0.5, hi=40, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, sig)

def load_record_counts(record_path):
    try:
        ann = wfdb.rdann(record_path, 'atr')
        labels = [sym2cls.get(s, 'Q') for s in ann.symbol]
        cnt = Counter(labels)
        return {k: cnt.get(k, 0) for k in AAMI10}, len(labels)
    except Exception as e:
        st.error(f"Error reading annotations: {e}")
        return {k: 0 for k in AAMI10}, 0

def flag_color(cls, p, c, h):
    if cls == "N":
        if p >= 90: return "Safe"
        elif p >= 80: return "Elevated"
        else: return "Dangerous"
    if cls in ("A", "a", "S", "j", "J"):
        if p > 5 or c * 48 > 3000: return "Dangerous"
        elif p > 0.5 or c * 48 > 100: return "Elevated"
        else: return "Safe"
    if cls == "V":
        if p > 10: return "Dangerous"
        elif p > 1 or h > 10: return "Elevated"
        else: return "Safe"
    if cls == "L":
        if p > 50: return "Dangerous"
        else: return "Safe"
    if cls == "R":
        if p > 50: return "Elevated"
        else: return "Safe"
    if cls == "/":
        if p > 90: return "Elevated"
        else: return "Safe"
    if cls == "E":
        if c >= 30: return "Elevated"
        else: return "Safe"
    if cls == "F":
        if c > 10: return "Elevated"
        else: return "Safe"
    if cls == "Q":
        if p > 20: return "Dangerous"
        elif p > 10: return "Elevated"
        else: return "Safe"
    return "Safe"

def assign_disease_zone(disease, counts, total_beats, duration_hours):
    data = disease_mappings[disease]
    p = {k: 100 * counts.get(k, 0) / total_beats for k in AAMI10}
    c = counts
    h = {k: c.get(k, 0) / duration_hours for k in AAMI10}
    markers_met = []
    if disease == "Coronary Artery Disease (Stable CAD, Silent Ischemia)":
        if h.get("V", 0) > 10 or p.get("V", 0) > 1: markers_met.append("yellow")
        if p.get("V", 0) > 10 or (p.get("R", 0) > 50): markers_met.append("red")
    elif disease == "Acute Myocardial Infarction (Heart Attack)":
        if p.get("L", 0) > 0 or p.get("R", 0) > 0: markers_met.append("yellow")
        if p.get("L", 0) > 0 or (h.get("V", 0) * 60 > 6): markers_met.append("red")
    elif disease == "Heart Failure (HFrEF & Pacing-Induced)":
        if p.get("L", 0) > 50 or p.get("/", 0) > 40: markers_met.append("yellow")
        if p.get("V", 0) > 10: markers_met.append("red")
    elif disease == "Arrhythmia (Future Atrial Fibrillation)":
        pac_c = c.get("A", 0) + c.get("a", 0)
        pac_24h = pac_c * 48
        pac_p = p.get("A", 0) + p.get("a", 0)
        if pac_24h > 100 or pac_p > 0.5: markers_met.append("yellow")
        if pac_p > 5 or pac_24h > 3000: markers_met.append("red")
    elif disease == "Stroke (Cardio-Embolic & Cryptogenic)":
        pac_p = p.get("A", 0) + p.get("a", 0)
        if pac_p > 1: markers_met.append("yellow")
        if pac_p > 5 or p.get("N", 0) < 80: markers_met.append("red")
    if "red" in markers_met: return "Dangerous"
    elif "yellow" in markers_met: return "Elevated"
    return "Safe"

def load_record(record_path, win=WINDOW):
    try:
        # Check if .hea file exists; create a minimal one if not
        hea_path = f"{record_path}.hea"
        if not os.path.exists(hea_path):
            with open(hea_path, 'w') as f:
                f.write(f"{record_path} 1 360 {win}\n")  # Minimal header: 1 channel, 360 Hz
        rec = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")
        ecg = bandpass(rec.p_signal[:, 0], fs=360)  # Use first channel
        beats, labels = [], []
        for pos, sym in zip(ann.sample, ann.symbol):
            cls = sym2cls.get(sym, 'Q')
            start, end = max(0, pos - win//2), min(len(ecg), pos + win//2)
            seg = ecg[start:end]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)), mode='constant')
            beats.append(seg)
            labels.append(cls)
        # Clean up generated .hea file
        if os.path.exists(hea_path):
            os.remove(hea_path)
        return np.asarray(beats), np.asarray(labels)
    except Exception as e:
        st.error(f"Error processing ECG: {e}")
        return np.array([]), np.array([])

# File upload
st.subheader("Upload ECG Files")
dat_file = st.file_uploader("Upload .dat file (ECG signal)", type=["dat"])
atr_file = st.file_uploader("Upload .atr file (Annotations)", type=["atr"])
hea_file = st.file_uploader("Upload .hea file (Header, optional)", type=["hea"])

if dat_file and atr_file:
    # Validate file names
    record_name = dat_file.name.split(".")[0]
    if dat_file.name.split(".")[0] != atr_file.name.split(".")[0]:
        st.error("Uploaded .dat and .atr files must have the same record name (e.g., 114.dat and 114.atr).")
    else:
        # Save uploaded files
        with open(f"{record_name}.dat", "wb") as f:
            f.write(dat_file.read())
        with open(f"{record_name}.atr", "wb") as f:
            f.write(atr_file.read())
        if hea_file:
            with open(f"{record_name}.hea", "wb") as f:
                f.write(hea_file.read())

        # Step 1: Burden Analysis
        st.subheader("Burden Analysis")
        counts, total_beats = load_record_counts(record_name)
        if total_beats == 0:
            st.error("No valid beats found in the uploaded record.")
        else:
            duration_hours = 0.5  # MIT-BIH records ~30 minutes
            rows = []
            for k in AAMI10:
                c = counts[k]
                p = 100 * c / total_beats if total_beats > 0 else 0
                h = c / duration_hours
                zone = flag_color(k, p, c, h)
                rows.append([k, c, f"{p:.2f} %", complication[k], zone])
            df_burden = pd.DataFrame(rows, columns=["Class", "# Beats", "%", "Possible Complication", "Zone"])
            
            # Style for display
            def hilite_zone(row):
                color_map = {"Safe": "#d4edda", "Elevated": "#fff3cd", "Dangerous": "#f8d7da"}
                return [f'background-color: {color_map.get(row.Zone, "white")}'] * len(row)
            st.dataframe(df_burden.style.apply(hilite_zone, axis=1))
            st.download_button("Download Burden Table", df_burden.to_csv(index=False), f"burden_{record_name}.csv")

        # Step 2: Disease Risk Assessment
        st.subheader("Disease Risk Assessment")
        disease_rows = []
        for disease in disease_mappings:
            zone = assign_disease_zone(disease, counts, total_beats, duration_hours)
            markers = disease_mappings[disease]["yellow"] if zone == "Elevated" else disease_mappings[disease]["red"] if zone == "Dangerous" else "None"
            comps = disease_mappings[disease]["comps"] if zone != "Safe" else "None triggered"
            disease_rows.append([disease, markers, zone, comps])
        df_diseases = pd.DataFrame(disease_rows, columns=["Disease", "Markers & Thresholds Met", "Zone", "Complications"])
        st.dataframe(df_diseases.style.apply(hilite_zone, axis=1))
        st.download_button("Download Disease Table", df_diseases.to_csv(index=False), f"diseases_{record_name}.csv")

        # Step 3: ML Classification (XGBoost)
        st.subheader("Machine Learning Classification (XGBoost)")
        try:
            X_raw, y_raw = load_record(record_name)
            if len(X_raw) == 0:
                st.error("Failed to load ECG data for ML analysis.")
            else:
                X_ds = resample(X_raw, 32, axis=1)
                scaler = StandardScaler()
                X_ds = scaler.fit_transform(X_ds)
                X_tr, X_te, y_tr, y_te = train_test_split(X_ds, y_raw, test_size=0.3, stratify=y_raw, random_state=RANDOM)
                
                # SMOTE Balancing
                cnt = Counter(y_tr)
                m = min(cnt.values())
                if m < 2:
                    X_tr_bal, y_tr_bal = X_tr, y_tr
                else:
                    k = max(1, min(5, m-1))
                    X_tr_bal, y_tr_bal = SMOTE(k_neighbors=k, random_state=RANDOM).fit_resample(X_tr, y_tr)
                
                # Train XGBoost
                present_classes = np.unique(y_tr_bal)
                le = LabelEncoder().fit(present_classes)
                y_tr_bal_int = le.transform(y_tr_bal)
                y_te_int = le.transform([cls if cls in present_classes else 'Q' for cls in y_te])
                xgb = XGBClassifier(tree_method="hist", max_depth=6, learning_rate=0.1, n_estimators=400,
                                    subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                    eval_metric="mlogloss", num_class=len(present_classes), random_state=RANDOM)
                xgb.fit(X_tr_bal, y_tr_bal_int)
                pred_xgb_int = xgb.predict(X_te)
                pred_xgb = le.inverse_transform(pred_xgb_int)
                
                # Display accuracy and classification report
                st.write(f"XGBoost Accuracy: {accuracy_score(y_te, pred_xgb):.4f}")
                report = classification_report(y_te, pred_xgb, labels=present_classes, target_names=present_classes, zero_division=0, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # Confusion Matrix
                st.subheader("Confusion Matrix (XGBoost)")
                cm = confusion_matrix(y_te, pred_xgb, labels=['A', 'V', 'N'])
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm, display_labels=['PAC', 'PVC', 'Normal']).plot(cmap='Blues', ax=ax)
                st.pyplot(fig)

                # SHAP Explanation
                st.subheader("SHAP Feature Importance (XGBoost)")
                explainer_xgb = shap.TreeExplainer(xgb)
                shap_values_xgb = explainer_xgb.shap_values(X_te[:50])  # Limit for speed
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values_xgb, X_te[:50], feature_names=[f"Sample_{i}" for i in range(32)], class_names=present_classes, plot_type="bar")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"ML processing failed: {e}")

        # Clean up temporary files
        try:
            os.remove(f"{record_name}.dat")
            os.remove(f"{record_name}.atr")
            if os.path.exists(f"{record_name}.hea"):
                os.remove(f"{record_name}.hea")
        except:
            pass

else:
    st.write("Please upload both .dat and .atr files to proceed.")
```
