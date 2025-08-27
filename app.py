import streamlit as st
import wfdb
import numpy as np
import pandas as pd
from collections import Counter
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
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

# Complications and disease mappings
complication = { ... }  # same as your original mapping
disease_mappings = { ... }  # same as your original mapping

# Helper functions
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
    # same as your original function
    ...

def assign_disease_zone(disease, counts, total_beats, duration_hours):
    # same as your original function
    ...

def load_record(record_path, win=WINDOW):
    try:
        hea_path = f"{record_path}.hea"
        if not os.path.exists(hea_path):
            with open(hea_path, 'w') as f:
                f.write(f"{record_path} 1 360 {win}\n")
        rec = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, "atr")
        ecg = bandpass(rec.p_signal[:, 0], fs=360)
        beats, labels = [], []
        for pos, sym in zip(ann.sample, ann.symbol):
            cls = sym2cls.get(sym, 'Q')
            start, end = max(0, pos - win//2), min(len(ecg), pos + win//2)
            seg = ecg[start:end]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)), mode='constant')
            beats.append(seg)
            labels.append(cls)
        if os.path.exists(hea_path):
            os.remove(hea_path)
        return np.asarray(beats), np.asarray(labels)
    except Exception as e:
        st.error(f"Error processing ECG: {e}")
        return np.array([]), np.array([])

# File upload
st.subheader("Upload ECG Files")
dat_file = st.file_uploader("Upload .dat file", type=["dat"])
atr_file = st.file_uploader("Upload .atr file", type=["atr"])
hea_file = st.file_uploader("Upload .hea file (optional)", type=["hea"])

if dat_file and atr_file:
    record_name = dat_file.name.split(".")[0]
    if record_name != atr_file.name.split(".")[0]:
        st.error("Uploaded .dat and .atr files must have the same record name.")
    else:
        with open(f"{record_name}.dat", "wb") as f:
            f.write(dat_file.read())
        with open(f"{record_name}.atr", "wb") as f:
            f.write(atr_file.read())
        if hea_file:
            with open(f"{record_name}.hea", "wb") as f:
                f.write(hea_file.read())

        # Burden analysis
        counts, total_beats = load_record_counts(record_name)
        if total_beats > 0:
            duration_hours = 0.5
            rows = []
            for k in AAMI10:
                c = counts[k]
                p = 100 * c / total_beats
                h = c / duration_hours
                zone = flag_color(k, p, c, h)
                rows.append([k, c, f"{p:.2f} %", complication[k], zone])
            df_burden = pd.DataFrame(rows, columns=["Class", "# Beats", "%", "Possible Complication", "Zone"])
            def hilite_zone(row):
                color_map = {"Safe": "#d4edda", "Elevated": "#fff3cd", "Dangerous": "#f8d7da"}
                return [f'background-color: {color_map.get(row.Zone, "white")}'] * len(row)
            st.dataframe(df_burden.style.apply(hilite_zone, axis=1))
        else:
            st.error("No valid beats found in the uploaded record.")

        # ML and Disease assessment as in your original code
        ...
        
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
