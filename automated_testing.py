import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

# 1. Pemuatan Data dan Rekonstruksi Pipeline
print("[INFO] Memuat dataset Kaggle.csv...")
df = pd.read_csv(os.path.join("data", "Kaggle.csv"), sep=";").dropna(axis=1, how="all")

def parse_num(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", "").replace(" ", "")
    return np.nan if s in ["-", "", "nan", "None"] else float(s)

df["BENEFIT_CLEAN"] = df["BENEFIT"].apply(parse_num)
df["PREMIUM_CLEAN"] = df["Premium"].apply(parse_num)
df["IS_LAPSE"] = (df["POLICY STATUS"].str.strip() == "Lapse").astype(int)

cat_cols = ["SEX", "PAYMENT MODE", "NON LAPSE GUARANTEED", "Full Benefit?", 
            "CHANNEL1", "CHANNEL2", "CHANNEL3", "POLICY TYPE 1", "POLICY TYPE 2", "POLICY TYPE 3"]
num_cols = ["ENTRY AGE", "SUBSTANDARD RISK", "NUMBER OF ADVANCE PREMIUM", 
            "INITIAL BENEFIT", "Policy Year (Decimal)", "Policy Year", "BENEFIT_CLEAN", "PREMIUM_CLEAN"]

for col in cat_cols: df[col] = df[col].astype(str)

X = df[cat_cols + num_cols]
y = df["IS_LAPSE"]

# Menggunakan satu split standar untuk ekstraksi probabilitas
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[INFO] Memuat model lapse_model.joblib...")
model_path = os.path.join("models", "lapse_model.joblib")
if not os.path.exists(model_path):
    raise FileNotFoundError("Model tidak ditemukan. Pastikan path sudah benar.")
pipeline = joblib.load(model_path)

# 2. Ekstraksi Probabilitas Aktual
y_prob = pipeline.predict_proba(X_test)[:, 1]

# 3. Otomatisasi Pencarian Keseimbangan Optimal (Threshold Tuning)
print("[INFO] Menjalankan kalkulasi matematis Precision-Recall Curve...\n")
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Hindari pembagian dengan nol saat kalkulasi F1
f1_scores = np.divide(
    2 * (precision * recall),
    (precision + recall),
    out=np.zeros_like(precision),
    where=(precision + recall) != 0
)

# Temukan indeks dengan F1-Score tertinggi
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
max_f1 = f1_scores[optimal_idx]
optimal_precision = precision[optimal_idx]
optimal_recall = recall[optimal_idx]

# Evaluasi pada Threshold Default (0.50) vs Optimal
y_pred_default = (y_prob >= 0.50).astype(int)
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

cm_default = confusion_matrix(y_test, y_pred_default)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

# 4. Laporan Hasil Audit Sistem
print("====================================================")
print(" HASIL AUDIT KESEIMBANGAN OTOMATIS (F1 MAXIMIZATION)")
print("====================================================")
print(f"Batas Probabilitas (Threshold) Optimal : {optimal_threshold:.4f}")
print(f"Skor F1 Maksimal                       : {max_f1:.4f}")
print(f"Precision pada Titik Optimal           : {optimal_precision:.4f}")
print(f"Recall pada Titik Optimal              : {optimal_recall:.4f}\n")

print("--- PERBANDINGAN CONFUSION MATRIX ---")
print("1. Kinerja Bawaan (Threshold 0.50):")
print(f"   False Positives (Alarm Palsu) : {cm_default[0][1]}")
print(f"   False Negatives (Kebocoran)   : {cm_default[1][0]}\n")

print(f"2. Kinerja Optimal (Threshold {optimal_threshold:.4f}):")
print(f"   False Positives (Alarm Palsu) : {cm_optimal[0][1]}")
print(f"   False Negatives (Kebocoran)   : {cm_optimal[1][0]}")
print("====================================================")
