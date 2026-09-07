import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Validasi File Model
model_path = os.path.join("models", "lapse_model.joblib")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di {model_path}. Jalankan train_and_export.py terlebih dahulu.")

print("[1/3] Memuat model yang sudah dilatih...")
model = joblib.load(model_path)

# 2. Rekonstruksi Data Uji (Wajib menggunakan parameter yang sama persis dengan saat training)
print("[2/3] Memuat dan menyiapkan dataset...")
df = pd.read_csv(os.path.join("data", "Kaggle.csv"), sep=";").dropna(axis=1, how="all")

def parse_numeric(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", "").replace(" ", "")
    return np.nan if s in ["-", "", "nan", "None"] else float(s)

df["BENEFIT_CLEAN"] = df["BENEFIT"].apply(parse_numeric)
df["PREMIUM_CLEAN"] = df["Premium"].apply(parse_numeric)
df["IS_LAPSE"] = (df["POLICY STATUS"].str.strip() == "Lapse").astype(int)

cat_cols = ["SEX", "PAYMENT MODE", "NON LAPSE GUARANTEED", "Full Benefit?", 
            "CHANNEL1", "CHANNEL2", "CHANNEL3", "POLICY TYPE 1", "POLICY TYPE 2", "POLICY TYPE 3"]
num_cols = ["ENTRY AGE", "SUBSTANDARD RISK", "NUMBER OF ADVANCE PREMIUM", 
            "INITIAL BENEFIT", "Policy Year (Decimal)", "Policy Year", "BENEFIT_CLEAN", "PREMIUM_CLEAN"]

for col in cat_cols:
    df[col] = df[col].astype(str)

X = df[cat_cols + num_cols]
y = df["IS_LAPSE"]

# WAJIB: random_state=42 dan stratify=y agar Test Set identik dengan saat training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Eksekusi Pengujian
print("[3/3] Menjalankan Pengujian Metrik...\n")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Inforce (0)", "Lapse (1)"]))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(f"True Negatives (Prediksi Aman, Aktual Aman)   : {cm[0][0]}")
print(f"False Positives(Prediksi Lapse, Aktual Aman)  : {cm[0][1]} (Alarm Palsu)")
print(f"False Negatives(Prediksi Aman, Aktual Lapse)  : {cm[1][0]} (Kebocoran Risiko)")
print(f"True Positives (Prediksi Lapse, Aktual Lapse) : {cm[1][1]}")
