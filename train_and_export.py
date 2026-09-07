import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Validasi Keberadaan Data
data_path = os.path.join("data", "Kaggle.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File data tidak ditemukan di path: {data_path}")

print("[1/5] Membaca dataset...")
df = pd.read_csv(data_path, sep=";").dropna(axis=1, how="all")

# 2. Pembersihan Format Angka & Labeling Target
print("[2/5] Membersihkan anomali tipe data...")


def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", "").replace(" ", "")
    return np.nan if s in ["-", "", "nan", "None"] else float(s)


df["BENEFIT_CLEAN"] = df["BENEFIT"].apply(parse_numeric)
df["PREMIUM_CLEAN"] = df["Premium"].apply(parse_numeric)
df["IS_LAPSE"] = (df["POLICY STATUS"].str.strip() == "Lapse").astype(int)

# --- FEATURE ENGINEERING ---
# 1. Rasio Beban Proteksi (Premi mahal dengan UP kecil menandakan risiko lapse tinggi)
df["PREMIUM_BENEFIT_RATIO"] = df["PREMIUM_CLEAN"] / (df["BENEFIT_CLEAN"] + 1)

# 2. Rasio Beban Usia
df["PREMIUM_AGE_RATIO"] = df["PREMIUM_CLEAN"] / (df["ENTRY AGE"] + 1)

# 3. Indikator Komitmen (Apakah nasabah pernah bayar di muka?)
df["HAS_ADVANCE"] = (df["NUMBER OF ADVANCE PREMIUM"] > 0).astype(str)

# Sanitasi nilai tak terhingga (infinity) jika ada pembagian nol
df.replace([float('inf'), float('-inf')], np.nan, inplace=True)

# 3. Definisi Fitur
cat_cols = [
    "SEX",
    "PAYMENT MODE",
    "NON LAPSE GUARANTEED",
    "Full Benefit?",
    "CHANNEL1",
    "CHANNEL2",
    "CHANNEL3",
    "POLICY TYPE 1",
    "POLICY TYPE 2",
    "POLICY TYPE 3",
]
num_cols = [
    "ENTRY AGE",
    "SUBSTANDARD RISK",
    "NUMBER OF ADVANCE PREMIUM",
    "INITIAL BENEFIT",
    "Policy Year (Decimal)",
    "Policy Year",
    "BENEFIT_CLEAN",
    "PREMIUM_CLEAN",
]

for col in cat_cols:
    df[col] = df[col].astype(str)

X = df[cat_cols + num_cols]
y = df["IS_LAPSE"]

# 4. Membangun Pipeline Preprocessing & Algoritma
print("[3/5] Membangun pipeline transformasi...")
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Unknown"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]
)

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            HistGradientBoostingClassifier(
    max_iter=300,              # Gandakan iterasi (dari 150) agar model belajar lebih lama
    learning_rate=0.05,        # Perlambat laju belajar untuk mengekstraksi pola yang lebih presisi
    max_leaf_nodes=63,         # Buka ruang memori cabang pohon keputusan
    l2_regularization=1.5,     # Terapkan penalti matematis agar tidak berbalik menjadi overfitting
    random_state=42
),
        ),
    ]
)

# 5. Training dan Penyimpanan Model
print("[4/5] Melatih model (HistGradientBoosting)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
full_pipeline.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

# 1. Ekstrak Probabilitas dari KEDUA Set
y_train_prob = full_pipeline.predict_proba(X_train)[:, 1]
y_test_prob = full_pipeline.predict_proba(X_test)[:, 1]

# 2. Kalkulasi Skor ROC-AUC
auc_train = roc_auc_score(y_train, y_train_prob)
auc_test = roc_auc_score(y_test, y_test_prob)

# 3. Analisis Jarak (Delta)
delta = auc_train - auc_test

print("=== DIAGNOSIS OVERFITTING / UNDERFITTING ===")
print(f"ROC-AUC Train Set : {auc_train:.4f}")
print(f"ROC-AUC Test Set  : {auc_test:.4f}")
print(f"Selisih (Delta)   : {delta:.4f}")

# Evaluasi Logis
if delta > 0.05:
    print("\nSTATUS: OVERFITTING")
    print("Tindakan: Tambah L2 regularization, kurangi max_iter, atau kurangi max_leaf_nodes.")
elif auc_train < 0.80 and auc_test < 0.80:
    print("\nSTATUS: UNDERFITTING / FITUR LEMAH")
    print("Tindakan: Model kekurangan kapasitas atau data butuh Feature Engineering tambahan (kombinasi variabel baru).")
else:
    print("\nSTATUS: OPTIMAL / ROBUST")
    print("Tindakan: Model stabil. Performa masa lalu sejalan dengan performa pada data baru.")

os.makedirs("models", exist_ok=True)
model_export_path = os.path.join("models", "lapse_model.joblib")
joblib.dump(full_pipeline, model_export_path)

print(
    f"[5/5] Selesai. Model artefak tersimpan di: {model_export_path}"
)