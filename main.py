import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join("models", "lapse_model.joblib")
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model {MODEL_PATH} belum ada. Jalankan train_and_export.py terlebih dahulu.")
    else:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[INFO] Model pipeline berhasil dimuat dari {MODEL_PATH}")
    yield


app = FastAPI(
    title="Insurance Lapse Prediction API",
    description="Backend API berbasis Machine Learning untuk evaluasi risiko terminasi/lapse polis asuransi.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["frontend-insurance-lapse-api.vercel.app"],  # ganti "*" dengan domain frontend spesifik saat sudah production, mis. ["https://frontend-anda.up.railway.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolicyInputSchema(BaseModel):
    entry_age: int = Field(..., ge=0, le=100, example=35, description="Usia masuk tertanggung")
    sex: Literal["M", "F"] = Field(..., example="M")
    payment_mode: Literal["Monthly", "Annually", "Quaterly", "Semiannually", "Single Premium"] = Field(
        ..., example="Monthly"
    )
    non_lapse_guaranteed: str = Field(default="NO NLG", example="NO NLG")
    substandard_risk: float = Field(default=0.0, example=0.0, description="Persentase ekstra risiko medis")
    advance_premium_count: int = Field(default=0, ge=0, example=0)
    initial_benefit: float = Field(default=0.0, ge=0.0, example=0.0)
    full_benefit: Literal["Y", "N"] = Field(default="N", example="N")
    policy_year_decimal: float = Field(..., ge=0.0, example=2.0)
    policy_year: int = Field(..., ge=1, example=2)
    channel1: str = Field(default="1", example="1")
    channel2: str = Field(default="1", example="1")
    channel3: str = Field(default="1", example="1")
    policy_type_1: str = Field(default="3", example="3")
    policy_type_2: str = Field(default="5", example="5")
    policy_type_3: str = Field(default="A", example="A")
    benefit_amount: float = Field(..., gt=0, example=100000.0, description="Uang Pertanggungan")
    premium_amount: float = Field(..., gt=0, example=250.0, description="Besaran Premi")


def categorize_lapse_risk(prob: float):
    if prob < 0.30:
        return "Tier 1: Low Churn Risk", "Pertahankan pola monitoring standar."
    elif prob < 0.60:
        return "Tier 2: Moderate Risk", "Kirim reminder tagihan otomatis H-7 jatuh tempo."
    elif prob < 0.80:
        return "Tier 3: High Risk", "Eskalasi ke tim Relationship Manager untuk konfirmasi pembayaran."
    else:
        return "Tier 4: Critical Risk", "Tawarkan opsi restrukturisasi polis / grace period khusus."


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Insurance Persistency ML API",
        "model_loaded": model_pipeline is not None,
    }


@app.post("/predict")
def predict_policy_status(payload: PolicyInputSchema):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model belum siap/gagal dimuat pada server.")

    # Rekonstruksi baris data sesuai skema training
    input_df = pd.DataFrame(
        [
            {
                "SEX": str(payload.sex),
                "PAYMENT MODE": str(payload.payment_mode),
                "NON LAPSE GUARANTEED": str(payload.non_lapse_guaranteed),
                "Full Benefit?": str(payload.full_benefit),
                "CHANNEL1": str(payload.channel1),
                "CHANNEL2": str(payload.channel2),
                "CHANNEL3": str(payload.channel3),
                "POLICY TYPE 1": str(payload.policy_type_1),
                "POLICY TYPE 2": str(payload.policy_type_2),
                "POLICY TYPE 3": str(payload.policy_type_3),
                "ENTRY AGE": payload.entry_age,
                "SUBSTANDARD RISK": payload.substandard_risk,
                "NUMBER OF ADVANCE PREMIUM": payload.advance_premium_count,
                "INITIAL BENEFIT": payload.initial_benefit,
                "Policy Year (Decimal)": payload.policy_year_decimal,
                "Policy Year": payload.policy_year,
                "BENEFIT_CLEAN": payload.benefit_amount,
                "PREMIUM_CLEAN": payload.premium_amount,
            }
        ]
    )

    input_df["PREMIUM_BENEFIT_RATIO"] = input_df["PREMIUM_CLEAN"] / (input_df["BENEFIT_CLEAN"] + 1)
    input_df["PREMIUM_AGE_RATIO"] = input_df["PREMIUM_CLEAN"] / (input_df["ENTRY AGE"] + 1)
    input_df["HAS_ADVANCE"] = (input_df["NUMBER OF ADVANCE PREMIUM"] > 0).astype(str)

    # Eksekusi inferensi
    lapse_proba = float(model_pipeline.predict_proba(input_df)[0, 1])
    OPTIMAL_THRESHOLD = 0.3879
    is_lapse = bool(lapse_proba >= OPTIMAL_THRESHOLD)
    tier_label, action_rec = categorize_lapse_risk(lapse_proba)

    return {
        "status": "success",
        "prediction": {
            "is_lapse_predicted": is_lapse,
            "lapse_probability": round(lapse_proba, 4),
            "lapse_probability_percentage": f"{lapse_proba * 100:.2f}%",
            "risk_tier": tier_label,
            "recommended_action": action_rec,
        },
    }
