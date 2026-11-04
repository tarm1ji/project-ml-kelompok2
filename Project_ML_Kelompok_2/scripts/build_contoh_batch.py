"""Buat CSV contoh untuk demo prediksi batch di aplikasi."""
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "student_dropout_risk_2026.csv"
ARTEFAK = BASE / "models" / "model_dropout.pkl"
OUT = BASE / "data" / "contoh_batch.csv"

ART = joblib.load(ARTEFAK)
FEATURES = list(ART["features"])
DERIVED = {"fail_rate", "engagement_score", "risk_flag_count"}
RAW = [f for f in FEATURES if f not in DERIVED]

df = pd.read_csv(DATA)
df = df.drop(columns=["gpa_semester_2", "semester", "dropout"], errors="ignore")
df = df[["person_id"] + RAW]

# Skor probabilitas untuk memilih contoh berisiko tinggi dan rendah.
kerja = df[RAW].copy()
kerja["fail_rate"] = kerja["units_failed"] / kerja["units_enrolled"].clip(lower=1)
kerja["engagement_score"] = (
    kerja["attendance_rate"] + kerja["assignments_submitted_rate"]
) / 2
kerja["risk_flag_count"] = (
    (kerja["units_failed"] >= 3).astype(int)
    + (kerja["attendance_rate"] < 70).astype(int)
    + (kerja["missed_deadlines"] > 2).astype(int)
    + (kerja["gpa_semester_1"] < 2).astype(int)
)
proba = ART["model"].predict_proba(kerja[FEATURES])[:, 1]
df = df.assign(proba=proba)

tinggi = df.sort_values("proba", ascending=False).head(5)
rendah = df.sort_values("proba").head(5)
contoh = pd.concat([tinggi, rendah]).drop(columns="proba")
contoh.to_csv(OUT, index=False)
print("Disimpan:", OUT)
print(contoh[["person_id", "gpa_semester_1", "units_failed", "attendance_rate"]].to_string(index=False))
