"""Buat gambar-gambar untuk PPT laporan."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BG = "#FBF8F3"
DARK = "#1F2937"
ACCENT = "#A93226"
BLUE = "#2E86C1"
GREEN = "#1E8449"
MUTED = "#6B7280"
GRID = "#E5E1DA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "text.color": DARK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def simpan(fig, nama):
    fig.savefig(FIG / nama, dpi=220, bbox_inches="tight")
    plt.close(fig)


with open(BASE / "reports" / "ringkasan_model.json", encoding="utf-8") as f:
    ringkasan = json.load(f)

df = pd.read_csv(BASE / "data" / "student_dropout_risk_2026.csv")
df_model = df.drop(columns=["person_id", "gpa_semester_2", "semester"]).copy()
df_model["fail_rate"] = df_model["units_failed"] / df_model["units_enrolled"]
df_model["engagement_score"] = (
    df_model["attendance_rate"] + df_model["assignments_submitted_rate"]
) / 2
df_model["risk_flag_count"] = (
    (df_model["units_failed"] >= 3).astype(int)
    + (df_model["attendance_rate"] < 70).astype(int)
    + (df_model["missed_deadlines"] > 2).astype(int)
    + (df_model["gpa_semester_1"] < 2).astype(int)
)
df_model = df_model.drop(columns=["units_passed"])

X = df_model.drop(columns=["dropout"])
y = df_model["dropout"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

artefak = joblib.load(BASE / "models" / "model_dropout.pkl")
model = artefak["model"]
threshold = float(artefak["threshold"])
proba = model.predict_proba(X_test)[:, 1]

# 1. Distribusi target -------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.2))
jumlah = y.value_counts().sort_index()
warna = [GREEN, ACCENT]
wedges, texts, autotexts = ax.pie(
    jumlah,
    labels=["Bertahan\n68%", "Dropout\n32%"],
    colors=warna,
    autopct="",
    startangle=90,
    wedgeprops={"width": 0.38, "edgecolor": BG, "linewidth": 3},
)
ax.text(0, 0.08, "100 rb", ha="center", va="center", fontsize=20, fontweight="bold")
ax.text(0, -0.16, "mahasiswa", ha="center", va="center", fontsize=10, color=MUTED)
ax.set_title("Distribusi target", fontsize=15, fontweight="bold", pad=16)
simpan(fig, "target.png")

# 2. EDA: dropout rate per GPA dan units_failed -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

gpa_bins = pd.cut(
    df["gpa_semester_1"],
    bins=[-0.01, 1.99, 2.49, 2.99, 3.49, 4.0],
    labels=["<2,0", "2,0–2,49", "2,5–2,99", "3,0–3,49", "3,5–4,0"],
)
gpa_rate = df.groupby(gpa_bins, observed=False)["dropout"].mean() * 100
bars = axes[0].bar(gpa_rate.index.astype(str), gpa_rate.values, color=ACCENT)
for b, v in zip(bars, gpa_rate.values):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                 ha="center", fontsize=9, color=DARK)
axes[0].set_title("Dropout menurut IPK semester 1", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Tingkat dropout (%)")
axes[0].set_ylim(0, 85)
axes[0].grid(axis="y", color=GRID, linewidth=0.8)
axes[0].set_axisbelow(True)

fail_bins = pd.cut(
    df["units_failed"], bins=[-0.1, 0, 2, 4, 20],
    labels=["0", "1–2", "3–4", "≥5"],
)
fail_rate = df.groupby(fail_bins, observed=False)["dropout"].mean() * 100
bars = axes[1].bar(fail_rate.index.astype(str), fail_rate.values, color=BLUE)
for b, v in zip(bars, fail_rate.values):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                 ha="center", fontsize=9, color=DARK)
axes[1].set_title("Dropout menurut mata kuliah gagal", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Tingkat dropout (%)")
axes[1].set_ylim(0, 85)
axes[1].grid(axis="y", color=GRID, linewidth=0.8)
axes[1].set_axisbelow(True)

fig.text(0.5, -0.03, "Sumber: dataset, dihitung ulang dari 100.000 baris.",
         ha="center", fontsize=8, color=MUTED)
simpan(fig, "eda.png")

# 3. Perbandingan model ------------------------------------------------------
cv = pd.DataFrame(ringkasan["baseline_cv"]).T.loc[
    ["Logistic Regression", "Random Forest", "XGBoost"]
]
tuning = pd.DataFrame(ringkasan["tuning"]).T.loc[cv.index]
test = pd.DataFrame(ringkasan["test_default"]).T.loc[cv.index]

fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(len(cv))
lebar = 0.26
b1 = ax.bar(x - lebar, cv["f1"], lebar, label="F1 5-fold CV (baseline)",
            color="#D5A17C", edgecolor=BG)
b2 = ax.bar(x, tuning["f1_cv"].astype(float), lebar,
            label="F1 CV (setelah tuning)", color="#E8C39E", edgecolor=BG)
b3 = ax.bar(x + lebar, test["F1"], lebar, label="F1 test (threshold 0,50)",
            color=ACCENT, edgecolor=BG)
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(cv.index)
ax.set_ylim(0.70, 0.755)
ax.set_ylabel("F1-score")
ax.set_title("Perbandingan F1 antar-model", fontsize=14, fontweight="bold", pad=14)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.14),
    fontsize=8, frameon=False, ncol=3,
)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.22)
simpan(fig, "model_compare.png")

# 4. Confusion matrix --------------------------------------------------------
pred = (proba >= threshold).astype(int)
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5.0, 4.3))
im = ax.imshow(cm, cmap="Reds")
label = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        ax.text(j, i, f"{label[i][j]}\n{val:,}", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if val > cm.max() / 2 else DARK)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Prediksi bertahan", "Prediksi dropout"])
ax.set_yticklabels(["Aktual bertahan", "Aktual dropout"])
ax.set_title(f"Confusion matrix (threshold {threshold:.2f})",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(np.arange(-0.5, 2), minor=True)
ax.set_yticks(np.arange(-0.5, 2), minor=True)
ax.grid(which="minor", color=BG, linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
simpan(fig, "confusion.png")

# 5. Feature importance ------------------------------------------------------
fitur = pd.DataFrame(ringkasan["top_features"]).head(10)
nama = {
    "gpa_semester_1": "IPK semester 1",
    "family_income_level_High": "Pendapatan: Tinggi",
    "family_income_level_Low": "Pendapatan: Rendah",
    "tuition_up_to_date": "Uang kuliah lancar",
    "lms_logins_per_week": "Login LMS/minggu",
    "missed_deadlines": "Tenggat terlewat",
    "weekly_work_hours": "Jam kerja/minggu",
    "attendance_rate": "Kehadiran",
    "stress_level": "Tingkat stres",
    "units_failed": "Mata kuliah gagal",
    "engagement_score": "Skor keterlibatan",
    "scholarship_holder": "Penerima beasiswa",
}
fitur["Label"] = fitur["Fitur"].map(nama).fillna(fitur["Fitur"])
fitur["Magnitudo"] = fitur["Nilai"].abs()
fitur = fitur.sort_values("Magnitudo")
warna = [ACCENT if v > 0 else BLUE for v in fitur["Nilai"]]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.barh(fitur["Label"], fitur["Nilai"], color=warna)
for b, v in zip(bars, fitur["Nilai"]):
    ax.text(v + (0.02 if v > 0 else -0.02), b.get_y() + b.get_height() / 2,
            f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
ax.axvline(0, color=DARK, linewidth=1)
ax.set_xlabel("Koefisien (merah = menaikkan risiko, biru = menurunkan risiko)")
ax.set_title("10 fitur paling berpengaruh — Logistic Regression",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(-1.55, 1.55)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
simpan(fig, "importance.png")

# 6. Threshold curve ---------------------------------------------------------
thresholds = np.arange(0.30, 0.86, 0.01)
rows = []
for t in thresholds:
    p, r, f, _ = precision_recall_fscore_support(
        y_test, (proba >= t).astype(int), average="binary", zero_division=0
    )
    rows.append({"threshold": t, "precision": p, "recall": r, "f1": f})
kurva = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(kurva["threshold"], kurva["precision"], label="Precision", color=BLUE)
ax.plot(kurva["threshold"], kurva["recall"], label="Recall", color=GREEN)
ax.plot(kurva["threshold"], kurva["f1"], label="F1", color=ACCENT, linewidth=2.5)
ax.axvline(threshold, color=DARK, linestyle="--", linewidth=1,
           label=f"Threshold terpilih ({threshold:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel("Skor")
ax.set_ylim(0, 1.02)
ax.set_title("Precision, recall, dan F1 terhadap threshold",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
simpan(fig, "threshold.png")

print("Gambar dibuat di:", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" -", p.name)
