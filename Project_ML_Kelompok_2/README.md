# Prediksi Risiko Dropout Mahasiswa

Project akhir mata kuliah Pembelajaran Mesin: memprediksi apakah seorang mahasiswa berisiko **dropout** pada **awal semester 2**, menggunakan data demografi, admisi, capaian akademik semester 1, dan perilaku belajar.

## Tujuan

1. Membangun model klasifikasi risiko dropout yang dapat dipakai untuk deteksi dini.
2. Membandingkan tiga algoritma: **Logistic Regression**, **Random Forest**, dan **XGBoost**.
3. Mengidentifikasi fitur yang paling berpengaruh.
4. Mengimplementasikan model sebagai aplikasi Streamlit untuk prediksi individu dan batch.

## Dataset

- Berkas: `data/student_dropout_risk_2026.csv`
- Sumber: [Kaggle — Student Academic Performance & Dropout Risk](https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk), oleh Mohan Krishna Thalla
- Lisensi: CC0 Public Domain
- Ukuran: 100.000 baris, 34 kolom
- Target: `dropout` (1 = dropout, 0 = bertahan)
- Distribusi target: 68% bertahan, 32% dropout
- Missing value: 0
- Duplikat: 0

**Skenario prediksi:** awal semester 2. Karena itu:

- `gpa_semester_2` **tidak dipakai** (leakage).
- `semester` **tidak dipakai** (tidak konsisten dengan skenario).
- `person_id` **tidak dipakai** (identifier).
- `units_passed` diganti dengan `fail_rate` karena redundan.

## Hasil Utama

Model terpilih: **Logistic Regression**, dipilih berdasarkan F1 cross-validation hasil tuning.

| Model | F1 CV (baseline) | F1 CV (tuning) | F1 test (threshold 0,50) |
|---|---:|---:|---:|
| Logistic Regression | 0,7392 | **0,7387** | 0,7381 |
| XGBoost | 0,7381 | 0,7379 | **0,7390** |
| Random Forest | 0,7275 | 0,7300 | 0,7317 |

Pada threshold hasil out-of-fold tuning (**0,57**), Logistic Regression mencapai:

| Metrik | Nilai |
|---|---:|
| Accuracy | 0,8287 |
| Precision | 0,7145 |
| Recall dropout | 0,7741 |
| F1 | **0,7431** |
| ROC-AUC | 0,8991 |
| PR-AUC | 0,8246 |

Confusion matrix pada data test: **TN 11.619, FP 1.980, FN 1.446, TP 4.955**. Artinya, dari 6.401 mahasiswa dropout, 4.955 berhasil ditangkap dan 1.446 terlewat.

> XGBoost memiliki F1 test tertinggi pada threshold default 0,50, tetapi Logistic Regression dipilih karena F1 CV-nya terbaik, lebih stabil, lebih mudah diinterpretasikan, dan setelah threshold tuning menghasilkan F1 test tertinggi (0,7431).

## Deliverable

| Deliverable | Lokasi |
|---|---|
| EDA + interpretasi | `notebooks/01_eda.ipynb` |
| Modeling + evaluasi | `notebooks/02_modeling.ipynb` |
| Model terlatih | `models/model_dropout.pkl` |
| Aplikasi Streamlit | `app.py` |
| PPT laporan project | `docs/Presentasi_Project_ML_Kelompok_2.pptx` |
| Skenario video demo | `docs/skenario_video.md` |
| Tabel studi literatur (6 jurnal) | `docs/tabel_studi_literatur.md` |
| Panduan deploy dan share aplikasi | `docs/cara_deploy_dan_share.md` |
| Contoh CSV prediksi batch | `data/contoh_batch.csv` |

## Struktur Project

```text
project-ml-kel-2/
├── data/
│   ├── student_dropout_risk_2026.csv
│   └── contoh_batch.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── models/
│   └── model_dropout.pkl
├── reports/
│   ├── figures/
│   ├── perbandingan_model_cv.csv
│   ├── perbandingan_model_test.csv
│   ├── perbandingan_threshold.csv
│   └── ringkasan_model.json
├── docs/
│   ├── Presentasi_Project_ML_Kelompok_2.pptx
│   ├── cara_deploy_dan_share.md
│   ├── skenario_video.md
│   └── tabel_studi_literatur.md
├── scripts/
│   ├── build_charts.py
│   ├── build_contoh_batch.py
│   ├── build_modeling.py
│   └── build_ppt.py
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

`requirements.txt` hanya berisi dependensi aplikasi agar deploy Streamlit ringan dan kompatibel. `requirements-dev.txt` menambahkan Jupyter, XGBoost, matplotlib, seaborn, dan alat pembuat PPT.

## Cara Menjalankan

### 1. Aktifkan virtual environment

```powershell
cd "C:\!Documents\project-ml-kel-2"
.\venv\Scripts\Activate.ps1
```

### 2. Install dependensi

Untuk menjalankan aplikasi saja:

```powershell
pip install -r requirements.txt
```

Untuk menjalankan notebook, melatih ulang model, atau membuat ulang PPT:

```powershell
pip install -r requirements-dev.txt
```

### 3. Jalankan notebook (opsional)

```powershell
jupyter notebook
```

Buka `notebooks/01_eda.ipynb`, lalu `notebooks/02_modeling.ipynb`. Notebook modeling membutuhkan waktu 10–15 menit karena menjalankan 5-fold cross-validation dan hyperparameter tuning.

### 4. Jalankan aplikasi

```powershell
streamlit run app.py
```

Aplikasi memiliki tiga tab:

- **Prediksi Individu** — form profil, akademik semester 1, dan kebiasaan belajar; menghasilkan probabilitas, kategori risiko, faktor risiko, dan rekomendasi.
- **Prediksi Batch** — unggah CSV (template tersedia di aplikasi), dapatkan daftar prioritas intervensi dan unduhan hasil. Contoh: `data/contoh_batch.csv`.
- **Tentang Model** — metrik test, threshold, feature importance, cara kerja model, dan keterbatasan.

### 5. Regenerasi gambar dan PPT (opsional)

```powershell
python scripts/build_charts.py
python scripts/build_ppt.py
```

## Ringkasan Metodologi

| Tahap | Keputusan |
|---|---|
| Split | 80% train, 20% test, stratified, `random_state=42` |
| CV | 5-fold StratifiedKFold |
| Preprocessing | `log1p` + `StandardScaler` untuk fitur miring; `OneHotEncoder` untuk kategorikal |
| Feature engineering | `fail_rate`, `engagement_score`, `risk_flag_count` |
| Imbalance | `class_weight="balanced"` / `scale_pos_weight`, tanpa SMOTE |
| Tuning | RandomizedSearchCV, scoring F1 |
| Metrik utama | F1, recall kelas dropout, PR-AUC |
| Threshold | Dicari dari out-of-fold prediction pada data training |

## Video Demo

Ikuti `docs/skenario_video.md`. Durasi target 6 menit 30 detik, dengan pembagian:

1. Pembukaan dan perkenalan anggota.
2. Latar belakang, dataset, EDA, preprocessing, dan algoritma.
3. Perbandingan model dan alasan pemilihan Logistic Regression.
4. Demo live prediksi individu dan batch.
5. Kesimpulan, keterbatasan, dan penutup.

Video wajib memperlihatkan aplikasi benar-benar dijalankan.

## Catatan Penting

- **Indikasi data sintetis:** `gpa_semester_2 < gpa_semester_1` pada 90,4% baris. Hasil tidak boleh digeneralisasi ke populasi nyata tanpa validasi ulang.
- **Asumsi timing:** fitur keterlibatan (`attendance_rate`, `assignments_submitted_rate`, `lms_logins_per_week`, `library_visits_per_month`, `missed_deadlines`) dianggap merekam semester 1.
- **Data test tidak dipakai** untuk pemilihan fitur, tuning, atau pemilihan threshold.
- **Accuracy bukan metrik utama** karena kelas dropout adalah kelas yang paling penting dideteksi.
- **Prediksi bukan vonis.** Hasil model dipakai untuk memprioritaskan pengecekan dan intervensi manusia.

## Dokumentasi Tambahan

- Cara deploy dan membagikan aplikasi: `docs/cara_deploy_dan_share.md`
- Studi literatur dan tabel 6 jurnal: `docs/tabel_studi_literatur.md`
- Skenario dan storyboard video: `docs/skenario_video.md`
- Ringkasan hasil model: `reports/ringkasan_model.json`

## Anggota Kelompok

1. ...
2. ...
3. ...

> Ganti nama, NIM, dan kelas pada README, slide judul, dan slide pembagian tugas sebelum presentasi.
