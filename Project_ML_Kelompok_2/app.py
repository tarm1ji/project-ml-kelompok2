from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Risiko Dropout",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Inter dipakai sebagai cadangan di Windows/Android, karena SF Pro
           hanya tersedia di perangkat Apple dan tidak boleh di-host sendiri */
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap");

        .stApp,
        .stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre):not([class*="material"]) {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                "SF Pro Display", "Inter", "Segoe UI", Roboto, "Helvetica Neue",
                Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .stApp code, .stApp pre {
            font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
            color: #D4D4D4;
        }

        .block-container {
            padding-top: 2.5rem;
            padding-left: clamp(1rem, 4vw, 3rem);
            padding-right: clamp(1rem, 4vw, 3rem);
            max-width: min(1320px, 100%);
        }
        h1 {
            font-weight: 600;
            letter-spacing: -0.01em;
            font-size: clamp(1.6rem, 4vw, 2.4rem);
            overflow-wrap: anywhere;
        }
        h2, h3, h4 { font-weight: 600; }
        footer, #MainMenu { visibility: hidden; }

        /* Tombol utama: biru aksen dengan teks putih */
        .stApp button[kind^="primary"] {
            background-color: #3B82F6;
            border-color: #3B82F6;
        }
        .stApp button[kind^="primary"]:hover {
            background-color: #2563EB;
            border-color: #2563EB;
        }
        .stApp button[kind^="primary"],
        .stApp button[kind^="primary"] p,
        .stApp button[kind^="primary"] span {
            color: #FFFFFF !important;
            font-weight: 600;
        }

        /* Sembunyikan tulisan "Press Enter to apply / submit form" di kolom input */
        [data-testid="InputInstructions"] { display: none; }
        [data-testid="stMetricValue"] { font-size: clamp(1.2rem, 3vw, 1.6rem); }
        [data-testid="stMetricLabel"] { opacity: 0.7; }

        /* Sembunyikan tombol +/- agar kolom angka terasa seperti input ketik biasa */
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] { display: none; }
        [data-testid="stNumberInput"] input { text-align: left; }

        /* Kolom otomatis turun ke baris baru saat ruang sempit (HP / zoom in) */
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 0.75rem; }
        [data-testid="stColumn"] {
            min-width: min(100%, 220px);
            flex: 1 1 220px;
        }

        /* Tanpa garis kotak di tab prediksi individu (form, container, pembungkus) */
        .stApp div:has(.layout-utama),
        [data-testid="stForm"],
        [data-testid="stHorizontalBlock"]:has(.layout-utama) [data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* Layout utama: form di kiri, hasil di kanan (menempel saat scroll) */
        [data-testid="stHorizontalBlock"]:has(.layout-utama) {
            align-items: flex-start;
        }
        [data-testid="stHorizontalBlock"]:has(.layout-utama) > [data-testid="stColumn"]:first-child {
            flex: 2 1 480px !important;
        }
        [data-testid="stHorizontalBlock"]:has(.layout-utama) > [data-testid="stColumn"]:last-child {
            flex: 1 1 340px !important;
            position: sticky;
            top: 4rem;
            max-height: calc(100vh - 5rem);
            overflow-y: auto;
        }

        @media (max-width: 640px) {
            .block-container { padding-top: 1.5rem; }
            [data-testid="stColumn"] { flex: 1 1 100%; }
            [data-testid="stHorizontalBlock"]:has(.layout-utama) > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
            }
            [data-testid="stHorizontalBlock"]:has(.layout-utama) > [data-testid="stColumn"]:last-child {
                position: static;
                max-height: none;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "models" / "model_dropout.pkl"


@st.cache_resource(show_spinner="Memuat model prediksi ...")
def muat_artefak(path: str):
    return joblib.load(path)


try:
    ARTEFAK = muat_artefak(str(ARTIFACT_PATH))
except FileNotFoundError:
    st.error(
        "Model belum ditemukan di `models/model_dropout.pkl`. "
        "Jalankan `notebooks/02_modeling.ipynb` terlebih dahulu."
    )
    st.stop()

MODEL = ARTEFAK["model"]
THRESHOLD = float(ARTEFAK["threshold"])
NAMA_MODEL = ARTEFAK.get("nama_model", "Model klasifikasi")
FITUR_MODEL = list(ARTEFAK["features"])
FITUR_KATEGORIKAL = list(ARTEFAK.get("categorical_features", []))
METRIK_DEFAULT = ARTEFAK.get("metrics_test_default", {})
METRIK_THRESHOLD = ARTEFAK.get("metrics_test_threshold", {})

FITUR_TURUNAN = ["fail_rate", "engagement_score", "risk_flag_count"]
FITUR_MENTAH = [f for f in FITUR_MODEL if f not in FITUR_TURUNAN]

# ---------------------------------------------------------------------------
# Label dan pilihan
# ---------------------------------------------------------------------------
LABEL_FITUR = {
    "age": "Usia",
    "gender": "Jenis kelamin",
    "nationality": "Kewarganegaraan",
    "first_generation": "Mahasiswa generasi pertama",
    "disability_status": "Memiliki disabilitas",
    "family_income_level": "Tingkat pendapatan keluarga",
    "scholarship_holder": "Penerima beasiswa",
    "tuition_up_to_date": "Uang kuliah lancar",
    "part_time_job": "Memiliki pekerjaan paruh waktu",
    "weekly_work_hours": "Jam kerja per minggu",
    "course_type": "Bidang studi",
    "study_mode": "Mode kuliah",
    "international_student": "Mahasiswa internasional",
    "admission_grade": "Nilai admisi",
    "previous_qualification_grade": "Nilai kualifikasi sebelumnya",
    "units_enrolled": "Mata kuliah yang diambil",
    "units_failed": "Mata kuliah yang gagal",
    "gpa_semester_1": "IPK semester 1",
    "attendance_rate": "Tingkat kehadiran (%)",
    "assignments_submitted_rate": "Tugas terkumpul (%)",
    "library_visits_per_month": "Kunjungan perpustakaan per bulan",
    "lms_logins_per_week": "Frekuensi buka e-learning per minggu",
    "extracurricular_activities": "Mengikuti kegiatan ekstrakurikuler",
    "missed_deadlines": "Tenggat yang terlewat",
    "stress_level": "Tingkat stres (1-10)",
    "mental_health_support": "Mendapat dukungan kesehatan mental",
    "satisfaction_score": "Kepuasan terhadap perkuliahan (1-10)",
    "commute_time_mins": "Waktu perjalanan ke kampus (menit)",
    "sleep_hours": "Jam tidur per hari",
    "fail_rate": "Rasio mata kuliah gagal",
    "engagement_score": "Skor keterlibatan",
    "risk_flag_count": "Jumlah indikator risiko",
}

GENDER = ["Female", "Male", "Other"]
GENDER_LABEL = {"Female": "Perempuan", "Male": "Laki-laki", "Other": "Lainnya"}

NATIONALITY = [
    "Australia", "Brazil", "Canada", "China", "Egypt", "Germany", "India",
    "Indonesia", "Mexico", "Nigeria", "Pakistan", "Philippines",
    "South Korea", "UK", "USA",
]

INCOME = ["Low", "Middle", "High"]
INCOME_LABEL = {"Low": "Rendah", "Middle": "Menengah", "High": "Tinggi"}

COURSE_TYPE = [
    "Arts & Humanities", "Business", "Health Sciences",
    "Law & Social Sciences", "STEM",
]
COURSE_LABEL = {
    "Arts & Humanities": "Seni & Humaniora",
    "Business": "Bisnis",
    "Health Sciences": "Ilmu Kesehatan",
    "Law & Social Sciences": "Hukum & Ilmu Sosial",
    "STEM": "STEM",
}

STUDY_MODE = ["Full-time", "Part-time", "Online"]
STUDY_LABEL = {
    "Full-time": "Penuh waktu",
    "Part-time": "Paruh waktu",
    "Online": "Daring",
}

LABEL_SINGKAT = {
    "family_income_level": "Pendapatan",
    "course_type": "Bidang",
    "study_mode": "Mode",
    "gender": "Gender",
    "nationality": "Negara",
}
NILAI_KATEGORI = {
    "High": "tinggi",
    "Middle": "menengah",
    "Low": "rendah",
    "Full-time": "penuh waktu",
    "Part-time": "paruh waktu",
    "Online": "daring",
    "Female": "perempuan",
    "Male": "laki-laki",
    "Other": "lainnya",
    "STEM": "STEM",
    "Business": "bisnis",
    "Health Sciences": "ilmu kesehatan",
    "Arts & Humanities": "seni & humaniora",
    "Law & Social Sciences": "hukum & sosial",
}


# ---------------------------------------------------------------------------
# Fungsi bantuan
# ---------------------------------------------------------------------------
def label_nilai(nilai: float) -> str:
    """Label ringkas untuk nilai skala 0-100."""
    if nilai < 60:
        return "Rendah"
    if nilai < 80:
        return "Cukup"
    return "Tinggi"


def siapkan_fitur(data: dict) -> pd.DataFrame:
    """Bangun fitur turunan dan urutkan kolom sesuai saat training."""
    baris = dict(data)
    baris["fail_rate"] = baris["units_failed"] / max(baris["units_enrolled"], 1)
    baris["engagement_score"] = (
        baris["attendance_rate"] + baris["assignments_submitted_rate"]
    ) / 2
    baris["risk_flag_count"] = int(
        (baris["units_failed"] >= 3)
        + (baris["attendance_rate"] < 70)
        + (baris["missed_deadlines"] > 2)
        + (baris["gpa_semester_1"] < 2)
    )
    return pd.DataFrame([baris], columns=FITUR_MODEL)


def siapkan_fitur_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Versi batch dari siapkan_fitur."""
    hasil = df.copy()
    hasil["fail_rate"] = hasil["units_failed"] / hasil["units_enrolled"].clip(lower=1)
    hasil["engagement_score"] = (
        hasil["attendance_rate"] + hasil["assignments_submitted_rate"]
    ) / 2
    hasil["risk_flag_count"] = (
        (hasil["units_failed"] >= 3).astype(int)
        + (hasil["attendance_rate"] < 70).astype(int)
        + (hasil["missed_deadlines"] > 2).astype(int)
        + (hasil["gpa_semester_1"] < 2).astype(int)
    )
    return hasil.reindex(columns=FITUR_MODEL)


def kategori_risiko(proba: float) -> tuple[str, str]:
    """Kembalikan (label, warna badge)."""
    if proba >= max(0.70, THRESHOLD + 0.15):
        return "Risiko tinggi", "red"
    if proba >= THRESHOLD:
        return "Risiko sedang", "orange"
    if proba >= max(0.0, THRESHOLD - 0.15):
        return "Perlu dipantau", "yellow"
    return "Risiko rendah", "green"


WARNA_HEX = {
    "red": "#EF4444",
    "orange": "#F97316",
    "yellow": "#EAB308",
    "green": "#22C55E",
}


def kartu_probabilitas(proba: float, warna: str) -> str:
    """HTML angka besar dan bar berwarna sesuai kategori risiko."""
    hex_warna = WARNA_HEX.get(warna, "#3B82F6")
    persen = min(max(proba, 0.0), 1.0) * 100
    batas = min(max(THRESHOLD, 0.0), 1.0) * 100
    return (
        '<div style="font-size:0.85rem;opacity:0.7">Probabilitas dropout</div>'
        f'<div style="font-size:2.6rem;font-weight:600;line-height:1.15;color:{hex_warna}">{proba:.1%}</div>'
        '<div style="position:relative;height:10px;border-radius:5px;'
        'background:rgba(128,128,128,0.25);margin:0.6rem 0 0.4rem">'
        f'<div style="width:{persen:.1f}%;height:100%;border-radius:5px;background:{hex_warna}"></div>'
        f'<div style="position:absolute;left:{batas:.1f}%;top:-4px;height:18px;'
        'border-left:2px solid rgba(128,128,128,0.9)"></div></div>'
        f'<div style="font-size:0.8rem;opacity:0.7">Garis abu-abu menandai ambang keputusan model ({THRESHOLD:.1%})</div>'
    )


def faktor_risiko(baris: dict) -> list[str]:
    """Indikator yang bisa dijelaskan ke pengguna, bukan SHAP."""
    faktor = []
    if baris["gpa_semester_1"] < 2.0:
        faktor.append("IPK semester 1 di bawah 2,0")
    if baris["units_failed"] >= 3:
        faktor.append(f"{baris['units_failed']} mata kuliah gagal")
    if baris["attendance_rate"] < 70:
        faktor.append(f"Kehadiran rendah ({baris['attendance_rate']:.0f}%)")
    if baris["assignments_submitted_rate"] < 60:
        faktor.append(f"Tugas terkumpul rendah ({baris['assignments_submitted_rate']:.0f}%)")
    if baris["missed_deadlines"] > 2:
        faktor.append(f"{baris['missed_deadlines']} tenggat terlewat")
    if baris["stress_level"] >= 8:
        faktor.append(f"Tingkat stres tinggi ({baris['stress_level']}/10)")
    if baris["satisfaction_score"] <= 4:
        faktor.append(f"Kepuasan kuliah rendah ({baris['satisfaction_score']}/10)")
    if baris["sleep_hours"] < 6:
        faktor.append(f"Jam tidur kurang ({baris['sleep_hours']:.1f} jam)")
    if baris["weekly_work_hours"] > 20:
        faktor.append(f"Jam kerja tinggi ({baris['weekly_work_hours']} jam/minggu)")
    if not baris["tuition_up_to_date"]:
        faktor.append("Uang kuliah belum lunas")
    if not baris["mental_health_support"] and baris["stress_level"] >= 7:
        faktor.append("Stres tinggi tanpa dukungan kesehatan mental")
    return faktor


def rekomendasi(proba: float, faktor: list[str]) -> list[str]:
    saran = []
    if proba >= max(0.70, THRESHOLD + 0.15):
        saran.append(
            "Prioritaskan sesi konseling individual dan susun rencana perbaikan "
            "akademik dalam 1-2 minggu ke depan."
        )
    elif proba >= THRESHOLD:
        saran.append(
            "Jadwalkan check-in dengan dosen wali untuk membahas hambatan "
            "belajar dan rencana semester ini."
        )
    elif proba >= max(0.0, THRESHOLD - 0.15):
        saran.append(
            "Pantau perkembangan akademik dan tawarkan bantuan sebelum masalah "
            "menjadi lebih besar."
        )
    else:
        saran.append(
            "Pertahankan kebiasaan belajar yang sudah baik dan lakukan pemantauan "
            "rutin."
        )

    if any("Kehadiran" in f or "Tugas" in f or "tenggat" in f for f in faktor):
        saran.append(
            "Cek hambatan kehadiran dan pengumpulan tugas; tawarkan fleksibilitas "
            "atau pendampingan belajar bila diperlukan."
        )
    if any("mata kuliah gagal" in f for f in faktor):
        saran.append(
            "Susun rencana remedial atau pengambilan ulang mata kuliah bersama "
            "pembimbing akademik."
        )
    if any("stres" in f.lower() or "Kepuasan" in f or "tidur" in f for f in faktor):
        saran.append(
            "Arahkan ke layanan konseling kampus dan bantu menata beban kerja, "
            "tidur, serta aktivitas harian."
        )
    if any("Uang kuliah" in f for f in faktor):
        saran.append(
            "Hubungkan dengan bagian keuangan atau skema bantuan biaya pendidikan."
        )
    return saran[:4]


@st.cache_data(show_spinner=False)
def ambil_importance(_model, fitur_kategorikal: tuple[str, ...]) -> pd.DataFrame:
    prep = _model.named_steps["prep"]
    model = _model.named_steps["model"]
    nama = list(prep.get_feature_names_out())

    if hasattr(model, "coef_"):
        nilai = np.asarray(model.coef_[0], dtype=float)
        arah = np.where(nilai >= 0, "Menaikkan risiko", "Menurunkan risiko")
    else:
        nilai = np.asarray(model.feature_importances_, dtype=float)
        arah = np.array(["Kontribusi"] * len(nilai))

    def rapikan(nama_fitur: str) -> str:
        if "__" not in nama_fitur:
            return LABEL_FITUR.get(nama_fitur, nama_fitur)
        prefix, sisa = nama_fitur.split("__", 1)
        if prefix == "categorical":
            for kolom in fitur_kategorikal:
                if sisa.startswith(kolom + "_"):
                    nilai_kategori = sisa[len(kolom) + 1:]
                    nilai_kategori = NILAI_KATEGORI.get(nilai_kategori, nilai_kategori)
                    return f"{LABEL_SINGKAT.get(kolom, kolom)}: {nilai_kategori}"
        return LABEL_FITUR.get(sisa, sisa)

    tabel = pd.DataFrame({
        "Fitur": [rapikan(n) for n in nama],
        "Nilai": nilai,
        "Arah": arah,
        "Magnitudo": np.abs(nilai),
    })
    return tabel.sort_values("Magnitudo", ascending=False).head(15)


def grafik_importance(tabel: pd.DataFrame) -> alt.Chart:
    domain = ["Menaikkan risiko", "Menurunkan risiko", "Kontribusi"]
    rentang = ["#EF4444", "#3B82F6", "#A1A1A6"]
    return (
        alt.Chart(tabel)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Nilai:Q", title="Nilai"),
            y=alt.Y(
                "Fitur:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=170, labelFontSize=11),
            ),
            color=alt.Color(
                "Arah:N",
                scale=alt.Scale(domain=domain, range=rentang),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Fitur:N", title="Fitur"),
                alt.Tooltip("Nilai:Q", title="Nilai", format=".4f"),
            ],
        )
        .properties(height=380)
    )


TEMPLATE_BATCH = pd.DataFrame([{
    "age": 21,
    "gender": "Female",
    "nationality": "Indonesia",
    "first_generation": 0,
    "disability_status": 0,
    "family_income_level": "Low",
    "scholarship_holder": 1,
    "tuition_up_to_date": 1,
    "part_time_job": 0,
    "weekly_work_hours": 0,
    "course_type": "STEM",
    "study_mode": "Full-time",
    "international_student": 0,
    "admission_grade": 133.1,
    "previous_qualification_grade": 139.6,
    "units_enrolled": 13,
    "units_failed": 2,
    "gpa_semester_1": 2.58,
    "attendance_rate": 76.1,
    "assignments_submitted_rate": 72.6,
    "library_visits_per_month": 3,
    "lms_logins_per_week": 7.1,
    "extracurricular_activities": 1,
    "missed_deadlines": 2,
    "stress_level": 7,
    "mental_health_support": 0,
    "satisfaction_score": 6,
    "commute_time_mins": 16,
    "sleep_hours": 6.5,
}])

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Deteksi Risiko Dropout Mahasiswa")
st.caption(
    "Prediksi pada awal semester 2 menggunakan profil mahasiswa, capaian "
    "akademik semester 1, dan perilaku belajar."
)

tab_individu, tab_batch, tab_model = st.tabs(
    ["Prediksi individu", "Prediksi batch", "Tentang model"]
)

# ---------------------------------------------------------------------------
# Tab 1: prediksi individu
# ---------------------------------------------------------------------------
with tab_individu:
    kol_form, kol_hasil = st.columns([2, 1], gap="large")

    with kol_form:
        st.markdown('<span class="layout-utama"></span>', unsafe_allow_html=True)
        st.subheader("Profil mahasiswa")
        st.write(
            "Isi data yang tersedia di awal semester 2. Data semester 2 seperti "
            "`gpa_semester_2` tidak dipakai karena belum tersedia saat prediksi."
        )

        with st.form("form_prediksi", border=False):
            bagian_profil = st.container()
            bagian_akademik = st.container()
            bagian_perilaku = st.container()

            with bagian_profil:
                st.markdown("#### Profil & admisi")
                st.caption(
                    "Data diri dan latar belakang mahasiswa. Arahkan kursor ke ikon "
                    "tanda tanya (?) di samping kolom untuk melihat penjelasannya."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input(
                        "Usia *", help="Usia mahasiswa saat ini, dalam tahun.", min_value=15, max_value=60, value=None, step=1,
                        placeholder="Isi usia (15-60)",
                    )
                    gender = st.segmented_control(
                        "Jenis kelamin *", GENDER, help="Pilih sesuai data di sistem akademik.", default=None,
                        format_func=lambda x: GENDER_LABEL[x],
                    )
                    nationality = st.selectbox(
                        "Kewarganegaraan *", NATIONALITY, help="Negara asal mahasiswa menurut dokumen kependudukan.", index=None,
                        placeholder="Pilih kewarganegaraan",
                    )
                    first_generation = st.checkbox("Mahasiswa generasi pertama", help="Centang jika mahasiswa adalah anggota keluarga pertama yang kuliah di perguruan tinggi (orang tua belum ada yang kuliah).")
                    disability_status = st.checkbox("Memiliki disabilitas", help="Centang jika mahasiswa tercatat memiliki disabilitas fisik, sensorik, atau kognitif.")

                with c2:
                    income = st.segmented_control(
                        "Pendapatan keluarga *", INCOME, help="Perkiraan kelompok penghasilan keluarga: Rendah, Menengah, atau Tinggi.", default=None,
                        format_func=lambda x: INCOME_LABEL[x],
                    )
                    tuition_pilihan = st.segmented_control(
                        "Uang kuliah lancar *", ["Ya", "Tidak"], help="Pilih Ya jika pembayaran uang kuliah tidak ada tunggakan.", default=None,
                    )
                    scholarship_holder = st.checkbox("Penerima beasiswa", help="Centang jika mahasiswa sedang menerima beasiswa.")
                    part_time_job = st.checkbox("Bekerja paruh waktu", help="Centang jika mahasiswa punya pekerjaan di luar kuliah. Jumlah jamnya diisi di tab Kebiasaan Belajar.")
                    international_student = st.checkbox("Mahasiswa internasional", help="Centang jika mahasiswa berasal dari luar negeri dan kuliah sebagai mahasiswa asing.")

                with c3:
                    course_type = st.selectbox(
                        "Bidang studi *", COURSE_TYPE, help="Rumpun program studi yang diambil.", index=None,
                        placeholder="Pilih bidang studi",
                        format_func=lambda x: COURSE_LABEL[x],
                    )
                    study_mode = st.segmented_control(
                        "Mode kuliah *", STUDY_MODE, help="Penuh waktu: kuliah tatap muka penuh. Paruh waktu: beban kuliah dikurangi. Daring: kuliah jarak jauh.", default=None,
                        format_func=lambda x: STUDY_LABEL[x],
                    )
                    nilai_tes_masuk = st.slider(
                        "Nilai tes masuk kampus (0-100)", min_value=0.0,
                        max_value=100.0, value=65.0, step=0.5, format="%.1f",
                        help="Nilai seleksi masuk kampus dalam skala 0 sampai 100. "
                             "Jika masuk lewat jalur tanpa tes (misalnya jalur rapor), "
                             "isi perkiraan nilai seleksinya. Geser slider, atau klik "
                             "lalu gunakan tombol panah keyboard.",
                    )
                    st.caption(f"Kategori: {label_nilai(nilai_tes_masuk)}")
                    nilai_sekolah = st.slider(
                        "Rata-rata nilai ijazah/rapor SMA atau SMK (0-100)",
                        min_value=0.0, max_value=100.0, value=65.0, step=0.5,
                        format="%.1f",
                        help="Rata-rata nilai pada ijazah atau rapor sekolah "
                             "menengah, dalam skala 0 sampai 100.",
                    )
                    st.caption(f"Kategori: {label_nilai(nilai_sekolah)}")
                    # Model dilatih pada skala 0-200, jadi nilai 0-100 dikonversi.
                    admission_grade = nilai_tes_masuk * 2
                    previous_qualification_grade = nilai_sekolah * 2

            with bagian_akademik:
                st.markdown("#### Capaian semester 1")
                st.caption(
                    "Hasil belajar mahasiswa selama semester 1. Arahkan kursor ke ikon "
                    "tanda tanya (?) untuk contoh pengisian."
                )
                c1, c2 = st.columns(2)
                with c1:
                    units_enrolled = st.number_input(
                        "Mata kuliah yang diambil *", help="Jumlah mata kuliah yang diambil pada semester 1.", min_value=1, max_value=30,
                        value=None, step=1, placeholder="Isi jumlah (1-30)",
                    )
                    units_failed = st.number_input(
                        "Mata kuliah yang gagal *", help="Jumlah mata kuliah semester 1 yang tidak lulus. Tidak boleh lebih besar dari jumlah yang diambil.", min_value=0, max_value=30,
                        value=None, step=1, placeholder="Isi jumlah (0-30)",
                    )
                    gpa_semester_1 = st.number_input(
                        "IPK semester 1 *", help="Indeks Prestasi pada semester 1, skala 0 sampai 4. Contoh: 3,25.", min_value=0.0, max_value=4.0, value=None,
                        step=0.01, format="%.2f", placeholder="Isi IPK (0-4)",
                    )
                with c2:
                    attendance_rate = st.number_input(
                        "Tingkat kehadiran (%) *", help="Persentase pertemuan kuliah yang dihadiri selama semester 1. Contoh: hadir 12 dari 14 pertemuan kira-kira 86%.", min_value=0.0, max_value=100.0,
                        value=None, step=0.1, placeholder="Isi persentase (0-100)",
                    )
                    assignments_submitted_rate = st.number_input(
                        "Tugas terkumpul (%) *", help="Persentase tugas yang dikumpulkan dari seluruh tugas semester 1. Contoh: 9 dari 10 tugas berarti 90%.", min_value=0.0, max_value=100.0,
                        value=None, step=0.1, placeholder="Isi persentase (0-100)",
                    )
                    missed_deadlines = st.number_input(
                        "Tenggat yang terlewat *", help="Berapa kali mahasiswa terlambat atau tidak mengumpulkan tugas sesuai batas waktu pada semester 1.", min_value=0, max_value=30,
                        value=None, step=1, placeholder="Isi jumlah (0-30)",
                    )
                st.caption(
                    "Fitur turunan dihitung otomatis: rasio mata kuliah gagal, "
                    "skor keterlibatan, dan jumlah indikator risiko."
                )

            with bagian_perilaku:
                st.markdown("#### Kebiasaan belajar")
                st.caption(
                    "Kebiasaan sehari-hari dan kondisi mahasiswa. Isi perkiraan terbaik "
                    "jika tidak tahu angka pastinya."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    library_visits = st.number_input(
                        "Kunjungan perpustakaan per bulan *", help="Rata-rata berapa kali mahasiswa ke perpustakaan dalam sebulan. Isi 0 jika hampir tidak pernah.", min_value=0, max_value=60,
                        value=None, step=1, placeholder="Isi jumlah (0-60)",
                    )
                    lms_logins = st.number_input(
                        "Berapa kali buka e-learning kampus per minggu *",
                        help="Berapa kali dalam seminggu mahasiswa membuka situs atau "
                             "aplikasi belajar online kampus, tempat dosen membagikan "
                             "materi dan tugas (misalnya e-learning kampus, Moodle, atau "
                             "Google Classroom). Contoh: buka hampir setiap hari berarti "
                             "sekitar 7 kali. Isi 0 jika hampir tidak pernah.",
                        min_value=0.0, max_value=60.0, value=None, step=1.0,
                        placeholder="Isi jumlah kali per minggu (0-60)",
                    )
                    weekly_work_hours = st.number_input(
                        "Jam kerja per minggu *", help="Total jam bekerja dalam seminggu. Isi 0 jika tidak bekerja.", min_value=0, max_value=80,
                        value=None, step=1, placeholder="Isi jam (0 jika tidak bekerja)",
                    )
                    extracurricular = st.checkbox("Mengikuti ekstrakurikuler", help="Centang jika mahasiswa aktif di UKM, organisasi, atau kegiatan di luar perkuliahan.")
                with c2:
                    stress_level = st.number_input(
                        "Tingkat stres (1-10) *", help="Penilaian diri: 1 berarti hampir tidak stres, 10 berarti sangat stres.", min_value=1, max_value=10,
                        value=None, step=1, placeholder="1 = sangat rendah, 10 = sangat tinggi",
                    )
                    satisfaction_score = st.number_input(
                        "Kepuasan terhadap perkuliahan (1-10) *", help="Penilaian diri: 1 berarti sangat tidak puas dengan kuliahnya, 10 berarti sangat puas.", min_value=1, max_value=10,
                        value=None, step=1, placeholder="1 = sangat tidak puas, 10 = sangat puas",
                    )
                    mental_health_support = st.checkbox(
                        "Mendapat dukungan kesehatan mental", help="Centang jika mahasiswa mendapat layanan konseling atau pendampingan kesehatan mental."
                    )
                with c3:
                    commute_time = st.number_input(
                        "Waktu perjalanan ke kampus (menit) *", help="Lama perjalanan sekali jalan dari tempat tinggal ke kampus, dalam menit.", min_value=0, max_value=240,
                        value=None, step=1, placeholder="Isi menit (0-240)",
                    )
                    sleep_hours = st.number_input(
                        "Jam tidur per hari *", help="Rata-rata jam tidur dalam sehari. Contoh: 6,5 untuk enam setengah jam.", min_value=0.0, max_value=16.0,
                        value=None, step=0.1, placeholder="Isi jam (0-16)",
                    )

                st.caption(
                    "Semua kolom bertanda * wajib diisi sebelum "
                    "menghitung risiko."
                )
                dikirim = st.form_submit_button(
                    "Hitung risiko dropout", type="primary", width="stretch"
                )


        if dikirim:
            wajib = {
                "Usia": age,
                "Jenis kelamin": gender,
                "Kewarganegaraan": nationality,
                "Pendapatan keluarga": income,
                "Uang kuliah lancar": tuition_pilihan,
                "Bidang studi": course_type,
                "Mode kuliah": study_mode,
                "Mata kuliah yang diambil": units_enrolled,
                "Mata kuliah yang gagal": units_failed,
                "IPK semester 1": gpa_semester_1,
                "Tingkat kehadiran": attendance_rate,
                "Tugas terkumpul": assignments_submitted_rate,
                "Tenggat yang terlewat": missed_deadlines,
                "Kunjungan perpustakaan": library_visits,
                "Frekuensi buka e-learning": lms_logins,
                "Jam kerja per minggu": weekly_work_hours,
                "Tingkat stres": stress_level,
                "Kepuasan terhadap perkuliahan": satisfaction_score,
                "Waktu perjalanan": commute_time,
                "Jam tidur": sleep_hours,
            }
            kosong = [nama for nama, nilai in wajib.items() if nilai is None]
            tuition_up_to_date = tuition_pilihan == "Ya"

            if kosong:
                st.session_state.pop("hasil_individu", None)
                st.error(
                    "Lengkapi dulu kolom berikut sebelum menghitung: "
                    + ", ".join(kosong)
                    + "."
                )
            elif units_failed > units_enrolled:
                st.error(
                    "Jumlah mata kuliah gagal tidak boleh melebihi jumlah mata kuliah "
                    "yang diambil. Periksa kembali data semester 1."
                )
            else:
                data_input = {
                    "age": age,
                    "gender": gender,
                    "nationality": nationality,
                    "first_generation": int(first_generation),
                    "disability_status": int(disability_status),
                    "family_income_level": income,
                    "scholarship_holder": int(scholarship_holder),
                    "tuition_up_to_date": int(tuition_up_to_date),
                    "part_time_job": int(part_time_job),
                    "weekly_work_hours": weekly_work_hours,
                    "course_type": course_type,
                    "study_mode": study_mode,
                    "international_student": int(international_student),
                    "admission_grade": admission_grade,
                    "previous_qualification_grade": previous_qualification_grade,
                    "units_enrolled": units_enrolled,
                    "units_failed": units_failed,
                    "gpa_semester_1": gpa_semester_1,
                    "attendance_rate": attendance_rate,
                    "assignments_submitted_rate": assignments_submitted_rate,
                    "library_visits_per_month": library_visits,
                    "lms_logins_per_week": lms_logins,
                    "extracurricular_activities": int(extracurricular),
                    "missed_deadlines": missed_deadlines,
                    "stress_level": stress_level,
                    "mental_health_support": int(mental_health_support),
                    "satisfaction_score": satisfaction_score,
                    "commute_time_mins": commute_time,
                    "sleep_hours": sleep_hours,
                }
                with st.spinner("Menghitung probabilitas ..."):
                    fitur = siapkan_fitur(data_input)
                    proba = float(MODEL.predict_proba(fitur)[0, 1])

                st.session_state["hasil_individu"] = {
                    "proba": proba,
                    "prediksi": int(proba >= THRESHOLD),
                    "data": data_input,
                }

    with kol_hasil:
        hasil = st.session_state.get("hasil_individu")
        if not hasil:
            with st.container():
                st.markdown("#### Hasil prediksi")
                st.caption(
                    "Isi formulir di sebelah kiri, lalu klik Hitung risiko "
                    "dropout. Hasil dan rekomendasi tampil di sini."
                )
        else:
            proba = hasil["proba"]
            prediksi = hasil["prediksi"]
            baris = hasil["data"]
            label_kategori, warna = kategori_risiko(proba)
            faktor = faktor_risiko(baris)

            with st.container():
                st.markdown("#### Hasil prediksi")
                st.markdown(kartu_probabilitas(proba, warna), unsafe_allow_html=True)
                st.badge(label_kategori, color=warna)
                st.markdown(
                    "**Status:** diprediksi **dropout**"
                    if prediksi
                    else "**Status:** diprediksi **bertahan**"
                )
                st.caption(
                    "Probabilitas bukan vonis. Gunakan sebagai bahan "
                    "pertimbangan untuk intervensi."
                )

            with st.container():
                st.markdown("**Faktor yang terdeteksi**")
                if faktor:
                    for f in faktor[:6]:
                        st.markdown(f"- {f}")
                else:
                    st.write("Tidak ada indikator risiko utama yang menonjol.")

            with st.container():
                st.markdown("**Rekomendasi tindak lanjut**")
                for i, saran in enumerate(rekomendasi(proba, faktor), start=1):
                    st.markdown(f"{i}. {saran}")

            with st.expander("Lihat data yang dipakai model"):
                tampil = pd.DataFrame([{
                    "Usia": baris["age"],
                    "IPK semester 1": baris["gpa_semester_1"],
                    "Mata kuliah diambil": baris["units_enrolled"],
                    "Mata kuliah gagal": baris["units_failed"],
                    "Kehadiran (%)": baris["attendance_rate"],
                    "Tugas terkumpul (%)": baris["assignments_submitted_rate"],
                    "Tenggat terlewat": baris["missed_deadlines"],
                    "Stres": baris["stress_level"],
                    "Kepuasan": baris["satisfaction_score"],
                    "Jam tidur": baris["sleep_hours"],
                    "Jam kerja/minggu": baris["weekly_work_hours"],
                }])
                st.dataframe(tampil, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: prediksi batch
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Prediksi banyak mahasiswa sekaligus")
    st.write(
        "Unggah CSV dengan kolom yang sama seperti template. Aplikasi akan "
        "menghitung probabilitas dropout dan mengurutkan mahasiswa dari risiko "
        "tertinggi."
    )

    unduh_template = st.download_button(
        "Unduh template CSV",
        data=TEMPLATE_BATCH.to_csv(index=False).encode("utf-8"),
        file_name="template_prediksi_dropout.csv",
        mime="text/csv",
    )

    berkas = st.file_uploader("Unggah CSV data mahasiswa", type=["csv"])

    if berkas is not None:
        try:
            df_unggah = pd.read_csv(berkas)
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")
            st.stop()

        hilang = [k for k in FITUR_MENTAH if k not in df_unggah.columns]
        if hilang:
            st.error(
                "Kolom berikut belum ada di CSV: "
                + ", ".join(f"`{k}`" for k in hilang)
            )
            st.stop()

        kerja = df_unggah.copy()
        kolom_numerik = [
            k for k in FITUR_MENTAH
            if k not in FITUR_KATEGORIKAL
        ]
        for kolom in kolom_numerik:
            kerja[kolom] = pd.to_numeric(kerja[kolom], errors="coerce")

        if kerja[FITUR_MENTAH].isna().any().any():
            jumlah_na = int(kerja[FITUR_MENTAH].isna().any(axis=1).sum())
            st.error(
                f"Ada {jumlah_na:,} baris dengan nilai kosong atau tidak valid. "
                "Lengkapi data terlebih dahulu; model ini tidak melakukan imputasi."
            )
            st.stop()

        if (kerja["units_failed"] > kerja["units_enrolled"]).any():
            jumlah_salah = int((kerja["units_failed"] > kerja["units_enrolled"]).sum())
            st.error(
                f"Ada {jumlah_salah:,} baris dengan mata kuliah gagal lebih banyak "
                "daripada mata kuliah yang diambil."
            )
            st.stop()

        with st.spinner("Menghitung prediksi ..."):
            fitur_batch = siapkan_fitur_batch(kerja)
            proba_batch = MODEL.predict_proba(fitur_batch)[:, 1]

        hasil_batch = kerja.copy()
        if "person_id" in df_unggah.columns:
            urutan_kolom = ["person_id"] + [
                c for c in hasil_batch.columns if c != "person_id"
            ]
            hasil_batch = hasil_batch[urutan_kolom]
        else:
            hasil_batch.insert(0, "baris", np.arange(1, len(hasil_batch) + 1))

        hasil_batch["probabilitas_dropout"] = proba_batch
        hasil_batch["prediksi"] = np.where(
            proba_batch >= THRESHOLD, "Dropout", "Bertahan"
        )
        hasil_batch["kategori_risiko"] = [
            kategori_risiko(p)[0] for p in proba_batch
        ]
        hasil_batch = hasil_batch.sort_values(
            "probabilitas_dropout", ascending=False
        ).reset_index(drop=True)

        st.divider()
        st.subheader("Ringkasan")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah mahasiswa", f"{len(hasil_batch):,}")
        m2.metric(
            "Diprediksi dropout",
            f"{int((hasil_batch['prediksi'] == 'Dropout').sum()):,}",
        )
        m3.metric(
            "Risiko tinggi",
            f"{int((hasil_batch['kategori_risiko'] == 'Risiko tinggi').sum()):,}",
        )
        m4.metric(
            "Rata-rata probabilitas",
            f"{hasil_batch['probabilitas_dropout'].mean():.1%}",
        )

        kolom_tampil = [
            c for c in [
                "person_id", "baris", "probabilitas_dropout", "prediksi",
                "kategori_risiko", "gpa_semester_1", "units_failed",
                "attendance_rate", "missed_deadlines",
            ] if c in hasil_batch.columns
        ]

        st.subheader("Daftar prioritas intervensi")
        st.caption(
            "Diurutkan dari probabilitas tertinggi. Tabel menampilkan maksimal "
            "500 baris; unduhan berisi seluruh data."
        )
        st.dataframe(
            hasil_batch[kolom_tampil].head(500),
            hide_index=True,
            width="stretch",
            column_config={
                "probabilitas_dropout": st.column_config.ProgressColumn(
                    "Probabilitas dropout",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
                "prediksi": st.column_config.TextColumn("Prediksi"),
                "kategori_risiko": st.column_config.TextColumn("Kategori risiko"),
                "gpa_semester_1": st.column_config.NumberColumn(
                    "IPK semester 1", format="%.2f"
                ),
                "attendance_rate": st.column_config.NumberColumn(
                    "Kehadiran (%)", format="%.1f"
                ),
            },
            alt="Tabel prioritas mahasiswa berisiko dropout",
        )

        st.download_button(
            "Unduh hasil prediksi lengkap",
            data=hasil_batch.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_dropout.csv",
            mime="text/csv",
            type="primary",
        )

# ---------------------------------------------------------------------------
# Tab 3: tentang model
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Tentang model")
    st.write(
        f"Aplikasi memakai **{NAMA_MODEL}** yang dilatih pada 80.000 baris data "
        "dan dievaluasi pada 20.000 baris holdout test. Data test tidak dipakai "
        "untuk tuning maupun pemilihan threshold."
    )

    st.markdown("#### Performa pada data test")
    metrik_tampil = METRIK_THRESHOLD or METRIK_DEFAULT
    kolom_metrik = st.columns(5)
    for kolom, nama_metrik, label in zip(
        kolom_metrik,
        ["F1", "Recall", "Precision", "ROC-AUC", "PR-AUC"],
        ["F1", "Recall dropout", "Precision", "ROC-AUC", "PR-AUC"],
    ):
        nilai = metrik_tampil.get(nama_metrik)
        kolom.metric(
            label,
            f"{nilai:.3f}" if isinstance(nilai, (int, float)) else "-",
            border=True,
        )

    st.caption(
        f"Metrik dihitung pada threshold {THRESHOLD:.2f}, yang dipilih dari "
        "out-of-fold prediction pada data training. Accuracy tidak dijadikan "
        "metrik utama karena kelas dropout hanya 32%."
    )

    st.markdown("#### Fitur paling berpengaruh")
    if hasattr(MODEL.named_steps["model"], "coef_"):
        st.caption(
            "Logistic Regression: nilai positif menaikkan risiko, nilai negatif "
            "menurunkan risiko. Besaran koefisien dipengaruhi skala fitur dan "
            "korelasi antarfitur."
        )
    else:
        st.caption(
            "Feature importance menunjukkan kontribusi prediktif, bukan hubungan "
            "kausal."
        )

    tabel_importance = ambil_importance(MODEL, tuple(FITUR_KATEGORIKAL))
    st.altair_chart(
        grafik_importance(tabel_importance),
        width="stretch",
        alt="Grafik fitur paling berpengaruh terhadap risiko dropout",
    )

    with st.expander("Bagaimana model bekerja?"):
        st.markdown(
            """
            1. **Input**: profil, nilai admisi, capaian semester 1, dan perilaku belajar.
            2. **Feature engineering**: `fail_rate`, `engagement_score`, dan
               `risk_flag_count` dihitung dari data semester 1.
            3. **Preprocessing**: fitur miring ditransformasi `log1p` lalu
               distandardisasi, fitur kategorikal di-one-hot. Semua transformasi
               dipelajari hanya dari data training.
            4. **Prediksi**: model menghasilkan probabilitas dropout. Jika
               probabilitas melewati threshold, mahasiswa ditandai berisiko.
            5. **Tindak lanjut**: hasil dipakai untuk memprioritaskan intervensi,
               bukan untuk memberi label permanen kepada mahasiswa.
            """
        )

    st.warning(
        "**Keterbatasan.** Dataset memiliki indikasi kuat sebagai data sintetis "
        "(`gpa_semester_2 < gpa_semester_1` pada 90,4% baris). Model belum "
        "divalidasi secara temporal dan tidak mengukur motivasi, kondisi keluarga, "
        "atau kesehatan mental secara menyeluruh. Jangan memakai prediksi sebagai "
        "satu-satunya dasar keputusan akademik."
    )
