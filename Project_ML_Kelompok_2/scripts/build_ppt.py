"""Bangun PPT laporan project dari hasil model dan gambar yang sudah dibuat."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "reports" / "figures"
OUT = BASE / "docs" / "Presentasi_Project_ML_Kelompok_2.pptx"
OUT.parent.mkdir(parents=True, exist_ok=True)

BG = RGBColor(0xFB, 0xF8, 0xF3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1F, 0x29, 0x37)
ACCENT = RGBColor(0xA9, 0x32, 0x26)
ACCENT_SOFT = RGBColor(0xF4, 0xE3, 0xE0)
MUTED = RGBColor(0x6B, 0x72, 0x80)
GREEN = RGBColor(0x1E, 0x84, 0x49)
BLUE = RGBColor(0x2E, 0x86, 0xC1)
LINE = RGBColor(0xE5, 0xE1, 0xDA)
PALE_BLUE = RGBColor(0xE8, 0xF1, 0xF8)
PALE_GREEN = RGBColor(0xE8, 0xF5, 0xEC)

FONT = "Segoe UI"
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def add_text(slide, text, x, y, w, h, size=14, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT,
             line_spacing=1.0, space_after=2):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, baris in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = baris
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def add_bullets(slide, items, x, y, w, h, size=15, color=DARK, gap=8,
                bullet_color=ACCENT, line_spacing=1.05):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(gap)
        p.level = 0
        run = p.add_run()
        run.text = "•  "
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.color.rgb = bullet_color
        run.font.bold = True
        run2 = p.add_run()
        run2.text = item
        run2.font.name = FONT
        run2.font.size = Pt(size)
        run2.font.color.rgb = color
    return box


def add_slide(title, kicker=None):
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG

    # garis aksen atas
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.07)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    if kicker:
        add_text(slide, kicker.upper(), 0.55, 0.35, 9.0, 0.3,
                 size=10, bold=True, color=ACCENT)
        y_title = 0.62
    else:
        y_title = 0.48

    add_text(slide, title, 0.55, y_title, 12.2, 0.65,
             size=27, bold=True, color=DARK)

    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(y_title + 0.72),
        Inches(1.10), Inches(0.045)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()

    # footer
    add_text(slide, "Project Akhir Pembelajaran Mesin  •  Kelompok 2",
             0.55, 7.12, 10.0, 0.22, size=8.5, color=MUTED)
    add_text(slide, f"{len(prs.slides):02d}", 12.3, 7.12, 0.5, 0.22,
             size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)
    return slide


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_card(slide, x, y, w, h, title, body, accent=ACCENT,
             fill=WHITE, title_size=14, body_size=11, body_color=MUTED):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = LINE
    shape.line.width = Pt(0.8)
    shape.adjustments[0] = 0.06

    stripe = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.12),
        Inches(0.06), Inches(h - 0.24)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background()

    add_text(slide, title, x + 0.20, y + 0.13, w - 0.32, 0.30,
             size=title_size, bold=True, color=DARK)
    add_text(slide, body, x + 0.20, y + 0.50, w - 0.32, h - 0.60,
             size=body_size, color=body_color, line_spacing=1.05)


def add_stat(slide, x, y, w, h, angka, label, accent=ACCENT, fill=WHITE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = LINE
    shape.line.width = Pt(0.8)
    shape.adjustments[0] = 0.06

    add_text(slide, angka, x + 0.16, y + 0.12, w - 0.32, 0.55,
             size=30, bold=True, color=accent)
    add_text(slide, label, x + 0.16, y + 0.70, w - 0.32, h - 0.76,
             size=10, color=MUTED, line_spacing=1.0)


def add_image_fit(slide, path, x, y, max_w, max_h):
    iw, ih = Image.open(path).size
    skala = min(max_w / iw, max_h / ih)
    w = iw * skala
    h = ih * skala
    left = x + (max_w - w) / 2
    top = y + (max_h - h) / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                             width=Inches(w), height=Inches(h))
    return left, top, w, h


def add_table(slide, data, x, y, w, h, col_widths, font_size=10,
              header_fill=ACCENT, header_color=WHITE, row_height=0.32,
              header_height=None, first_col_bold=False, align_center=False):
    baris = len(data)
    kolom = len(data[0])
    shape = slide.shapes.add_table(baris, kolom, Inches(x), Inches(y),
                                   Inches(w), Inches(h))
    table = shape.table
    for i, cw in enumerate(col_widths):
        table.columns[i].width = Inches(cw)
    tinggi_header = header_height if header_height is not None else max(row_height, 0.36)
    for i in range(baris):
        table.rows[i].height = Inches(tinggi_header if i == 0 else row_height)
        for j in range(kolom):
            cell = table.cell(i, j)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else BG
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if (align_center or i == 0 or j == 0) else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(data[i][j])
            run.font.name = FONT
            run.font.size = Pt(font_size)
            run.font.bold = bool(i == 0 or (first_col_bold and j == 0))
            run.font.color.rgb = header_color if i == 0 else DARK
    return table


def add_arrow(slide, x, y, w=0.34, h=0.30, color=ACCENT):
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
import json

with open(BASE / "reports" / "ringkasan_model.json", encoding="utf-8") as f:
    R = json.load(f)

cv = pd.DataFrame(R["baseline_cv"]).T
test = pd.DataFrame(R["test_default"]).T


# ---------------------------------------------------------------------------
# 1. Judul
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(BLANK)
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = BG

band = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.22)
)
band.fill.solid()
band.fill.fore_color.rgb = ACCENT
band.line.fill.background()

add_text(slide, "PROJECT AKHIR MATA KULIAH PEMBELAJARAN MESIN",
         0.75, 0.90, 11.5, 0.30, size=11, bold=True, color=ACCENT)
add_text(slide, "Deteksi Risiko Dropout\nMahasiswa pada Awal Semester 2",
         0.75, 1.35, 11.8, 1.55, size=39, bold=True, color=DARK,
         line_spacing=0.98)
add_text(slide,
         "Memanfaatkan data akademik semester 1 dan perilaku belajar untuk "
         "memprioritaskan intervensi.",
         0.78, 3.12, 10.7, 0.60, size=16, color=MUTED)

line = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, Inches(0.78), Inches(3.95), Inches(1.3), Inches(0.05)
)
line.fill.solid()
line.fill.fore_color.rgb = ACCENT
line.line.fill.background()

add_text(slide, "Kelompok 2  •  Nama Anggota 1  •  Nama Anggota 2  •  Nama Anggota 3",
         0.78, 4.15, 11.8, 0.30, size=13, bold=True, color=DARK)
add_text(slide, "NIM ...   •   NIM ...   •   NIM ...",
         0.78, 4.52, 8.5, 0.28, size=11.5, color=DARK)
add_text(slide, "Kelas: ...   |   2026", 0.78, 4.88, 8.0, 0.28,
         size=11.5, color=MUTED)

add_stat(slide, 9.55, 4.95, 2.95, 1.25, "32%", "mahasiswa dropout\ndalam dataset", ACCENT)
add_stat(slide, 9.55, 6.30, 2.95, 0.62, "100.000", "baris data", GREEN)
add_notes(slide,
          "Perkenalkan anggota kelompok, judul, dan konteks singkat. Sebutkan "
          "bahwa project berfokus pada prediksi awal semester 2 dan aplikasi "
          "untuk membantu prioritas intervensi.")

# ---------------------------------------------------------------------------
# 2. Latar belakang
# ---------------------------------------------------------------------------
slide = add_slide("Latar Belakang & Masalah", "01  •  Konteks")
add_text(slide,
         "Dropout bukan sekadar angka retensi. Di baliknya ada mahasiswa yang "
         "kehilangan kesempatan belajar dan institusi yang kehilangan sumber daya.",
         0.55, 1.55, 7.35, 1.05, size=16, color=DARK, line_spacing=1.08)

add_bullets(slide, [
    "32% mahasiswa dalam dataset dropout — cukup besar untuk membutuhkan skrining sistematis.",
    "Sinyal risiko sudah terlihat di semester 1: IPK, mata kuliah gagal, kehadiran, tenggat, dan keterlibatan LMS.",
    "Intervensi akan lebih efektif jika dilakukan sebelum semester 2 berjalan, bukan setelah nilai akhir keluar.",
    "Pertanyaan project: bisakah model menandai risiko menggunakan data yang tersedia di awal semester 2?",
], 0.55, 2.85, 7.55, 3.5, size=14, gap=10)

add_stat(slide, 8.55, 1.60, 4.15, 1.18, "32%", "proporsi dropout (32.004 dari 100.000)", ACCENT)
add_stat(slide, 8.55, 2.95, 4.15, 1.18, "68 : 32", "rasio bertahan : dropout", GREEN)
add_stat(slide, 8.55, 4.30, 4.15, 1.18, "0", "missing value dan duplikat", BLUE)

add_text(slide,
         "Tantangan utama:  gpa_semester_2 adalah leakage  •  kelas tidak seimbang  •  "
         "accuracy saja menyesatkan",
         0.55, 6.45, 12.2, 0.32, size=11, bold=True, color=ACCENT)
add_notes(slide,
          "Jelaskan mengapa masalah ini penting. Tekankan bahwa data semester 2 "
          "belum tersedia saat prediksi, sehingga gpa_semester_2 harus dibuang. "
          "Sebutkan juga tantangan ketidakseimbangan kelas dan pemilihan metrik.")

# ---------------------------------------------------------------------------
# 3. Tujuan & kontribusi
# ---------------------------------------------------------------------------
slide = add_slide("Tujuan & Kontribusi", "02  •  Arah project")
tujuan = [
    ("Model early warning", "Memprediksi probabilitas dropout pada awal semester 2 dengan data yang benar-benar tersedia."),
    ("Bandingkan 3 algoritma", "Logistic Regression, Random Forest, dan XGBoost diuji dengan protokol yang sama."),
    ("Jelaskan faktor risiko", "Mengidentifikasi fitur paling berpengaruh dan membaca arah hubungannya."),
    ("Aplikasi siap pakai", "Streamlit untuk prediksi individu, prediksi batch, dan rekomendasi tindak lanjut."),
]
for i, (judul, isi) in enumerate(tujuan):
    x = 0.55 + (i % 2) * 6.15
    y = 1.60 + (i // 2) * 1.62
    add_card(slide, x, y, 5.85, 1.38, judul, isi, accent=ACCENT if i % 2 == 0 else BLUE,
             title_size=15, body_size=11)

add_text(slide, "Kontribusi project", 0.55, 5.20, 3.0, 0.30,
         size=12, bold=True, color=ACCENT)
add_bullets(slide, [
    "Skenario awal semester 2 dengan kontrol leakage yang eksplisit.",
    "Evaluasi berfokus pada F1, recall dropout, dan PR-AUC — bukan hanya accuracy.",
    "Threshold dipilih dari out-of-fold prediction, bukan dari data test.",
    "Aplikasi tidak hanya menampilkan probabilitas, tetapi juga faktor risiko dan rekomendasi.",
], 0.55, 5.58, 12.1, 1.35, size=11.5, gap=3)
add_notes(slide,
          "Sampaikan empat tujuan utama dan kontribusi. Poin pembeda: skenario "
          "awal semester 2, kontrol leakage, threshold dari data training, dan "
          "aplikasi yang memberi rekomendasi, bukan hanya angka.")

# ---------------------------------------------------------------------------
# 4. Studi literatur 1
# ---------------------------------------------------------------------------
slide = add_slide("Studi Literatur (1/2)", "03  •  Dasar penelitian")
data = [
    ["No.", "Penulis & tahun", "Fokus & dataset", "Metode", "Hasil utama"],
    ["1", "Mduma, Kalegele, & Machuve (2019)",
     "Survei ML untuk prediksi dropout; literatur jurnal, buku, studi kasus.",
     "Decision tree, naive Bayes, neural network, ensemble.",
     "Riset terkonsentrasi di negara maju; data tingkat sekolah dan keterbatasan sumber daya perlu diperhatikan."],
    ["2", "Delen (2011)",
     "Attrition mahasiswa tahun pertama; data institusional 8 tahun.",
     "ANN, decision tree, logistic regression.",
     "ANN terbaik dengan akurasi sekitar 81%; variabel pendidikan dan keuangan paling penting."],
    ["3", "Hoffait & Schyns (2017)",
     "Deteksi dini mahasiswa berpotensi kesulitan; data pendaftaran Universitas Liège.",
     "Random forest, logistic regression, dan ANN; adaptasi algoritma untuk kelas yang paling penting.",
     "Algoritma yang disesuaikan membantu menemukan mahasiswa berisiko lebih akurat untuk remediasi dini."],
]
add_table(slide, data, 0.55, 1.55, 12.2, 4.9,
          [0.45, 2.05, 3.15, 2.65, 3.90], font_size=9.2,
          row_height=1.35, header_height=0.45)
add_text(slide, "Benang merah: data akademik + keterlibatan belajar adalah sinyal utama; "
                "evaluasi perlu melampaui accuracy.",
         0.55, 6.60, 12.2, 0.32, size=10.5, bold=True, color=ACCENT)
add_notes(slide,
          "Ringkas tiga literatur pertama. Soroti bahwa ANN/ensemble sering unggul, "
          "tetapi data dan metrik berbeda-beda. Delen menekankan faktor pendidikan "
          "dan keuangan; Hoffait menekankan deteksi dini dan kelas berisiko.")

# ---------------------------------------------------------------------------
# 5. Studi literatur 2
# ---------------------------------------------------------------------------
slide = add_slide("Studi Literatur (2/2)", "03  •  Dasar penelitian")
data = [
    ["No.", "Penulis & tahun", "Fokus & dataset", "Metode", "Hasil utama"],
    ["4", "Gray & Perkins (2019)",
     "Keterlibatan awal dan kehadiran; data perkuliahan.",
     "Learning analytics + machine learning.",
     "Mahasiswa dapat diidentifikasi sejak minggu ke-3 dengan akurasi sekitar 97%."],
    ["5", "Tomasevic, Gvozdenovic, & Vranes (2020)",
     "Prediksi performa ujian dan risiko gagal; data mata kuliah.",
     "Perbandingan supervised data mining, termasuk ANN dan ensemble.",
     "ANN terbaik; data keterlibatan dan performa masa lalu lebih informatif daripada demografi."],
    ["6", "Sghir, Adadi, & Lahmer (2023)",
     "Systematic review predictive learning analytics 2012–2022.",
     "PRISMA; review model ML/DL, fitur, preprocessing, dan metrik.",
     "Data keterlibatan penting; interpretabilitas dan validasi masih menjadi celah."],
]
add_table(slide, data, 0.55, 1.55, 12.2, 4.9,
          [0.45, 2.15, 3.05, 2.65, 3.90], font_size=9.2,
          row_height=1.35, header_height=0.45)

add_text(slide, "Celah yang dijawab project", 0.55, 6.50, 3.4, 0.30,
         size=11, bold=True, color=ACCENT)
add_text(slide, "Prediksi dropout tingkat mahasiswa pada awal semester 2, "
                "perbandingan tiga model, threshold tuning, dan aplikasi yang dapat "
                "dijelaskan ke pengguna.",
         3.70, 6.47, 9.0, 0.50, size=10.5, color=DARK)
add_notes(slide,
          "Lanjutkan tiga literatur berikutnya. Gray & Perkins menunjukkan nilai "
          "data keterlibatan awal. Tomasevic menunjukkan demografi kurang penting "
          "dibanding keterlibatan. Sghir menunjukkan celah interpretabilitas dan "
          "validasi, yang menjadi dasar project ini.")

# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------
slide = add_slide("Dataset", "04  •  Bahan analisis")
add_bullets(slide, [
    "Sumber: Kaggle — Student Academic Performance & Dropout Risk (Mohan Krishna Thalla), lisensi CC0.",
    "100.000 baris × 34 kolom; 0 missing value; 0 duplikat.",
    "Target: dropout (1 = dropout, 0 = bertahan) dengan komposisi 68% : 32%.",
    "Skenario: hanya data sampai akhir semester 1 yang boleh dipakai.",
], 0.55, 1.65, 7.25, 2.4, size=13, gap=9)

add_image_fit(slide, FIG / "target.png", 8.05, 1.55, 4.6, 3.4)

add_text(slide, "Kelompok fitur", 0.55, 4.15, 4.0, 0.30,
         size=12, bold=True, color=ACCENT)
for i, (judul, isi, warna) in enumerate([
    ("Profil & admisi", "Usia, gender, kewarganegaraan, pendapatan, beasiswa, nilai admisi.", BLUE),
    ("Capaian semester 1", "Mata kuliah diambil/gagal, IPK semester 1, kehadiran, tugas.", ACCENT),
    ("Perilaku belajar", "LMS, perpustakaan, tenggat, stres, kepuasan, tidur, kerja.", GREEN),
]):
    add_card(slide, 0.55 + i * 4.10, 4.55, 3.80, 1.20, judul, isi,
             accent=warna, title_size=12, body_size=9.5)

add_text(slide,
         "Dibuang: person_id (identifier)  •  semester (tidak sesuai skenario)  •  "
         "gpa_semester_2 (leakage)",
         0.55, 6.40, 12.2, 0.30, size=10.5, bold=True, color=ACCENT)
add_notes(slide,
          "Jelaskan sumber dataset dan karakteristiknya. Tekankan alasan membuang "
          "gpa_semester_2: pada awal semester 2 nilainya belum ada. Sebutkan tiga "
          "kelompok fitur yang dipakai.")

# ---------------------------------------------------------------------------
# 7. EDA
# ---------------------------------------------------------------------------
slide = add_slide("EDA: Sinyal Risiko Sudah Terlihat", "05  •  Eksplorasi data")
add_image_fit(slide, FIG / "eda.png", 0.55, 1.45, 12.2, 4.05)
add_text(slide,
         "IPK < 2,0 → dropout 73,9%   •   IPK 3,5–4,0 → 2,0%   •   gagal ≥5 mata kuliah → 66,3%   •   "
         "pendapatan rendah 42,9% vs tinggi 20,0%",
         0.55, 5.68, 12.2, 0.38, size=10.5, bold=True, color=ACCENT)
add_bullets(slide, [
    "Pola bertingkat dan konsisten: semakin rendah capaian akademik, semakin tinggi risiko.",
    "Korelasi terkuat: IPK semester 1 (−0,569) dan jumlah mata kuliah gagal (+0,467).",
    "Keterlibatan (kehadiran, tugas, LMS, tenggat) memberi sinyal tambahan yang dapat diintervensi.",
], 0.55, 6.10, 12.1, 0.85, size=10, gap=1)
add_notes(slide,
          "Bacakan dua grafik: risiko turun tajam saat IPK naik, dan melonjak saat "
          "jumlah mata kuliah gagal bertambah. Tambahkan temuan pendapatan. Tekankan "
          "bahwa ini asosiasi, bukan kausalitas.")

# ---------------------------------------------------------------------------
# 8. Preprocessing
# ---------------------------------------------------------------------------
slide = add_slide("Preprocessing & Feature Engineering", "06  •  Menjaga integritas evaluasi")
langkah = [
    ("1", "Buang leakage", "person_id, semester,\ngpa_semester_2"),
    ("2", "Feature engineering", "fail_rate, engagement_score,\nrisk_flag_count"),
    ("3", "Split 80 : 20", "stratified,\nrandom_state=42"),
    ("4", "Pipeline", "log1p + scale;\none-hot kategori"),
    ("5", "Class weight", "balanced /\nscale_pos_weight"),
]
for i, (no, judul, isi) in enumerate(langkah):
    x = 0.55 + i * 2.55
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.62),
        Inches(2.22), Inches(1.36)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = LINE
    add_text(slide, no, x + 0.12, 1.68, 0.35, 0.28, size=12, bold=True, color=ACCENT)
    add_text(slide, judul, x + 0.12, 1.96, 1.95, 0.30, size=11, bold=True)
    add_text(slide, isi, x + 0.12, 2.27, 1.95, 0.60, size=8.8, color=MUTED)
    if i < 4:
        add_arrow(slide, x + 2.27, 2.14, w=0.23, h=0.24)

add_text(slide, "Keputusan yang sengaja dihindari", 0.55, 3.25, 5.8, 0.30,
         size=12, bold=True, color=ACCENT)
data = [
    ["Keputusan", "Alasan"],
    ["Tanpa imputasi", "Tidak ada missing value (0 dari 100.000 baris)."],
    ["Tanpa hapus outlier", "Nilai ekstrem masih wajar secara domain dan bisa menjadi sinyal risiko."],
    ["Tanpa SMOTE", "Rasio 68:32 masih wajar; class_weight lebih sederhana dan menghindari overfitting sintetis."],
]
add_table(slide, data, 0.55, 3.62, 7.55, 2.55,
          [1.75, 5.80], font_size=9.5, row_height=0.62, first_col_bold=True)

add_card(slide, 8.45, 3.58, 4.25, 1.45, "Data test tetap bersih",
         "Split dilakukan sebelum preprocessing, tuning, dan threshold. Data test "
         "baru dipakai sekali pada evaluasi akhir.", accent=GREEN,
         title_size=13, body_size=10)
add_card(slide, 8.45, 5.20, 4.25, 1.45, "Hasil transformasi",
         "32 fitur input → 56 fitur setelah one-hot. Semua transformer di-fit "
         "hanya pada data training.", accent=BLUE,
         title_size=13, body_size=10)
add_notes(slide,
          "Jelaskan alur preprocessing dan alasan setiap keputusan. Tekankan bahwa "
          "semua transformasi berada dalam pipeline dan hanya di-fit pada training. "
          "Data test tidak pernah dipakai untuk memilih fitur, tuning, atau threshold.")

# ---------------------------------------------------------------------------
# 9. Model development
# ---------------------------------------------------------------------------
slide = add_slide("Pengembangan Model", "07  •  Tiga algoritma, satu protokol")
cards = [
    ("Logistic Regression", "Linear, interpretable.\nBaseline kuat untuk data tabular.\nKoefisien menunjukkan arah risiko.", ACCENT),
    ("Random Forest", "Ensemble pohon, menangkap interaksi non-linear.\nLebih tahan outlier.\nFeature importance berbasis impurity.", GREEN),
    ("XGBoost", "Gradient boosting, kuat untuk data tabular.\nMenangani interaksi kompleks.\nscale_pos_weight untuk kelas minoritas.", BLUE),
]
for i, (judul, isi, warna) in enumerate(cards):
    add_card(slide, 0.55 + i * 4.10, 1.58, 3.85, 1.65, judul, isi,
             accent=warna, title_size=15, body_size=9.5)

add_text(slide, "Protokol eksperimen", 0.55, 3.50, 5.5, 0.30,
         size=12, bold=True, color=ACCENT)
add_bullets(slide, [
    "80.000 train / 20.000 test, stratified, random_state=42.",
    "5-fold StratifiedKFold untuk baseline dan pemilihan model.",
    "RandomizedSearchCV (3-fold, scoring F1) pada data training.",
    "Threshold dicari dari out-of-fold prediction, bukan data test.",
], 0.55, 3.88, 6.0, 1.80, size=10.5, gap=4)

data = [
    ["Model", "Parameter terbaik"],
    ["Logistic Regression", "C=10, class_weight=balanced"],
    ["Random Forest", "200 trees, max_depth=None, min_samples_split=10"],
    ["XGBoost", "400 trees, lr=0.03, depth=5, subsample=0.7"],
]
add_table(slide, data, 6.95, 3.55, 5.85, 1.30,
          [1.70, 4.15], font_size=8.8, row_height=0.30, first_col_bold=True)

add_card(slide, 0.55, 5.78, 12.25, 0.78,
         "Mengapa tiga model ini?",
         "Mewakili tiga tingkat kompleksitas: linear, bagging, dan boosting. "
         "Perbandingannya menunjukkan apakah hubungan non-linear benar-benar memberi nilai tambah.",
         accent=ACCENT, title_size=11, body_size=9.5)
add_notes(slide,
          "Jelaskan karakteristik tiga algoritma dan protokol yang sama. Sebutkan "
          "parameter terbaik. Tekankan bahwa data test tidak dipakai untuk tuning, "
          "dan threshold dipilih dari OOF prediction.")

# ---------------------------------------------------------------------------
# 10. Hasil perbandingan
# ---------------------------------------------------------------------------
slide = add_slide("Evaluasi & Perbandingan Model", "08  •  Hasil pada data test")
add_image_fit(slide, FIG / "model_compare.png", 0.45, 1.55, 7.55, 3.95)

data = [["Model", "Acc", "Prec", "Recall", "F1"]]
for nama in ["Logistic Regression", "Random Forest", "XGBoost"]:
    row = test.loc[nama]
    data.append([nama, f"{row['Accuracy']:.3f}", f"{row['Precision']:.3f}",
                 f"{row['Recall']:.3f}", f"{row['F1']:.3f}"])
add_table(slide, data, 8.15, 1.62, 4.60, 1.55,
          [1.55, 0.65, 0.70, 0.80, 0.80], font_size=8.2,
          row_height=0.34, align_center=True)

add_card(slide, 8.15, 3.40, 4.55, 0.98, "F1 CV terbaik",
         "Logistic Regression — 0,7387", accent=ACCENT,
         title_size=11, body_size=10)
add_card(slide, 8.15, 4.52, 4.55, 0.98, "F1 test @0,50",
         "XGBoost — 0,7390", accent=BLUE,
         title_size=11, body_size=10)
add_card(slide, 8.15, 5.64, 4.55, 0.98, "F1 final LR @0,57",
         "0,7431  •  recall 77,4%  •  precision 71,4%", accent=GREEN,
         title_size=11, body_size=10)
add_text(slide,
         "Selisih F1 test antar-model hanya 0,0073 — performa ketiganya kompetitif.",
         0.55, 5.80, 7.3, 0.30, size=10, bold=True, color=MUTED)
add_notes(slide,
          "Bandingkan F1 CV dan F1 test. LR unggul di CV, XGBoost unggul tipis di "
          "test threshold 0,50. Setelah threshold LR dioptimalkan, F1-nya menjadi "
          "0,7431. Tekankan selisih kecil dan pemilihan model tidak hanya dari test.")

# ---------------------------------------------------------------------------
# 11. Threshold dan confusion matrix
# ---------------------------------------------------------------------------
slide = add_slide("Threshold & Confusion Matrix", "09  •  Dari probabilitas ke keputusan")
add_image_fit(slide, FIG / "confusion.png", 0.45, 1.55, 4.55, 3.75)
add_image_fit(slide, FIG / "threshold.png", 5.25, 1.55, 7.50, 3.75)

add_text(slide, "Threshold 0,57 dipilih dari out-of-fold prediction", 0.55, 5.48, 7.0, 0.30,
         size=11, bold=True, color=ACCENT)
add_bullets(slide, [
    "TP 4.955  •  FN 1.446  •  FP 1.980  •  TN 11.619",
    "Recall 77,4%: sekitar tiga dari empat mahasiswa dropout tertangkap.",
    "Precision 71,4%: dari yang ditandai, sekitar tujuh dari sepuluh benar-benar dropout.",
], 0.55, 5.86, 12.15, 0.95, size=10.5, gap=2)
add_notes(slide,
          "Jelaskan trade-off threshold. Di 0,57, recall 77,4% dan precision 71,4%. "
          "False negative 1.446 adalah kelompok yang paling mahal karena tidak "
          "mendapat intervensi. False positive tetap bisa diberi pengecekan ringan.")

# ---------------------------------------------------------------------------
# 12. Feature importance
# ---------------------------------------------------------------------------
slide = add_slide("Faktor yang Paling Berpengaruh", "10  •  Interpretasi model")
add_image_fit(slide, FIG / "importance.png", 0.45, 1.50, 7.75, 4.55)

add_card(slide, 8.45, 1.60, 4.30, 1.25, "IPK semester 1",
         "Koefisien −1,26; fitur paling dominan. IPK rendah menaikkan risiko.",
         accent=BLUE, title_size=12, body_size=9.2)
add_card(slide, 8.45, 2.98, 4.30, 1.25, "Faktor finansial",
         "Pendapatan tinggi dan uang kuliah lancar menurunkan risiko; pendapatan rendah menaikkannya.",
         accent=ACCENT, title_size=12, body_size=9.2)
add_card(slide, 8.45, 4.36, 4.30, 1.25, "Perilaku belajar",
         "Tenggat terlewat, jam kerja, stres, kehadiran, dan LMS ikut berkontribusi.",
         accent=GREEN, title_size=12, body_size=9.2)
add_text(slide,
         "Feature importance = kontribusi prediktif, bukan hubungan kausal.",
         8.45, 5.88, 4.3, 0.45, size=9.5, bold=True, color=MUTED)
add_notes(slide,
          "Bacakan fitur teratas. IPK semester 1 paling dominan, diikuti faktor "
          "finansial dan perilaku. Ingatkan bahwa koefisien dipengaruhi korelasi "
          "antarfitur dan tidak boleh ditafsirkan sebagai sebab-akibat.")

# ---------------------------------------------------------------------------
# 13. Pemilihan model
# ---------------------------------------------------------------------------
slide = add_slide("Mengapa Logistic Regression?", "11  •  Pemilihan model")
add_text(slide,
         "Model dipilih berdasarkan F1 cross-validation, bukan accuracy atau satu "
         "angka test.",
         0.55, 1.58, 7.0, 0.60, size=14, color=MUTED)

alasan = [
    ("F1 CV terbaik", "0,7387 — sedikit di atas XGBoost (0,7379) dan Random Forest (0,7300)."),
    ("Stabil", "Standar deviasi F1 antar-fold 0,0022; gap train–test mendekati nol."),
    ("Interpretable", "Koefisien memberi arah risiko dan memudahkan penjelasan kepada pengguna."),
    ("Threshold-aware", "Setelah threshold 0,57, F1 test 0,7431 — lebih tinggi dari XGBoost di 0,50."),
]
for i, (judul, isi) in enumerate(alasan):
    add_card(slide, 0.55, 2.30 + i * 0.98, 7.15, 0.84, judul, isi,
             accent=[ACCENT, GREEN, BLUE, ACCENT][i],
             title_size=12, body_size=9.5)

add_text(slide, "Perbandingan singkat", 8.05, 1.58, 4.5, 0.30,
         size=11, bold=True, color=ACCENT)
data = [
    ["", "LR", "XGB", "RF"],
    ["F1 CV", "0,739", "0,738", "0,730"],
    ["F1 test", "0,738", "0,739", "0,732"],
    ["Recall", "0,817", "0,813", "0,763"],
    ["Accuracy", "0,814", "0,816", "0,821"],
]
add_table(slide, data, 8.05, 1.95, 4.65, 1.55,
          [1.15, 1.05, 1.05, 1.05], font_size=8.8,
          row_height=0.30, align_center=True)

add_card(slide, 8.05, 3.85, 4.65, 1.55, "Catatan penting",
         "Random Forest punya accuracy tertinggi (0,821), tetapi recall dan F1 "
         "terendah. Accuracy bukan metrik utama untuk early warning.",
         accent=ACCENT, title_size=12, body_size=9.5)
add_text(slide,
         "Model yang dipakai di aplikasi: Logistic Regression + threshold 0,57.",
         8.05, 5.62, 4.65, 0.60, size=11, bold=True, color=GREEN)
add_notes(slide,
          "Jelaskan empat alasan memilih LR. Tekankan bahwa RF punya accuracy "
          "tertinggi tetapi recall terendah, sehingga tidak cocok untuk early "
          "warning. XGBoost kompetitif, tetapi LR lebih stabil dan interpretable.")

# ---------------------------------------------------------------------------
# 14. Arsitektur aplikasi
# ---------------------------------------------------------------------------
slide = add_slide("Implementasi Aplikasi", "12  •  Streamlit")
alur = [
    ("Input", "Form profil,\nakademik, perilaku\natau CSV batch"),
    ("Feature engineering", "fail_rate,\nengagement_score,\nrisk_flag_count"),
    ("Preprocessing", "log1p + scale,\none-hot;\npipeline tersimpan"),
    ("Model", "Logistic Regression\n+ threshold 0,57"),
    ("Output", "Probabilitas,\nkategori, faktor,\nrekomendasi"),
]
for i, (judul, isi) in enumerate(alur):
    x = 0.45 + i * 2.55
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.68),
        Inches(2.20), Inches(1.42)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE if i % 2 == 0 else ACCENT_SOFT
    shape.line.color.rgb = LINE
    add_text(slide, judul, x + 0.13, 1.80, 1.95, 0.30, size=11, bold=True)
    add_text(slide, isi, x + 0.13, 2.15, 1.95, 0.85, size=8.8, color=MUTED)
    if i < 4:
        add_arrow(slide, x + 2.25, 2.24, w=0.23, h=0.24)

add_text(slide, "Teknologi & output", 0.55, 3.45, 5.0, 0.30,
         size=12, bold=True, color=ACCENT)
add_bullets(slide, [
    "Streamlit 1.66, joblib, pandas, scikit-learn, Altair.",
    "Model + threshold + daftar fitur dimuat dari satu artefak (model_dropout.pkl).",
    "Prediksi individu: probabilitas, kategori risiko, faktor risiko, dan rekomendasi.",
    "Prediksi batch: unggah CSV, daftar prioritas, unduh hasil lengkap.",
], 0.55, 3.82, 6.10, 1.70, size=10.5, gap=4)

add_card(slide, 7.05, 3.48, 5.75, 1.35, "Input → Process → Output",
         "Alur aplikasi mengikuti kebutuhan tugas: pengguna memasukkan data, "
         "model menjalankan pipeline yang sama seperti saat evaluasi, lalu hasil "
         "ditampilkan dalam bahasa yang dapat ditindaklanjuti.",
         accent=GREEN, title_size=12, body_size=9.2)
add_card(slide, 7.05, 5.00, 5.75, 1.35, "Menjaga konsistensi",
         "Preprocessing dan threshold disimpan bersama model. Aplikasi tidak "
         "menghitung ulang aturan secara terpisah, sehingga hasil konsisten.",
         accent=BLUE, title_size=12, body_size=9.2)
add_notes(slide,
          "Jelaskan arsitektur input-proses-output. Tekankan bahwa aplikasi memuat "
          "pipeline dan threshold yang sama dengan evaluasi, sehingga tidak ada "
          "perbedaan preprocessing antara notebook dan aplikasi.")

# ---------------------------------------------------------------------------
# 15. Demo individu
# ---------------------------------------------------------------------------
slide = add_slide("Demo: Prediksi Individu", "13  •  Aplikasi dijalankan")
with TemporaryDirectory() as tmp:
    crop_result = Path(tmp) / "app_result_crop.png"
    Image.open(FIG / "app_result.png").crop((0, 360, 1600, 1000)).save(crop_result)
    add_image_fit(slide, crop_result, 0.35, 1.48, 12.65, 4.70)

add_card(slide, 0.55, 6.30, 3.95, 0.78, "97,5%",
         "probabilitas dropout pada contoh profil berisiko tinggi", accent=ACCENT,
         title_size=18, body_size=8.5)
add_card(slide, 4.72, 6.30, 3.95, 0.78, "Risiko tinggi",
         "faktor: IPK 1,2; 6 mata kuliah gagal; kehadiran 45%", accent=ACCENT,
         title_size=14, body_size=8.5)
add_card(slide, 8.89, 6.30, 3.95, 0.78, "Rekomendasi",
         "konseling individual, remedial, dan pengecekan hambatan belajar", accent=GREEN,
         title_size=14, body_size=8.5)
add_notes(slide,
          "Tunjukkan aplikasi benar-benar berjalan. Isi contoh profil berisiko "
          "tinggi, klik prediksi, lalu tunjukkan probabilitas, kategori, faktor, "
          "dan rekomendasi. Sebutkan bahwa angka ini contoh input, bukan hasil "
          "untuk mahasiswa nyata.")

# ---------------------------------------------------------------------------
# 16. Demo batch
# ---------------------------------------------------------------------------
slide = add_slide("Demo: Prediksi Batch", "14  •  Aplikasi dijalankan")
with TemporaryDirectory() as tmp:
    crop_batch = Path(tmp) / "app_batch_crop.png"
    Image.open(FIG / "app_batch.png").crop((0, 120, 1600, 1170)).save(crop_batch)
    add_image_fit(slide, crop_batch, 0.35, 1.45, 8.30, 4.85)

add_text(slide, "Alur batch", 8.95, 1.60, 3.8, 0.30,
         size=12, bold=True, color=ACCENT)
add_bullets(slide, [
    "Unduh template CSV.",
    "Unggah data seluruh angkatan.",
    "Model menghitung probabilitas dan mengurutkan prioritas.",
    "Unduh hasil lengkap untuk tim bimbingan.",
], 8.95, 1.98, 3.85, 1.70, size=10.2, gap=4)

add_card(slide, 8.95, 3.85, 3.85, 1.25, "Validasi input",
         "Kolom wajib diperiksa; nilai kosong dan mata kuliah gagal > diambil "
         "ditolak dengan pesan yang jelas.", accent=BLUE,
         title_size=11, body_size=9.2)
add_card(slide, 8.95, 5.25, 3.85, 1.25, "Output", 
         "Tabel prioritas, ringkasan jumlah berisiko, dan unduhan CSV.",
         accent=GREEN, title_size=11, body_size=9.2)
add_notes(slide,
          "Tunjukkan tab batch: unduh template, unggah CSV, lihat ringkasan dan "
          "tabel prioritas, lalu unduh hasil. Tekankan bahwa output diurutkan dari "
          "probabilitas tertinggi agar sumber daya bimbingan dapat diprioritaskan.")

# ---------------------------------------------------------------------------
# 17. Kesimpulan
# ---------------------------------------------------------------------------
slide = add_slide("Kesimpulan", "15  •  Jawaban atas tujuan")
add_text(slide,
         "Model early warning berhasil dibangun dan diimplementasikan, dengan "
         "batasan yang perlu dipahami.",
         0.55, 1.55, 12.0, 0.55, size=15, color=MUTED)

data = [
    ["Poin", "Hasil"],
    ["Model terpilih", "Logistic Regression; F1 CV 0,7387."],
    ["Performa final", "F1 0,7431; recall 77,4%; precision 71,4%; ROC-AUC 0,899; PR-AUC 0,825."],
    ["Dampak", "4.955 dari 6.401 mahasiswa dropout di data test berhasil ditangkap."],
    ["Faktor utama", "IPK semester 1, pendapatan, status uang kuliah, LMS, tenggat, kehadiran, stres."],
    ["Aplikasi", "Prediksi individu + batch, lengkap dengan faktor risiko dan rekomendasi."],
]
add_table(slide, data, 0.55, 2.35, 12.2, 3.10,
          [2.10, 10.10], font_size=10.5, row_height=0.50, first_col_bold=True)

add_card(slide, 0.55, 5.75, 12.2, 0.85, "Pesan utama",
         "Model ini adalah alat skrining untuk memprioritaskan intervensi manusia — "
         "bukan vonis otomatis. Keputusan akhir tetap berada pada pembimbing akademik.",
         accent=GREEN, title_size=11, body_size=9.5)
add_notes(slide,
          "Ringkas hasil: LR terpilih, F1 final 0,7431, recall 77,4%, dan aplikasi "
          "sudah berjalan. Tutup dengan pesan bahwa model adalah alat bantu, bukan "
          "pengganti keputusan manusia.")

# ---------------------------------------------------------------------------
# 18. Keterbatasan & pengembangan
# ---------------------------------------------------------------------------
slide = add_slide("Keterbatasan & Pengembangan", "16  •  Kejujuran ilmiah")
add_text(slide, "Keterbatasan", 0.55, 1.60, 5.8, 0.30,
         size=12, bold=True, color=ACCENT)
add_bullets(slide, [
    "Indikasi data sintetis: gpa_semester_2 < gpa_semester_1 pada 90,4% baris.",
    "Asumsi timing: fitur keterlibatan dianggap merekam semester 1; dataset tidak mendokumentasikan periodenya.",
    "Split acak, bukan validasi temporal antar-angkatan.",
    "Faktor motivasi, keluarga, dan kesehatan mental tidak terukur lengkap.",
], 0.55, 1.98, 5.90, 2.10, size=9.5, gap=5)

add_text(slide, "Potensi pengembangan", 6.95, 1.60, 5.8, 0.30,
         size=12, bold=True, color=GREEN)
add_bullets(slide, [
    "Validasi temporal pada data angkatan berikutnya.",
    "SHAP untuk interpretasi per mahasiswa.",
    "Cost-sensitive learning dengan biaya FN/FP eksplisit.",
    "Monitoring drift dan deployment ke server kampus.",
    "Survival analysis untuk memperkirakan waktu dropout.",
], 6.95, 1.98, 5.80, 2.10, size=9.5, gap=5)

add_card(slide, 0.55, 4.55, 6.05, 1.45, "Keterbatasan aplikasi",
         "Aplikasi membutuhkan input lengkap dan belum terhubung ke SIAKAD. "
         "Prediksi bergantung pada kualitas dan konsistensi data yang diunggah.",
         accent=ACCENT, title_size=12, body_size=9.5)
add_card(slide, 6.75, 4.55, 6.05, 1.45, "Rencana pengembangan",
         "Integrasi API SIAKAD, antrean intervensi, umpan balik hasil intervensi, "
         "dan evaluasi berkala performa model.",
         accent=GREEN, title_size=12, body_size=9.5)
add_notes(slide,
          "Sampaikan keterbatasan secara terbuka, terutama indikasi data sintetis "
          "dan tidak adanya validasi temporal. Jelaskan rencana pengembangan agar "
          "model lebih dapat dipercaya dan terhubung ke proses kampus.")

# ---------------------------------------------------------------------------
# 19. Pembagian tugas
# ---------------------------------------------------------------------------
slide = add_slide("Pembagian Tugas", "17  •  Kerja kelompok")
data = [
    ["Anggota", "Fokus tugas", "Kontribusi"],
    ["Nama Anggota 1\n(NIM ...)", "Dataset, EDA, preprocessing", "Pembersihan data, analisis eksploratif, pipeline, dokumentasi."],
    ["Nama Anggota 2\n(NIM ...)", "Modeling & evaluasi", "Training 3 model, tuning, threshold, confusion matrix, analisis error."],
    ["Nama Anggota 3\n(NIM ...)", "Aplikasi & presentasi", "Streamlit, pengujian aplikasi, PPT, video demo."],
]
add_table(slide, data, 0.55, 1.65, 12.2, 3.30,
          [2.35, 3.20, 6.65], font_size=10, row_height=0.80,
          header_height=0.50, first_col_bold=True)
add_text(slide,
         "Seluruh anggota memahami dataset, preprocessing, algoritma, evaluasi, dan aplikasi.",
         0.55, 5.35, 12.2, 0.35, size=11, bold=True, color=ACCENT)
add_text(slide,
         "Ganti nama, NIM, dan kelas sesuai kelompok sebelum presentasi.",
         0.55, 5.80, 12.2, 0.30, size=9, color=MUTED)
add_notes(slide,
          "Sesuaikan nama dan NIM. Jelaskan pembagian tugas, tetapi tegaskan bahwa "
          "semua anggota memahami keseluruhan project dan siap menjawab pertanyaan.")

# ---------------------------------------------------------------------------
# 20. Referensi
# ---------------------------------------------------------------------------
slide = add_slide("Referensi", "18  •  Sumber literatur")
refs = [
    "Mduma, N., Kalegele, K., & Machuve, D. (2019). A survey of machine learning approaches and techniques for student dropout prediction. Data Science Journal, 18(1), 14. https://doi.org/10.5334/dsj-2019-014",
    "Delen, D. (2011). Predicting student attrition with data mining methods. Journal of College Student Retention, 13(1), 17–35. https://doi.org/10.2190/CS.13.1.B",
    "Hoffait, A.-S., & Schyns, M. (2017). Early detection of university students with potential difficulties. Decision Support Systems, 101, 1–11. https://doi.org/10.1016/j.dss.2017.05.003",
    "Gray, C. C., & Perkins, D. (2019). Utilizing early engagement and machine learning to predict student outcomes. Computers & Education, 131, 22–32. https://doi.org/10.1016/j.compedu.2018.12.006",
    "Tomasevic, N., Gvozdenovic, N., & Vranes, S. (2020). An overview and comparison of supervised data mining techniques for student exam performance prediction. Computers & Education, 143, 103676. https://doi.org/10.1016/j.compedu.2019.103676",
    "Sghir, N., Adadi, A., & Lahmer, M. (2023). Recent advances in predictive learning analytics: A decade systematic review (2012–2022). Education and Information Technologies, 28(7), 8299–8333. https://doi.org/10.1007/s10639-022-11536-0",
]
for i, ref in enumerate(refs):
    add_text(slide, f"{i+1}.  {ref}", 0.55, 1.58 + i * 0.82, 12.2, 0.75,
             size=8.8, color=DARK, line_spacing=1.0)
add_text(slide,
         "Dataset: https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk (CC0)",
         0.55, 6.55, 12.2, 0.30, size=8.5, color=MUTED)
add_notes(slide,
          "Tunjukkan enam referensi utama. Dataset bersumber dari Kaggle dengan "
          "lisensi CC0. Detail tabel studi literatur tersedia di dokumen terpisah.")

# ---------------------------------------------------------------------------
# 21. Terima kasih
# ---------------------------------------------------------------------------
slide = prs.slides.add_slide(BLANK)
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = DARK
band = slide.shapes.add_shape(
    MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.14)
)
band.fill.solid()
band.fill.fore_color.rgb = ACCENT
band.line.fill.background()

add_text(slide, "Terima kasih", 0.85, 1.15, 11.0, 0.85,
         size=42, bold=True, color=WHITE)
add_text(slide, "Pertanyaan & diskusi", 0.88, 2.10, 8.0, 0.45,
         size=18, color=RGBColor(0xE8, 0xD5, 0xD2))
add_text(slide,
         "Project files: notebooks • models • reports • app.py • docs",
         0.88, 3.10, 10.0, 0.35, size=12, color=RGBColor(0xD1, 0xD5, 0xDB))
add_text(slide,
         "Model adalah alat bantu untuk memprioritaskan intervensi, bukan pengganti keputusan manusia.",
         0.88, 3.60, 11.5, 0.65, size=12, color=RGBColor(0xD1, 0xD5, 0xDB))
add_text(slide, "Kelompok 2  •  Pembelajaran Mesin  •  2026",
         0.88, 6.35, 8.0, 0.30, size=10, color=RGBColor(0x9C, 0xA3, 0xAF))
add_notes(slide, "Buka sesi tanya jawab. Siapkan notebook, aplikasi, dan laporan.")

prs.save(OUT)
print("PPT disimpan:", OUT)
print("Jumlah slide:", len(prs.slides))
