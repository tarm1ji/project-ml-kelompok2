# Skenario Video Demo Aplikasi

**Project:** Deteksi Risiko Dropout Mahasiswa pada Awal Semester 2  
**Durasi target:** 6 menit 30 detik (maksimal 7 menit)  
**Format:** 1920 × 1080, 16:9, 30 fps  
**Pembicara:** Anggota 1 (latar belakang–EDA), Anggota 2 (preprocessing–pemilihan model), Anggota 3 (demo aplikasi–kesimpulan)

> Ganti nama dan NIM pada bagian pembukaan. Video harus menampilkan aplikasi yang benar-benar dijalankan, bukan hanya screenshot.

---

## Persiapan sebelum merekam

1. Aktifkan virtual environment dan jalankan aplikasi:

   ```powershell
   cd "C:\!Documents\project-ml-kel-2"
   .\venv\Scripts\Activate.ps1
   streamlit run app.py
   ```

2. Buka `http://localhost:8501` di browser. Jika port dipakai, gunakan `--server.port 8502`.
3. Siapkan berkas `data/contoh_batch.csv` untuk demo prediksi batch.
4. Tutup notifikasi, tab browser lain, dan aplikasi chat. Sembunyikan bookmark bar.
5. Gunakan resolusi 1920 × 1080, zoom browser 90–100%, dan mode layar penuh saat demo.
6. Rekam dengan OBS Studio atau PowerPoint Record. Tes mikrofon dan level audio.
7. Latihan minimal satu kali. Targetkan durasi 6:30 agar masih aman di bawah 7:00.
8. Jangan memotong proses klik tombol prediksi. Video harus menunjukkan aplikasi berjalan dan merespons input.

---

## Struktur waktu

| Waktu | Segmen | Visual |
|---|---|---|
| 00:00–00:25 | Pembukaan dan perkenalan | Slide 1, kamera/logo kampus |
| 00:25–01:00 | Latar belakang masalah | Slide 2 |
| 01:00–01:35 | Dataset dan skenario prediksi | Slide 6 |
| 01:35–02:10 | EDA dan temuan utama | Slide 7 |
| 02:10–02:55 | Preprocessing dan feature engineering | Slide 8 |
| 02:55–03:35 | Algoritma dan protokol eksperimen | Slide 9 |
| 03:35–04:15 | Perbandingan model dan pemilihan | Slide 10–11 |
| 04:15–05:35 | Demo prediksi individu | Aplikasi live |
| 05:35–06:15 | Demo prediksi batch | Aplikasi live |
| 06:15–06:50 | Kesimpulan dan keterbatasan | Slide 17–18 |
| 06:50–07:00 | Penutup | Slide 21 |

---

## Naskah dan arahan

### 00:00–00:25 — Pembukaan

**Visual:** Slide judul. Tampilkan nama kelompok dan anggota. Bisa diselingi webcam kecil di pojok.

**Narasi:**
> Selamat pagi. Kami Kelompok 2 dari mata kuliah Pembelajaran Mesin. Saya [Nama 1], bersama [Nama 2] dan [Nama 3]. Project kami berjudul **Deteksi Risiko Dropout Mahasiswa pada Awal Semester 2**. Kami membangun model untuk membantu kampus menemukan mahasiswa yang perlu diprioritaskan untuk intervensi, sebelum semester 2 berjalan.

**Aksi:** Tunjukkan slide judul selama 2–3 detik sebelum pindah.

---

### 00:25–01:00 — Latar belakang

**Visual:** Slide 2 “Latar Belakang & Masalah”.

**Narasi:**
> Dropout bukan hanya soal angka retensi. Di baliknya ada mahasiswa yang kehilangan kesempatan belajar, dan institusi yang kehilangan sumber daya. Pada dataset kami, 32 persen mahasiswa dropout. Sinyal risikonya sudah terlihat di semester 1: IPK, mata kuliah gagal, kehadiran, tugas, dan keterlibatan LMS. Pertanyaannya, bisakah kita memakai data yang tersedia di awal semester 2 untuk menandai mahasiswa berisiko lebih dini?

**Aksi:** Sorot angka 32% dan tiga kartu statistik di sisi kanan.

---

### 01:00–01:35 — Dataset dan skenario

**Visual:** Slide 6 “Dataset”.

**Narasi:**
> Kami memakai dataset **Student Academic Performance and Dropout Risk** dari Kaggle, lisensi CC0, berisi 100.000 baris dan 34 kolom. Targetnya biner: satu untuk dropout, nol untuk bertahan, dengan komposisi 68 banding 32. Karena prediksi dilakukan di awal semester 2, kami membuang `person_id`, `semester`, dan yang paling penting `gpa_semester_2` karena nilainya belum tersedia saat prediksi. Fiturnya mencakup profil dan admisi, capaian semester 1, serta perilaku belajar.

**Aksi:** Tunjukkan tiga kartu kelompok fitur dan baris “Dibuang”.

---

### 01:35–02:10 — EDA

**Visual:** Slide 7 “EDA: Sinyal Risiko Sudah Terlihat”.

**Narasi:**
> EDA kami menunjukkan pola yang sangat jelas. Mahasiswa dengan IPK di bawah 2 memiliki tingkat dropout 73,9 persen, sedangkan IPK 3,5 sampai 4 hanya 2 persen. Yang gagal lima mata kuliah atau lebih mencapai 66,3 persen. Pendapatan keluarga rendah juga berisiko lebih tinggi, 42,9 persen, dibanding pendapatan tinggi, 20 persen. Korelasi terkuat ada pada IPK semester 1 dan jumlah mata kuliah gagal. Ini asosiasi, bukan bukti kausalitas.

**Aksi:** Gerakkan kursor dari grafik kiri ke kanan, lalu ke baris insight di bawah.

---

### 02:10–02:55 — Preprocessing dan feature engineering

**Visual:** Slide 8 “Preprocessing & Feature Engineering”.

**Narasi:**
> Alur preprocessing kami dijaga ketat agar evaluasi tidak bocor. Kami membuang kolom yang tidak boleh dipakai, lalu membuat tiga fitur turunan: `fail_rate`, `engagement_score`, dan `risk_flag_count`. Setelah itu data dibagi 80 persen training dan 20 persen testing secara stratified. Di dalam pipeline, fitur miring ditransformasi `log1p` lalu distandardisasi, sedangkan fitur kategorikal di-one-hot. Kami memakai `class_weight` atau `scale_pos_weight` untuk kelas minoritas. Kami sengaja tidak melakukan imputasi karena tidak ada missing value, tidak menghapus outlier karena nilai ekstrem masih wajar, dan tidak memakai SMOTE karena rasio 68 banding 32 masih dapat ditangani dengan class weight. Data test tidak pernah dipakai untuk memilih fitur, tuning, atau threshold.

**Aksi:** Ikuti panah lima langkah, lalu tunjukkan tabel “Keputusan yang sengaja dihindari”.

---

### 02:55–03:35 — Algoritma dan protokol

**Visual:** Slide 9 “Pengembangan Model”.

**Narasi:**
> Kami membandingkan tiga algoritma yang mewakili tiga tingkat kompleksitas. Logistic Regression sebagai model linear yang mudah diinterpretasikan. Random Forest sebagai ensemble pohon yang menangkap interaksi non-linear. Dan XGBoost sebagai gradient boosting yang kuat untuk data tabular. Semua diuji dengan protokol yang sama: 5-fold StratifiedKFold, RandomizedSearchCV dengan scoring F1, dan threshold yang dicari dari out-of-fold prediction pada data training, bukan dari data test.

**Aksi:** Sorot tiga kartu algoritma, lalu daftar protokol.

---

### 03:35–04:15 — Perbandingan dan pemilihan model

**Visual:** Slide 10 dan 11.

**Narasi:**
> Hasilnya, ketiga model sangat kompetitif. Pada cross-validation, Logistic Regression unggul tipis dengan F1 0,7387, diikuti XGBoost 0,7379 dan Random Forest 0,7300. Pada test dengan threshold 0,50, XGBoost sedikit lebih tinggi, 0,7390, dibanding Logistic Regression 0,7381. Setelah threshold dioptimalkan menjadi 0,57, Logistic Regression mencapai F1 0,7431, recall 77,4 persen, precision 71,4 persen, ROC-AUC 0,899, dan PR-AUC 0,825. Kami memilih Logistic Regression karena F1 cross-validation-nya terbaik, stabil, mudah dijelaskan, dan setelah threshold tuning memberikan F1 test tertinggi. Accuracy tetap kami laporkan, tetapi bukan metrik utama karena kelas dropout lebih sedikit dan false negative lebih mahal.

**Aksi:** Tunjukkan grafik perbandingan, lalu tabel dan tiga kartu ringkasan di slide 11.

---

### 04:15–05:35 — Demo prediksi individu

**Visual:** Aplikasi Streamlit live, tab **Prediksi Individu**.

**Aksi dan narasi:**
> Sekarang kami jalankan aplikasinya. Ini adalah aplikasi Streamlit yang memuat model, threshold, dan daftar fitur dari satu artefak, sehingga preprocessing-nya sama dengan saat evaluasi.

1. Tunjukkan header dan tiga tab.
   > Ada tiga fitur utama: prediksi individu, prediksi batch, dan penjelasan tentang model.

2. Tetap di tab **Profil & Admisi**, lalu pindah ke tab **Semester 1**.
   > Saya akan memasukkan contoh profil mahasiswa berisiko tinggi. Data ini hanya contoh, bukan mahasiswa nyata.

3. Isi nilai berikut:
   - Mata kuliah yang gagal: **6**
   - IPK semester 1: **1,2**
   - Tingkat kehadiran: **45%**
   - Tugas terkumpul: **40%**
   - Tenggat terlewat: **8**

4. Pindah ke tab **Kebiasaan Belajar**, isi:
   - Jam kerja per minggu: **30**
   - Jam tidur: **4,5**
   - Tingkat stres: **10**
   - Kepuasan kuliah: **2**
   - Hilangkan centang “Uang kuliah lancar”.

5. Klik **Hitung risiko dropout**.
   > Model menghitung probabilitas dropout. Hasilnya di atas 95 persen, masuk kategori risiko tinggi. Aplikasi tidak hanya menampilkan angka, tetapi juga faktor yang terdeteksi: IPK di bawah 2, enam mata kuliah gagal, kehadiran rendah, tugas rendah, tenggat terlewat, dan stres tinggi tanpa dukungan kesehatan mental.

6. Scroll ke **Rekomendasi tindak lanjut**.
   > Rekomendasinya berupa konseling individual, pengecekan hambatan kehadiran dan tugas, serta rencana remedial. Ini yang membuat model dapat ditindaklanjuti, bukan sekadar prediksi.

7. Buka expander **Lihat data yang dipakai model** sebentar.
   > Pengguna juga bisa memeriksa data yang masuk ke model.

**Catatan:** Jangan menyebut angka pasti jika tampilan dapat berbeda karena pembulatan. Cukup katakan “di atas 95 persen” atau “risiko tinggi”.

---

### 05:35–06:15 — Demo prediksi batch

**Visual:** Tab **Prediksi Batch** di aplikasi live.

**Aksi dan narasi:**
> Untuk penggunaan tingkat program studi, tersedia prediksi batch.

1. Klik tab **Prediksi Batch**.
   > Pengguna dapat mengunduh template CSV, lalu mengunggah data mahasiswa.

2. Klik **Unduh template CSV** sebentar, lalu unggah `data/contoh_batch.csv`.
   > Saya unggah contoh berisi sepuluh mahasiswa dengan profil beragam.

3. Tunggu tabel muncul, lalu scroll ke **Ringkasan** dan **Daftar prioritas intervensi**.
   > Aplikasi memvalidasi kolom, menghitung probabilitas, dan mengurutkan mahasiswa dari risiko tertinggi. Ringkasan menunjukkan jumlah mahasiswa, jumlah yang diprediksi dropout, jumlah risiko tinggi, dan rata-rata probabilitas. Tabel prioritas memudahkan tim bimbingan menentukan siapa yang dihubungi lebih dulu.

4. Klik **Unduh hasil prediksi lengkap**.
   > Hasil lengkap dapat diunduh untuk tindak lanjut.

---

### 06:15–06:50 — Kesimpulan dan keterbatasan

**Visual:** Slide 17 lalu slide 18.

**Narasi:**
> Kesimpulannya, model early warning berhasil dibangun dan diimplementasikan. Logistic Regression dengan threshold 0,57 mencapai F1 0,7431 dan recall 77,4 persen. Dari 6.401 mahasiswa dropout di data test, 4.955 berhasil ditangkap. Faktor utamanya adalah IPK semester 1, pendapatan keluarga, status uang kuliah, login LMS, tenggat terlewat, kehadiran, dan stres. Namun, kami perlu jujur pada keterbatasan. Dataset ini memiliki indikasi kuat sebagai data sintetis, split yang dipakai acak dan belum temporal, serta faktor motivasi dan kondisi keluarga tidak terukur lengkap. Karena itu, model ini adalah alat bantu untuk memprioritaskan intervensi manusia, bukan vonis otomatis.

**Aksi:** Tunjukkan angka 4.955 dari 6.401 dan kartu “Pesan utama”.

---

### 06:50–07:00 — Penutup

**Visual:** Slide 21 “Terima kasih”.

**Narasi:**
> Terima kasih. Kami siap menerima pertanyaan dan diskusi.

**Aksi:** Diam 2 detik, lalu hentikan rekaman.

---

## Checklist akhir sebelum upload

- [ ] Durasi di bawah 7 menit.
- [ ] Semua anggota kelompok disebutkan.
- [ ] Judul project disebutkan.
- [ ] Latar belakang, dataset, preprocessing, algoritma, perbandingan model, dan model terpilih dijelaskan.
- [ ] Aplikasi benar-benar dijalankan dan prediksi individu diperlihatkan.
- [ ] Prediksi batch diperlihatkan, termasuk unggah CSV dan hasil prioritas.
- [ ] Kesimpulan dan keterbatasan disampaikan.
- [ ] Audio jelas, tidak ada notifikasi, dan teks terbaca di 1080p.
- [ ] Nama file: `Video_Demo_Project_ML_Kelompok_2.mp4`.

## Jika aplikasi gagal saat rekaman

1. Hentikan rekaman, jangan berpura-pura aplikasi berjalan.
2. Cek terminal: pastikan `streamlit run app.py` masih aktif.
3. Jalankan ulang di port lain: `streamlit run app.py --server.port 8502`.
4. Pastikan `models/model_dropout.pkl` ada dan `data/contoh_batch.csv` tidak kosong.
5. Mulai ulang rekaman dari awal segmen demo; hindari menyambung dengan screenshot.
