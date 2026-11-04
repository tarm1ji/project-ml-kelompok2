# Cara Membagikan Aplikasi ke Anggota Kelompok

Ada tiga pilihan. Pilih sesuai kebutuhan.

| Kebutuhan | Cara | Link berlaku |
|---|---|---|
| Demo cepat, laptop masih menyala | Cloudflare Tunnel | Sementara, selama laptop dan terminal aktif |
| Teman satu Wi-Fi/kelas | Alamat IP lokal | Selama satu jaringan dan laptop menyala |
| Link permanen untuk dikumpulkan | Streamlit Community Cloud | Selama aplikasi tidak dihapus |

Untuk tugas, cara paling rapi adalah **Streamlit Community Cloud** karena menghasilkan link publik yang bisa dibuka kapan saja.

---

## Opsi 1 — Streamlit Community Cloud (rekomendasi)

### Yang dibutuhkan

- Akun GitHub.
- Akun Streamlit Community Cloud (bisa login dengan GitHub).
- Repository GitHub berisi project ini.

### Langkah 1 — Upload project ke GitHub

1. Extract `Project_ML_Kelompok_2.zip`.
2. Buat repository baru di GitHub, misalnya `project-ml-kelompok-2`.
3. Pilih **Public** agar paling mudah.
4. Upload isi folder project ke repository. Jangan upload folder `venv`.
5. Pastikan struktur repository seperti ini:

```text
project-ml-kelompok-2/
├── app.py
├── requirements.txt
├── models/
│   └── model_dropout.pkl
├── data/
├── docs/
├── reports/
├── notebooks/
└── .streamlit/
    └── config.toml
```

File yang wajib ada untuk aplikasi:

- `app.py`
- `requirements.txt`
- `models/model_dropout.pkl`
- `.streamlit/config.toml` (opsional, untuk tema)

### Langkah 2 — Deploy

1. Buka [share.streamlit.io](https://share.streamlit.io).
2. Login dengan GitHub.
3. Klik **Create app** atau **New app**.
4. Pilih repository `project-ml-kelompok-2`.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Buka **Advanced settings** jika perlu, lalu pilih Python versi 3.12 atau 3.13.
8. Klik **Deploy**.

Tunggu proses build selesai. Setelah berhasil, kamu akan mendapat link seperti:

```text
https://project-ml-kelompok-2.streamlit.app
```

Link itu yang dibagikan ke kelompok.

### Langkah 3 — Bagikan

Kirim link ke grup, misalnya:

> Ini link aplikasi project ML Kelompok 2: https://project-ml-kelompok-2.streamlit.app  
> Bisa dibuka di HP atau laptop. Kalau sedang tidur, tunggu beberapa detik lalu refresh.

### Jika build gagal

- **ModuleNotFoundError** → cek `requirements.txt` sudah ada di root repository.
- **Model belum ditemukan** → pastikan `models/model_dropout.pkl` ikut ter-upload dan namanya sama.
- **Versi Python bermasalah** → ganti Python version di Advanced settings ke 3.12 atau 3.13.
- **App tidur** → buka link, tunggu 10–30 detik, lalu refresh.
- **Repo private tidak terbaca** → buat repo public atau beri akses Streamlit ke repo private.

---

## Opsi 2 — Cloudflare Tunnel (link sementara, cepat)

Cocok untuk demo mendadak tanpa upload ke GitHub. Laptop harus tetap menyala dan terminal tidak boleh ditutup.

### Langkah

1. Install Cloudflare Tunnel sekali saja:

   ```powershell
   winget install --id Cloudflare.cloudflared
   ```

2. Jalankan aplikasi:

   ```powershell
   cd "C:\!Documents\project-ml-kel-2"
   .\venv\Scripts\Activate.ps1
   streamlit run app.py
   ```

3. Buka terminal **baru**, lalu jalankan:

   ```powershell
   cloudflared tunnel --url http://localhost:8501
   ```

4. Tunggu sampai muncul link seperti:

   ```text
   https://nama-acak.trycloudflare.com
   ```

5. Kirim link tersebut ke grup.

### Catatan

- Link hanya aktif selama laptop menyala dan kedua terminal masih berjalan.
- Siapa pun yang punya link bisa membuka aplikasi. Jangan upload data mahasiswa asli.
- Jika laptop sleep atau internet putus, link mati.
- Jangan tutup jendela Streamlit maupun Cloudflare.

---

## Opsi 3 — Satu Wi-Fi yang sama

Cocok saat presentasi di kelas dan semua orang terhubung ke Wi-Fi yang sama.

### Langkah

1. Jalankan aplikasi:

   ```powershell
   cd "C:\!Documents\project-ml-kel-2"
   .\venv\Scripts\Activate.ps1
   streamlit run app.py
   ```

2. Cari alamat IPv4 laptop:

   ```powershell
   ipconfig
   ```

   Contoh hasil: `192.168.1.2`.

3. Minta teman membuka:

   ```text
   http://192.168.1.2:8501
   ```

### Jika tidak bisa dibuka

- Pastikan satu Wi-Fi dan bukan Wi-Fi tamu yang mengisolasi perangkat.
- Saat Windows Firewall bertanya, izinkan akses untuk **Private networks**.
- Jika tetap gagal, gunakan Cloudflare Tunnel atau Streamlit Cloud.

---

## Ringkasan cepat

- **Mau link permanen untuk dikumpulkan:** pakai Streamlit Community Cloud.
- **Mau demo sekarang tanpa upload:** pakai Cloudflare Tunnel.
- **Semua di ruangan yang sama:** pakai IP lokal.

Untuk versi yang dibagikan ke kelompok, gunakan **Streamlit Community Cloud**.
