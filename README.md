# Panduan Demo: Insurance Lapse Prediction System

Dokumen ini berisi dua versi penjelasan yang bisa kamu pakai sesuai audiensnya saat demo:
**Bagian 1** untuk audiens umum/non-teknis, **Bagian 2** untuk audiens yang menanyakan detail teknis.

---

## BAGIAN 1 — Penjelasan Sederhana

### Apa masalah yang diselesaikan?

Perusahaan asuransi jiwa kehilangan banyak pendapatan setiap tahun karena nasabah **berhenti membayar premi** di tengah jalan, sehingga polisnya batal (istilah industrinya: **"lapse"**). Selama ini, perusahaan biasanya baru tahu seorang nasabah akan berhenti bayar **setelah** nasabah itu benar-benar telat atau berhenti — sudah terlambat untuk mencegahnya.

### Apa solusinya?

Program ini adalah **sistem peringatan dini** yang memprediksi, dari data profil seorang nasabah (usia, cara bayar, besar premi, jenis polis, dll), **seberapa besar kemungkinan nasabah itu akan berhenti bayar** — sebelum itu benar-benar terjadi. Dengan begitu, tim asuransi bisa menghubungi nasabah yang berisiko tinggi lebih dulu, menawarkan solusi (reminder, keringanan, dsb), alih-alih kehilangan nasabah begitu saja.

### Bagaimana cara kerjanya (analogi sederhana)?

Bayangkan sistem ini seperti **dokter yang memeriksa "kesehatan" sebuah polis asuransi**:

1. Kamu masukkan data nasabah ke formulir (seperti mengisi rekam medis).
2. Sistem "mendiagnosis" berdasarkan pengalaman dari **185.560 polis nasabah di masa lalu** — pola-pola mana yang biasanya berakhir dengan nasabah berhenti bayar, dan mana yang tidak.
3. Sistem memberi "hasil diagnosis": persentase risiko, kategori risiko (rendah/sedang/tinggi/kritis), dan rekomendasi tindakan konkret untuk tim asuransi.

### Alur saat demo berlangsung

1. Buka halaman **form** di browser.
2. Isi data contoh nasabah (usia, cara bayar premi, besar premi, dll).
3. Klik **"Analisis risiko lapse"**.
4. Dalam hitungan detik, muncul panel hasil: kategori risiko, persentase, dan saran tindakan.

### Poin yang bagus disampaikan ke audiens

- Sistem ini belajar dari **data historis riil** (bukan aturan manual yang ditulis orang), sehingga polanya lebih akurat dan bisa diperbarui kapan saja saat ada data baru.
- Hasilnya bukan sekadar angka — langsung disertai **rekomendasi tindakan** yang bisa ditindaklanjuti tim operasional.
- Sistem ini adalah **alat bantu**, bukan pengganti keputusan manusia — prediksi bersifat probabilistik, bukan kepastian mutlak.

---

## BAGIAN 2 — Penjelasan Teknis

### Arsitektur singkat

Ada tiga komponen utama, berjalan sebagai layanan terpisah:

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Model Machine Learning | scikit-learn (HistGradientBoostingClassifier) | "Otak" yang menghitung probabilitas lapse |
| Backend API | FastAPI (Python), di-host di Railway | Menerima data nasabah, menjalankan model, mengembalikan hasil |
| Frontend | Next.js (React), di-host di Railway/Vercel | Formulir input dan tampilan hasil untuk pengguna |

Alur datanya: **Frontend → (HTTP POST) → Backend API → Model → hasil dikirim balik → ditampilkan di Frontend.**

### Bagaimana model dilatih

- **Data**: 185.560 baris data historis polis (`Kaggle.csv`), berisi profil nasabah dan status akhir polisnya (Lapse, Inforce/aktif, Surrender, Death, Expired).
- **Target**: model memprediksi biner — apakah polis akan **Lapse** atau tidak (status lain digabung jadi satu kelas "non-lapse").
- **Pengolahan data**: pembersihan format angka (menghapus koma ribuan, dsb), pembuatan fitur turunan (misalnya rasio premi terhadap manfaat, rasio premi terhadap usia), encoding untuk data kategorikal (jenis kelamin, mode pembayaran, dll), dan scaling untuk data numerik.
- **Algoritma**: Gradient Boosting berbasis histogram — dipilih karena efisien untuk data besar dan tahan terhadap data yang hilang.
- **Kalibrasi**: ambang keputusan "lapse/tidak" tidak memakai default 50%, tapi hasil optimasi khusus (≈38,79%) berdasarkan F1-score terbaik dari data uji.

### Kenapa hasilnya dibagi jadi 4 tingkatan risiko?

Alih-alih hanya bilang "ya/tidak", sistem membagi probabilitas jadi 4 tier (Rendah, Sedang, Tinggi, Kritis), masing-masing dengan rekomendasi tindakan berbeda — ini membuat output lebih actionable untuk tim operasional dibanding sekadar angka probabilitas mentah.

### Arsitektur deployment

- Backend dan frontend berjalan sebagai **container Docker terpisah**, di-deploy ke **Railway**.
- Model (`lapse_model.joblib`) dimuat sekali saat server backend menyala, disimpan di memori — bukan dibaca ulang dari disk setiap ada request, supaya respons cepat.
- Frontend berkomunikasi dengan backend murni lewat **HTTP API**, sehingga backend juga bisa dipanggil dari sistem lain di luar frontend ini (misalnya nanti diintegrasikan ke sistem internal perusahaan).

### Hal yang jujur perlu disampaikan kalau ditanya (jangan overclaim)

- Ini masih **versi demo/prototipe**, dideploy di infrastruktur trial gratis — belum di lingkungan produksi perusahaan yang sesungguhnya.
- Model ini menangkap **korelasi**, bukan hubungan sebab-akibat. Contoh: nasabah dengan premi besar cenderung jarang lapse, tapi itu tidak berarti "menaikkan premi nasabah lain" akan menurunkan risiko lapse mereka — justru berpotensi sebaliknya.
- Akurasi model bergantung pada seberapa mirip data nasabah baru dengan pola di data historis yang dipakai untuk melatih — perlu dievaluasi ulang secara berkala kalau dipakai jangka panjang.

---

## Antisipasi Pertanyaan Saat Demo

**Q: "Datanya dari mana?"**
A: Dataset historis polis asuransi jiwa publik (sumber Kaggle), 185 ribuan baris, dipakai untuk melatih pola — bukan data nasabah asli perusahaan.

**Q: "Seberapa akurat modelnya?"**
A: (Isi dengan angka ROC-AUC/precision-recall dari `evaluate_model.py` kalau kamu punya angka terbaru di tangan — sebutkan sebagai referensi performa pada data uji, bukan jaminan performa di dunia nyata.)

**Q: "Bisa diintegrasikan ke sistem kami?"**
A: Ya — backend-nya adalah REST API standar, bisa dipanggil dari sistem apa pun (bukan cuma dari frontend yang didemokan ini), asal formatnya menyesuaikan skema data yang sudah ditentukan.

**Q: "Datanya aman?"**
A: (Jawab sesuai kebijakan aktual perusahaan soal data nasabah — dokumen ini tidak bisa menjawab ini untukmu karena bergantung kebijakan internal.)
