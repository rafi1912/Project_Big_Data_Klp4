# ADR 0004 — Dashboard Streamlit hanya membaca MongoDB; Power BI adalah ekspor

- Status: diterima
- Konteks: dashboard awal membaca file di `reports/` dan `data/gold/`, sedangkan arsitektur proyek
  menempatkan MongoDB sebagai lapisan penyajian. Dua sumber kebenaran membuat dashboard bisa
  menampilkan sesuatu yang tidak ada di basis data, dan sebaliknya.
- Keputusan: arsitektur = **Spark → MongoDB → {Streamlit, ekspor Power BI}**. Setiap halaman Streamlit
  hanya membaca MongoDB (`load_mongo.py`, `stream.py`, `synthesis.py` yang mengisinya). Bila MongoDB
  tidak terjangkau, dashboard menampilkan perintah untuk menyalakannya lalu berhenti — **tidak ada
  cadangan diam-diam ke file**. Power BI tidak punya konektor MongoDB bawaan yang sederhana, jadi tabel
  yang sama diekspor sebagai CSV datar ke `outputs/powerbi/` (di-gitignore).
- Konsekuensi: menjalankan dashboard butuh MongoDB. Untuk demo tanpa Kaggle ada
  `scripts/seed_demo.py` (data sintetis + agregat tanpa identitas).
