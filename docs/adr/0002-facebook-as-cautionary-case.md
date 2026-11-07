# ADR 0002 — Facebook dilaporkan sebagai kasus peringatan, bukan hasil utama

- Status: diterima
- Konteks: `khajahussainsk/facebook-spam-dataset` berisi 600 profil (497 label 0 / 103 label 1, bukan
  500/100 seperti kartu Kaggle) yang tersusun **dua blok menurut label**. `profile id` berkorelasi
  dengan label; setelah dikeluarkan pun akurasi tetap ≈ 0,98 karena blok file tercermin di fitur lain.
  Kesalahan model terkumpul di baris yang labelnya tidak cocok dengan bloknya
  (`docs/facebook_findings.md`).
- Keputusan: Facebook tetap dijalankan dan dilaporkan, tetapi sebagai **kasus peringatan**: angka
  akurasinya tidak dikutip sebagai kemampuan deteksi dan tidak dipakai untuk mengurutkan platform.
  `profile id` menjadi kolom identitas di `config.yaml`.
- Konsekuensi: perbandingan kuantitatif lintas platform hanya untuk X (Treiman) dan Instagram; dashboard
  dan laporan menandai Facebook dengan perannya (`cautionary_artefact`).
