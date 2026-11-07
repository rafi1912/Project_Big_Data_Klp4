# ADR 0005 — Nama level risiko menggambarkan kekuatan bukti, bukan vonis

- Status: diterima
- Konteks: dokumen konsep §13 menamai level 0–4 dengan kesimpulan tentang akun ("Authentic / Low Risk",
  "Inauthentic / Spam", …). §15 menyatakan keluaran model adalah indikasi berbasis data, bukan bukti
  mutlak niat pemilik akun, dan pengguna sasaran adalah moderator yang memutuskan.
- Keputusan: level, urutan dan artinya tetap, tetapi **nama yang ditampilkan menggambarkan kekuatan bukti
  dan tindakan yang pantas**: 0 *Tidak ada indikasi*, 1 *Indikasi lemah – pantau*, 2 *Indikasi kuat
  tingkat akun – prioritas tinjauan manusia*, 3 *Dugaan koordinasi antar-akun*, 4 *Amplifikasi
  terkoordinasi*, ditambah **NA** *Tidak dapat dinilai* (kosong bukan nol). Nama §13 disimpan
  berdampingan (`name_concept_doc`). Level 3–4 butuh bukti koordinasi yang tidak ada di dataset mana pun,
  jadi tidak pernah diberikan. Tidak ada aksi otomatis.
- Konsekuensi: setiap tampilan level (MongoDB, Power BI, dashboard) membawa kedua nama; ambang ditentukan
  dengan cross-fitting dan evidence gate (`src/synthesis.py`).
