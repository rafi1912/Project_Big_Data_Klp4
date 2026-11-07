# ADR 0001 — Dataset X sintetis dipertahankan sebagai kontrol negatif

- Status: diterima
- Konteks: `goyaladi/twitter-bot-detection-dataset` (50.000 baris) adalah dataset X yang disebut di
  Worksheet 2–3. Kosakata kolom `Tweet` persis 971 kata daftar `lorem` milik library Faker (en_US),
  setiap kolom numerik terdistribusi seragam, dan setiap model berakhir di ROC-AUC ≈ 0,50
  (`src/twitter_faker_check.py`, `outputs/metrics/twitter_synthetic*.json`). Data ini dibangkitkan,
  bukan dikumpulkan.
- Keputusan: dataset ini **tidak dibuang** dan **tidak dilaporkan sebagai kemampuan deteksi**. Ia
  dijalankan dengan prosedur yang sama persis seperti dataset nyata sebagai **kontrol negatif**: bukti
  bahwa pipeline tidak "menemukan" sinyal di data acak. Hasil utama X memakai dataset nyata Treiman
  (37.438 akun).
- Konsekuensi: config, README, laporan dan dashboard menandainya sintetis; evidence gate level risiko
  gagal di dataset ini sehingga semua akunnya berstatus *tidak dapat dinilai* (NA). Data demo dashboard
  memakai dataset ini karena tidak ada identitas akun nyata di dalamnya.
