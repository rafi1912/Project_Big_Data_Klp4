# ADR 0008 — TikTok dianalisis tanpa supervisi; tidak ada label yang dibuat-buat

- Status: diterima
- Konteks: `haziqhalifi/tiktok-2025-dataset` (7.225 video) tidak punya label bot/autentik. Membuat label
  dari aturan lalu "melatih" model pada label itu hanya mengukur aturan itu sendiri.
- Keputusan: TikTok tidak punya classifier dan tidak punya akurasi. Pipeline membangun profil perilaku per
  author (handle diganti kunci pseudonim), indikator anomali tri-state (menyala / dinilai dan tidak
  menyala / tidak dapat dinilai) dengan bobot bukti yang disebutkan, dan tier tinjauan manusia. Di
  `config.yaml` TikTok ada di blok `unlabelled_datasets`, sehingga ikut diunduh tetapi tidak pernah masuk
  ingest/process/train. Level risikonya dibatasi di Level 1; author dengan satu video berstatus NA.
- Konsekuensi: TikTok tidak muncul di tabel metrik klasifikasi atau perbandingan akurasi; ia muncul di
  perbandingan konsep sebagai "fitur tidak tersedia/tidak dapat dinilai" bila memang begitu.
