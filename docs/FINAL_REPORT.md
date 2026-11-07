# Laporan Akhir — Deteksi Akun Bot / Tidak Autentik Lintas Platform

**Kelompok "Penyayang Claude"** · Teknologi Big Data · Sekolah Vokasi IPB University
Anggota: Syafiq Fathinas Syaddid, Rafi Fakhri Alaudin, Osanna Patricia Limbong, Aldio Lisafron
Tanggal: 5 November 2026

Dokumen ini menyatukan empat track platform (X/Twitter, Instagram, Facebook, TikTok) dan track
sintesis. Semua angka berasal dari file di repositori ini, bukan dari ingatan atau laporan sesi:

| Sumber | Isi |
|---|---|
| `outputs/metrics/{twitter_real,twitter_synthetic,instagram,facebook,tiktok}.json` | hasil tiap track |
| `outputs/metrics/cross_platform_comparison.json` | hasil sintesis (perbandingan konsep, level risiko, verifikasi MongoDB) |
| `docs/{twitter,instagram,facebook,tiktok}_findings.md` | temuan rinci tiap track |
| `docs/mongodb_powerbi.md` | koneksi MongoDB, skema koleksi, tabel Power BI |
| `src/synthesis.py` | kode sintesis (dapat dijalankan ulang) |

Nama metrik dan nama kolom sengaja tidak diterjemahkan.

> **Catatan versi repo.** Laporan ini ditulis untuk run analisis 5 November 2026 dan dibawa ke repo
> kelompok tanpa mengubah angka. Run ulang dari repo (file Kaggle yang sama, sha256 identik) memberi
> metrik yang **identik** untuk X, Instagram, Facebook, TikTok dan sintesis (perbandingannya ada di PR
> penggabungan); satu-satunya beda adalah bagian `tiktok.json` yang menyalin importance track lain, karena
> run TikTok acuan dijalankan sebelum run X final. Yang berubah hanya lokasi file:
> data mentah dibaca dari `data/landing/<kunci>/`, kredensial MongoDB di `.env`. Bagian §3 menggambarkan
> status arsitektur jalur analisis **sebelum** digabung dengan pipeline operasional; arsitektur repo
> sekarang (termasuk model tersimpan di `models/` dan streaming) ada di
> [architecture.md](architecture.md). Sesuai [ADR 0007](adr/0007-no-dataset-or-per-account-output.md),
> keluaran per akun (`outputs/predictions/`, `outputs/powerbi/`, `data/curated/`) dan file metrik yang
> memuat id akun atau id baris (`cross_platform_comparison.json`, `facebook.json`, `instagram.json`)
> hanya dibuat ulang secara lokal dan tidak di-commit; contoh id akun di laporan ini disamarkan.

---

## 0. Ringkasan

1. **Pengguna sasaran adalah moderator *trust & safety* yang menyaring akun untuk ditinjau manusia.**
   Ini bukan sistem pemblokiran otomatis. Setiap skor, level, dan bendera adalah alasan untuk
   *melihat* sebuah akun, bukan bukti niat, dan tidak pernah menjadi dasar sanksi otomatis.
2. **Hanya satu hasil yang benar-benar mendarat di target ~80% Bu Gema dengan data nyata yang
   besar: X/Twitter, Random Forest accuracy 0,8538** (95% CI 0,8456–0,8616) melawan baseline
   mayoritas 0,6664, MCC 0,6749, ROC-AUC 0,9231, pada 7.474 akun uji dari 37.438 akun nyata.
3. **Instagram 92–93% tinggi tetapi dapat dipertanggungjawabkan** (CV 0,9239 ± 0,0203, ROC-AUC
   0,979). Datanya kecil (696 akun), bersih, seimbang, dan dikurasi satu orang. Angka 95,3% dari split
   seed 42 adalah undian yang beruntung dan tidak boleh dikutip.
4. **Facebook ~95–97% adalah persis kasus "mendekati 99%" yang diperingatkan Bu Gema.** Kami
   melaporkannya sebagai **kegagalan data**, bukan keberhasilan model. File-nya berupa dua blok yang
   terurut menurut label, dan model lebih baik menebak blok (0,990) daripada label (0,947).
5. **Dataset X yang disebut di Worksheet 2–3 (`goyaladi`) adalah data sintetis buatan Faker.**
   Hasilnya 0,4945 melawan baseline 0,501. Dataset ini dipertahankan sebagai **kontrol negatif**.
6. **TikTok tidak punya label, jadi tidak ada hasil supervised.** Yang ada hanya indikator perilaku
   untuk tinjauan manusia, dan transfer model dari platform lain dinilai tidak dapat dipertanggungjawabkan.
7. **Temuan lintas platform utama (§7):** X dan Instagram adalah satu-satunya pasangan yang mendukung
   perbandingan kuantitatif, dan keduanya mempelajari **konsep yang sama: akun yang sepi dan kosong.**
   Ketujuh indikator yang terukur di kedua platform menunjuk ke arah yang sama. **Audiens kecil**
   adalah satu-satunya konsep yang searah di ketiga dataset berlabel, dan konsep ini masuk dua besar
   permutation importance di setiap model. Facebook "spam" justru berlawanan arah pada aktivitas dan umur akun.
8. **Level risiko 0–4 (§8):** Level 0–2 dapat dihitung untuk X, Instagram, dan Facebook. Di X,
   Level 2 berisi 23,6% akun dengan presisi 0,890. Nama level diubah dari nama §13 menjadi nama
   berbasis kekuatan bukti, sesuai prinsip §15, dan pemetaannya dinyatakan di §8.1. **Level 3 dan 4 tidak dapat dicapai dengan data
   apa pun di proyek ini**, karena tidak ada bukti koordinasi maupun amplifikasi.
9. **MongoDB dan Power BI kini ada dan sudah diverifikasi (§9).** Ada 25.335 dokumen akun beserta
   metrik, importance, dan tabel level, yang ditulis Spark ke MongoDB lalu di-query balik. Tabel
   level risiko yang dihitung ulang di dalam MongoDB identik dengan hasil Python. Sepuluh CSV siap
   dipakai untuk Power BI.

---

## 1. Pengguna sasaran (Catatan Bu Gema no. 1)

**Pengguna:** moderator *trust & safety*, yaitu orang yang setiap hari harus memutuskan akun mana
yang ditinjau lebih dulu, dengan kapasitas tinjauan terbatas.

**Apa yang sistem berikan kepadanya:** antrean akun yang diurutkan menurut skor model. Setiap akun
diberi level risiko dengan arti operasional yang jelas (§8), plus alasan yang dapat dibaca: konsep
perilaku apa yang mendorong skor (§7).

**Apa yang sistem TIDAK lakukan:** sistem tidak memblokir, menangguhkan, atau menandai akun secara
publik. Tidak ada level yang memicu aksi otomatis. Level tertinggi yang dapat dicapai (Level 2)
berarti "prioritas tinjauan manusia".

**Mengapa ini penting untuk cara membaca angka:**
- *False positive* berarti orang sungguhan yang masuk antrean moderator. Di X, Level 2 tetap
  memasukkan 194 dari 4.981 manusia di set uji (3,9%). Di Instagram, semua 18 false positive RF
  adalah akun asli yang sepi: punya foto, 0–28 post, dan sebagian besar tanpa bio
  (`instagram_findings.md` §8). Mereka adalah alasan langkah tinjauan manusia ada.
- Presisi bergantung pada prevalensi. Prevalensi bot di dunia nyata tidak diketahui dan hampir pasti
  jauh lebih rendah daripada di dataset ini (33% di X, 50% di Instagram). Pada prevalensi 5%,
  presisi Level 2 di X turun dari 0,890 ke sekitar 0,46 (§8.4). Moderator harus tahu ini.
- Label sumber berbeda konsep: X "bot", Instagram "fake/spammer", Facebook "spam". Target
  `0 = Authentic / 1 = Inauthentic` adalah konvensi operasional, bukan klaim tentang niat.

---

## 2. Data dan peran setiap dataset

Cakupan proyek bergeser dari tiga dataset (Worksheet 3; Tujuan Khusus 3 hanya menyebut Instagram,
Twitter, dan Facebook) ke empat platform (keputusan kelompok 05/11, menambahkan TikTok), dan dalam
praktik menjadi **lima dataset**, karena X dijalankan dua kali (§12).

| Dataset | Platform | Unit | Baris | Positif | Konsep label | Peran dalam studi |
|---|---|---|---|---|---|---|
| `danieltreiman/twitter-human-bots-dataset` | X | akun | 37.438 | 33,2% | bot | **hasil utama** |
| `free4ever1/instagram-fake-spammer-genuine-accounts` | Instagram | akun | 696 (576 + 120) | 50,0% | fake/spammer | pendukung, sampel kecil |
| `khajahussainsk/facebook-spam-dataset` | Facebook | akun | 600 | 17,2% | spam | **kasus peringatan** |
| `goyaladi/twitter-bot-detection-dataset` | X | tweet (1 per user) | 50.000 | 50,0% | label file sintetis | **kontrol negatif** |
| `haziqhalifi/tiktok-2025-dataset` | TikTok | author (dari 7.225 video) | 6.693 author | – | **tidak ada label** | indikator tanpa label |

Catatan koreksi: Instagram berjumlah **576 + 120 = 696** baris (dicek langsung dari `train.csv` dan
`test.csv`). Angka "579 + 120" yang sempat beredar tidak sesuai dengan file.

---

## 3. Arsitektur yang benar-benar berjalan

Worksheet 2 menggambarkan: Kaggle → batch ingest → data lake HDFS/S3 (raw CSV, curated Parquet) →
Spark → Spark MLlib → MongoDB → Power BI. Status sebenarnya per 5 November 2026:

| Komponen | Status | Catatan jujur |
|---|---|---|
| Kaggle → batch ingest | ✅ empat sumber | file mentah di `data/landing/<kunci>/` |
| Data lake HDFS/S3 | ⚠️ **filesystem lokal** | `data/landing` berperan sebagai zona raw dan `data/curated` sebagai zona curated. Tidak ada HDFS atau S3/MinIO |
| Curated Parquet | ⚠️ sebagian | `tiktok_videos_clean.parquet` dan `cross_platform_account_scores.parquet` (25.335 baris, dari sintesis). Track X, Instagram, dan Facebook membaca CSV mentah langsung |
| Spark | ✅ | Spark 3.5.3 local mode, semua sesi lewat `common.get_spark` |
| Spark MLlib | ✅ X, Instagram, Facebook | RF + LR. TikTok sengaja tanpa classifier (tidak ada label) |
| MongoDB | ✅ **baru** | `mongo:7.0` di Docker, ditulis Spark lewat `mongo-spark-connector` 10.4.0 (§9) |
| Power BI | ⚠️ data siap, dashboard belum | 10 CSV datar di `outputs/powerbi/`. Dashboard-nya harus dibuat kelompok |
| Model tersimpan | ❌ | `outputs/models/` kosong. Akun baru belum bisa diberi skor tanpa melatih ulang (§12) |

---

## 4–7. Empat pertanyaan analitik Worksheet 3

Judul §4–§7 adalah rumusan **asli** Worksheet 3, dalam urutan worksheet. Pemetaan ke Tujuan Khusus:

| Pertanyaan analitik | Tujuan Khusus | Bagian |
|---|---|---|
| 1. Fitur metadata profil dan pola aktivitas apa yang paling membedakan? | TK 1 — mengidentifikasi fitur metadata profil yang paling berpengaruh | §4 |
| 2. Akun dengan karakteristik seperti apa yang berkemungkinan tinggi diklasifikasikan bot/spam? | TK 2 — mengklasifikasikan setiap akun | §5 (+ level risiko §8) |
| 3. Seberapa akurat kombinasi fitur metadata dan aktivitas memprediksi label? | TK 2 | §6 |
| 4. Apakah pola perilaku bot di satu platform juga berlaku di platform lain? | TK 3 — membandingkan karakteristik bot antarplatform | §7 |

**Cakupan.** Tujuan Khusus 3 hanya menyebut **Instagram, Twitter, dan Facebook**. TikTok ditambahkan
sebagai platform keempat lewat keputusan kelompok pada 05/11. TikTok tidak berlabel, sehingga TikTok
tidak ikut dalam jawaban kuantitatif Q1–Q4 dan hanya dibahas sebagai batas dari apa yang dapat
diklaim (§7.4, §8).

## 4. Q1 — "Fitur metadata profil dan pola aktivitas apa yang paling membedakan akun bot/spam dari akun manusia asli?"

*Tujuan Khusus 1.* **Jawaban singkat:** fitur yang paling membedakan adalah **ukuran audiens** (sedikit
followers/teman) di ketiga platform. Setelah itu, di X: **likes yang diberikan** dan jumlah akun yang
diikuti. Di Instagram: **digit di username**, kelengkapan profil, dan jumlah post. Empat fitur yang
disebut Tujuan Khusus 1 (rasio followers/following, umur akun, kelengkapan profil, verified) memang
membedakan, tetapi **tidak satu pun yang terkuat**.

### 4.1 Fitur paling berpengaruh, dan apakah ukuran importance-nya bisa dipercaya

Worksheet 3 memperingatkan bias impurity importance. Track X membuktikannya pada data yang
jawabannya sudah diketahui: di dataset sintetis, impurity importance memberi 90% bobot pada kata-kata
TF-IDF yang murni noise, sementara permutation importance di test set menunjukkan tidak ada fitur
yang berarti. Karena itu peringkat di bawah memakai **permutation importance pada data uji** sebagai
ukuran utama dan impurity sebagai pembanding.

| Dataset | Top-3 permutation (penurunan ROC-AUC test) | Kesesuaian dengan impurity |
|---|---|---|
| X real | `log_favourites_count` 0,056 · `log_followers_count` 0,047 · `log_friends_count` 0,027 | urutan sedikit berubah: likes yang diberikan naik ke #1, `description_length` turun dari #9 ke #12 |
| Instagram | `log1p_n_followers` 0,054 · `nums_length_username` 0,029 · `profile_pic` 0,011 | Spearman ρ = 0,84 |
| Facebook | `log1p_n_urls_shared` 0,019 · `log1p_n_friends` 0,014 · `frac_posts_with_urls` 0,014 | ρ = 0,71, tetapi **tidak ada satu kolom pun yang penting sendirian**. Mengacak 5 kolom sekaligus menurunkan ROC-AUC 0,197 |

Rincian per konsep perilaku ada di §7.3 dan [`outputs/figures/synthesis_importance_by_concept.png`](../outputs/figures/synthesis_importance_by_concept.png).

### 4.2 Empat fitur yang disebut Tujuan Khusus 1

Kekuatan pemisah ditulis sebagai δ (Cliff's delta univariat; §7.2) dan porsi permutation importance
konsep itu di model RF.

| Fitur TK 1 | X real | Instagram | Facebook | Kesimpulan |
|---|---|---|---|---|
| Rasio followers/following | δ −0,19; 4,4% importance | δ −0,52 | tidak berbeda (δ −0,01) | lebih lemah daripada jumlah followers itu sendiri |
| Umur akun | δ −0,28 (bot lebih muda); 7,5% | tidak tersedia | δ +0,47 (spam lebih *tua*) | arahnya berlawanan antar platform |
| Kelengkapan profil | bio: δ −0,23; panjang bio δ −0,33; 4,3% | bio δ −0,61; tanpa foto δ +0,57; 13,1% | tidak tersedia | kuat di Instagram, lemah di X |
| Status verified | δ −0,26: 2,5% bot verified vs 28,9% manusia; 6,7% | tidak tersedia | tidak tersedia | penanda kepercayaan yang jelas, tetapi hanya ada di X |

### 4.3 Pola perilakunya, per platform

- **X (bot):** akun bot **tidak dipakai seperti akun manusia**. Bot memberi lebih sedikit like
  (median 1.131 vs 3.649), punya lebih sedikit followers (25 vs 1.046), dan mengikuti lebih sedikit
  akun (31 vs 474). Bot lebih muda, lebih jarang punya bio, jarang terverifikasi (hanya 4,2% akun
  verified adalah bot), dan geolokasinya mati. Secara marginal bot memposting *lebih sedikit*
  per hari, tetapi setelah dikondisikan pada fitur lain koefisien LR-nya positif (+0,57). Polanya
  adalah *laju posting tinggi untuk akun yang tidak diikuti siapa pun dan tidak menyukai apa pun*.
- **Instagram (fake):** akun palsu **kosong dan bernama mesin**. Followers sedikit (median 43,5 vs
  577), post nol (median 0 vs 65), tanpa bio (hanya 12,6% punya bio vs 73,9%), 57,2% tanpa foto
  profil (vs 0,6%), dan username penuh digit (median 30% digit vs 0%).
- **Facebook (spam), jika dibaca apa adanya:** profil siaran. Teman sedikit (47 vs 1.245),
  komunitas banyak (841 vs 51), tautan banyak (1.897 vs 46), post banyak (2.751 vs 765), dan akun
  lebih tua. **Pola ini tidak boleh dikutip sebagai temuan tentang spammer Facebook** (§6, §7.3 f).
- **TikTok:** tidak dapat dijawab, karena tidak ada label. Dari 24 author yang cukup datanya, nol
  yang memposting lebih teratur daripada posting acak.

## 5. Q2 — "Akun dengan karakteristik seperti apa yang memiliki kemungkinan tinggi diklasifikasikan sebagai bot/spam?"

*Tujuan Khusus 2.* Pertanyaan ini dijawab dengan **profil nilai fitur**, bukan dengan ambang. Untuk
setiap dataset, kami mengambil **10% akun dengan skor RF out-of-sample tertinggi** dan membandingkannya
dengan semua akun yang diberi skor serta dengan tiap kelas (`comparison.worksheet_q2_high_score_profiles`;
tabel Power BI `fact_high_score_profile.csv`; koleksi MongoDB `high_score_profiles`). Kelompok ini
kurang lebih sama dengan bagian atas Level 2 (§8).

**Jawaban singkat:** di X, akun berskor tinggi adalah akun **yang tidak terverifikasi, geolokasinya
mati, hampir tanpa followers, dan tidak mengikuti siapa pun, tetapi tetap memposting**. Di Instagram,
akun berskor tinggi adalah **akun yang benar-benar kosong**: tanpa post, tanpa bio, tanpa foto, dan
username sekitar sepertiganya digit. Di Facebook, profil berskor tinggi adalah profil blok "spam"
pada file-nya. Di data sintetis, **tidak ada karakteristik apa pun** yang membuat akun berskor tinggi.

**X real:** 748 akun teratas dari 7.474 (skor ≥ 0,902). **98,3% bot** (base rate 33,4%), dan 13 manusia.

| Indikator (median, atau % akun) | Top 10% | Semua akun uji | Manusia | Bot |
|---|---|---|---|---|
| Followers | **9** (IQR 4–16) | 356 | 962 | 26 |
| Akun yang diikuti | **0** (IQR 0–20) | 301,5 | 474 | 38 |
| Total post seumur akun | 1.679 | 4.210 | 7.676 | 2.195 |
| Post per hari | 0,65 | 1,51 | 2,46 | 0,84 |
| Likes yang diberikan | 1.200 | 2.062 | 3.560 | 1.126 |
| Umur akun (hari) | 2.775 | 3.209 | 3.388 | 2.878 |
| Punya bio | 64,3% | 81,1% | 88,1% | 67,3% |
| Verified | **0%** | 19,7% | 28,4% | 2,4% |
| Geolokasi aktif | **2,8%** | 46,0% | 57,7% | 22,7% |
| Avatar default | 0,4% | 1,5% | 0,6% | 3,4% |

Catatan: porsi followers di top 10% justru *lebih tinggi* (0,75 vs 0,60). Ini karena hampir semua akun
itu tidak mengikuti siapa pun, sehingga rasionya tidak berarti ketika kedua hitungan sangat kecil.
Rasio adalah ukuran yang buruk untuk akun sepi, dan inilah alasan rasio kalah kuat dari jumlah mentah
(§4.2). Avatar default **bukan** penanda di X: top 10% malah lebih jarang memakainya.

**Instagram:** 70 akun teratas dari 696 (skor ≥ 0,9998). **100% fake** (base rate 50%), tanpa satu
akun asli pun.

| Indikator | Top 10% | Semua | Genuine | Fake |
|---|---|---|---|---|
| Followers | **21** (IQR 10–32) | 165,5 | 577 | 43,5 |
| Akun yang diikuti | 27 | 252 | 438 | 76 |
| Jumlah post | **0** (IQR 0–0: semuanya nol) | 9 | 65 | 0 |
| Punya bio | **0%** | 43,3% | 73,9% | 12,6% |
| Tanpa foto profil | **95,7%** | 28,9% | 0,6% | 57,2% |
| Tautan di bio | 0% | 11,4% | 22,7% | 0% |
| Porsi digit di username | **0,345** | 0 | 0 | 0,30 |
| Akun private | 22,9% | 36,9% | 43,4% | 30,5% |

**Facebook (kasus peringatan):** 60 akun teratas dari 600 (skor ≥ 0,91). 88,3% spam (base rate
17,2%). Ketujuh akun legit di dalamnya **semuanya baris
berlabel legit di blok spam**, yaitu baris yang profilnya mengikuti blok, bukan label (§6).

| Indikator (median) | Top 10% | Semua | Legit | Spam |
|---|---|---|---|---|
| Teman | **45** | 1.159 | 1.245 | 47 |
| Akun yang diikuti | 65 | 1.132 | 1.235 | 64 |
| Total post | 3.021 | 876,5 | 765 | 2.751 |
| URL yang dibagikan | **1.995,5** (41× median) | 49 | 46 | 1.897 |
| Komunitas | **854** | 56 | 51 | 841 |
| Likes per post | 0,28 | 0,88 | 0,97 | 0,32 |
| Umur akun (hari) | 2.007 | 1.136 | 1.102 | 1.840 |

**X sintetis (kontrol negatif):** 988 akun teratas hanya 47,0% bot, *di bawah* base rate 49,9%, dan
524 di antaranya manusia. Profilnya praktis sama dengan semua akun: followers 4.673 vs 4.948,
verified 48,5% vs 49,8%, retweet 54 vs 50. Pada data tanpa sinyal, "akun berskor tinggi" tidak punya
karakteristik apa pun. Ini kontrol yang menunjukkan bahwa profil di atas memang berasal dari data,
bukan dari cara kami mengukurnya.

**TikTok:** tidak dapat dijawab, karena tanpa model tidak ada "kemungkinan diklasifikasikan". Sepuluh
author Level 1 (§8.3) masing-masing punya satu indikator perilaku: burst posting (3 author) atau
caption duplikat (7 author). Penjelasan wajarnya adalah akun klip siaran langsung, kreator dengan
caption tetap, dan unggah ulang (`tiktok_findings.md` §5.3).

**Implikasi bagi moderator:** di X, top 10% berisi 13 manusia, yaitu akun asli yang sepi dan tidak
terverifikasi. Mereka tidak dapat dibedakan dari bot dengan metadata profil saja. Itulah alasan
"berskor tinggi" berarti "tinjau", bukan "bot".

## 6. Q3 — "Seberapa akurat kombinasi fitur metadata dan aktivitas dapat digunakan untuk memprediksi label bot atau manusia pada sebuah akun?"

*Tujuan Khusus 2.* **Jawaban singkat:** pada data nyata yang besar (X), **cukup akurat: 0,8538**
melawan baseline 0,6664, dengan ROC-AUC 0,9231. Di Instagram lebih tinggi (0,924 ± 0,020), tetapi pada
sampel kecil yang dikurasi. Angka Facebook bukan ukuran akurasi deteksi spam.

Tabel berikut **diurutkan menurut bobot bukti, bukan menurut akurasi** (alasannya di §7.1).

| Dataset | Angka yang boleh dikutip | Baseline mayoritas | RF vs LR | Verdict |
|---|---|---|---|---|
| **X real** | RF acc **0,8538** (CI 0,8456–0,8616), bal. acc 0,8409, F1 0,7854, MCC 0,6749, ROC-AUC 0,9231, PR-AUC 0,8798 | 0,6664 | RF jelas lebih baik: LR 0,7877, CI tidak tumpang tindih | **dipercaya**. Split vendor memberi 0,855 |
| **Instagram** | RF acc **0,9239 ± 0,0203** (CV 5×10), ROC-AUC 0,979 ± 0,009, MCC 0,849 ± 0,040; 20 seed lain: 0,870–0,961 | ≈0,50 (0,5234 di test) | **tidak dapat dibedakan** (McNemar p = 1,0) | dapat dipertanggungjawabkan tetapi sempit |
| **Facebook** | RF 0,9474 (seed 42) / 0,9673 ± 0,0181 (CV) — **tidak dikutip sebagai kinerja** | 0,7895 (test) / 0,8283 (file) | tidak dapat dibedakan (McNemar p = 1,0) | **kegagalan data** |
| X sintetis | RF 0,4945, MCC −0,011, ROC-AUC 0,4925 | 0,501 | keduanya di baseline | kontrol negatif: tidak ada sinyal |
| TikTok | tidak ada | – | – | tidak ada hasil supervised |

Kesimpulan Q3: model benar-benar belajar sesuatu di X dan Instagram. Random Forest hanya terbukti
lebih baik daripada baseline Logistic Regression di X, dan di sana sinyalnya non-linear. Di
Instagram dan Facebook, LR sama baiknya. Di Facebook bahkan satu ambang `#friends <= 126` sama
baiknya dengan RF. Akurasi pada ambang 0,5 juga bukan satu-satunya cara memakai skor. Sebagai
antrean (§8), Level 2 di X memberi presisi 0,890 untuk 23,6% akun.

## 7. Q4 — "Apakah pola perilaku bot yang ditemukan pada satu platform juga berlaku pada platform lainnya?"

*Tujuan Khusus 3 (Instagram, Twitter, Facebook).* **Jawaban singkat: sebagian.** Inti pola "akun
sepi dan kosong" berlaku sama di X dan Instagram, dan **audiens kecil** berlaku di ketiga platform.
Namun sebagian besar sinyal tetap khas platform: 62% permutation importance model X dan 35% model
Instagram berada pada konsep yang tidak diukur platform lainnya (likes yang diberikan di X, digit
username di Instagram), dan konsep terbesar Facebook (berbagi tautan, 33%) tidak ada di X maupun
Instagram. Facebook berlawanan arah pada aktivitas dan
umur akun, dan arah itu sendiri tidak dapat dipercaya. Pola bot karenanya **sebagian lintas platform,
sebagian spesifik platform**, dan model satu platform tidak dapat dipindahkan ke platform lain.

### 7.1 Mengapa platform TIDAK diperingkat menurut akurasi

Jika diperingkat menurut akurasi, urutannya Facebook (0,97) > Instagram (0,93) > X (0,85). Urutan
itu **terbalik dari kenyataan**. Angka Facebook adalah artefak file berdua-blok yang terurut menurut
label. Angka X adalah satu-satunya yang diperoleh pada dataset besar, nyata, dan dikumpulkan secara
independen.

Sintesis ini juga mengukur **seberapa mudah setiap *file* dipisahkan pada indikatornya sendiri**,
yaitu rata-rata |Cliff's δ| per indikator (`comparison.file_separability`):

| File | Indikator | Rata-rata \|δ\| | Maks \|δ\| | Indikator dengan \|δ\| ≥ 0,6 |
|---|---|---|---|---|
| X real | 11 | 0,32 | 0,62 | 1 dari 11 |
| Instagram | 10 | 0,55 | 0,83 | 5 dari 10 |
| Facebook | 11 | **0,74** | 0,94 | **9 dari 11** |

Urutan keterpisahan file ini sama persis dengan urutan akurasi. Artinya, peringkat akurasi mengukur
**seberapa mudah file-nya**, bukan seberapa mudah bot di platform itu dideteksi. Sinyal perilaku
nyata jarang memisahkan dua kelas sebesar |δ| > 0,8 di hampir setiap kolom. Dua kelompok baris yang
dibuat secara terpisah, seperti di Facebook, justru menghasilkan pola seperti itu.

Maka pernyataan yang benar adalah:
- **Hanya X dan Instagram yang mendukung perbandingan kuantitatif yang sungguhan.** Keduanya data
  nyata, labelnya bermakna, dan tidak ada kebocoran. Instagram kecil dan dikurasi, X besar.
- **Facebook dilaporkan sebagai kasus peringatan**, tidak diperingkat.
- **TikTok tidak punya hasil supervised sama sekali.**

### 7.2 Yang dibandingkan: konsep perilaku, bukan skor

Setiap kolom dipetakan ke **konsep perilaku** (ukuran audiens, ikatan keluar, volume posting, dan
seterusnya). Untuk setiap konsep yang terukur di data berlabel, kami menghitung ROC-AUC univariat
dari nilai mentahnya untuk kelas tidak autentik, dengan 95% CI bootstrap (1.000 resample, seed 42).
AUC < 0,5 berarti nilai yang lebih tinggi lebih autentik. Arah hanya diklaim jika CI tidak memuat 0,5.
δ = 2·AUC − 1 adalah besar efeknya. Gambar: [`outputs/figures/synthesis_concept_direction.png`](../outputs/figures/synthesis_concept_direction.png).

**Indikator yang terukur di ≥ 2 platform berlabel:**

| Indikator | X real: AUC (δ) | Instagram: AUC (δ) | Facebook*: AUC (δ) | Searah? |
|---|---|---|---|---|
| Ukuran audiens (followers; FB: teman) | 0,192 (−0,62) | 0,085 (−0,83) | 0,061 (−0,88) | **ya, ketiganya**: audiens kecil → tidak autentik |
| Akun yang diikuti | 0,240 (−0,52) | 0,239 (−0,52) | 0,059 (−0,88) | **ya, ketiganya**: mengikuti sedikit → tidak autentik |
| Porsi followers f/(f+following) | 0,403 (−0,19) | 0,243 (−0,52) | 0,493 (CI memuat 0,5) | X = IG, FB tidak berbeda |
| Total post seumur akun | 0,314 (−0,37) | 0,084 (−0,83) | **0,923 (+0,85)** | X = IG; **FB berlawanan** |
| Post per hari | 0,332 (−0,34) | – | **0,909 (+0,82)** | **FB berlawanan** |
| Umur akun | 0,359 (−0,28) | – | **0,737 (+0,47)** | **FB berlawanan** |
| Punya bio | 0,383 (−0,23) | 0,194 (−0,61) | – | ya |
| Panjang bio | 0,336 (−0,33) | 0,180 (−0,64) | – | ya |
| Tanpa foto profil sendiri | 0,512 (+0,02) | 0,783 (+0,57) | – | arah sama, **besar efek berbeda 24×** |

\* Facebook ditandai: arahnya menggambarkan dua blok file, bukan spammer (`facebook_findings.md` §8.7).

### 7.3 Temuan

**(a) X dan Instagram mempelajari konsep yang sama.** Ketujuh indikator yang terukur di kedua
platform menunjuk ke arah yang sama. Polanya adalah akun dengan audiens kecil, sedikit ikatan keluar,
sedikit aktivitas seumur hidup, dan profil kosong. Ini masuk akal. "Bot" di dataset Treiman sebagian
besar adalah akun sepi atau dorman (bukan spammer follow-everyone), dan "fake" Instagram adalah
akun kosong.

**(b) Besar efeknya berbeda, dan perbedaannya sistematis.** Di Instagram |δ| 1,0 sampai 2,6 kali
lebih besar daripada di X: audiens 1,35×, akun diikuti 1,00×, porsi followers 2,65×, total post
2,23×, punya bio 2,62×, panjang bio 1,95×. Ini konsisten dengan sampel Instagram yang dipilih dari
kasus yang jelas. Pengecualian besarnya adalah **foto profil (24×)**. Di Instagram 57,2% akun palsu
tanpa foto vs 0,6% akun asli. Di X hanya 3,1% bot vs 0,7% manusia yang memakai avatar default, dan
koefisien LR-nya bahkan negatif (−0,10). "Tanpa foto → palsu" sebagian besar adalah **artefak seleksi
sampel Instagram**, bukan sinyal yang berpindah antar platform.

**(c) Audiens kecil adalah satu-satunya konsep yang searah di ketiga dataset berlabel**, dan konsep
ini masuk dua besar permutation importance di setiap model: X 21% (#2), Instagram 51% (#1),
Facebook 14% (#2).

**(d) Arah marginal ≠ arah kondisional.** Di Instagram, mengikuti sedikit akun menunjuk ke "palsu"
secara marginal (δ −0,52), tetapi koefisien LR-nya **positif** (+0,41). Artinya, *pada jumlah followers
yang sama*, mengikuti lebih banyak akun lebih mencurigakan, sesuai pola follow-for-follow. Di X, laju
posting juga berbalik (marginal −0,34, kondisional +0,57). Perbandingan lintas platform harus
menyebut ukuran mana yang dipakai.

**(e) Konsep bersama hanya menanggung sebagian sinyal.** Konsep yang diukur oleh kedua model, X dan
Instagram (audiens, ikatan keluar, kelengkapan profil), menanggung **38% permutation importance X**
(45% impurity) dan **65% milik Instagram** (61% impurity). Sisanya bersandar pada konsep khas platform:

| Model | Konsep khas platform terbesar (permutation share) |
|---|---|
| X real | **likes yang DIBERIKAN 25%** (konsep terbesar di X, tidak ada padanannya di platform lain), laju posting 12%, umur akun 7,5%, verified 6,7% |
| Instagram | **digit di username 28%**. Tidak diturunkan untuk X/TikTok karena berasal dari handle (identifier) |
| Facebook | **berbagi tautan 33%**, keanggotaan grup 12% |

Inilah alasan kuantitatif mengapa model dari satu platform **tidak dapat dipindah** ke platform
lain. Kesimpulan ini sejalan dengan temuan track TikTok bahwa 97–100% importance setiap model berada
pada fitur yang tidak ada di TikTok.

**(f) Facebook menunjuk ke arah sebaliknya.** Facebook sejalan di audiens (teman sedikit) dan ikatan
keluar, tetapi **berlawanan** pada total post, laju post, dan umur akun: "spam" di sini berarti akun
lebih tua yang memposting lebih banyak. Ini memang bisa mencerminkan konsep label yang berbeda (spam
siaran ≠ akun palsu kosong). Namun karena 9 dari 11 indikator Facebook punya |δ| ≥ 0,6 dan model
lebih baik menebak blok file daripada label, **arah Facebook tidak dapat dipakai sebagai bukti apa pun**.

### 7.4 Konsep yang TIDAK dapat dibandingkan, dan alasannya

| Konsep | Alasan |
|---|---|
| Likes yang diberikan (X `favourites_count`) vs likes yang diterima (FB `likes/post`, TikTok `likes`) | kata yang sama, **arah berlawanan**: yang satu aktivitas akun, yang lain respons audiens. Tidak boleh dipetakan |
| Teman Facebook vs followers X/Instagram | pertemanan Facebook bersifat timbal balik, sehingga sekaligus audiens dan ikatan keluar. Dibandingkan sebagai *parsial* |
| Digit di username | hanya Instagram yang menyediakannya sebagai fitur. Menurunkannya dari `screen_name` X atau handle TikTok berarti memakai identifier (prinsip §15 konsep: identifier bukan fitur) |
| Tautan di bio (IG) vs tautan di post (FB `#urlshared`) | konsep berbeda: profil vs perilaku posting |
| Volume video TikTok vs total post | `video_count` TikTok = video viral yang tertangkap dalam 6 minggu (median 1), bukan total post |
| **Semua konsep di TikTok** | tidak ada label, jadi arah tidak dapat diukur untuk konsep apa pun. Followers, following, umur akun, dan kelengkapan profil juga tidak ada di data TikTok |

---

## 8. Lapisan level risiko 0–4 (dokumen konsep §13)

Urutan dan makna level mengikuti §13 dokumen konsep: level 0–4 berada di atas probabilitas model,
dan level 3–4 memerlukan bukti koordinasi serta amplifikasi. **Nama levelnya kami ubah**, dan
pemetaannya dinyatakan terbuka di §8.1 agar kedua dokumen tidak berbeda diam-diam.

### 8.1 Definisi, dan pemetaan ke nama §13

**Pemetaan nama.** Kolom §13 dikutip verbatim dari dokumen konsep. Kolom kanan adalah nama yang
dipakai di laporan ini, di MongoDB (`risk_level_definitions.name_id`), dan di Power BI
(`dim_risk_level.name_id`). Nama §13 disimpan berdampingan di field `name_concept_doc` dan
`description_concept_doc`.

| Level | Kategori di konsep §13 | Deskripsi di konsep §13 | Nama di laporan ini |
|---|---|---|---|
| 0 | Authentic / Low Risk | "Tidak menunjukkan indikasi kuat perilaku tidak autentik dari fitur yang tersedia." | Tidak ada indikasi |
| 1 | Suspicious Automation | "Ada indikasi otomatisasi atau pola aktivitas yang tidak biasa, tetapi bukti masih terbatas." | Indikasi lemah - pantau |
| 2 | Inauthentic / Spam | "Indikasi perilaku tidak autentik atau spam lebih kuat." | Indikasi kuat tingkat akun - prioritas tinjauan manusia |
| 3 | Coordinated Manipulation | "Ada indikasi pola koordinasi antar akun yang dapat didukung oleh data jaringan/waktu/konten." | Dugaan koordinasi antar-akun |
| 4 | High-Risk Coordinated Influence | "Indikasi koordinasi dan amplifikasi kuat serta mempunyai potensi impact yang lebih besar." | Amplifikasi terkoordinasi |
| NA | *(tidak ada di §13)* | – | Tidak dapat dinilai |

**Mengapa namanya diubah.** Nama kategori §13 seperti "Inauthentic / Spam" dan "Authentic / Low Risk"
menyatakan *kesimpulan* tentang sebuah akun. Hal itu bertentangan dengan prinsip §15 dokumen konsep
sendiri: hasil model adalah indikasi berbasis data, bukan bukti mutlak mengenai niat pemilik akun.
Karena pengguna sasarannya moderator yang memutuskan (§1), nama di laporan ini menyatakan **kekuatan
bukti dan aksi yang pantas**, bukan vonis. Isi, urutan, dan ambang tiap level tidak berubah; hanya
labelnya.

**Mengapa ada tier NA.** §13 tidak punya padanan untuk "tidak dapat dinilai". Tanpa tier ini, akun
yang tidak punya bukti apa pun akan jatuh ke Level 0, yang di §13 bernama "Authentic / Low Risk". Itu
melanggar prinsip *missing is not zero*. Saat ini **16.355 akun berada di NA**: 6.483 author TikTok
yang perilakunya tidak dapat dinilai (95,4% author hanya punya satu video), dan seluruh 9.872 baris X
sintetis, karena model di dataset itu gagal gerbang bukti (§8.2).

**Syarat dan aksi tiap level:**

| Level | Nama | Syarat | Aksi yang disarankan | Dapat dicapai? |
|---|---|---|---|---|
| 0 | Tidak ada indikasi | skor < t1 (TikTok: perilaku dapat dinilai dan tidak ada indikator yang menyala). **Bukan berarti autentik** | tidak masuk antrean | ya |
| 1 | Indikasi lemah - pantau | t1 ≤ skor < t2 | pantau / sampel QA | ya |
| 2 | Indikasi kuat tingkat akun - prioritas tinjauan manusia | skor ≥ t2 | antrean tinjauan manusia, urut skor. **Tidak pernah sanksi otomatis** | ya, untuk X, IG, FB |
| 3 | Dugaan koordinasi antar-akun | Level 2 + bukti akun bertindak bersama akun lain: waktu posting sinkron, konten sama/mirip lintas akun, atau ikatan jaringan/infrastruktur | eskalasi ke analis investigasi | **tidak**, tidak ada data |
| 4 | Amplifikasi terkoordinasi | Level 3 + bukti kelompok itu memperkuat target yang sama (burst lintas akun pada post/URL/hashtag yang sama) dengan jangkauan terukur | eskalasi sebagai investigasi kampanye | **tidak**, tidak ada data |
| NA | Tidak dapat dinilai | tidak ada bukti yang dapat dihitung, atau model tidak punya sinyal. **Missing is not zero: NA ≠ Level 0** | jangan dilaporkan "aman" | – |

### 8.2 Cara ambang diturunkan (bukan angka bulat)

- Skor yang dipakai adalah probabilitas RF **out-of-sample** dari tiap track. X: set uji yang
  ditahan. Instagram/Facebook: set uji + skor *out-of-fold* CV untuk akun latih.
- **t1** memaksimalkan F2 (recall dibobot 2×), karena Level 1 tidak berakibat apa pun bagi akun,
  jadi jaringnya lebar. **t2** memaksimalkan F0,5 (presisi dibobot 2×), karena setiap akun Level 2
  menghabiskan waktu moderator dan bisa berarti orang sungguhan masuk antrean. β adalah pilihan
  kebijakan yang dinyatakan terbuka dan dapat diubah.
- **Cross-fitting 2-fold** (berstrata, seed 42): ambang diturunkan dari satu paruh lalu diterapkan
  ke paruh lainnya. Setiap angka di tabel adalah out-of-sample terhadap model *dan* terhadap ambang
  (aturan kontrak no. 6). Ambang "deployment" untuk akun baru dihitung dari semua akun, lengkap
  dengan CI bootstrap.
- **Gerbang bukti (evidence gate):** level hanya diberikan jika batas bawah Wilson 95% presisi
  Level ≥ 2 melebihi base rate. Gerbang ini ditambahkan setelah kontrol negatif menunjukkan masalah:
  pada model tanpa sinyal, potongan F-beta menaruh **99% akun di Level 2 dengan presisi = base rate**.
  Tanpa gerbang, lapisan level akan memproduksi "risiko" dari noise.

### 8.3 Tabel level (cross-fitted)

Gambar: [`outputs/figures/synthesis_risk_levels.png`](../outputs/figures/synthesis_risk_levels.png).

**X real** (7.474 akun uji, base rate 0,334). Ambang deployment t1 = 0,356 (CI bootstrap
0,282–0,369), t2 = 0,725 (0,700–0,742). Ambang per paruh: A 0,320 / 0,728, B 0,357 / 0,713, jadi stabil.

| Level | Akun | Volume | Presisi (95% CI) | Porsi semua bot | Porsi semua manusia |
|---|---|---|---|---|---|
| 0 | 3.996 | 53,5% | 0,059 (0,052–0,066) | 9,4% | 75,5% |
| 1 | 1.715 | 22,9% | 0,402 (0,379–0,426) | 27,7% | 20,6% |
| **2** | **1.763** | **23,6%** | **0,890 (0,875–0,904)** | **62,9%** | **3,9% (194 orang)** |
| ≥ 1 | 3.478 | 46,5% | 0,650 | recall 0,906 | 24,5% (1.219 orang) |

**Instagram** (696 akun, base rate 0,50). t1 = 0,212 (CI 0,137–0,401), t2 = 0,614 (0,513–0,816).
Ambangnya **tidak stabil** (paruh A 0,220 / 0,677, paruh B 0,390 / 0,745), konsekuensi dari data kecil.

| Level | Akun | Volume | Presisi (95% CI) | Porsi semua fake | Porsi semua genuine |
|---|---|---|---|---|---|
| 0 | 313 | 45,0% | 0,048 (0,029–0,078) | 4,3% | 85,6% |
| 1 | 76 | 10,9% | 0,461 (0,353–0,572) | 10,1% | 11,8% |
| **2** | **307** | **44,1%** | **0,971 (0,945–0,985)** | **85,6%** | **2,6% (9 akun)** |

**Facebook** (600 akun, base rate 0,172), *kasus peringatan*. t1 = 0,365, t2 = 0,425. Di paruh A,
F2 dan F0,5 jatuh di ambang yang sama (0,555), sehingga **Level 1 praktis kosong (1 akun)**. Skor RF
bimodal tanpa wilayah tengah. Ini tanda lain bahwa file-nya dua blok, bukan satu populasi.

| Level | Akun | Volume | Presisi (95% CI) | Porsi semua spam |
|---|---|---|---|---|
| 0 | 498 | 83,0% | 0,026 (0,015–0,044) | 12,6% |
| 1 | 1 | 0,2% | 0,0 | 0% |
| 2 | 101 | 16,8% | 0,891 (0,815–0,938) | 87,4% |

**X sintetis** (kontrol negatif): **gerbang gagal.** Presisi Level ≥ 2 adalah 0,4998 (batas bawah
0,4899) melawan base rate 0,499. Semua 9.872 baris disimpan sebagai **NA**.

**TikTok** (6.693 author): tidak ada probabilitas dan tidak ada label. Level diberikan dengan aturan
dari indikator perilaku track TikTok dan **dibatasi maksimal Level 1**, karena tanpa label presisi
potongan mana pun tidak dapat diukur. Sinyal lemah yang terkontaminasi popularitas (outlier
engagement, IsolationForest) tidak pernah menaikkan level.

| Level | Author | Asal |
|---|---|---|
| NA | 6.483 | 5.667 `behaviour_not_assessable` + 816 `weak_signal_only` yang perilakunya tidak dapat dinilai (author satu video) |
| 0 | 200 | 188 `no_indicator_fired` + 12 `weak_signal_only` yang dapat dinilai dan bersih |
| 1 | 10 | `secondary_context` (burst ATAU caption duplikat) |
| 2–4 | 0 | tidak dapat dicapai |

### 8.4 Yang harus diketahui moderator: presisi bergantung pada prevalensi

Presisi di atas berlaku pada prevalensi sampel. Dengan TPR/FPR cross-fitted, Level ≥ 2 di X pada
prevalensi yang **diasumsikan** (ilustrasi saja, prevalensi sebenarnya tidak diketahui):

| Prevalensi bot | Presisi Level ≥ 2 | Akun ditandai per 10.000 | Di antaranya orang sungguhan |
|---|---|---|---|
| 33,4% (sampel) | 0,890 | – | – |
| 10% | 0,64 | 980 | 351 |
| 5% | 0,46 | 685 | 370 |
| 1% | 0,14 | 449 | 386 |

Pada prevalensi rendah, sebagian besar antrean Level 2 adalah manusia. Inilah alasan paling kuat
mengapa sistem ini hanya boleh menjadi alat triase untuk tinjauan manusia.

### 8.5 Mengapa Level 3 dan 4 tidak dapat dicapai

Level 3–4 memerlukan bukti **antar-akun**. Kelima dataset hanya berisi informasi **per akun**:

| Bukti | X real | X sintetis | Instagram | Facebook | TikTok |
|---|---|---|---|---|---|
| Timestamp per post | ✗ snapshot profil | ◐ 1 tweet/user, waktu sintetis | ✗ | ✗ | ◐ 95,4% author 1 video, hanya video viral |
| Konten per post | ✗ | ◐ kata Faker. "Konten sama" terjadi **by construction**, jadi akan menjadi temuan palsu | ✗ | ✗ | ◐ 29 caption lintas author, 23 di antaranya hanya hashtag tren (`#fyp`) |
| Graf interaksi / follow | ✗ hanya hitungan | ✗ | ✗ hanya hitungan | ✗ hanya hitungan | ✗ |
| Burst lintas akun pada target sama | ✗ | ✗ | ✗ | ✗ | ✗ hanya total per video |
| Infrastruktur bersama (device/IP) | ✗ | ✗ | ✗ | ✗ | ✗ |

Memberi Level 3 atau 4 dengan data ini adalah fabrikasi (prinsip §15: tidak ada klaim koordinasi
tanpa bukti). **Data yang dibutuhkan:**
1. Timeline per akun: setiap post dengan timestamp, teks, URL, hashtag, dan referensi
   (retweet/quote/reply/duet/stitch), untuk **banyak akun dalam jendela waktu yang sama**.
2. Graf interaksi atau follow (siapa me-retweet, membagikan, membalas, atau mengikuti siapa).
3. Log event interaksi berstempel waktu per target, agar burst lintas akun dan jangkauannya dapat
   diukur (syarat Level 4).
4. Ground truth di tingkat **kampanye**, misalnya daftar takedown operasi terkoordinasi dari platform
   dengan provenance yang jelas, bukan hanya label per akun.
5. Kandidat dataset untuk dijajaki (isinya perlu diverifikasi dulu): Cresci-2017 versi lengkap,
   yang berisi tweet akun social spambot (di box ini hanya ada file users-nya), dan TwiBot-22
   (tweet + graf relasi).

---

## 9. MongoDB dan Power BI: dibangun dan diverifikasi

Detail koneksi dan skema lengkap ada di **`docs/mongodb_powerbi.md`**. Ringkasnya:

- Service `mongo` (`mongo:7.0`) dibatasi `mem_limit 512m`, cache WiredTiger 0,25 GB, dan
  port `127.0.0.1:${MONGO_PORT:-27017}` (hanya loopback, akses jarak jauh lewat SSH tunnel). Pemakaian memori
  setelah dimuat sekitar 160 MiB. Compose file: `deploy/mongodb/docker-compose.yml`.
  Kredensial: `.env` (chmod 600).
- Database `socmed_bot_detection`. `account_scores` (25.335 dokumen) ditulis **oleh Spark** lewat
  `mongo-spark-connector_2.12:10.4.0`. Koleksi kecil lainnya ditulis dengan pymongo.
- Handle atau `screen_name` **tidak** dimuat. Akun dirujuk dengan `account_uid` dan `source_ref`
  (id X, baris file IG, profile id FB, kunci pseudonim TikTok).

**Verifikasi round-trip** dengan `mongosh` di dalam container. Output asli lengkap disimpan saat run (tidak di-commit); di sini Q2 disusun ulang menjadi kolom, `Long('…')` ditulis sebagai angka biasa, dan Q4–Q6 dipersingkat:

```text
// Q1 collections and document counts
account_scores           25335
concept_comparison       32
coordination_evidence    25
cross_platform_summary   1
datasets                 5
feature_importance       338
high_score_profiles      37
model_metrics            5
risk_level_definitions   6
risk_level_summary       34

// Q2 accounts per dataset and risk level
facebook           0   498      instagram          0   313      twitter_real       0   3996
facebook           1   1        instagram          1   76       twitter_real       1   1715
facebook           2   101      instagram          2   307      twitter_real       2   1763
tiktok             0   200      tiktok             1   10       tiktok             NA  6483
twitter_synthetic  NA  9872

// Q3 precision per level recomputed from stored labels (X real)
db.account_scores.aggregate([{$match:{dataset:"twitter_real"}},
  {$group:{_id:"$risk_level", n:{$sum:1}, bots:{$sum:"$label"}}}, ...])
{ _id: 0, n: 3996, bots: 234,  precision: 0.0586 }
{ _id: 1, n: 1715, bots: 690,  precision: 0.4023 }
{ _id: 2, n: 1763, bots: 1569, precision: 0.89 }

// Q4 top of the X human-review queue
db.account_scores.find({dataset:"twitter_real", suggested_action:"human_review"})
  .sort({rf_p_inauthentic:-1}).limit(3)
{ account_uid: 'x_<id>', rf_p_inauthentic: 0.984797, risk_level_name: 'Indikasi kuat tingkat akun - prioritas tinjauan manusia' }
...

// Q5 a TikTok author that cannot be assessed is stored as null, not 0
{ account_uid: 'tt_<author_key>', label: null, risk_level: null, risk_level_code: 'NA',
  risk_level_basis: 'not_assessable', tiktok_video_count: 1 }

// Q6 unreachable levels
db.risk_level_definitions.find({reachable:false}) -> code '3' (Dugaan koordinasi antar-akun),
                                                     code '4' (Amplifikasi terkoordinasi)

// Q8 level names: concept doc section 13 vs this report
db.risk_level_definitions.find({}, {code:1, name_concept_doc:1, name_id:1}).sort({code:1})
0   Authentic / Low Risk               Tidak ada indikasi
1   Suspicious Automation              Indikasi lemah - pantau
2   Inauthentic / Spam                 Indikasi kuat tingkat akun - prioritas tinjauan manusia
3   Coordinated Manipulation           Dugaan koordinasi antar-akun
4   High-Risk Coordinated Influence    Amplifikasi terkoordinasi
NA  null                               Tidak dapat dinilai
```

`src/synthesis.py` juga menjalankan pemeriksaan otomatis dan **gagal (assert)** bila ada yang tidak
cocok. Pemeriksaannya: jumlah dokumen = jumlah baris pandas; jumlah per dataset sama; tabel level
yang dihitung ulang di MongoDB **identik** dengan tabel cross-fitted Python; dataset yang gagal
gerbang tersimpan seluruhnya sebagai null; akurasi RF X yang dibaca balik = 0,8538; dan level 0–4
membawa nama laporan ini maupun nama §13, sedangkan NA tidak punya nama §13. Hasilnya
tercatat di `cross_platform_comparison.json` → `mongodb.verification_queries`.

**Power BI:** sepuluh CSV di `outputs/powerbi/` (UTF-8, null = kosong):
`fact_account_scores` (25.335), `dim_dataset` (5), `dim_risk_level` (6, membawa `name_id` **dan**
`name_concept_doc`), `fact_model_metrics` (152), `fact_feature_importance` (338),
`fact_importance_by_concept` (50), `fact_concept_comparison` (32), `fact_high_score_profile` (37),
`fact_risk_level_summary` (34), `fact_coordination_evidence` (25). Kolom kode level (`code`,
`risk_level_code`, `level_code`) harus bertipe **Text** di Power Query, karena `"NA"` adalah kode,
bukan null. Relasi dan saran halaman dashboard ada di `docs/mongodb_powerbi.md` §3. Kelompok dapat membangun dashboard tanpa menyentuh Spark.

---

## 10. Catatan Bu Gema, satu per satu

### Catatan 1 — Pengguna sasaran harus dinyatakan eksplisit
Dijawab di §1: moderator *trust & safety* yang menyaring akun untuk tinjauan manusia, bukan sistem
pemblokiran otomatis. Pilihan ini membentuk seluruh desain: keluaran berupa antrean terurut, level
dengan aksi "tinjau" (tidak pernah "blokir"), NA yang tidak dihitung sebagai aman, dan perhatian
khusus pada false positive.

### Catatan 2 — Parameter harus mencakup seluruh dataset agar hasil tidak bias
Kami membacanya sebagai: **evaluasi dan aturan harus berlaku untuk seluruh data, bukan satu potongan
yang kebetulan menguntungkan**, tanpa membiarkan data uji ikut menyetel parameter. Pemenuhannya:
- *Evaluasi atas seluruh data:* Instagram dan Facebook memberi skor out-of-sample untuk **setiap**
  akun (test holdout + out-of-fold CV), dengan CV 5×10 dan 20 seed split lain. Angka seed 42 Instagram
  (0,953) terbukti +1,45 SD di atas rata-rata CV, dan inilah contoh bias yang dicegah catatan ini.
  X diperiksa ulang dengan split vendor (0,855 vs 0,854).
- *Level risiko atas seluruh data:* ambang di-cross-fit sehingga **semua** akun mendapat level
  out-of-sample, bukan hanya separuhnya.
- *Tidak membuang kasus sulit:* TikTok mempertahankan 6.693 author, termasuk 95,4% yang hanya punya
  satu video. Fitur yang tidak terdefinisi dibiarkan NA dan tidak dihapus. Di X, `lang` yang kosong
  menjadi level kategori sendiri dan tidak dibuang.
- *Batasnya:* parameter pra-proses (scaler, kosakata TF-IDF, level `lang`) sengaja hanya di-fit
  pada data latih. "Seluruh dataset" berlaku untuk evaluasi, bukan untuk fitting, agar tidak bocor.

### Catatan 3 — Model boleh diganti
- RF tetap model utama dan LR baseline (Worksheet 3), supaya hasil antar-track dapat dibandingkan.
- **Bukti kapan model penting:** di X, RF mengalahkan LR jauh (0,854 vs 0,788, CI tidak tumpang
  tindih) karena sinyalnya non-linear. Di Instagram dan Facebook keduanya tidak dapat dibedakan
  (McNemar p = 1,0), sehingga model yang lebih sederhana sudah cukup. Di Facebook satu ambang
  `#friends <= 126` menyamai RF.
- **Di TikTok model memang diganti:** tanpa label, classifier tidak bisa dilatih maupun divalidasi,
  jadi diganti dengan indikator perilaku berbasis aturan.
- **Mengganti model tidak memperbaiki data:** tidak ada model yang bisa menemukan sinyal di data
  sintetis X, dan model apa pun akan mempelajari blok-blok Facebook. Opsi lanjutan untuk X adalah
  Gradient-Boosted Trees, tetapi kurva CV RF sudah mendatar (maxDepth 16 → 20: +0,0015 PR-AUC), jadi
  peningkatannya kemungkinan kecil.

### Catatan 4 — Target sekitar 80%; rendah boleh tetapi tidak di bawah 50%; mendekati 99% harus dicurigai

| Dataset | Angka | Terhadap catatan 4 |
|---|---|---|
| **X real** | **0,8538** | **inilah hasil yang mendarat di target ~80%**, pada data nyata yang besar, tanpa di-tuning ke arah 80%, dan sudah diaudit kebocorannya |
| Instagram | 0,924 ± 0,020 (CV) | tinggi tetapi dapat dipertanggungjawabkan: data kecil dan bersih, tidak bocor, kurva belajar mendatar. Bukan 99%, dan 95,3% dari satu split tidak dikutip |
| **Facebook** | 0,947 / 0,967 | **persis kasus "mendekati 99%" yang diperingatkan.** Audit menemukan file dua blok terurut label, satu ambang menyamai RF, error model = baris yang tidak sesuai bloknya, dan nilai tampak dibuat. **Dilaporkan sebagai kegagalan data, bukan keberhasilan model** |
| X sintetis | 0,4945 | di bawah 50%, tetapi bukan model yang rusak: data tanpa sinyal tersebar ±1% di sekitar 50%. Dilaporkan apa adanya (tidak di-*re-run* sampai lewat 0,5) sebagai bukti datanya kosong |
| TikTok | – | tidak ada akurasi, dan tidak dibuat-buat |

### Catatan 5 — Standardisasi dan aturan setiap dataset harus dinyatakan

| Dataset | Label → target | Identifier dibuang | Missing | Transformasi & rekayasa fitur | Penyeimbangan & split |
|---|---|---|---|---|---|
| **X real** | `account_type`: bot → 1, human → 0 | `id`, `screen_name`, indeks baris. Juga dibuang: path gambar profil (missing bergantung label, sehingga bocor), URL gambar, `split` vendor, `statuses_count` (= laju × umur), `created_at` (= tanggal koleksi − umur) | bio NULL → `has_description` 0 / panjang 0 (nilai sebenarnya); `lang` NULL → level `__missing__`; lokasi `unknown` → `has_location` 0. Semua input dipastikan bebas NULL | ln(1+x) pada followers, friends, favourites, tweets/day; `follower_share` = (f+1)/(f+fr+2); boolean → 0/1; `lang` one-hot (< 30 akun latih → `__other__`), di-fit pada latih | bobot kelas n/(2·n_kelas) = 0,748 / 1,509; stratified 80:20 seed 42 (29.964 / 7.474); CV 3-fold dalam latih untuk `maxDepth` / `regParam` (PR-AUC); LR dengan StandardScaler |
| **X sintetis** | `Bot Label` | `User ID`, `Username`, `Location` | `Hashtags` NULL = tidak ada hashtag (0 adalah nilai sebenarnya) | `Created At` → jam, hari, hari sejak 2020-01-01; panjang tweet; TF-IDF `Tweet` & `Hashtags` (vocab 1.000, minDF 5) di-fit di dalam fold (1.949 dimensi) | sama seperti X real. Peran: kontrol negatif |
| **Instagram** | `fake` | tidak ada kolom ID; `account_key` = file + baris, hanya provenance | tidak ada missing (0 dari 7.656 sel) | log1p pada `#posts`, `#followers`, `#follows`; 11 kolom profil dipakai semua | seimbang, tanpa bobot; dua file vendor digabung lalu stratified 80:20 seed 42 (589 / 107); CV 5×10 dalam latih; 20 seed lain; partisi vendor sebagai cek; tanpa tuning |
| **Facebook** | `Label`: spam → 1 | `profile id` (identifier **dan** kebocoran label, r = 0,58) | `fpurls` 2 baris → direkonstruksi `#urlshared / #postshared` (definisi kolomnya sendiri; cocok di 594/598 baris), bukan 0 atau median | log1p pada 7 kolom hitungan; nama kolom asli dipertahankan (lihat §12 no. 2) | bobot ≈ 0,6 / 3,0; stratified seed 42 (505 / 95); CV 5×10; tanpa tuning |
| **TikTok** | **tidak ada**; tidak dibuat | `video_id`, `video_url`, `author` → kunci pseudonim SHA-256 bergaram | fitur waktu NA di bawah jumlah video yang dibutuhkan (≥ 2 / ≥ 3 / ≥ 5); 0 komentar pada video > 140 ribu plays → *unknown*; `fetch_time` dibuang (99,9% kosong) | video → author (handle: hapus `@`, trim, huruf kecil); `views`/`plays` dan `create_time`/`posted_time` digabung; hashtag = gabungan kolom + caption; judul musik placeholder dikecualikan | tanpa split (tanpa label); indikator tri-state (1 / 0 / NA) |
| **Sintesis** | – | handle/`screen_name` tidak dimuat ke MongoDB/Power BI | NA ≠ Level 0; dataset gagal gerbang → semua NA | AUC univariat pada nilai mentah (invarian terhadap transformasi monoton) | ambang di-cross-fit 2-fold seed 42; gerbang bukti |

---

## 11. Keterbatasan yang harus diketahui pembaca

1. **Label adalah penilaian penyusun dataset**, bukan putusan platform, dan berbeda konsep antar
   platform (bot ≠ fake ≠ spam).
2. **Snapshot lama:** X Agustus 2020 (API v1.1), Instagram Maret 2019. Platform dan taktik operator
   bot sudah berubah.
3. **Prevalensi sampel ≠ prevalensi nyata.** Presisi di lapangan akan lebih rendah (§8.4).
4. **Hanya metadata profil** di X, Instagram, dan Facebook. Tidak ada konten, waktu, atau jaringan,
   padahal sinyal-sinyal itulah yang menangkap otomasi canggih dan koordinasi.
5. **Instagram kecil dan dikurasi.** Satu akun mengubah akurasi 0,93 poin, dan ambang level tidak stabil.
6. **TikTok tersaring popularitas** (setiap video ≥ 140,8 ribu plays). Akun berjangkauan rendah, tempat
   sebagian besar spam berada, tidak ada di sampel.

---

## 12. Yang harus diperbaiki kelompok

1. **Diagram arsitektur Worksheet 2 masih hanya Twitter dengan satu jalur ingest.** Diagram harus
   menjadi **empat sumber** (X, Instagram, Facebook, TikTok), masing-masing dengan batch ingest ke
   zona raw. Cabang TikTok digambar terpisah: agregasi author → indikator, tanpa classifier MLlib.
   MongoDB dan Power BI kini benar-benar ada (§9). HDFS/S3 dalam praktiknya adalah filesystem lokal,
   jadi pilih salah satu: pasang HDFS atau MinIO, atau beri label jujur di diagram.
2. **Dokumen konsep §4.1 memuat nama kolom Facebook hasil tebakan yang tidak ada.**
   `avg_comments` → **`avgcomment/post`**, `avg_likes` → **`likes/post`**, `avg_tags` →
   **`tags/post`**, `avg_hashtags` → **`#tags/post`**. Skema lengkap yang sudah diverifikasi ada di
   `facebook_findings.md` §3.
3. **Cakupan bergeser dari tiga dataset (Worksheet 3) ke empat platform (chat kelompok 05/11)**, dan
   dalam praktik menjadi lima dataset. Tujuan Khusus 3 masih hanya menyebut Instagram, Twitter, dan
   Facebook. Dokumen tertulis belum menyusul. Perbarui Worksheet 3 (termasuk TK 3), dokumen konsep,
   laporan, dan slide, dan nyatakan bahwa TikTok masuk sebagai platform tanpa label.
4. **Ganti dataset X utama di Worksheet 2–3** dari `goyaladi` (sintetis Faker) ke `danieltreiman`.
   Pertahankan `goyaladi` sebagai kontrol negatif dan jelaskan alasannya.
5. **Dokumen konsep §13:** pakai nama level berbasis kekuatan bukti dari §8.1, atau setidaknya rujuk
   pemetaannya, karena nama "Inauthentic / Spam" dan "Authentic / Low Risk" bertentangan dengan §15.
   Tambahkan tier NA dan aturan "NA ≠ Level 0", tambahkan gerbang bukti, dan nyatakan bahwa Level 3–4
   belum dapat dicapai serta data apa yang dibutuhkan (§8.5).
6. **Jangan mengutip** Instagram 95,3% (pakai 92,4% ± 2,0) dan **jangan menyajikan** Facebook
   ~97% sebagai keberhasilan. Jangan pula mengklaim RF lebih baik dari LR di Instagram/Facebook.
7. **Jangan membuat visual atau slide yang memeringkat platform menurut akurasi** (§7.1).
8. **Simpan model yang sudah dilatih** (`PipelineModel.save` ke `outputs/models/`, yang saat ini
   kosong). Tanpa itu, ambang deployment di §8.3 tidak bisa dipakai untuk akun baru.
9. **Lengkapi zona curated:** X, Instagram, dan Facebook belum menulis Parquet curated, padahal
   arsitektur mengklaimnya.
10. **Bangun dashboard Power BI** dari CSV di `outputs/powerbi/`. Datanya sudah siap.

## 13. Catatan atas angka track lain (tidak diubah di file mereka)

- **Facebook:** temuan track menyebut 97% sebagai "batas atas pada benchmark yang mudah". Laporan
  ini lebih keras: karena model lebih baik menebak blok daripada label, angka itu **bukan ukuran
  deteksi spam Facebook sama sekali**, bahkan bukan batas atas. Ini perbedaan pembingkaian, bukan
  perbedaan angka.
- **Instagram:** brief proyek menulis "579 + 120 baris". File berisi **576 + 120 = 696**, sesuai
  temuan track. Brief juga mengutip ROC-AUC 0,9926. Itu angka split seed 42. Yang seharusnya dikutip
  adalah CV 0,979 ± 0,009.
- **X, TikTok:** tidak ada ketidaksepakatan.

## 14. Reproduksi

Dari root repo, setelah `pip install -r requirements.txt`, Java 17 di PATH (atau `JAVA_HOME`), dan
`.env` berisi kredensial MongoDB (salin dari `.env.example`):

```bash
python scripts/download_datasets.py
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env up -d
python src/twitter_faker_check.py && python src/twitter_pipeline.py
python src/instagram_pipeline.py && python src/facebook_pipeline.py && python src/tiktok_pipeline.py
python src/synthesis.py
```

Sintesis berjalan sekitar satu menit (Spark 2 core, driver 2 GB). Opsi `--skip-mongo` hanya menjalankan
analisis dan CSV, dan `--no-connector` memuat MongoDB lewat pymongo saja. Saat JVM dinyalakan pertama
kali, connector Spark-MongoDB di-resolve dari Maven Central (cache di `~/.ivy2`).

| File yang ditulis track sintesis | Isi | Di-commit |
|---|---|---|
| `src/synthesis.py` | pipeline sintesis | ya |
| `outputs/metrics/cross_platform_comparison.json` | semua angka sintesis + hasil verifikasi MongoDB | tidak (memuat id akun) |
| [`outputs/figures/synthesis_concept_direction.png`](../outputs/figures/synthesis_concept_direction.png) | arah tiap konsep perilaku per platform | ya |
| [`outputs/figures/synthesis_importance_by_concept.png`](../outputs/figures/synthesis_importance_by_concept.png) | porsi importance per konsep | ya |
| [`outputs/figures/synthesis_risk_levels.png`](../outputs/figures/synthesis_risk_levels.png) | distribusi skor, kurva presisi/recall, level | ya |
| `outputs/powerbi/*.csv` | 10 tabel untuk Power BI | tidak (per akun) |
| `data/curated/cross_platform_account_scores.parquet` | tabel akun terpadu (zona curated) | tidak (per akun) |
| `deploy/mongodb/docker-compose.yml` | layanan MongoDB | ya |
| `.env` | kredensial (lokal, chmod 600, jangan dibagikan) | tidak |
| `docs/mongodb_powerbi.md` | koneksi, skema koleksi, tabel Power BI | ya |
| `outputs/logs/synthesis.log` | log run terakhir | tidak |
