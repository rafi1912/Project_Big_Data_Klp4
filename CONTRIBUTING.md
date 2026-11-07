# Panduan kontribusi (SOP)

Kelompok "Penyayang Claude" — Teknologi Big Data, SV IPB University. Dokumen ini adalah SOP kerja
di repo ini. Singkatnya: **issue → branch → PR kecil → review Rafi → merge**, dan `main` selalu
bisa dijalankan.

## 1. Alur kerja

1. **Issue dulu.** Buat satu *tracking issue* untuk satu pekerjaan besar (berisi checklist PR-nya)
   dan **satu issue per bug**. Tidak perlu satu issue per PR.
2. **Branch** dari `main` (atau dari branch PR sebelumnya bila bertumpuk, lihat §6) dengan
   awalan:

   | Awalan | Untuk | Contoh |
   |---|---|---|
   | `feature/` | fitur baru | `feature/dashboard-mongodb` |
   | `fix/` | perbaikan bug | `fix/snake-column-collision` |
   | `chore/` | CI, dependensi, rapi-rapi, **dan dokumentasi** | `chore/docs-final-report` |

3. **Pull request** ke branch dasar, isi template (`.github/pull_request_template.md`).
4. **Review** oleh Rafi (`rafi1912`), lalu **merge**. Tidak ada push langsung ke `main`.

Model branch: *trunk-based*. `main` harus selalu bisa di-install dan dijalankan. Tidak ada branch
`release`; versi yang dikumpulkan ditandai dengan **git tag**, mis. `v1.0-submission`.

## 2. Ukuran PR dan commit

- Satu PR = **satu perhatian** (satu bug, satu fitur, satu refactor). PR besar dipecah menjadi
  beberapa PR bertumpuk.
- Target **< 500 baris per PR** dan **5–100 baris per commit**, satu perubahan logis per commit.
  Pengecualian yang disepakati: memasukkan file yang sudah ada dan sudah pernah dijalankan
  (mis. memindahkan skrip analisis dari repo lain) boleh lebih besar, cukup satu commit per file.
- **Tidak ada commit WIP** ("wip", "fix lagi", "coba") di branch bersama. Rapikan dulu sebelum push.

## 3. Pesan commit dan judul PR

Pakai [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipe>(<scope opsional>)<! bila breaking>: <deskripsi singkat, bahasa Inggris>
```

Tipe: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`. Tanda `!` menandai **breaking change**
(perilaku atau format keluaran berubah sehingga pengguna harus melakukan sesuatu), mis.
`fix(common)!: keep '#'-prefixed columns distinct in snake()`. Judul PR memakai format yang sama
dan dicek otomatis oleh CI.

Nomor issue **tidak** ditulis di judul. Tulis di badan PR:

- `Closes #12` — PR ini menyelesaikan issue 12 (issue tertutup otomatis saat merge ke `main`).
- `Part of #10` — PR ini bagian dari tracking issue 10.

## 4. Syarat sebelum minta review

- **Lint dan test lolos** (`ruff check .` dan `pytest`; CI menjalankan keduanya di setiap PR).
- **Test unit** untuk logika yang bisa jalan tanpa dataset (fungsi murni, DataFrame kecil
  di memori). Untuk perilaku yang butuh dataset asli, tempel **bukti run** (potongan output
  perintah) di bagian *Test Evidence*.
- **README/docs diperbarui** di PR yang sama dengan perubahan perilakunya.
- **Screenshot** untuk setiap perubahan tampilan dashboard.
- **Catatan migrasi** untuk breaking change: langkah persis yang harus dijalankan pengguna
  (mis. `rm -rf data/bronze data/silver && bash run_all.sh`).
- **Jangan pernah commit** `.env`, `kaggle.json`, token/API key, file log, dataset, atau keluaran
  per-akun. `.gitignore` sudah menutup sebagian besar; tetap periksa `git status` sebelum commit.

Pasang alat lokal sekali saja:

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest
```

## 5. Etika review

- Reviewer memberi label pada komentar: **wajib** (harus diselesaikan sebelum merge) atau
  **nit** (saran kecil, boleh diabaikan dengan alasan).
- Penulis PR **menanggapi setiap komentar**: perbaiki, atau jelaskan kenapa tidak. Reviewer yang
  menandai percakapan *resolved*.
- Kritik kode, bukan orang. Bila diskusi melebar, pindahkan ke issue.
