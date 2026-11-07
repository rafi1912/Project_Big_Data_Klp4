<!-- Judul PR: conventional commit, mis. `fix(common)!: keep '#'-prefixed columns distinct in snake()`.
     Nomor issue TIDAK di judul, tetapi di baris "Closes / Part of" di bawah. -->

## Summary
<!-- Satu-dua kalimat: apa yang berubah. -->

## Problem
<!-- Masalah/kebutuhan apa yang diselesaikan. Tautkan bukti (log error, angka). -->

## Solution
<!-- Cara menyelesaikannya dan alasan memilih cara ini. -->

## Test Evidence
<!-- Tempel potongan output perintah yang benar-benar dijalankan (ruff, pytest, run pipeline).
     Perubahan UI: lampirkan screenshot. -->

## Deployment Prerequisite / Breaking Changes
<!-- Tulis "Tidak ada" bila tidak ada. Untuk breaking change (`!` di judul): langkah migrasi persisnya. -->

## Notes
<!-- Hal yang perlu diketahui reviewer: keterbatasan, keputusan yang ditunda, urutan merge stacked PR. -->

Closes #
Part of #

## Checklist
- [ ] `ruff check .` lolos
- [ ] `pytest` lolos (test unit untuk logika yang bisa jalan tanpa dataset; selain itu bukti run di atas)
- [ ] README/docs diperbarui untuk perilaku yang berubah
- [ ] Tidak ada rahasia atau data per-akun yang ikut ter-commit (`.env`, `kaggle.json`, token, log, dataset)
