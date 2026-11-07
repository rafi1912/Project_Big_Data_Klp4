# ADR 0003 — Spark adalah syarat arsitektur, bukan pilihan kecepatan

- Status: diterima
- Konteks: dataset proyek kecil (600 sampai 50.000 baris). Diukur di mesin yang sama, operasi agregasi
  yang setara: **pandas 0,01 s vs Spark 2,54 s pada 600 baris**, dan **0,06 s vs 0,86 s pada 37.438
  baris**. Pada ukuran ini Spark lebih lambat karena biaya JVM dan penjadwalan.
- Keputusan: semua tahap pemrosesan dan pemodelan utama tetap memakai Spark (Spark SQL, MLlib,
  Structured Streaming), karena mata kuliah dan arsitektur Worksheet 2 menuntut rantai
  Kaggle → data lake → Spark → MongoDB → dashboard yang sama yang akan dipakai pada data berskala besar.
  Statistik kecil sesudah agregasi (bootstrap, uji statistik, gambar) boleh dihitung di pandas/numpy
  dan disebutkan demikian.
- Konsekuensi: laporan tidak mengklaim Spark mempercepat analisis ini. Ukuran sesi dibatasi
  (`get_spark`: driver 2 GB, UI mati) supaya muat di mesin kecil.
