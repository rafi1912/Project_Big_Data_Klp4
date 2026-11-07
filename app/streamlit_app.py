"""Dashboard Streamlit — Sistem Big Data Deteksi Akun Bot/Spam Media Sosial.
Jalankan:  streamlit run app/streamlit_app.py

Dashboard HANYA membaca MongoDB (Spark -> MongoDB -> dashboard). Isinya ditulis oleh
src/load_mongo.py (hasil ingest/process/train, dipanggil run_all.sh), src/stream.py (skor streaming),
src/synthesis.py (level risiko & perbandingan lintas platform), atau scripts/seed_demo.py (demo).
Tidak ada cadangan berupa file: bila MongoDB tidak terjangkau, halaman berhenti dengan instruksi.
Database yang diisi seed_demo.py bertanda `meta`/`demo_seed`: setiap halaman lalu diawali banner "Mode demo".
"""
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
from load_mongo import DEMO_MARKER_ID, META_COLLECTION  # noqa: E402

START_CMD = "docker compose -f deploy/mongodb/docker-compose.yml --env-file .env up -d"
PAGES = ["Arsitektur & Data Lake", "Kualitas Data", "EDA", "Model & Evaluasi", "Streaming (simulasi)",
         "Antrean Review Moderator", "Level Risiko", "Perbandingan Lintas Platform"]
PIPELINE_PAGES = PAGES[1:5]
LEVEL_COLOR = {"0": "#9ec5f4", "1": "#f2b84b", "2": "#e34948", "NA": "#bdbdbd"}
st.set_page_config(page_title="Deteksi Akun Bot — Big Data", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner=False)
def connect():
    client, db = common.get_mongo(timeout_ms=2500)
    client.admin.command("ping")          # gagal di sini = MongoDB mati / kredensial salah
    return db


try:
    db = connect()
except Exception as e:                    # noqa: BLE001 - pesan apa pun, tanggapannya sama
    st.error(f"**MongoDB tidak dapat dihubungi** ({type(e).__name__}). Dashboard hanya membaca MongoDB "
             "dan tidak memakai file hasil sebagai cadangan. Nyalakan MongoDB dari root repo:\n\n"
             f"```bash\n{START_CMD}\n```\n\nPastikan `.env` berisi user/password yang sama (lihat `.env.example`), "
             "lalu isi datanya: `python scripts/seed_demo.py` (demo tanpa Kaggle) atau jalur lengkap "
             "(`bash run_all.sh`, pipeline analisis, `python src/synthesis.py`).")
    st.stop()


demo = db[META_COLLECTION].find_one({"_id": DEMO_MARKER_ID})
if demo:                                  # banner di atas SETIAP halaman selama database berisi data demo
    when = demo.get("seeded_at")
    st.info("**Mode demo** — isi MongoDB ini berasal dari `scripts/seed_demo.py`"
            + (f" (diisi {when:%Y-%m-%d %H:%M} UTC)" if when else "")
            + ": hanya dataset **sintetis** `twitter_bot_detection` (username buatan Faker) dan **agregat hasil "
            "analisis tanpa identitas akun**. Hasil batch dataset nyata, antrean moderator untuk akun nyata, dan "
            "skor streaming dataset lain baru muncul setelah run lengkap: `python scripts/download_datasets.py` → "
            "`bash run_all.sh` → pipeline analisis (`src/*_pipeline.py`) → `python src/synthesis.py` (lihat README).")


def find(coll, query=None, sort=None, limit=0) -> pd.DataFrame:
    cur = db[coll].find(query or {}, {"_id": 0})
    if sort:
        cur = cur.sort(sort)
    if limit:
        cur = cur.limit(limit)
    return pd.DataFrame(list(cur))


def one(coll, query):
    return db[coll].find_one(query, {"_id": 0})


def need(ok: bool, msg: str):
    if not ok:
        st.info(msg)
        st.stop()


NO_PIPELINE = ("Belum ada hasil pipeline batch di MongoDB. Jalankan `bash run_all.sh` (memuat hasil ke MongoDB "
               "di langkah terakhir) atau `python scripts/seed_demo.py` untuk data demo.")
NO_SYNTHESIS = ("Belum ada hasil sintesis di MongoDB. Jalankan pipeline analisis lalu `python src/synthesis.py`, "
                "atau `python scripts/seed_demo.py` untuk data demo.")

# ---------------------------------------------------------------- sidebar
cfg_df = find("pipeline_datasets")
ready = sorted(db["pipeline_dq"].distinct("dataset"))
info = {r["dataset"]: r for r in cfg_df.to_dict("records")} if len(cfg_df) else {}
st.sidebar.title("🤖 Deteksi Akun Bot")
st.sidebar.caption("Teknologi Big Data · data dibaca dari MongoDB")
page = st.sidebar.radio("Halaman", PAGES)
ds = None
if page in PIPELINE_PAGES:
    need(bool(ready), NO_PIPELINE)
    ds = st.sidebar.selectbox("Dataset", ready,
                              format_func=lambda k: f"{info.get(k, {}).get('platform', '?')} · {k}")
    if info.get(ds, {}).get("note"):
        st.sidebar.info(info[ds]["note"])
st.sidebar.caption(f"MongoDB: `{db.name}`")

# ---------------------------------------------------------------- 1
if page == "Arsitektur & Data Lake":
    st.title("Arsitektur sistem Big Data")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#EAF2FB"];
      src [label="Kaggle\\n(data/landing)"];
      ing [label="Ingestion\\nSpark batch + manifest sha256", fillcolor="#E3F4EC"];
      lake [label="Bronze / Silver / Gold\\nParquet", fillcolor="#FFF4D6"];
      ml  [label="Modeling\\nSpark MLlib", fillcolor="#EFE6F8"];
      st_ [label="Streaming scoring\\nStructured Streaming", fillcolor="#FDE7E2"];
      ana [label="Pipeline analisis per platform\\n+ sintesis level risiko (Spark)", fillcolor="#EFE6F8"];
      mg  [label="MongoDB\\nlapisan penyajian", fillcolor="#E3F4EC"];
      ui  [label="Dashboard\\nStreamlit"];
      pbi [label="Ekspor Power BI\\n(CSV)"];
      src->ing->lake->ml->st_; src->ana; lake->mg; ml->mg; st_->mg; ana->mg; mg->ui; mg->pbi; }
    """)
    st.caption("Spark memproses (batch + streaming), MongoDB menyajikan. Dashboard ini hanya membaca MongoDB; "
               "tabel Power BI adalah ekspor CSV dari data yang sama.")
    man = find("pipeline_ingest_manifests")
    need(len(man) > 0, NO_PIPELINE)
    rows = [{"dataset": r["dataset"], "platform": info.get(r["dataset"], {}).get("platform"),
             "file": ", ".join(f["name"] for f in r["files"]), "baris": sum(f["rows"] for f in r["files"]),
             "kaggle": info.get(r["dataset"], {}).get("kaggle")} for r in man.to_dict("records")]
    t = pd.DataFrame(rows)
    st.subheader("Isi data lake (zona bronze)")
    st.dataframe(t, width="stretch", hide_index=True)
    total = int(t["baris"].sum())
    ok = total >= 100_000
    if demo:
        st.info(f"Total baris **data demo**: **{total:,}** — angka ini hanya menghitung data demo (dataset "
                "sintetis `twitter_bot_detection`), **bukan** sistem lengkap. Syarat minimum 100.000 baris dinilai "
                "dari run lengkap (lihat README, bagian *Syarat 100.000 baris*).")
    else:
        (st.success if ok else st.warning)(
            f"Total baris seluruh sumber berlabel: **{total:,}** — "
            + ("memenuhi syarat minimum 100.000 baris." if ok else
               "**belum** memenuhi syarat minimum 100.000 baris; tambahkan dataset (lihat README)."))
    st.subheader("Isi MongoDB")
    counts = pd.DataFrame([{"koleksi": c, "dokumen": db[c].estimated_document_count()}
                           for c in sorted(db.list_collection_names())])
    st.dataframe(counts, width="stretch", hide_index=True)

# ---------------------------------------------------------------- 2
elif page == "Kualitas Data":
    dq = one("pipeline_dq", {"dataset": ds})
    st.title(f"Kualitas data · {ds}")
    c = st.columns(4)
    c[0].metric("Baris mentah", f"{dq['rows_raw']:,}")
    c[1].metric("Duplikat dibuang", f"{dq['duplicate_rows_removed']:,}")
    c[2].metric("Tanpa label dibuang", f"{dq['rows_without_label_removed']:,}")
    c[3].metric("Baris bersih (silver)", f"{dq['rows_clean']:,}")
    left, right = st.columns(2)
    lc = pd.DataFrame({"label": ["0 (bukan bot/spam)", "1 (bot/spam)"],
                       "jumlah": [dq["label_counts"].get("0", 0), dq["label_counts"].get("1", 0)]})
    left.plotly_chart(px.bar(lc, x="label", y="jumlah", title="Distribusi label"), width="stretch")
    nulls = pd.Series(dq["null_pct_by_column"], dtype=float).sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if len(nulls):
        right.plotly_chart(px.bar(nulls, title="% nilai kosong per kolom", labels={"value": "%", "index": "kolom"}),
                           width="stretch")
    else:
        right.success("Tidak ada nilai kosong pada kolom fitur.")
    st.markdown(f"**Kolom label:** `{dq['label_column']}` · **nilai asli label:** `{dq['label_raw_values']}` "
                "→ verifikasi pemetaan ke 0/1 sesuai dokumentasi dataset.")
    if dq["leakage_suspects"]:
        st.error(f"Fitur dengan korelasi sangat tinggi terhadap label (dikeluarkan dari model): {dq['leakage_suspects']}")
    else:
        st.info("Tidak ada fitur dengan korelasi mencurigakan (|r| ≥ ambang) terhadap label.")
    if dq["duplicate_record_ids"]:
        st.warning(f"{dq['duplicate_record_ids']:,} ID rekaman berulang — pertimbangkan split per-akun.")

# ---------------------------------------------------------------- 3
elif page == "EDA":
    dq = one("pipeline_dq", {"dataset": ds})
    st.title(f"Eksplorasi data · {ds}")
    sample = find("pipeline_eda_sample", {"dataset": ds})
    need(len(sample) > 0, "Sampel EDA belum dimuat untuk dataset ini.")
    feats = [f for f in dq["numeric_features"] if f in sample.columns]
    f = st.selectbox("Fitur", feats)
    sample["kelas"] = sample["label"].map({0: "bukan bot/spam", 1: "bot/spam"})
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(sample, x=f, color="kelas", barmode="overlay", opacity=.6, nbins=40,
                                title=f"Distribusi {f}"), width="stretch")
    b.plotly_chart(px.box(sample, x="kelas", y=f, color="kelas", title=f"Sebaran {f} per kelas"), width="stretch")
    corr = pd.Series(dq["corr_with_label"], dtype=float).dropna().sort_values()
    st.plotly_chart(px.bar(corr, orientation="h", title="Korelasi tiap fitur dengan label",
                           labels={"value": "r (Pearson)", "index": "fitur"}), width="stretch")
    st.caption("Korelasi mendekati 0 untuk semua fitur berarti fitur tidak membedakan kelas.")
    fs = find("pipeline_feature_stats", {"dataset": ds})
    if len(fs):
        st.subheader("Statistik fitur per kelas")
        st.dataframe(fs.drop(columns="dataset").round(3), width="stretch", hide_index=True)
    mc = find("pipeline_monthly_counts", {"dataset": ds})
    if len(mc):
        mc["kelas"] = mc["label"].map({0: "bukan bot/spam", 1: "bot/spam"})
        st.plotly_chart(px.line(mc.sort_values("month"), x="month", y="count", color="kelas",
                                title="Jumlah rekaman per bulan"), width="stretch")

# ---------------------------------------------------------------- 4
elif page == "Model & Evaluasi":
    st.title(f"Model & evaluasi · {ds}")
    mt = one("pipeline_metrics", {"dataset": ds})
    need(mt is not None, "Model belum dilatih untuk dataset ini.")
    v = mt["verdict"]
    (st.error if v.startswith("LEMAH") else st.warning if v.startswith("WASPADA") else st.success)(v)
    t = mt["test"]
    c = st.columns(5)
    c[0].metric("ROC-AUC (uji)", f"{mt['test_auc']:.3f}")
    c[1].metric("Akurasi", f"{t['accuracy']:.3f}", f"{t['accuracy'] - mt['baseline_majority_accuracy']:+.3f} vs baseline")
    c[2].metric("Precision bot", f"{t['precision_bot']:.3f}")
    c[3].metric("Recall bot", f"{t['recall_bot']:.3f}")
    c[4].metric("F1 bot", f"{t['f1_bot']:.3f}")
    st.caption(f"Model terbaik (dipilih pada data validasi): **{mt['best_model']}** · "
               f"train/val/uji = {mt['n_train']:,}/{mt['n_val']:,}/{mt['n_test']:,} · "
               f"baseline mayoritas = {mt['baseline_majority_accuracy']:.3f}")
    a, b = st.columns(2)
    cm = go.Figure(go.Heatmap(z=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]], x=["Prediksi: bukan", "Prediksi: bot"],
                              y=["Aktual: bukan", "Aktual: bot"], text=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]],
                              texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title="Confusion matrix (data uji)", yaxis_autorange="reversed")
    a.plotly_chart(cm, width="stretch")
    roc = find("pipeline_roc", {"dataset": ds})
    fr = go.Figure(go.Scatter(x=roc.get("fpr"), y=roc.get("tpr"), mode="lines", name="model"))
    fr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fr.update_layout(title="Kurva ROC", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b.plotly_chart(fr, width="stretch")
    st.subheader("Perbandingan model")
    st.dataframe(pd.DataFrame(mt["models"]).T.round(4).rename_axis("model").reset_index(), width="stretch",
                 hide_index=True)
    a, b = st.columns(2)
    imp = find("pipeline_feature_importance", {"dataset": ds})
    if len(imp):
        a.plotly_chart(px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                              title="Fitur paling berpengaruh"), width="stretch")
    pr = find("pipeline_test_predictions", {"dataset": ds})
    if len(pr):
        pr["kelas"] = pr["label"].map({0: "bukan bot/spam", 1: "bot/spam"})
        b.plotly_chart(px.histogram(pr, x="prob_bot", color="kelas", barmode="overlay", opacity=.6, nbins=40,
                                    title="Skor bot di data uji per kelas"), width="stretch")
    with st.expander("Detail fitur & keterbatasan"):
        st.write({"fitur numerik": mt["features_numeric"], "fitur kategorikal": mt["features_categorical"],
                  "kolom dilewati (kardinalitas tinggi)": mt["features_skipped"],
                  "dikeluarkan (indikasi leakage)": mt["leakage_excluded"]})
        st.write("Skor model adalah sinyal bantu untuk review manusia, bukan dasar tunggal untuk menghukum akun.")

# ---------------------------------------------------------------- 5
elif page == "Streaming (simulasi)":
    st.title(f"Streaming scoring · {ds}")
    st.caption("Micro-batch akun (dari data uji, disimulasikan oleh `simulate_feed.py`) diberi skor oleh "
               "`stream.py` (Spark Structured Streaming) dan ditulis ke MongoDB `stream_scores`. "
               "Di produksi sumbernya Kafka.")
    s = find("stream_scores", {"dataset": ds})
    need(len(s) > 0, "Belum ada skor streaming di MongoDB. Jalankan:\n\n```bash\ncd src\npython simulate_feed.py "
                     f"{ds} --reset --batches 5 --size 200 --interval 3 &\npython stream.py {ds}\n```")
    thr = info.get(ds, {}).get("alert_threshold", 0.8)
    c = st.columns(4)
    c[0].metric("Akun diberi skor", f"{len(s):,}")
    c[1].metric("Micro-batch", s["batch_id"].nunique())
    c[2].metric(f"Alert (skor ≥ {thr})", int(s["alert"].sum()))
    c[3].metric("Rata-rata skor bot", f"{s['prob_bot'].mean():.3f}")
    per = (s.groupby("batch_id").agg(akun=("record_id", "count"), alert=("alert", "sum"), skor=("prob_bot", "mean"))
           .reset_index())
    per["alert_rate"] = per["alert"] / per["akun"]
    a, b = st.columns(2)
    a.plotly_chart(px.line(per, x="batch_id", y="alert_rate", markers=True, title="Proporsi alert per micro-batch"),
                   width="stretch")
    b.plotly_chart(px.histogram(s, x="prob_bot", nbins=30, title="Distribusi skor bot"), width="stretch")
    if "label" in s.columns and s["label"].notna().any() and s["alert"].sum() > 0:
        prec = s.loc[s.alert == 1, "label"].mean()
        st.info(f"Precision alert terhadap kunci jawaban simulasi: **{prec:.3f}** "
                f"(base rate bot = {s['label'].mean():.3f}). Jika ≈ base rate, alert tidak informatif.")
    st.subheader("Alert terbaru")
    cols = [x for x in ["scored_at", "batch_id", "record_id", "username_ref", "prob_bot", "label"] if x in s.columns]
    st.dataframe(s[s.alert == 1].sort_values("scored_at", ascending=False)[cols].head(50), width="stretch",
                 hide_index=True)
    if st.checkbox("Auto-refresh tiap 5 detik"):
        time.sleep(5)
        st.rerun()

# ---------------------------------------------------------------- 6
elif page == "Antrean Review Moderator":
    st.title("Antrean review moderator")
    st.warning("Skor dan level adalah **sinyal bantu** untuk memprioritaskan pemeriksaan manusia — bukan dasar "
               "tunggal untuk menghapus atau menghukum akun. Tidak ada aksi otomatis.")
    sets = sorted(db["account_scores"].distinct("dataset"))
    need(bool(sets), NO_SYNTHESIS)
    d = st.selectbox("Dataset", sets, key="queue_dataset")
    codes = sorted(db["account_scores"].distinct("risk_level_code", {"dataset": d}))
    default = [c for c in ("2",) if c in codes] or codes
    pick = st.multiselect("Level risiko", codes, default=default,
                          format_func=lambda c: f"{c} · {(one('risk_level_definitions', {'code': c}) or {}).get('name_id', c)}")
    top = int(st.number_input("Tampilkan N akun teratas", 5, 500, 25))
    acc = find("account_scores", {"dataset": d})
    scored = acc["rf_p_inauthentic"].notna().any()
    labelled = acc["label"].notna()
    if scored:
        thr = st.slider("Ambang skor model (untuk menilai precision)", 0.0, 1.0, 0.5, 0.01)
        lab = acc[labelled]
        sel = lab[lab["rf_p_inauthentic"] >= thr]
        c = st.columns(4)
        c[0].metric("Akun di level terpilih", f"{acc['risk_level_code'].isin(pick).sum():,}")
        c[1].metric("Akun berlabel di atas ambang", f"{len(sel):,}")
        c[2].metric("Precision pada ambang ini", f"{sel['label'].mean():.3f}" if len(sel) else "—")
        c[3].metric("Base rate (acak)", f"{lab['label'].mean():.3f}" if len(lab) else "—")
        st.caption("Precision dan base rate dihitung dari label evaluasi (hanya untuk menilai antrean). Bila "
                   "precision ≈ base rate, antrean tidak lebih baik daripada memilih akun secara acak. Label tidak "
                   "ditampilkan di tabel operasional.")
    else:
        st.caption("Dataset ini tidak berlabel dan tidak punya skor model: urutan memakai level dan tier indikator.")
    q = acc[acc["risk_level_code"].isin(pick)]
    q = q.sort_values(["rf_p_inauthentic"] if scored else ["risk_level_code", "tiktok_review_tier"],
                      ascending=False, na_position="last")
    cols = ["account_uid", "source_ref", "rf_p_inauthentic", "risk_level_code", "risk_level_name", "suggested_action"]
    if not scored:
        cols += [c for c in acc.columns if c.startswith("ind_") or c.startswith("tiktok_")]
    if q["risk_level_basis"].eq("evidence_gate_failed_no_signal").any():
        st.error("Model dataset ini **gagal evidence gate** (tidak lebih baik daripada acak): semua akun berstatus "
                 "*Tidak dapat dinilai* (NA), bukan Level 0.")
    st.dataframe(q[[c for c in cols if c in q.columns]].head(top), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 7
elif page == "Level Risiko":
    st.title("Level risiko 0–4")
    defs = find("risk_level_definitions", sort=[("code", 1)])
    need(len(defs) > 0, NO_SYNTHESIS)
    st.caption("Nama level menggambarkan **kekuatan bukti dan tindakan yang pantas**, bukan vonis tentang pemilik "
               "akun. Ambang ditentukan dengan cross-fitting 2 bagian: setiap akun dinilai dengan ambang yang "
               "diturunkan dari bagian data lainnya.")
    show = defs[["code", "name_id", "name_concept_doc", "suggested_action", "reachable"]].rename(
        columns={"code": "kode", "name_id": "nama (laporan ini)", "name_concept_doc": "nama di dokumen konsep §13",
                 "suggested_action": "tindakan", "reachable": "dapat dicapai"})
    st.dataframe(show, width="stretch", hide_index=True)
    st.info("Level 3 dan 4 membutuhkan bukti koordinasi/amplifikasi antar akun (waktu unggah, konten bersama, "
            "graf interaksi) yang tidak ada di dataset mana pun, sehingga **tidak pernah diberikan**. "
            "NA = tidak dapat dinilai, bukan Level 0.")
    summ = find("risk_level_summary")
    per = summ[summ["kind"] == "per_level"].copy() if len(summ) else summ
    if len(per):
        per["level"] = per["level_code"].astype(str)
        st.plotly_chart(px.bar(per, x="dataset", y="n_accounts", color="level", barmode="stack",
                               color_discrete_map=LEVEL_COLOR, title="Jumlah akun per level (cross-fitted)"),
                        width="stretch")
        withp = per[per["precision_in_level"].notna()].copy()
        if len(withp):
            withp["err_hi"] = withp["precision_ci95_high"] - withp["precision_in_level"]
            withp["err_lo"] = withp["precision_in_level"] - withp["precision_ci95_low"]
            fig = px.scatter(withp, x="level", y="precision_in_level", color="dataset", error_y="err_hi",
                             error_y_minus="err_lo", title="Precision di setiap level (IK 95% Wilson)")
            fig.update_traces(marker_size=11)
            st.plotly_chart(fig, width="stretch")
        st.subheader("Ringkasan per level")
        cols = ["dataset", "level_code", "n_accounts", "volume_share", "n_inauthentic", "precision_in_level",
                "precision_ci95_low", "precision_ci95_high", "t1_deployment", "t2_deployment"]
        st.dataframe(per[[c for c in cols if c in per.columns]], width="stretch", hide_index=True)
    other = summ[summ["kind"] != "per_level"] if len(summ) else summ
    if len(other):
        with st.expander("Kumulatif, tidak dapat dicapai, dan tidak dapat dinilai"):
            st.dataframe(other, width="stretch", hide_index=True)
    agg = list(db["account_scores"].aggregate([
        {"$group": {"_id": {"dataset": "$dataset", "level": "$risk_level_code"}, "n": {"$sum": 1}}}]))
    if agg:
        st.subheader("Hitungan langsung dari `account_scores` (agregasi di MongoDB)")
        t = pd.DataFrame([{"dataset": a["_id"]["dataset"], "level": a["_id"]["level"], "akun": a["n"]} for a in agg])
        st.dataframe(t.pivot_table(index="dataset", columns="level", values="akun", fill_value=0), width="stretch")

# ---------------------------------------------------------------- 8
else:
    st.title("Perbandingan lintas platform")
    meta = find("datasets")
    need(len(meta) > 0, NO_SYNTHESIS)
    st.warning("**Bukan peringkat akurasi.** Mengurutkan platform menurut akurasi akan menaruh file Facebook yang "
             "tersusun per label di atas, dan X (satu-satunya dataset nyata yang besar) di bawah. Baris diurutkan "
             "menurut bobot bukti.")
    if "evidential_weight_order" in meta.columns:
        meta = meta.sort_values("evidential_weight_order")
    cols = ["dataset", "platform", "dataset_role", "n_rows", "positive_rate", "label_concept", "verdict",
            "number_to_quote"]
    st.dataframe(meta[[c for c in cols if c in meta.columns]], width="stretch", hide_index=True)
    rows = []
    for m in db["model_metrics"].find({"models": {"$exists": True}}, {"_id": 0}):
        for model, block in (m.get("models") or {}).items():
            met = (block or {}).get("metrics") or {}
            base = (met.get("baseline_majority_class") or {}).get("accuracy")
            rows.append({"dataset": m["dataset"], "model": model, "accuracy": met.get("accuracy"),
                         "baseline_mayoritas": base, "balanced_accuracy": met.get("balanced_accuracy"),
                         "mcc": met.get("mcc"), "roc_auc": met.get("roc_auc"), "pr_auc": met.get("pr_auc")})
    if rows:
        st.subheader("Metrik data uji vs baseline kelas mayoritas")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    cc = find("concept_comparison")
    if len(cc):
        st.subheader("Arah setiap indikator perilaku per dataset")
        cc = cc[cc["auc"].notna()].copy()
        cc["err_hi"] = cc["auc_ci95_high"] - cc["auc"]
        cc["err_lo"] = cc["auc"] - cc["auc_ci95_low"]
        fig = px.scatter(cc, x="auc", y="indicator", color="dataset", error_x="err_hi", error_x_minus="err_lo",
                         title="ROC-AUC univariat indikator (0,5 = tidak membedakan)")
        fig.add_vline(x=0.5, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.caption("AUC > 0,5: nilai tinggi lebih sering pada akun tidak autentik; < 0,5: sebaliknya. Arah hanya "
                   "diklaim bila interval kepercayaan 95% tidak memuat 0,5.")
    ev = find("coordination_evidence")
    if len(ev):
        st.subheader("Bukti koordinasi yang tersedia (syarat Level 3–4)")
        st.dataframe(ev.pivot_table(index="evidence_type", columns="dataset", values="status", aggfunc="first"),
                     width="stretch")
