"""Dashboard Streamlit — Sistem Big Data Deteksi Akun Bot/Spam Media Sosial.
Jalankan:  streamlit run app/streamlit_app.py
Dashboard hanya MEMBACA keluaran pipeline (reports/ dan data/gold/); tidak melatih apa pun.
"""
import json, time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
st.set_page_config(page_title="Deteksi Akun Bot — Big Data", page_icon="🤖", layout="wide")


def jload(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ready():
    return [k for k in CFG["datasets"] if (ROOT / CFG["paths"]["reports"] / k / "dq.json").exists()]


def rep(k, name): return ROOT / CFG["paths"]["reports"] / k / name
def gold(k, name): return ROOT / CFG["paths"]["gold"] / k / name


datasets = ready()
st.sidebar.title("🤖 Deteksi Akun Bot")
st.sidebar.caption("Teknologi Big Data · rancangan & prototipe pipeline")
page = st.sidebar.radio("Halaman", ["Arsitektur & Data Lake", "Kualitas Data", "EDA", "Model & Evaluasi",
                                    "Streaming (simulasi)", "Antrean Review Moderator"])
if not datasets:
    st.warning("Belum ada hasil pipeline. Jalankan `bash run_all.sh` terlebih dahulu.")
    st.stop()
ds = st.sidebar.selectbox("Dataset", datasets, format_func=lambda k: f"{CFG['datasets'][k]['platform']} · {k}")
st.sidebar.info(CFG["datasets"][ds].get("note", ""))
dq, mt = jload(rep(ds, "dq.json")), jload(rep(ds, "metrics.json"))

# ------------------------------------------------------------------ 1
if page == "Arsitektur & Data Lake":
    st.title("Arsitektur sistem Big Data")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#EAF2FB"];
      src [label="Sumber data\\n(CSV Kaggle / log platform)"];
      ing [label="Ingestion\\nSpark batch + manifest sha256", fillcolor="#E3F4EC"];
      br  [label="Bronze\\nParquet mentah", fillcolor="#FFF4D6"];
      pr  [label="Processing\\nSpark: cleaning, fitur", fillcolor="#EFE6F8"];
      si  [label="Silver\\nParquet bersih", fillcolor="#FFF4D6"];
      go  [label="Gold\\nagregat + EDA", fillcolor="#FFF4D6"];
      ml  [label="Modeling\\nSpark MLlib", fillcolor="#EFE6F8"];
      st_ [label="Streaming scoring\\nStructured Streaming", fillcolor="#FDE7E2"];
      ui  [label="Dashboard\\nStreamlit", fillcolor="#E3F4EC"];
      src->ing->br->pr->si; pr->go; si->ml; ml->st_; st_->ui; go->ui; ml->ui; }
    """)
    st.caption("Batch: melatih/menilai model. Streaming: memberi skor micro-batch akun baru. "
               "Pola ini setara Lambda Architecture (batch layer + speed layer).")
    rows = []
    for k in datasets:
        m = jload(rep(k, "ingest_manifest.json")) or {"files": []}
        rows.append({"dataset": k, "platform": CFG["datasets"][k]["platform"], "file": ", ".join(f["name"] for f in m["files"]),
                     "baris": sum(f["rows"] for f in m["files"]), "kaggle": CFG["datasets"][k]["kaggle"]})
    t = pd.DataFrame(rows)
    st.subheader("Isi data lake (zona bronze)")
    st.dataframe(t, width="stretch", hide_index=True)
    total = int(t["baris"].sum())
    ok = total >= 100_000
    (st.success if ok else st.warning)(
        f"Total baris seluruh sumber: **{total:,}** — "
        + ("memenuhi syarat minimum 100.000 baris." if ok else
           "**belum** memenuhi syarat minimum 100.000 baris; tambahkan dataset (lihat README)."))

# ------------------------------------------------------------------ 2
elif page == "Kualitas Data":
    st.title(f"Kualitas data · {ds}")
    c = st.columns(4)
    c[0].metric("Baris mentah", f"{dq['rows_raw']:,}")
    c[1].metric("Duplikat dibuang", f"{dq['duplicate_rows_removed']:,}")
    c[2].metric("Tanpa label dibuang", f"{dq['rows_without_label_removed']:,}")
    c[3].metric("Baris bersih (silver)", f"{dq['rows_clean']:,}")
    l, r = st.columns(2)
    lc = pd.DataFrame({"label": ["0 (bukan bot/spam)", "1 (bot/spam)"],
                       "jumlah": [dq["label_counts"].get("0", 0), dq["label_counts"].get("1", 0)]})
    l.plotly_chart(px.bar(lc, x="label", y="jumlah", title="Distribusi label"), width="stretch")
    nulls = pd.Series(dq["null_pct_by_column"]).sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    if len(nulls):
        r.plotly_chart(px.bar(nulls, title="% nilai kosong per kolom", labels={"value": "%", "index": "kolom"}),
                       width="stretch")
    else:
        r.success("Tidak ada nilai kosong pada kolom fitur.")
    st.markdown(f"**Kolom label:** `{dq['label_column']}` · **nilai asli label:** `{dq['label_raw_values']}` "
                "→ verifikasi pemetaan ke 0/1 sesuai dokumentasi dataset.")
    if dq["leakage_suspects"]:
        st.error(f"Fitur dengan korelasi sangat tinggi terhadap label (dikeluarkan dari model): {dq['leakage_suspects']}")
    else:
        st.info("Tidak ada fitur dengan korelasi mencurigakan (|r| ≥ ambang) terhadap label.")
    if dq["duplicate_record_ids"]:
        st.warning(f"{dq['duplicate_record_ids']:,} ID rekaman berulang — pertimbangkan split per-akun.")

# ------------------------------------------------------------------ 3
elif page == "EDA":
    st.title(f"Eksplorasi data · {ds}")
    sample = pd.read_parquet(gold(ds, "eda_sample.parquet"))
    feats = dq["numeric_features"]
    f = st.selectbox("Fitur", feats)
    sample["kelas"] = sample["label"].map({0: "bukan bot/spam", 1: "bot/spam"})
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(sample, x=f, color="kelas", barmode="overlay", opacity=.6, nbins=40,
                                title=f"Distribusi {f}"), width="stretch")
    b.plotly_chart(px.box(sample, x="kelas", y=f, color="kelas", title=f"Sebaran {f} per kelas"),
                   width="stretch")
    corr = pd.Series(dq["corr_with_label"]).dropna().sort_values()
    st.plotly_chart(px.bar(corr, orientation="h", title="Korelasi tiap fitur dengan label",
                           labels={"value": "r (Pearson)", "index": "fitur"}), width="stretch")
    st.caption("Korelasi mendekati 0 untuk semua fitur berarti fitur tidak membedakan kelas.")
    fs = gold(ds, "feature_stats_by_label.csv")
    if fs.exists():
        st.subheader("Statistik fitur per kelas")
        st.dataframe(pd.read_csv(fs).round(3), width="stretch", hide_index=True)
    mc = gold(ds, "monthly_counts.csv")
    if mc.exists():
        m = pd.read_csv(mc); m["kelas"] = m["label"].map({0: "bukan bot/spam", 1: "bot/spam"})
        st.plotly_chart(px.line(m, x="month", y="count", color="kelas", title="Jumlah rekaman per bulan"),
                        width="stretch")

# ------------------------------------------------------------------ 4
elif page == "Model & Evaluasi":
    st.title(f"Model & evaluasi · {ds}")
    if not mt:
        st.warning("Model belum dilatih untuk dataset ini."); st.stop()
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
    roc = pd.read_csv(gold(ds, "roc.csv"))
    fr = go.Figure(go.Scatter(x=roc.fpr, y=roc.tpr, mode="lines", name="model"))
    fr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fr.update_layout(title="Kurva ROC", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b.plotly_chart(fr, width="stretch")
    st.subheader("Perbandingan model")
    st.dataframe(pd.DataFrame(mt["models"]).T.round(4).rename_axis("model").reset_index(), width="stretch", hide_index=True)
    imp = pd.read_csv(gold(ds, "feature_importance.csv"))
    st.plotly_chart(px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                           title="Fitur paling berpengaruh"), width="stretch")
    with st.expander("Detail fitur & keterbatasan"):
        st.write({"fitur numerik": mt["features_numeric"], "fitur kategorikal": mt["features_categorical"],
                  "kolom dilewati (kardinalitas tinggi)": mt["features_skipped"],
                  "dikeluarkan (indikasi leakage)": mt["leakage_excluded"]})
        st.write("Skor model adalah sinyal bantu untuk review manusia, bukan dasar tunggal untuk menghukum akun.")

# ------------------------------------------------------------------ 5
elif page == "Streaming (simulasi)":
    st.title(f"Streaming scoring · {ds}")
    st.caption("Micro-batch akun (dari data uji, disimulasikan oleh `simulate_feed.py`) diberi skor oleh "
               "`stream.py` (Spark Structured Streaming). Di produksi sumbernya Kafka.")
    sdir = ROOT / CFG["paths"]["stream_out"] / ds
    if not sdir.exists() or not any(sdir.glob("*.parquet")):
        st.info("Belum ada skor streaming. Jalankan:\n\n```bash\ncd src\npython simulate_feed.py "
                f"{ds} --reset --batches 5 --size 200 --interval 3 &\npython stream.py {ds}\n```")
        st.stop()
    s = pd.read_parquet(sdir)
    thr = CFG["modeling"]["alert_threshold"]
    c = st.columns(4)
    c[0].metric("Akun diberi skor", f"{len(s):,}"); c[1].metric("Micro-batch", s["batch_id"].nunique())
    c[2].metric(f"Alert (skor ≥ {thr})", int(s["alert"].sum())); c[3].metric("Rata-rata skor bot", f"{s['prob_bot'].mean():.3f}")
    per = s.groupby("batch_id").agg(akun=("record_id", "count"), alert=("alert", "sum"), skor=("prob_bot", "mean")).reset_index()
    per["alert_rate"] = per["alert"] / per["akun"]
    a, b = st.columns(2)
    a.plotly_chart(px.line(per, x="batch_id", y="alert_rate", markers=True, title="Proporsi alert per micro-batch"), width="stretch")
    b.plotly_chart(px.histogram(s, x="prob_bot", nbins=30, title="Distribusi skor bot"), width="stretch")
    if "label" in s.columns and s["alert"].sum() > 0:
        prec = s.loc[s.alert == 1, "label"].mean()
        st.info(f"Precision alert terhadap kunci jawaban simulasi: **{prec:.3f}** "
                f"(base rate bot = {s['label'].mean():.3f}). Jika ≈ base rate, alert tidak informatif.")
    st.subheader("Alert terbaru")
    cols = [x for x in ["scored_at", "batch_id", "record_id", "username_ref", "prob_bot", "label"] if x in s.columns]
    st.dataframe(s[s.alert == 1].sort_values("scored_at", ascending=False)[cols].head(50), width="stretch", hide_index=True)
    if st.checkbox("Auto-refresh tiap 5 detik"):
        time.sleep(5); st.rerun()

# ------------------------------------------------------------------ 6
else:
    st.title(f"Antrean review moderator · {ds}")
    st.warning("Skor adalah **sinyal bantu** untuk memprioritaskan pemeriksaan manusia — bukan dasar tunggal "
               "untuk menghapus atau menghukum akun.")
    if not mt:
        st.stop()
    p = pd.read_parquet(gold(ds, "test_predictions.parquet"))
    thr = st.slider("Ambang skor bot", 0.0, 1.0, float(CFG["modeling"]["alert_threshold"]), 0.01)
    top = st.number_input("Tampilkan N akun teratas", 5, 500, 25)
    sel = p[p.prob_bot >= thr].sort_values("prob_bot", ascending=False)
    c = st.columns(3)
    c[0].metric("Akun di atas ambang", f"{len(sel):,}")
    c[1].metric("Precision pada ambang ini", f"{sel['label'].mean():.3f}" if len(sel) else "—")
    c[2].metric("Base rate bot (uji)", f"{p['label'].mean():.3f}")
    cols = [x for x in ["record_id", "username_ref", "prob_bot", "label"] if x in sel.columns]
    st.dataframe(sel[cols].head(int(top)).rename(columns={"label": "label_sebenarnya"}), width="stretch", hide_index=True)
