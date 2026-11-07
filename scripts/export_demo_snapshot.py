"""Ekspor agregat hasil analisis TANPA identitas akun dari MongoDB ke demo/analysis_aggregates.json.

Dijalankan setelah run lengkap (pipeline analisis + src/synthesis.py) untuk memperbarui data demo
yang dibaca scripts/seed_demo.py. Yang diekspor hanya koleksi agregat (metrik model, ringkasan level
risiko, perbandingan konsep, ...), tidak pernah account_scores, pipeline_* atau stream_scores.
Seluruh isi dipindai pola identitas akun; bila ada yang cocok, file TIDAK ditulis.

  python scripts/export_demo_snapshot.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402

DEST = ROOT / "demo" / "analysis_aggregates.json"
AGGREGATES = ["model_metrics", "datasets", "feature_importance", "risk_level_definitions", "risk_level_summary",
              "concept_comparison", "coordination_evidence", "high_score_profiles", "cross_platform_summary"]
# Pola identitas yang pernah muncul di keluaran proyek ini (lihat docs/adr): account_uid lintas platform,
# handle baris Instagram, id numerik panjang (id X), kunci author TikTok, daftar profile id Facebook,
# dan nama field per-akun.
ID_PATTERNS = {
    "account_uid": r"\b(?:x|xs|ig|fb|tt)_[0-9a-f]{3,}\b",
    "instagram_row_handle": r"\b(?:train|test)(?:\.csv:|_r)\d+",
    "long_numeric_id": r"(?<![\d.])\d{7,}(?![\d.])",
    "tiktok_author_key": r"\b[0-9a-f]{16}\b",
    "facebook_profile_id_list": r"profile[ _]ids?\D{0,12}\d+(?:, \d+)+",
    "per_account_field": r'"(?:account_uid|source_ref|screen_name|username|username_ref|record_id|'
                         r'profile_id|user_id|author|author_key)"\s*:',
}


def render(snap: dict) -> str:
    """JSON dengan satu dokumen per baris: mudah direview dan diff-nya kecil saat diperbarui."""
    out = ["{"]
    for i, (coll, docs) in enumerate(snap.items()):
        out.append(f"  {json.dumps(coll)}: [")
        out += ["    " + json.dumps(d, ensure_ascii=False, separators=(", ", ": "), default=str)
                + ("," if j < len(docs) - 1 else "") for j, d in enumerate(docs)]
        out.append("  ]" + ("," if i < len(snap) - 1 else ""))
    return "\n".join(out + ["}"]) + "\n"


def find_identifiers(text: str) -> dict:
    """Pola identitas yang cocok di `text` (kosong = aman). Fungsi murni, mudah diuji."""
    hits = {name: sorted(set(re.findall(p, text)))[:5] for name, p in ID_PATTERNS.items()}
    return {k: v for k, v in hits.items() if v}


def main():
    client, db = common.get_mongo(timeout_ms=3000)
    client.admin.command("ping")
    snap = {c: list(db[c].find({}, {"_id": 0})) for c in AGGREGATES}
    if not snap["model_metrics"]:
        raise SystemExit("Koleksi agregat kosong: jalankan dulu src/synthesis.py (bukan database demo).")
    text = render(snap)
    hits = find_identifiers(text)
    if hits:
        raise SystemExit(f"DIBATALKAN: pola identitas akun ditemukan, file tidak ditulis: {hits}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(text, encoding="utf-8")
    print(f"wrote {DEST.relative_to(ROOT)}: " + ", ".join(f"{c}={len(v)}" for c, v in snap.items()))
    client.close()


if __name__ == "__main__":
    main()
