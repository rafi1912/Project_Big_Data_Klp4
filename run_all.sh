#!/usr/bin/env bash
# Menjalankan pipeline batch end-to-end untuk semua dataset yang ada di data/landing/<kunci>/
set -euo pipefail
cd "$(dirname "$0")/src"
python ingest.py     # 1. landing  -> bronze
python process.py    # 2. bronze   -> silver + gold + laporan kualitas data
python train.py      # 3. silver   -> model terbaik + metrik
echo
echo "Batch selesai. Streaming demo (contoh dataset twitter_bot_detection):"
echo "  python simulate_feed.py twitter_bot_detection --reset --batches 5 --size 200 --interval 3 &"
echo "  python stream.py twitter_bot_detection"
echo "Dashboard:  cd .. && streamlit run app/streamlit_app.py"
