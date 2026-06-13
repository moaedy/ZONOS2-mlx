#!/bin/bash
# Keep the ZONOS2-mlx web demo running, restarting it if it ever exits.
#   ./serve.sh                # full bf16
#   ./serve.sh --quantize 4   # ~6 GB
# Logs to zonos2_app.log. Stop with:  pkill -f app.py
cd "$(dirname "$0")" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
while true; do
    python app.py "$@"
    echo "[serve] app exited ($(date)); restarting in 2s..." >&2
    sleep 2
done
