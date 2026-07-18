#!/bin/bash
# Deploy the bug-out box onto the Pi. Run FROM the Pi (over ssh/tailscale).
#   ssh caleb@bugout-box 'bash -s' < deploy/deploy.sh
set -e
REPO=https://github.com/masonlabs-git/bugout-box
DIR=/home/caleb/bugout-box

echo "== 1. code =="
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only
else git clone "$REPO" "$DIR"; fi

echo "== 2. python deps =="
pip3 install --break-system-packages -q flask webrtcvad requests || \
  pip3 install -q flask webrtcvad requests

echo "== 3. reclaim RAM: stop WALL-E robot services (box owns cam + audio) =="
sudo systemctl disable --now walle-node walle-cam 2>/dev/null || true

echo "== 4. pin the model resident =="
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment=OLLAMA_KEEP_ALIVE=-1\nEnvironment=OLLAMA_MAX_LOADED_MODELS=1\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart ollama 2>/dev/null || true

echo "== 5. install services =="
sudo cp "$DIR"/deploy/bugout-box.service "$DIR"/deploy/bugout-dashboard.service \
  /etc/systemd/system/
sudo systemctl daemon-reload

echo "== 6. sanity: index present? =="
python3 - <<PY
import os
os.environ.setdefault("BOX_VAULT", "/media/caleb/Expansion/vault")
from box.retrieval import connect
n = connect().execute("SELECT count(*) FROM chunks").fetchone()[0]
print(f"index chunks: {n}")
PY

echo "== DONE. Start with: sudo systemctl enable --now bugout-box bugout-dashboard =="
