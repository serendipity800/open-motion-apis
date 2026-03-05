#!/bin/bash
# Start all motion_api backends + motion store server.
# Usage: bash scripts/start_all.sh [--device cuda:0]

set -e
DEVICE=${1:-cuda:0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

echo "=== Starting motion API servers (device=$DEVICE) ==="

echo "[1/5] MoMask (port 8081)..."
cd "$ROOT/../momask-codes"
conda run -n momask --no-capture-output \
  python "$ROOT/motion_api/backends/momask_server.py" \
    --port 8081 --device "$DEVICE" --dataset t2m \
  > "$LOGDIR/momask.log" 2>&1 &
echo "  PID=$!"

echo "[2/5] MDM (port 8082)..."
cd "$ROOT/../motion-diffusion-model"
conda run -n mdm --no-capture-output \
  python "$ROOT/motion_api/backends/mdm_server.py" \
    --port 8082 --device "$DEVICE" \
    --model-path ./save/humanml_enc_512_50steps/model000750000.pt \
  > "$LOGDIR/mdm.log" 2>&1 &
echo "  PID=$!"

echo "[3/5] MLD (port 8083)..."
cd "$ROOT/../motion-latent-diffusion"
conda run -n mld --no-capture-output \
  python "$ROOT/motion_api/backends/mld_server.py" \
    --port 8083 --device "$DEVICE" \
    --cfg ./configs/config_mld_humanml3d.yaml \
  > "$LOGDIR/mld.log" 2>&1 &
echo "  PID=$!"

echo "[4/5] T2M-GPT (port 8084)..."
cd "$ROOT/../T2M-GPT"
conda run -n T2M-GPT --no-capture-output \
  python "$ROOT/motion_api/backends/t2m_gpt_server.py" \
    --port 8084 --device "$DEVICE" \
  > "$LOGDIR/t2m_gpt.log" 2>&1 &
echo "  PID=$!"

echo "[5/5] Motion store server (port 8090)..."
cd "$ROOT"
python -m motion_store.server --port 8090 \
  > "$LOGDIR/motion_store.log" 2>&1 &
echo "  PID=$!"

echo ""
echo "Waiting 60s for model loading..."
sleep 60

echo ""
echo "=== Health checks ==="
for port in 8081 8082 8083 8084 8090; do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
  echo "  port $port: HTTP $status"
done

echo ""
echo "Logs: $LOGDIR"
