#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-ssl-el}"
CONTAINER_NAME="${CONTAINER_NAME:-pequi-ssl-head}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
DISPLAY="${DISPLAY:-:0.0}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}') }"

printf '\n[head] Building Docker image %s...\n' "$IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$REPO_ROOT"

printf '\n[head] Removing old container %s if it exists...\n' "$CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

printf '\n[head] Starting Ray head node on port %s with host IP %s...\n' "$RAY_PORT" "$HOST_IP"

docker run --gpus all \
  --name "$CONTAINER_NAME" \
  --network host \
  --shm-size=11g \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$REPO_ROOT":/workspace \
  -v "$REPO_ROOT/volumes/videos:/ws/videos" \
  -v "$REPO_ROOT/volumes/dgx_checkpoints/PPO_selfplay_rec:/root/ray_results/PPO_selfplay_rec" \
  -v "$REPO_ROOT/src:/ws/src" \
  -v "$REPO_ROOT/config.yaml:/ws/config.yaml" \
  -p 5678:5678 \
  -p 5679:5679 \
  -w /workspace \
  -it "$IMAGE_NAME" \
  bash -lc "ray start --head --node-ip=${HOST_IP} --dashboard-host=0.0.0.0 --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --block"
