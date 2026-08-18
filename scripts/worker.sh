#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-ssl-el}"
CONTAINER_NAME="${CONTAINER_NAME:-pequi-ssl-worker}"
HEAD_IP="${1:-${HEAD_IP:-}}"
RAY_PORT="${RAY_PORT:-6379}"
DISPLAY="${DISPLAY:-:0.0}"
WORKER_CPUS="${WORKER_CPUS:-4}"
WORKER_NODE_IP="${WORKER_NODE_IP:-$(hostname -I | awk '{print $1}')}"

if [[ -z "$HEAD_IP" ]]; then
  echo "Uso: $0 IP_DO_COMPUTADOR_DO_HEAD"
  echo "Exemplo: $0 192.168.0.10"
  echo "Ou exporte HEAD_IP=192.168.0.10 antes de rodar."
  exit 1
fi

printf '\n[worker] Building Docker image %s...\n' "$IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$REPO_ROOT"

printf '\n[worker] Removing old container %s if it exists...\n' "$CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

printf '\n[worker] Connecting to Ray head at %s:%s with %s CPUs reserved for the worker and node IP %s...\n' "$HEAD_IP" "$RAY_PORT" "$WORKER_CPUS" "$WORKER_NODE_IP"

docker run --gpus all \
  --cpus "$WORKER_CPUS" \
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
  bash -lc "ray start --address='${HEAD_IP}:${RAY_PORT}' --node-ip=${WORKER_NODE_IP} --num-cpus=${WORKER_CPUS} --block"
