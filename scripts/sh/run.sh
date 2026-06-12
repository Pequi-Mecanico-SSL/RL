CONTAINER_NAME="andre_pequi_ssl_rl"
VOLUME_HF_CACHE="./volume:/ws/volume"
VOLUME_RAG_CHAT="./scripts:/ws/scripts"
GPU_DEVICE="all"
IMAGE="pequi-ssl-rl"
CPU_CORES="10"


# Docker run command
docker run --name "$CONTAINER_NAME" \
  --volume "$VOLUME_HF_CACHE" \
  --volume "$VOLUME_RAG_CHAT" \
  --cpus "$CPU_CORES" \
  --gpus "$GPU_DEVICE" \
  --ipc=host \
  --shm-size=5g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -dit "$IMAGE" \
  /bin/bash
