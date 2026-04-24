CONTAINER_NAME="andre_pequi_ssl_rl"
VOLUME_HF_CACHE="./volume:/ws/volume"
VOLUME_RAG_CHAT="./scripts:/ws/scripts"
GPU_DEVICE="device=1"
IMAGE="pequi-ssl-rl"
CPU_CORES="10"


# Docker run command
docker run --name $CONTAINER_NAME \
--volume $VOLUME_HF_CACHE \
--volume $VOLUME_RAG_CHAT \
--cpus $CPU_CORES \
--gpus "\"$GPU_DEVICE\"" \
-dit $IMAGE \
/bin/bash
