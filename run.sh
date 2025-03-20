CONTAINER_NAME="andre_pequi_ssl_rl"
VOLUME_HF_CACHE="./volume:/ws/volume"
VOLUME_RAG_CHAT="./scripts:/ws/scripts"
GPU_DEVICE="device=0"
IMAGE="pequi-ssl-rl"
CPU_CORES="10"




docker stop andre_pequi_ssl_rl
docker rm andre_pequi_ssl_rl
# Docker run command
docker run --name $CONTAINER_NAME \
--volume ".:/ws" \
--volume "./dgx_checkpoints/:/root/ray_results/" \
--network=host \
-e DISPLAY=$DISPLAY \
--volume $VOLUME_HF_CACHE \
--volume $VOLUME_RAG_CHAT \
--cpus $CPU_CORES \
--gpus "\"$GPU_DEVICE\"" \
-dit $IMAGE \
/bin/bash
docker attach andre_pequi_ssl_rl
