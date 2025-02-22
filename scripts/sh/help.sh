docker rm pequi-ssl
docker build -t ssl-el .
docker run --gpus all --name pequi-ssl \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/videos:/ws/videos \
    -v $(pwd)/dgx_checkpoints/PPO_selfplay_rec:/root/ray_results/PPO_selfplay_rec \
    -it ssl-el