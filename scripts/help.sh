docker rm pequi-ssl
docker build -t ssl-el .
docker run --gpus all --name pequi-ssl \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/volumes/videos:/ws/videos \
    -v $(pwd)/volumes/dgx_checkpoints/PPO_selfplay_rec:/root/ray_results/PPO_selfplay_rec \
    -v $(pwd)/src:/ws/src \
    -p 5678:5678 \
    -p 5679:5679 \
    -it ssl-el