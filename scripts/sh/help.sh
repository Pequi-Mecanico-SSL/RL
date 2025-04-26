docker rm pequi-ssl
docker build -t ssl-el .
docker run --gpus all --name pequi-ssl \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/volumes/videos:/ws/videos \
    -v $(pwd)/volumes/dgx_checkpoints/PPO_selfplay_rec:/root/ray_results/PPO_selfplay_rec \
    -p 7860:7860 \
    -v $(pwd)/RL_GUI.py:/ws/RL_GUI.py \
    -it ssl-el