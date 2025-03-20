# Descrição: Dockerfile para criar um contêiner com o ambiente de desenvolvimento do rSim

# Use uma imagem base ros2 iron
FROM osrf/ros:iron-desktop

# Defina o diretório de trabalho dentro do contêiner (ws == workspace)
WORKDIR /ws

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    pip \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    libode-dev \
    python3-opengl \
    python3-pyqt5 \
    python3-pyqtgraph \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências do Python listadas em requirements.txt
RUN pip install --no-cache-dir setuptools==65.5.0 pip==21 wheel==0.38.0
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118 --ignore-installed

# Instalar rSim
RUN pip install git+https://github.com/Pequi-Mecanico-SSL/rSim.git

RUN mkdir videos
COPY record_video.py ../usr/local/lib/python3.10/site-packages/gymnasium/wrappers/record_video.py
COPY video_recorder.py ../usr/local/lib/python3.10/site-packages/gymnasium/wrappers/monitoring/video_recorder.py

# Copy the rSoccer directory
RUN mkdir /rsoccer_gym
COPY rsoccer_gym rsoccer_gym

COPY rllib_multiagent.py .
COPY action_dists.py .
COPY custom_torch_model.py .
COPY config.yaml .
COPY render_episode.py .
COPY sim2real /ws/sim2real
COPY sim2real.py .

RUN mkdir /ws/volume
RUN mkdir /ws/scripts

# Iniciar o contêiner com o bash
CMD ["/bin/bash"]
