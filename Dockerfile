# Descrição: Dockerfile para criar um contêiner com o ambiente de desenvolvimento do rSim

# Use uma imagem base Python oficial
FROM python:3.10-slim-bookworm

# Defina o diretório de trabalho dentro do contêiner (ws == workspace)
WORKDIR /ws

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
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

RUN pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências do Python listadas em requirements.txt
RUN pip install --no-cache-dir setuptools==65.5.0 pip==21 wheel==0.38.0
RUN pip install --no-cache-dir -r requirements.txt

# Instalar o mesmo contrato do rSim de forma reproduzivel. O projeto usa um
# pybind11 antigo; CMake 4 requer declarar a policy minima explicitamente.
ARG RSIM_COMMIT=c30ec84bc07de9ae60c6d5a1d5e9283832b46d9a
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
RUN pip install "git+https://github.com/Pequi-Mecanico-SSL/rSim.git@${RSIM_COMMIT}"

RUN mkdir videos
COPY scripts/gymnasium/record_video.py ../usr/local/lib/python3.10/site-packages/gymnasium/wrappers/record_video.py
COPY scripts/gymnasium/video_recorder.py ../usr/local/lib/python3.10/site-packages/gymnasium/wrappers/monitoring/video_recorder.py


RUN mkdir /ws/scripts
COPY scripts /ws/scripts

# Copy the rSoccer directory
RUN mkdir /ws/rSoccer
COPY rSoccer /ws/rSoccer
RUN cd /ws/rSoccer && pip install .

COPY RL_train.py .
COPY RL_eval.py .
COPY config.yaml .
COPY rewards.py .
# COPY sim2real /ws/sim2real
# COPY sim2real.py .

# RUN mkdir /ws/volume

# Iniciar o contêiner com o bash
CMD ["/bin/bash"]