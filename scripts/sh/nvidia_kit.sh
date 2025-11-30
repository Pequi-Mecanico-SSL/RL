#!/bin/bash

if ! command -v curl &> /dev/null
then
    echo "O curl não está instalado. Instalando..."
    sudo apt update
    sudo apt install -y curl
else
    echo "O curl já está instalado."
fi

# Adiciona a chave GPG do repositório NVIDIA
echo "Adicionando chave GPG do repositório NVIDIA..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /tmp/nvidia-gpgkey
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /tmp/nvidia-gpgkey

# Adiciona o repositório do NVIDIA Container Toolkit
echo "Adicionando o repositório NVIDIA Container Toolkit..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o /tmp/nvidia-container-toolkit.list
sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' /tmp/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Atualiza a lista de pacotes do repositório
echo "Atualizando a lista de pacotes..."
sudo apt-get update > /dev/null

# Instala o NVIDIA Container Toolkit
echo "Instalando o NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo "Instalação concluída!"
