#!/bin/bash

echo "Removing old Docker versions..."
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    echo -e "\tRemoving $pkg..."
    sudo apt-get remove $pkg > /dev/null
done

echo "Adding Docker gpgp key..."
sudo apt-get update > /dev/null
sudo apt-get install ca-certificates curl > /dev/null
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update > /dev/null

echo "Installing Docker..."
for pkg in docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; do
    echo -e "\tInstalling $pkg..."
    sudo apt-get install -y $pkg > /dev/null
done

echo "Adding user to Docker group..."

# Verifica se o grupo 'docker' existe antes de criá-lo
if ! getent group docker > /dev/null 2>&1; then
    sudo groupadd docker
    echo -e "\tGroup 'docker' created."
else
    echo -e "\tGroup 'docker' already exists."
fi

# Adiciona o usuário ao grupo 'docker'
sudo usermod -aG docker $USER
echo -e "\tUser $USER added to 'docker' group."

# Atualiza a sessão para reconhecer a nova permissão
newgrp docker

echo "Docker installation completed!"