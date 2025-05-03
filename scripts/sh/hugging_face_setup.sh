#!/bin/bash

pip install -U huggingface-hub huggingface_hub[cli] torch tensorflow gym ray[rllib]
sudo apt update
sudo apt install -y git-lfs

cd volumes/dgx_checkpoints

rm -rf .git
git init
git lfs install

{
    echo "*.bin filter=lfs diff=lfs merge=lfs -text"
    echo "*.pth filter=lfs diff=lfs merge=lfs -text"
    echo "*.ckpt filter=lfs diff=lfs merge=lfs -text"
    echo "*.h5 filter=lfs diff=lfs merge=lfs -text"
    echo "*.zip filter=lfs diff=lfs merge=lfs -text"
    echo "*.csv filter=lfs diff=lfs merge=lfs -text"
    echo "*.json filter=lfs diff=lfs merge=lfs -text"
    echo "*.tfevents.* filter=lfs diff=lfs merge=lfs -text"
    echo "events.out.tfevents.* filter=lfs diff=lfs merge=lfs -text"
    echo "*.pkl filter=lfs diff=lfs merge=lfs -text"
    echo "**/policy_*.* filter=lfs diff=lfs merge=lfs -text"
    echo "**/checkpoint_*/** filter=lfs diff=lfs merge=lfs -text"
} > .gitattributes

git lfs track
git remote add origin git@hf.co:pmecssl/SSL-RL
git rm -r --cached .
git add .

git commit -m "Upload LFS"

git push -f origin main

cd -