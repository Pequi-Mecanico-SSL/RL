#!/usr/bin/env bash
# Relanca o run C2 (iter250->260) quando o preflight de recursos passar.
# Gates: MemAvailable >= 9,5e9 em 2 medicoes/20 s, 0 CUDA apps, VRAM >= 10240 MiB, disco >= 30 GB.
set -u
cd "$(dirname "$0")/.."
LOG=training_runs/h1_runC2/launcher.log
mkdir -p training_runs/h1_runC2/ray_results

# Restore unico e verificado por hash (condicao do policy-verifier)
mapfile -t CANDS < <(ls -d "$PWD"/training_runs/h1_runC/ray_results/h1_runC_iter260/PPO_Soccer_*/checkpoint_000002 2>/dev/null)
if [ "${#CANDS[@]}" -ne 1 ]; then
  echo "$(date -Is) ERRO: esperado exatamente 1 restore, achados ${#CANDS[@]}" >> "$LOG"; exit 1
fi
RESTORE="${CANDS[0]}"
EXPECTED=$(awk '/policy_blue\/policy_state.pkl/{print $1}' experiment_results/h1ext_runC2_preflight_manifest.txt)
ACTUAL=$(sha256sum "$RESTORE/policies/policy_blue/policy_state.pkl" | awk '{print $1}')
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "$(date -Is) ERRO: hash do restore divergente ($ACTUAL != $EXPECTED)" >> "$LOG"; exit 1
fi

measure() {
  MEM=$(grep MemAvailable /proc/meminfo | awk '{print $2*1024}')
  CUDA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  DISK=$(df -B1 --output=avail . | tail -1)
  echo "$(date -Is) mem=$MEM cuda=$CUDA vram=$VRAM disk=$DISK" >> "$LOG"
  [ "$MEM" -ge 9500000000 ] && [ "$CUDA" -eq 0 ] && [ "$VRAM" -ge 10240 ] && [ "$DISK" -ge 30000000000 ]
}

while true; do
  if measure; then
    sleep 20
    if measure; then
      echo "$(date -Is) PREFLIGHT PASS — lancando run C2 (restore $RESTORE)" >> "$LOG"
      nohup scripts/watchdog_host_memory.sh h1_runC2 training_runs/h1_runC2/watchdog.log >/dev/null 2>&1 &
      WD_PID=$!
      docker run --rm --name h1_runC2 --gpus all --memory=7g --memory-swap=7g --shm-size=1536m \
        -w /campaign -v "$PWD:/campaign" \
        -v "$PWD/training_runs/h1_runC2/ray_results:/root/ray_results" \
        -v /tmp/contrato_hist/rewards.py:/campaign/rewards.py:ro \
        -v /home/marcos/Documentos/RL-policy-improvement/rSoccer:/campaign/rSoccer:ro \
        -e PYTHONPATH=/campaign:/campaign/rSoccer rl-policy-training:c684c2b \
        python RL_train.py --config config.control-1w.yaml \
        --restore "$RESTORE" \
        --stop-timesteps 10015200 --experiment-name h1_runC2_iter260 \
        > training_runs/h1_runC2/train.log 2>&1
      RC=$?
      kill "$WD_PID" 2>/dev/null
      echo "$(date -Is) EXIT=$RC run C2 encerrado; watchdog ($WD_PID) finalizado" >> "$LOG"
      exit "$RC"
    fi
  fi
  sleep 60
done
