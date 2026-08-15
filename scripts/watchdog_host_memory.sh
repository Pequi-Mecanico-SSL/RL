#!/bin/bash
# Watchdog ativo de memória: mata o container de treino se MemAvailable do host
# ou a memória do container violarem os limites pré-registrados da campanha.
# Uso: watchdog_host_memory.sh <container_name> <logfile>
set -u
NAME="$1"; LOG="$2"
HOST_MIN_BYTES=2500000000        # abort imediato: MemAvailable < 2,5 GB
CONT_MAX_BYTES=6979321856        # abort se sustentado >= 4,0 s: container >= 6,5 GiB
SUSTAIN_SECS=4

CID=""
CGDIR=""
BREACH_START=0
while true; do
  if [[ -z "$CID" ]]; then
    CID=$(docker ps -q --no-trunc --filter "name=$NAME" | head -1)
    [[ -n "$CID" ]] && CGDIR="/sys/fs/cgroup/system.slice/docker-$CID.scope"
  fi
  AVAIL=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
  CONT=0
  if [[ -n "$CID" ]]; then
    if [[ -r "$CGDIR/memory.current" ]]; then
      # paridade com docker stats: usage - inactive_file (memory.current inclui page cache)
      CUR=$(cat "$CGDIR/memory.current")
      INACT=$(awk '/^inactive_file/{print $2}' "$CGDIR/memory.stat")
      CONT=$((CUR - INACT))
    else
      echo "$(date -Is) container encerrou; watchdog saindo" >> "$LOG"; exit 0
    fi
  fi
  echo "$(date -Is) avail=$AVAIL cont=$CONT" >> "$LOG"
  if (( AVAIL < HOST_MIN_BYTES )); then
    echo "$(date -Is) ABORT host: avail=$AVAIL — matando $CID" >> "$LOG"
    [[ -n "$CID" ]] && docker kill "$CID" >> "$LOG" 2>&1
    exit 1
  fi
  NOW=$(date +%s%N)
  if (( CONT >= CONT_MAX_BYTES )); then
    if (( BREACH_START == 0 )); then BREACH_START=$NOW; fi
    ELAPSED_MS=$(( (NOW - BREACH_START) / 1000000 ))
    if (( ELAPSED_MS >= SUSTAIN_SECS * 1000 )); then
      echo "$(date -Is) ABORT container sustentado ${ELAPSED_MS}ms: cont=$CONT — matando $CID" >> "$LOG"
      [[ -n "$CID" ]] && docker kill "$CID" >> "$LOG" 2>&1
      exit 1
    fi
  else
    BREACH_START=0
  fi
  sleep 0.5
done
