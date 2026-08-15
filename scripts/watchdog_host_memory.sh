#!/bin/bash
# Watchdog ativo de memória: mata o container de treino se MemAvailable do host
# ou a memória do container violarem os limites pré-registrados da campanha.
# Uso: watchdog_host_memory.sh <container_name> <logfile>
set -u
NAME="$1"; LOG="$2"
HOST_MIN_BYTES=2500000000        # abort: MemAvailable < 2,5 GB
CONT_MAX_BYTES=6979321856        # abort: container >= 6,5 GiB

CID=""
while true; do
  if [[ -z "$CID" ]]; then
    CID=$(docker ps -q --filter "name=$NAME" | head -1)
  fi
  AVAIL=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
  CONT=0
  if [[ -n "$CID" ]]; then
    # paridade com docker stats: usage - inactive_file (memory.current inclui page cache)
    CGDIR="/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' "$CID" 2>/dev/null).scope"
    if [[ -r "$CGDIR/memory.current" ]]; then
      CUR=$(cat "$CGDIR/memory.current")
      INACT=$(awk '/^inactive_file/{print $2}' "$CGDIR/memory.stat")
      CONT=$((CUR - INACT))
    fi
    # container terminou?
    if ! docker ps -q --no-trunc | grep -q "$(docker inspect -f '{{.Id}}' "$CID" 2>/dev/null)"; then
      echo "$(date -Is) container encerrou; watchdog saindo" >> "$LOG"; exit 0
    fi
  fi
  echo "$(date -Is) avail=$AVAIL cont=$CONT" >> "$LOG"
  if (( AVAIL < HOST_MIN_BYTES )) || (( CONT >= CONT_MAX_BYTES )); then
    echo "$(date -Is) ABORT: avail=$AVAIL cont=$CONT — matando $CID" >> "$LOG"
    [[ -n "$CID" ]] && docker kill "$CID" >> "$LOG" 2>&1
    exit 1
  fi
  sleep 1
done
