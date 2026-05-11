#!/usr/bin/env bash
# Diagnostico: verifica se ha trafego UDP SSL-Vision na porta do grSim (default 10020).
# Uso no host Linux (com grSim rodando):
#   ./scripts/sh/check_grsim_vision_udp.sh
#   ./scripts/sh/check_grsim_vision_udp.sh 10020
#
# Opcional: captura pacotes (precisa tcpdump e costuma precisar sudo)
set -euo pipefail

PORT="${1:-10020}"

echo "=== Portas UDP relevantes (ss) ==="
ss -ulpn 2>/dev/null | grep -E ":${PORT}\\b|:20011\\b|:10300\\b|:10301\\b|:10302\\b" || echo "(nenhuma encontrada ou ss sem permissao)"

echo ""
echo "=== Captura udp port ${PORT} (3s, ate 8 pacotes) ==="
if command -v tcpdump >/dev/null 2>&1; then
  if timeout 3 tcpdump -n -i any "udp port ${PORT}" -c 8 2>&1; then
    :
  else
    echo "tcpdump encerrou sem pacotes ou sem permissao (tente: sudo $0 ${PORT})"
  fi
else
  echo "tcpdump nao instalado; instale com: sudo apt install tcpdump"
fi

echo ""
echo "Multicast esperado: 224.5.23.2:${PORT} (vision). Comandos: UDP 20011 (grSim_Packet)."
echo "Dois containers no mesmo host: policy usa SO_REUSEPORT na ${PORT}; senao use um so receiver."
