#!/usr/bin/env python3
"""
Teste rapido de recepcao SSL-Vision (multicast) sem carregar PyTorch/policy.

Uso no host (grSim rodando, mesma rede que envia vision):
  cd /caminho/RL && python3 scripts/check_grsim_multicast_recv.py

Ou no container rl-policy:
  python3 /app/scripts/check_grsim_multicast_recv.py

Variaveis: VISION_PORT (default 10020), VISION_ADDRESS (default 224.5.23.2), VISION_BIND_ADDR (opcional)
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import time


def main() -> int:
    port = int(os.environ.get("VISION_PORT", "10020"))
    group = os.environ.get("VISION_ADDRESS", "224.5.23.2")
    bind_addr = os.environ.get("VISION_BIND_ADDR", "").strip()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    sock.bind((bind_addr, port))
    mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(1.0)

    print(f"Escutando {group}:{port} bind={bind_addr or '*'} (5s)...")
    n = 0
    t0 = time.time()
    while time.time() - t0 < 5.0:
        try:
            data, addr = sock.recvfrom(65535)
            n += 1
            print(f"  pacote {n}: {len(data)} bytes de {addr}")
            if n >= 5:
                break
        except socket.timeout:
            continue
    sock.close()
    if n == 0:
        print("Nenhum pacote recebido. Verifique: grSim aberto, vision na mesma porta, firewall, network_mode host.")
        return 1
    print(f"OK: {n} pacote(s) UDP na janela.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
