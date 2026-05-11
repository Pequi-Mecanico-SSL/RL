"""Valida que a policy esta efetivamente movendo robos no grSim.

Coleta posicoes em t=0, espera 3 s (policy mandando comandos), coleta de novo
e imprime os deltas em milimetros.

Uso (dentro do container `rl_policy_deploy`):
    python3 scripts/validate_motion.py

Tambem pode ser copiado para /tmp:
    docker cp scripts/validate_motion.py rl_policy_deploy:/tmp/v.py
    docker exec rl_policy_deploy python3 /tmp/v.py
"""
import socket
import struct
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
import ssl_vision_wrapper_pb2


def make_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 10020))
    mreq = struct.pack("4sl", socket.inet_aton("224.5.23.2"), socket.INADDR_ANY)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.settimeout(2.0)
    return s


def collect_positions(sock, duration=1.0):
    positions = {"blue": {}, "yellow": {}}
    end = time.time() + duration
    while time.time() < end:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            break
        w = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
        w.ParseFromString(data)
        for r in w.detection.robots_blue:
            positions["blue"][r.robot_id] = (r.x, r.y, r.orientation)
        for r in w.detection.robots_yellow:
            positions["yellow"][r.robot_id] = (r.x, r.y, r.orientation)
    return positions


def main():
    sock = make_socket()
    print("[t=0s] coletando posicoes iniciais por 1 s...")
    p0 = collect_positions(sock, 1.0)
    print(f"  blue ids: {sorted(p0['blue'].keys())}")
    print(f"  yellow ids: {sorted(p0['yellow'].keys())}")
    for i in range(3):
        print(f"  blue_{i}: {p0['blue'].get(i)}")

    print("\nesperando 3 s (policy enviando comandos)...")
    time.sleep(3.0)

    print("[t=4s] coletando posicoes finais por 1 s...")
    p1 = collect_positions(sock, 1.0)
    for i in range(3):
        print(f"  blue_{i}: {p1['blue'].get(i)}")

    print("\n=== DELTAS (mm) ===")
    for team in ("blue", "yellow"):
        for rid in sorted(p0[team]):
            if rid not in p1[team]:
                continue
            x0, y0, _ = p0[team][rid]
            x1, y1, _ = p1[team][rid]
            dx, dy = x1 - x0, y1 - y0
            d = (dx * dx + dy * dy) ** 0.5
            print(f"  {team}_{rid}: dx={dx:+7.1f} dy={dy:+7.1f} dist={d:6.1f} mm")


if __name__ == "__main__":
    main()
