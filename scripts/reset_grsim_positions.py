"""Reposiciona os robos blue 0..2, yellow 0..2 e a bola no centro do campo.

Util para testes de deploy quando os robos nascem encostados na parede.
Envia um grSim_Replacement via UDP para 127.0.0.1:20011.

Uso (dentro do container):
    python3 scripts/reset_grsim_positions.py
"""
import socket
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
import grSim_Packet_pb2

BLUE = [(-1.5, 0.0, 0.0), (-2.0, 1.0, 0.0), (-2.0, -1.0, 0.0)]
YELLOW = [(1.5, 0.0, 180.0), (2.0, 1.0, 180.0), (2.0, -1.0, 180.0)]


def main():
    pkt = grSim_Packet_pb2.grSim_Packet()
    rep = pkt.replacement

    for i, (x, y, d) in enumerate(BLUE):
        r = rep.robots.add()
        r.x, r.y, r.dir = x, y, d
        r.id = i
        r.yellowteam = False
        r.turnon = True

    for i, (x, y, d) in enumerate(YELLOW):
        r = rep.robots.add()
        r.x, r.y, r.dir = x, y, d
        r.id = i
        r.yellowteam = True
        r.turnon = True

    rep.ball.x = 0.0
    rep.ball.y = 0.0
    rep.ball.vx = 0.0
    rep.ball.vy = 0.0

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = pkt.SerializeToString()
    s.sendto(data, ("127.0.0.1", 20011))
    print(f"Replacement enviado ({len(data)} bytes).")


if __name__ == "__main__":
    main()
