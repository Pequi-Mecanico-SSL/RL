"""Visualizador top-down 2D do SSL-Vision em tempo real.

Le multicast 224.5.23.2:10020 e mostra o campo com robos azuis, amarelos e bola.
Roda fora de container (host) com network host ou via UDP local.

Uso:
    python scripts/vision_viewer.py
"""
import socket
import struct
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, ".")
import ssl_vision_wrapper_pb2

MCAST_GRP = "224.5.23.2"
MCAST_PORT = 10020

# Campo Division B (mm)
FIELD_LEN = 9000
FIELD_W = 6000


def make_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(False)
    return s


def drain_latest(sock):
    """Retorna o ultimo frame com deteccao recebido (drena fila)."""
    last = None
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except BlockingIOError:
            break
        w = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
        try:
            w.ParseFromString(data)
        except Exception:
            continue
        det = w.detection
        if det.robots_blue or det.robots_yellow or det.balls:
            last = det
    return last


def main():
    sock = make_socket()
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.canvas.manager.set_window_title("SSL-Vision viewer (Ctrl-C para sair)")

    last_render = 0.0
    while True:
        det = drain_latest(sock)
        now = time.time()
        if det is None or (now - last_render) < 0.05:
            plt.pause(0.01)
            continue
        last_render = now

        ax.clear()
        # campo
        ax.add_patch(
            patches.Rectangle(
                (-FIELD_LEN / 2, -FIELD_W / 2),
                FIELD_LEN,
                FIELD_W,
                fill=False,
                edgecolor="white",
                linewidth=2,
            )
        )
        ax.axhline(0, color="white", linewidth=0.5)
        ax.axvline(0, color="white", linewidth=0.5)

        for r in det.robots_blue:
            ax.add_patch(patches.Circle((r.x, r.y), 90, color="#3399ff"))
            ax.text(r.x, r.y, str(r.robot_id), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
            # heading
            ax.plot(
                [r.x, r.x + 150 * np.cos(r.orientation)],
                [r.y, r.y + 150 * np.sin(r.orientation)],
                color="white", linewidth=1.5,
            )

        for r in det.robots_yellow:
            ax.add_patch(patches.Circle((r.x, r.y), 90, color="#ffcc00"))
            ax.text(r.x, r.y, str(r.robot_id), ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold")
            ax.plot(
                [r.x, r.x + 150 * np.cos(r.orientation)],
                [r.y, r.y + 150 * np.sin(r.orientation)],
                color="black", linewidth=1.5,
            )

        for b in det.balls:
            ax.add_patch(patches.Circle((b.x, b.y), 50, color="orange"))

        ax.set_xlim(-FIELD_LEN / 2 - 500, FIELD_LEN / 2 + 500)
        ax.set_ylim(-FIELD_W / 2 - 500, FIELD_W / 2 + 500)
        ax.set_aspect("equal")
        ax.set_facecolor("#0a5d2c")
        ax.set_title(f"blue={len(det.robots_blue)}  yellow={len(det.robots_yellow)}  ball={len(det.balls)}")
        plt.pause(0.001)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nencerrado.")
