"""Reposiciona os robos blue 0..2, yellow 0..2 e a bola no centro do campo.

Util para testes de deploy quando os robos nascem encostados na parede.
Envia um grSim_Replacement via UDP para 127.0.0.1:20011.

Uso (dentro do container):
    python3 scripts/reset_grsim_positions.py
"""
import socket
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
import grSim_Packet_pb2

BLUE = [(-1.5, 0.0, 0.0), (-2.0, 1.0, 0.0), (-2.0, -1.0, 0.0)]
YELLOW = [(1.5, 0.0, 180.0), (2.0, 1.0, 180.0), (2.0, -1.0, 180.0)]


def send_zero_commands(sock, repeats=5):
    for _ in range(repeats):
        for yellowteam in (False, True):
            packet = grSim_Packet_pb2.grSim_Packet()
            packet.commands.timestamp = time.time()
            packet.commands.isteamyellow = yellowteam
            for robot_id in range(3):
                command = packet.commands.robot_commands.add()
                command.id = robot_id
                command.wheelsspeed = False
                command.kickspeedx = 0.0
                command.kickspeedz = 0.0
                command.veltangent = 0.0
                command.velnormal = 0.0
                command.velangular = 0.0
                command.spinner = False
            sock.sendto(packet.SerializeToString(), ("127.0.0.1", 20011))
        time.sleep(0.02)


def build_off_packet():
    # A imagem grSim preserva orientacao/estado de robos que ja estao ligados.
    # O off mantem cada ativo na propria posicao para evitar colisao em (0,0),
    # que injetava velocidades persistentes no corpo.
    off_pkt = grSim_Packet_pb2.grSim_Packet()
    for yellowteam in (False, True):
        for robot_id in range(11):
            r = off_pkt.replacement.robots.add()
            r.x = r.y = r.dir = 0.0
            r.id = robot_id
            r.yellowteam = yellowteam
            r.turnon = False
            if robot_id < 3:
                positions = YELLOW if yellowteam else BLUE
                r.x, r.y, r.dir = positions[robot_id]
    return off_pkt


def build_on_packet():
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

    # Mantem desligados os IDs que nao pertencem ao contrato 3v3.
    for yellowteam in (False, True):
        for robot_id in range(3, 11):
            r = rep.robots.add()
            r.x = r.y = r.dir = 0.0
            r.id = robot_id
            r.yellowteam = yellowteam
            r.turnon = False

    rep.ball.x = 0.0
    rep.ball.y = 0.0
    rep.ball.vx = 0.0
    rep.ball.vy = 0.0
    return pkt


def perform_kickoff(sock, address=("127.0.0.1", 20011),
                    zeros_before=25, zeros_after=50):
    """Reposiciona o mundo 3v3 no kickoff com zero-command antes/depois."""
    original_target = address

    def _zero(repeats):
        for _ in range(repeats):
            for yellowteam in (False, True):
                packet = grSim_Packet_pb2.grSim_Packet()
                packet.commands.timestamp = time.time()
                packet.commands.isteamyellow = yellowteam
                for robot_id in range(3):
                    command = packet.commands.robot_commands.add()
                    command.id = robot_id
                    command.wheelsspeed = False
                    command.kickspeedx = 0.0
                    command.kickspeedz = 0.0
                    command.veltangent = 0.0
                    command.velnormal = 0.0
                    command.velangular = 0.0
                    command.spinner = False
                sock.sendto(packet.SerializeToString(), original_target)
            time.sleep(0.02)

    _zero(zeros_before)
    sock.sendto(build_off_packet().SerializeToString(), original_target)
    time.sleep(0.1)
    data = build_on_packet().SerializeToString()
    sock.sendto(data, original_target)
    _zero(zeros_after)
    return len(data)


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_bytes = perform_kickoff(s)
    print(f"Replacement enviado ({n_bytes} bytes).")


if __name__ == "__main__":
    main()
