#!/usr/bin/env python3
"""Valida experimentalmente o contrato de comandos do grSim.

Este programa nao carrega checkpoint nem usa a policy. Ele reposiciona um robo,
envia um comando cartesiano local conhecido e estima a velocidade resultante a
partir do SSL-Vision. As policies devem estar paradas durante a execucao.
"""

import argparse
import math
import select
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, ".")
import grSim_Packet_pb2
import ssl_vision_wrapper_pb2


VISION_GROUP = "224.5.23.2"
VISION_PORT = 10020
COMMAND_ADDRESS = ("127.0.0.1", 20011)
N_ROBOTS = 3


@dataclass(frozen=True)
class Case:
    name: str
    team: str
    robot_id: int
    heading_deg: float
    tangent: float = 0.0
    normal: float = 0.0
    angular: float = 0.0


def angle_delta(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def make_vision_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("", VISION_PORT))
    membership = struct.pack(
        "4sl", socket.inet_aton(VISION_GROUP), socket.INADDR_ANY
    )
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.setblocking(False)
    return sock


def add_robot_command(commands, robot_id, tangent=0.0, normal=0.0, angular=0.0):
    command = commands.robot_commands.add()
    command.id = robot_id
    command.kickspeedx = 0.0
    command.kickspeedz = 0.0
    command.veltangent = tangent
    command.velnormal = normal
    command.velangular = angular
    command.spinner = False
    command.wheelsspeed = False


def send_team_command(
    sock: socket.socket,
    team: str,
    target_id: int = -1,
    tangent: float = 0.0,
    normal: float = 0.0,
    angular: float = 0.0,
) -> None:
    packet = grSim_Packet_pb2.grSim_Packet()
    packet.commands.timestamp = time.time()
    packet.commands.isteamyellow = team == "yellow"
    for robot_id in range(N_ROBOTS):
        if robot_id == target_id:
            add_robot_command(
                packet.commands, robot_id, tangent, normal, angular
            )
        else:
            add_robot_command(packet.commands, robot_id)
    sock.sendto(packet.SerializeToString(), COMMAND_ADDRESS)


def send_all_zero(sock: socket.socket, repeats: int = 10) -> None:
    for _ in range(repeats):
        send_team_command(sock, "blue")
        send_team_command(sock, "yellow")
        time.sleep(0.02)


def replacement_packet(case: Case, turn_on_active: bool):
    packet = grSim_Packet_pb2.grSim_Packet()
    replacement = packet.replacement
    parked = {
        "blue": [(-3.4, -2.1), (-3.4, 0.0), (-3.4, 2.1)],
        "yellow": [(3.4, -2.1), (3.4, 0.0), (3.4, 2.1)],
    }
    for team in ("blue", "yellow"):
        for robot_id in range(11):
            robot = replacement.robots.add()
            robot.id = robot_id
            robot.yellowteam = team == "yellow"
            robot.x = robot.y = robot.dir = 0.0
            robot.turnon = turn_on_active and robot_id < N_ROBOTS
            if robot_id < N_ROBOTS:
                robot.x, robot.y = parked[team][robot_id]
                robot.dir = 0.0 if team == "blue" else 180.0
            if team == case.team and robot_id == case.robot_id:
                robot.x = 0.0
                robot.y = 0.0
                robot.dir = case.heading_deg
    replacement.ball.x = 0.0
    replacement.ball.y = 2.6
    replacement.ball.vx = replacement.ball.vy = 0.0
    return packet


def replace_world(sock: socket.socket, case: Case) -> None:
    # A imagem atual preserva o heading de robos que ja estao ligados. A
    # transicao off->on torna posicao e orientacao iniciais reproduziveis.
    off_packet = replacement_packet(case, turn_on_active=False)
    sock.sendto(off_packet.SerializeToString(), COMMAND_ADDRESS)
    time.sleep(0.10)
    on_packet = replacement_packet(case, turn_on_active=True)
    sock.sendto(on_packet.SerializeToString(), COMMAND_ADDRESS)


def drain(sock: socket.socket) -> None:
    while True:
        try:
            sock.recvfrom(65535)
        except BlockingIOError:
            return


def run_pulse(
    vision_sock: socket.socket,
    command_sock: socket.socket,
    case: Case,
    duration: float,
) -> Dict:
    send_all_zero(command_sock)
    replace_world(command_sock, case)
    send_all_zero(command_sock)
    time.sleep(0.35)
    drain(vision_sock)

    samples: Dict[int, List[Tuple[float, float, float, float, int]]] = {}
    started = time.monotonic()
    next_command = started
    while time.monotonic() - started < duration:
        now = time.monotonic()
        if now >= next_command:
            send_team_command(
                command_sock,
                case.team,
                case.robot_id,
                case.tangent,
                case.normal,
                case.angular,
            )
            next_command += 1.0 / 60.0

        readable, _, _ = select.select([vision_sock], [], [], 0.005)
        if not readable:
            continue
        while True:
            try:
                data, _ = vision_sock.recvfrom(65535)
            except BlockingIOError:
                break
            wrapper = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
            wrapper.ParseFromString(data)
            detection = wrapper.detection
            robots = (
                detection.robots_yellow
                if case.team == "yellow"
                else detection.robots_blue
            )
            for robot in robots:
                if robot.robot_id != case.robot_id:
                    continue
                samples.setdefault(detection.camera_id, []).append(
                    (
                        time.monotonic(),
                        robot.x / 1000.0,
                        robot.y / 1000.0,
                        robot.orientation,
                        detection.frame_number,
                    )
                )

    send_all_zero(command_sock, repeats=25)
    if not samples:
        raise RuntimeError(f"{case.name}: nenhuma amostra SSL-Vision")

    camera_id, camera_samples = max(samples.items(), key=lambda item: len(item[1]))
    unique = {}
    for sample in camera_samples:
        unique[sample[4]] = sample
    ordered = sorted(unique.values(), key=lambda sample: sample[0])
    # Descarta a aceleracao inicial e a cauda anterior ao comando zero.
    usable = [sample for sample in ordered if sample[0] - started >= 0.25]
    if len(usable) < 20:
        raise RuntimeError(
            f"{case.name}: somente {len(usable)} frames unicos utilizaveis"
        )

    values = np.asarray([sample[:4] for sample in usable], dtype=np.float64)
    t = values[:, 0] - values[0, 0]
    x, y = values[:, 1], values[:, 2]
    theta = np.unwrap(values[:, 3])
    vx = float(np.polyfit(t, x, 1)[0])
    vy = float(np.polyfit(t, y, 1)[0])
    omega = float(np.polyfit(t, theta, 1)[0])
    return {
        "camera": camera_id,
        "frames": len(usable),
        "vx": vx,
        "vy": vy,
        "omega": omega,
        "speed": math.hypot(vx, vy),
        "dx": float(x[-1] - x[0]),
        "dy": float(y[-1] - y[0]),
        "dtheta": angle_delta(float(theta[-1]), float(theta[0])),
        "initial_heading": float(theta[0]),
    }


def evaluate(case: Case, result: Dict) -> Tuple[bool, str]:
    theta = result.get("initial_heading", math.radians(case.heading_deg))
    heading_error = abs(angle_delta(theta, math.radians(case.heading_deg)))
    if case.angular == 0.0 and heading_error > math.radians(7.0):
        return False, (
            f"replacement_heading={math.degrees(theta):+.1f}deg "
            f"expected={case.heading_deg:+.1f}deg"
        )
    expected_vx = case.tangent * math.cos(theta) - case.normal * math.sin(theta)
    expected_vy = case.tangent * math.sin(theta) + case.normal * math.cos(theta)
    expected_speed = math.hypot(expected_vx, expected_vy)

    if expected_speed == 0.0 and case.angular == 0.0:
        passed = result["speed"] <= 0.05 and abs(result["omega"]) <= 0.10
        detail = f"drift={result['speed']:.3f}m/s omega={result['omega']:.3f}rad/s"
        return passed, detail

    if expected_speed > 0.0:
        parallel = (
            result["vx"] * expected_vx + result["vy"] * expected_vy
        ) / expected_speed
        cross = abs(
            result["vx"] * expected_vy - result["vy"] * expected_vx
        ) / expected_speed
        direction_error = math.degrees(math.atan2(cross, parallel))
        speed_error = abs(parallel - expected_speed) / expected_speed
        passed = (
            parallel > 0.0
            and direction_error <= 10.0
            and cross <= max(0.08, 0.20 * expected_speed)
            and speed_error <= 0.30
            and abs(result["omega"]) <= 0.20
        )
        detail = (
            f"v=({result['vx']:+.3f},{result['vy']:+.3f})m/s "
            f"parallel={parallel:+.3f} cross={cross:.3f} "
            f"dir_error={direction_error:.1f}deg"
        )
        return passed, detail

    angular_error = abs(result["omega"] - case.angular) / abs(case.angular)
    passed = (
        result["omega"] * case.angular > 0.0
        and angular_error <= 0.30
        and result["speed"] <= 0.08
    )
    detail = (
        f"omega={result['omega']:+.3f}rad/s expected={case.angular:+.3f} "
        f"translation={result['speed']:.3f}m/s"
    )
    return passed, detail


def cases() -> List[Case]:
    return [
        Case("zero_blue_0", "blue", 0, 0.0),
        Case("tangent_blue_0_h0", "blue", 0, 0.0, tangent=0.5),
        Case("tangent_blue_1_h0", "blue", 1, 0.0, tangent=0.5),
        Case("normal_blue_2_h0", "blue", 2, 0.0, normal=0.5),
        Case("angular_blue_0_pos", "blue", 0, 0.0, angular=0.8),
        Case("angular_blue_0_neg", "blue", 0, 0.0, angular=-0.8),
        Case("tangent_yellow_0_h180", "yellow", 0, 180.0, tangent=0.5),
        Case("tangent_yellow_1_h180", "yellow", 1, 180.0, tangent=0.5),
        Case("normal_yellow_2_h180", "yellow", 2, 180.0, normal=0.5),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.25)
    args = parser.parse_args()
    vision_sock = make_vision_socket()
    command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    failures = []
    try:
        for case in cases():
            result = run_pulse(vision_sock, command_sock, case, args.duration)
            passed, detail = evaluate(case, result)
            status = "PASS" if passed else "FAIL"
            print(
                f"{status:4s} {case.name:27s} {detail} "
                f"heading0={math.degrees(result['initial_heading']):+.1f}deg "
                f"frames={result['frames']} camera={result['camera']}",
                flush=True,
            )
            if not passed:
                failures.append(case.name)
    finally:
        send_all_zero(command_sock, repeats=25)
        vision_sock.close()
        command_sock.close()
    if failures:
        print("DETERMINISTIC_GRSIM_FAIL " + ",".join(failures))
        return 1
    print(f"DETERMINISTIC_GRSIM_OK cases={len(cases())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())