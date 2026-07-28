#!/usr/bin/env python3
"""Mede trajetorias da policy passivamente pelos pacotes SSL-Vision."""

import argparse
import json
import math
import select
import socket
import struct
import time

import ssl_vision_wrapper_pb2


def make_vision_socket(group, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("", port))
    membership = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.setblocking(False)
    return sock


def distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


GOAL_LINE_X = 4.5
GOAL_HALF_WIDTH = 0.5
CONTACT_ON_M = 0.13
CONTACT_OFF_M = 0.20


class EventTracker:
    """Deteccao edge-triggered de gols e toques na bola."""

    def __init__(self):
        self.goals = []
        self.touches = {}
        self._ball_in_goal = False
        self._in_contact = set()

    def update(self, latest, elapsed_s):
        ball = latest.get("ball")
        if ball is None:
            return
        in_goal = abs(ball[0]) > GOAL_LINE_X and abs(ball[1]) < GOAL_HALF_WIDTH
        if in_goal and not self._ball_in_goal:
            side = "blue_scores" if ball[0] > 0 else "yellow_scores"
            self.goals.append({"t_s": round(elapsed_s, 2), "event": side})
        self._ball_in_goal = in_goal

        for key, position in latest.items():
            if key == "ball":
                continue
            gap = distance(position, ball)
            if gap < CONTACT_ON_M and key not in self._in_contact:
                self._in_contact.add(key)
                self.touches[key] = self.touches.get(key, 0) + 1
            elif gap > CONTACT_OFF_M:
                self._in_contact.discard(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--vision-group", default="224.5.23.2")
    parser.add_argument("--vision-port", type=int, default=10020)
    args = parser.parse_args()

    sock = make_vision_socket(args.vision_group, args.vision_port)
    tracks = {}
    latest = {}
    events = EventTracker()
    packets = 0
    started = time.monotonic()

    while time.monotonic() - started < args.duration:
        readable, _, _ = select.select([sock], [], [], 0.05)
        if not readable:
            continue
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            wrapper = ssl_vision_wrapper_pb2.SSL_WrapperPacket()
            wrapper.ParseFromString(data)
            detection = wrapper.detection
            packets += 1

            if detection.balls:
                ball = detection.balls[0]
                position = (ball.x / 1000.0, ball.y / 1000.0)
                tracks.setdefault((detection.camera_id, "ball"), {})[
                    detection.frame_number
                ] = position
                latest["ball"] = position

            for team, robots in (
                ("blue", detection.robots_blue),
                ("yellow", detection.robots_yellow),
            ):
                for robot in robots:
                    if robot.robot_id >= 3:
                        continue
                    key = f"{team}_{robot.robot_id}"
                    position = (robot.x / 1000.0, robot.y / 1000.0)
                    tracks.setdefault((detection.camera_id, key), {})[
                        detection.frame_number
                    ] = position
                    latest[key] = position

            events.update(latest, time.monotonic() - started)

    sock.close()
    if packets == 0:
        raise RuntimeError("nenhum estado completo recebido do SSL-Vision")

    selected = {}
    entity_names = {entity_name for _, entity_name in tracks}
    for entity_name in entity_names:
        candidates = [
            (camera_id, samples)
            for (camera_id, name), samples in tracks.items()
            if name == entity_name
        ]
        camera_id, samples = max(candidates, key=lambda item: len(item[1]))
        ordered = [position for _, position in sorted(samples.items())]
        selected[entity_name] = (camera_id, ordered)

    if "ball" not in selected:
        raise RuntimeError("bola ausente dos pacotes SSL-Vision")

    entities = {}
    ball_final = selected["ball"][1][-1]
    for key in sorted(selected):
        camera_id, positions = selected[key]
        first, last = positions[0], positions[-1]
        entity = {
            "camera_id": camera_id,
            "frames": len(positions),
            "initial": [round(value, 4) for value in first],
            "final": [round(value, 4) for value in last],
            "displacement_m": round(distance(first, last), 4),
        }
        if key != "ball":
            entity["final_ball_distance_m"] = round(
                distance(last, ball_final), 4
            )
        entities[key] = entity

    print(json.dumps({
        "duration_s": args.duration,
        "packets": packets,
        "entities": entities,
        "goals": events.goals,
        "touches": events.touches,
    }, sort_keys=True))


if __name__ == "__main__":
    main()