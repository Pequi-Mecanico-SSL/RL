#!/usr/bin/env python3
"""Compara checkpoints em estados fixos antes da camada de comandos grSim."""

import argparse
import glob
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.action_dists_inferece import InferenceBetaDist
from model.model_inferece import InferenceModel
from sim2real.state_to_obs import frame_to_observations


def load_model(checkpoint: str) -> InferenceModel:
    model = InferenceModel(input_size=616, output_size=8)
    policy_file = Path(checkpoint) / "policies/policy_blue/policy_state.pkl"
    with policy_file.open("rb") as stream:
        policy_state = pickle.load(stream)
    weights = {}
    for name, value in policy_state["weights"].items():
        split = name.split(".")
        if split[0] in ("_logits", "_value_branch"):
            mapped = split[0] + "." + split[-1]
        elif len(split) >= 3:
            mapped = split[0] + "." + str(int(split[1]) * 2) + "." + split[-1]
        else:
            mapped = name
        weights[mapped] = torch.as_tensor(value)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def canonical_frame():
    return {
        "robots_blue": {
            "robot_0": [-1.5, 0.0, 0.0],
            "robot_1": [-2.0, 1.0, 0.0],
            "robot_2": [-2.0, -1.0, 0.0],
        },
        "robots_yellow": {
            "robot_0": [1.5, 0.0, 180.0],
            "robot_1": [2.0, 1.0, 180.0],
            "robot_2": [2.0, -1.0, 180.0],
        },
        "ball": [0.0, 0.0],
    }


def observations(frame, warm_stack: bool):
    actions = {
        **{f"blue_{i}": np.zeros(4) for i in range(3)},
        **{f"yellow_{i}": np.zeros(4) for i in range(3)},
    }
    stacks = {name: np.zeros(616) for name in actions}
    repeats = 8 if warm_stack else 1
    for step in range(repeats):
        frame_to_observations(frame, actions, stacks, steps=step)
    return stacks


def beta_variance(distribution: InferenceBetaDist) -> np.ndarray:
    alpha = distribution.dist.concentration1
    beta = distribution.dist.concentration0
    variance = 4.0 * alpha * beta / (
        (alpha + beta).square() * (alpha + beta + 1.0)
    )
    return variance.detach().cpu().numpy()


def boundary_time(position, velocity):
    candidates = []
    for coordinate, speed, limit in zip(position, velocity, (4.5, 3.0)):
        if speed > 1e-9:
            candidates.append((limit - coordinate) / speed)
        elif speed < -1e-9:
            candidates.append((-limit - coordinate) / speed)
    positive = [value for value in candidates if value >= 0.0]
    return min(positive) if positive else math.inf


def analyze(checkpoint: str, warm_stack: bool):
    frame = canonical_frame()
    stacked = observations(frame, warm_stack)
    model = load_model(checkpoint)
    rows = []
    for team in ("blue", "yellow"):
        names = [f"{team}_{i}" for i in range(3)]
        tensor = torch.as_tensor(np.stack([stacked[name] for name in names])).float()
        with torch.no_grad():
            logits, values = model(tensor)
        signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
        distribution = InferenceBetaDist(logits, signal=signal)
        actions = distribution.deterministic_sample().cpu().numpy()
        variances = beta_variance(distribution)
        for robot_id, action in enumerate(actions):
            position = np.asarray(
                frame[f"robots_{team}"][f"robot_{robot_id}"][:2], dtype=float
            )
            ball = np.asarray(frame["ball"], dtype=float)
            direction = ball - position
            direction /= np.linalg.norm(direction)
            velocity = 1.5 * action[:2]
            speed = np.linalg.norm(velocity)
            if speed > 1.5:
                velocity *= 1.5 / speed
            rows.append({
                "name": names[robot_id],
                "action": action,
                "variance": variances[robot_id],
                "toward_ball": float(np.dot(velocity, direction)),
                "boundary_time": boundary_time(position, velocity),
                "value": float(values[robot_id].reshape(-1)[0]),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_glob")
    args = parser.parse_args()
    checkpoints = sorted(glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise SystemExit("nenhum checkpoint encontrado")
    for checkpoint in checkpoints:
        print(f"\n=== {os.path.basename(checkpoint)} ===")
        for warm_stack in (False, True):
            mode = "warm8" if warm_stack else "reset1"
            rows = analyze(checkpoint, warm_stack)
            print(f"-- {mode}")
            for row in rows:
                action = np.array2string(row["action"], precision=3, floatmode="fixed")
                mean_var = float(np.mean(row["variance"]))
                print(
                    f"{row['name']:8s} action={action} var={mean_var:.3f} "
                    f"toward_ball={row['toward_ball']:+.3f}m/s "
                    f"boundary={row['boundary_time']:.2f}s value={row['value']:+.2f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())