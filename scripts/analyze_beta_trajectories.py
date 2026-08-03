#!/usr/bin/env python3
"""Mede a distribuicao Beta em trajetorias do ambiente historico."""

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("execute com CUDA_VISIBLE_DEVICES vazio")

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/scripts")

from model.action_dists_inferece import InferenceBetaDist
from model.model_inferece import InferenceModel
from rewards import DENSE_REWARDS, SPARSE_REWARDS
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv


ACTION_NAMES = ("x", "y", "omega", "kick")


def load_model(checkpoint: Path) -> InferenceModel:
    model = InferenceModel(input_size=616, output_size=8)
    with (checkpoint / "policies/policy_blue/policy_state.pkl").open("rb") as stream:
        policy_state = pickle.load(stream)
    mapped_weights = {}
    for name, value in policy_state["weights"].items():
        split = name.split(".")
        if split[0] in ("_logits", "_value_branch"):
            mapped = split[0] + "." + split[-1]
        elif len(split) >= 3:
            mapped = split[0] + "." + str(int(split[1]) * 2) + "." + split[-1]
        else:
            mapped = name
        mapped_weights[mapped] = torch.as_tensor(value)
    model.load_state_dict(mapped_weights, strict=True)
    model.eval()
    return model


def load_env_config(experiment: Path) -> dict:
    with (experiment / "params.json").open(encoding="utf-8") as stream:
        config = json.load(stream)["env_config"]
    return {
        "init_pos": {
            "blue": {int(key): value for key, value in config["init_pos"]["blue"].items()},
            "yellow": {int(key): value for key, value in config["init_pos"]["yellow"].items()},
            "ball": config["init_pos"]["ball"],
        },
        "field_type": config["field_type"],
        "fps": config["fps"],
        "match_time": config["match_time"],
        "stack_observation": 8,
        "render_mode": None,
        "dense_rewards": DENSE_REWARDS,
        "sparse_rewards": SPARSE_REWARDS,
    }


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).tolist(),
        "p05": np.quantile(values, 0.05, axis=0).tolist(),
        "p50": np.quantile(values, 0.50, axis=0).tolist(),
        "p95": np.quantile(values, 0.95, axis=0).tolist(),
    }


def run(checkpoint: Path, episodes: int, seed_offset: int) -> dict:
    model = load_model(checkpoint)
    env_config = load_env_config(checkpoint.parent)
    alpha_rows = []
    beta_rows = []
    mean_rows = []
    std_rows = []
    sample_rows = []
    terminals = Counter()

    for episode in range(episodes):
        seed = seed_offset + episode
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = SSLMultiAgentEnv(**env_config)
        try:
            observations, _ = env.reset(seed=seed)
            done = truncated = {"__all__": False}
            info = {}
            while not done["__all__"] and not truncated["__all__"]:
                names = [f"{team}_{robot}" for team in ("blue", "yellow") for robot in range(3)]
                batch = torch.as_tensor(
                    np.stack([observations[name] for name in names]), dtype=torch.float32
                )
                with torch.no_grad():
                    logits, _ = model(batch)
                    distribution = InferenceBetaDist(logits)
                    alpha = distribution.dist.concentration1
                    beta = distribution.dist.concentration0
                    means = 2.0 * distribution.dist.mean - 1.0
                    stddev = 2.0 * distribution.dist.stddev
                    samples = 2.0 * distribution.dist.rsample() - 1.0
                alpha_rows.append(alpha.numpy())
                beta_rows.append(beta.numpy())
                mean_rows.append(means.numpy())
                std_rows.append(stddev.numpy())
                sample_rows.append(samples.numpy())
                actions = {}
                sampled = samples.numpy()
                for index, name in enumerate(names):
                    action = sampled[index].copy()
                    if name.startswith("yellow_"):
                        action *= np.asarray([-1.0, 1.0, -1.0, 1.0])
                    actions[name] = action
                observations, _, done, truncated, info = env.step(actions)
            score = info.get("blue_0", {}).get("score", {"blue": 0, "yellow": 0})
            if score["blue"] > score["yellow"]:
                terminals["blue_goal"] += 1
            elif score["yellow"] > score["blue"]:
                terminals["yellow_goal"] += 1
            else:
                terminals["timeout"] += 1
        finally:
            env.close()

    alpha = np.concatenate(alpha_rows)
    beta = np.concatenate(beta_rows)
    means = np.concatenate(mean_rows)
    stddev = np.concatenate(std_rows)
    samples = np.concatenate(sample_rows)
    return {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "agent_steps": int(alpha.shape[0]),
        "action_names": ACTION_NAMES,
        "terminals": dict(terminals),
        "alpha": summarize(alpha),
        "beta": summarize(beta),
        "deterministic_action": summarize(means),
        "distribution_stddev": summarize(stddev),
        "sample_action": summarize(samples),
        "sample_abs_gt_0_8": (np.abs(samples) > 0.8).mean(axis=0).tolist(),
        "kick_positive_rate": float((samples[:, 3] > 0.0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, args.episodes, args.seed_offset)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())