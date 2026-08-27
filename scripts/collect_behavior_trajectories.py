#!/usr/bin/env python3
"""Coleta trajetorias comportamentais no env historico (CPU-only).

Roda episodios como o harness de avaliacao, mas grava por step: bola, robos,
acoes e os componentes DENSOS do reward chamando as MESMAS funcoes que o env
usa (r_speed/r_dist/r_off/r_def de rewards.py historico) sobre frame/last_frame.
Saida: um .npz por confronto + metadados JSON embutidos.

Uso (dentro do container historico, CUDA_VISIBLE_DEVICES vazio):
  python scripts/collect_behavior_trajectories.py \
    --checkpoint <ckpt_blue> --yellow-checkpoint <ckpt_yellow> \
    --episodes 30 --seed-start 200 --output experiment_results/behavior/x.npz
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_checkpoints_cpu import (  # noqa: E402
    LEGACY_ENV,
    SSLMultiAgentEnv,
    compute_actions,
    load_env_config,
    load_model,
)
from rewards import DENSE_REWARDS  # noqa: E402

N_ROBOTS = 3
AGENTS = [f"blue_{i}" for i in range(N_ROBOTS)] + [f"yellow_{i}" for i in range(N_ROBOTS)]


def dense_components(env) -> np.ndarray:
    """Componentes nao-ponderados por agente, na ordem de DENSE_REWARDS."""
    values = np.zeros((len(DENSE_REWARDS), len(AGENTS)), dtype=np.float32)
    for c, (_, func, attrs) in enumerate(DENSE_REWARDS):
        kwargs = {attr: getattr(env, attr) for attr in attrs}
        result = func(env.field, env.frame, env.last_frame,
                      left="blue", right="yellow", **kwargs)
        for a, agent in enumerate(AGENTS):
            values[c, a] = result[agent]
    return values


def run_episode(models, env_config, seed, blue_stochastic, yellow_stochastic):
    stochastic_by_team = {"blue": blue_stochastic, "yellow": yellow_stochastic}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not LEGACY_ENV:
        raise RuntimeError("coletor requer o env historico (stack interno)")
    env = SSLMultiAgentEnv(**env_config)
    rows = {
        "ball": [], "robots": [], "actions": [], "rewards": [], "components": [],
    }
    try:
        observations, _ = env.reset(seed=seed)
        done = truncated = {"__all__": False}
        while not done["__all__"] and not truncated["__all__"]:
            actions = compute_actions(models, observations, stochastic_by_team)
            observations, rewards, done, truncated, info = env.step(actions)
            ball = env.frame.ball
            rows["ball"].append([ball.x, ball.y, ball.v_x, ball.v_y])
            robot_row = []
            for team_frame in (env.frame.robots_blue, env.frame.robots_yellow):
                for i in range(N_ROBOTS):
                    r = team_frame[i]
                    robot_row.extend([r.x, r.y, r.theta, r.v_x, r.v_y])
            rows["robots"].append(robot_row)
            rows["actions"].append(np.concatenate([actions[a] for a in AGENTS]))
            rows["rewards"].append([rewards[a] for a in AGENTS])
            rows["components"].append(dense_components(env))
        score = info.get("blue_0", {}).get("score", {"blue": 0, "yellow": 0})
        if score["blue"] > score["yellow"]:
            terminal = "blue_goal"
        elif score["yellow"] > score["blue"]:
            terminal = "yellow_goal"
        else:
            terminal = "timeout"
        return {
            "seed": seed,
            "terminal": terminal,
            "steps": len(rows["ball"]),
            "ball": np.asarray(rows["ball"], dtype=np.float32),
            "robots": np.asarray(rows["robots"], dtype=np.float32),
            "actions": np.asarray(rows["actions"], dtype=np.float32),
            "rewards": np.asarray(rows["rewards"], dtype=np.float32),
            "components": np.asarray(rows["components"], dtype=np.float32),
            "field": (env.field.length, env.field.width, env.field.goal_width),
            "fps": env_config["fps"],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--yellow-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--blue-mode", choices=("deterministic", "stochastic"),
                        default="deterministic")
    parser.add_argument("--yellow-mode", choices=("deterministic", "stochastic"),
                        default="stochastic")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env_config = load_env_config(str(Path(args.checkpoint).parent))
    models = {
        "blue": load_model(args.checkpoint),
        "yellow": load_model(args.yellow_checkpoint),
    }
    episodes = []
    started = time.monotonic()
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        ep = run_episode(models, env_config, seed,
                         args.blue_mode == "stochastic",
                         args.yellow_mode == "stochastic")
        episodes.append(ep)
        print(f"seed={seed} {ep['terminal']} steps={ep['steps']} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    meta = {
        "checkpoint": args.checkpoint,
        "yellow_checkpoint": args.yellow_checkpoint,
        "blue_mode": args.blue_mode,
        "yellow_mode": args.yellow_mode,
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "agents": AGENTS,
        "robot_cols": ["x", "y", "theta", "v_x", "v_y"],
        "component_names": [f.__name__ for _, f, _ in DENSE_REWARDS],
        "component_weights": [w for w, _, _ in DENSE_REWARDS],
        "field_length_width_goalwidth": list(episodes[0]["field"]),
        "fps": episodes[0]["fps"],
        "terminals": [ep["terminal"] for ep in episodes],
        "seeds": [ep["seed"] for ep in episodes],
        "steps": [ep["steps"] for ep in episodes],
    }
    arrays = {"meta_json": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    for i, ep in enumerate(episodes):
        for key in ("ball", "robots", "actions", "rewards", "components"):
            arrays[f"ep{i:03d}_{key}"] = ep[key]
    np.savez_compressed(args.output, **arrays)
    print(f"salvo: {args.output} ({len(episodes)} episodios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
