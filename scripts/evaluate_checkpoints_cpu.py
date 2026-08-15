#!/usr/bin/env python3
"""Avalia checkpoints no ambiente historico sem Ray, render ou GPU.

O processo deve ser iniciado com CUDA_VISIBLE_DEVICES vazio. Cada episodio cria
um ambiente novo para impedir vazamento de stack/last_actions entre episodios.
Os resultados sao anexados em JSONL imediatamente apos cada episodio.
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
    raise RuntimeError("CUDA_VISIBLE_DEVICES deve estar vazio para esta avaliacao")

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import inspect
import types

try:
    # Caminho normal: o pacote rSoccer resolve sem ciclo (ex.: commit de
    # treino c684c2b, cujo __init__.py e vazio).
    import rSoccer.rsoccer_gym.Entities  # noqa: F401
except ImportError:
    # v1.2.0: ssl_judge importa `rSoccer.rsoccer_gym.Entities`, mas
    # `rSoccer/__init__.py` reimporta o env parcialmente inicializado.
    # Stub com o pacote ja resolvido quebra o ciclo sem alterar o submodulo.
    import rsoccer_gym.Entities as _entities

    _stub = types.ModuleType("rSoccer")
    _stub_gym = types.ModuleType("rSoccer.rsoccer_gym")
    _stub.rsoccer_gym = _stub_gym
    _stub_gym.Entities = _entities
    sys.modules["rSoccer"] = _stub
    sys.modules["rSoccer.rsoccer_gym"] = _stub_gym
    sys.modules["rSoccer.rsoccer_gym.Entities"] = _entities

from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

# O contrato do env difere por versao do rSoccer: o commit de treino do
# baseline (c684c2b) calcula as 77 obs e o stack de 8 internamente
# (stack_observation=8); o v1.2.0 exige judge + StackWrapper externo.
ENV_PARAMS = set(inspect.signature(SSLMultiAgentEnv.__init__).parameters)
LEGACY_ENV = "stack_observation" in ENV_PARAMS
if not LEGACY_ENV:
    from rsoccer_gym.judges.ssl_judge import Judge
    from rsoccer_gym.Utils.Utils import StackWrapper
    from observations import OBSERVATIONS
from rewards import DENSE_REWARDS, SPARSE_REWARDS

sys.path.insert(0, "/app/scripts")
from model.action_dists_inferece import InferenceBetaDist
from model.model_inferece import InferenceModel


def load_model(checkpoint: str) -> InferenceModel:
    model = InferenceModel(input_size=616, output_size=8)
    policy_file = Path(checkpoint) / "policies/policy_blue/policy_state.pkl"
    with policy_file.open("rb") as stream:
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
    model.to("cpu")
    model.eval()
    return model


def load_env_config(experiment: str) -> dict:
    with open(os.path.join(experiment, "params.json"), encoding="utf-8") as stream:
        params = json.load(stream)
    config = params["env_config"]
    base = {
        "init_pos": {
            "blue": {int(key): value for key, value in config["init_pos"]["blue"].items()},
            "yellow": {int(key): value for key, value in config["init_pos"]["yellow"].items()},
            "ball": config["init_pos"]["ball"],
        },
        "field_type": config["field_type"],
        "fps": config["fps"],
        "match_time": config["match_time"],
        "render_mode": None,
        "dense_rewards": DENSE_REWARDS,
        "sparse_rewards": SPARSE_REWARDS,
    }
    if LEGACY_ENV:
        base["stack_observation"] = 8
    else:
        base["judge"] = Judge
    return base


def compute_actions(models: dict, observations: dict, stochastic_by_team: dict) -> dict:
    result = {}
    with torch.no_grad():
        for team in ("blue", "yellow"):
            names = [f"{team}_{robot_id}" for robot_id in range(3)]
            batch = torch.as_tensor(
                np.stack([observations[name] for name in names]), dtype=torch.float32
            )
            logits, _ = models[team](batch)
            signal = [-1, 1, -1, 1] if team == "yellow" else [1, 1, 1, 1]
            distribution = InferenceBetaDist(logits, signal=signal)
            stochastic = stochastic_by_team[team]
            values = distribution.sample() if stochastic else distribution.deterministic_sample()
            values = values.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("acao NaN/Inf")
            for index, name in enumerate(names):
                result[name] = np.clip(values[index], -1.0, 1.0)
    return result


def run_episode(models: dict, env_config: dict, seed: int, stochastic: bool,
                yellow_stochastic: bool = None) -> dict:
    if yellow_stochastic is None:
        yellow_stochastic = stochastic
    stochastic_by_team = {"blue": stochastic, "yellow": yellow_stochastic}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if LEGACY_ENV:
        env = SSLMultiAgentEnv(**env_config)
    else:
        env = StackWrapper(
            SSLMultiAgentEnv(**env_config), stack_size=8, observation_funcs=OBSERVATIONS
        )
    started = time.monotonic()
    try:
        observations, _ = env.reset(seed=seed)
        returns = {name: 0.0 for name in observations}
        done = truncated = {"__all__": False}
        steps = 0
        info = {}
        while not done["__all__"] and not truncated["__all__"]:
            actions = compute_actions(models, observations, stochastic_by_team)
            observations, rewards, done, truncated, info = env.step(actions)
            for name, reward in rewards.items():
                returns[name] += float(reward)
            steps += 1
        score = info.get("blue_0", {}).get("score", {"blue": 0, "yellow": 0})
        if score["blue"] > score["yellow"]:
            terminal = "blue_goal"
        elif score["yellow"] > score["blue"]:
            terminal = "yellow_goal"
        else:
            terminal = "timeout"
        blue_return = float(np.mean([returns[f"blue_{i}"] for i in range(3)]))
        yellow_return = float(np.mean([returns[f"yellow_{i}"] for i in range(3)]))
        return {
            "seed": seed,
            "mode": "stochastic" if stochastic else "deterministic",
            "yellow_mode": "stochastic" if yellow_stochastic else "deterministic",
            "steps": steps,
            "sim_seconds": steps / env_config["fps"],
            "wall_seconds": time.monotonic() - started,
            "terminal": terminal,
            "score_blue": int(score["blue"]),
            "score_yellow": int(score["yellow"]),
            "blue_return_mean": blue_return,
            "yellow_return_mean": yellow_return,
            "blue_return_per_step": blue_return / steps,
            "yellow_return_per_step": yellow_return / steps,
        }
    finally:
        env.close()


def completed_keys(output: str) -> set:
    keys = set()
    if not os.path.exists(output):
        return keys
    with open(output, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            keys.add((row["checkpoint"], row["mode"], row["seed"]))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--yellow-checkpoint", default=None,
                        help="checkpoint fixo para a yellow (cross-play); default = espelho do blue")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="primeiro seed absoluto; roda seeds [seed_start, seed_start+episodes)")
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="deterministic")
    parser.add_argument("--yellow-mode", choices=("deterministic", "stochastic", "same"), default="same",
                        help="modo fixo da yellow; 'same' replica o modo do blue (comportamento antigo)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(args.checkpoint_glob))
    if not checkpoints:
        raise SystemExit("nenhum checkpoint encontrado")
    modes = (False, True) if args.mode == "both" else (args.mode == "stochastic",)
    complete = completed_keys(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    yellow_model = load_model(args.yellow_checkpoint) if args.yellow_checkpoint else None
    yellow_name = (
        str(Path(args.yellow_checkpoint).parent.name + "/" + Path(args.yellow_checkpoint).name)
        if args.yellow_checkpoint else "mirror"
    )

    with open(args.output, "a", encoding="utf-8", buffering=1) as output:
        for checkpoint in checkpoints:
            experiment = str(Path(checkpoint).parent)
            env_config = load_env_config(experiment)
            model = load_model(checkpoint)
            models = {"blue": model, "yellow": yellow_model or model}
            checkpoint_name = str(Path(experiment).name + "/" + Path(checkpoint).name)
            for stochastic in modes:
                mode = "stochastic" if stochastic else "deterministic"
                yellow_stochastic = None if args.yellow_mode == "same" else (args.yellow_mode == "stochastic")
                for seed in range(args.seed_start, args.seed_start + args.episodes):
                    if (checkpoint_name, mode, seed) in complete:
                        continue
                    row = run_episode(models, env_config, seed, stochastic, yellow_stochastic)
                    row["checkpoint"] = checkpoint_name
                    row["yellow_checkpoint"] = yellow_name
                    row["field_type"] = env_config["field_type"]
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    print(
                        f"{checkpoint_name} {mode} seed={seed} "
                        f"{row['terminal']} steps={row['steps']} "
                        f"wall={row['wall_seconds']:.2f}s",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())