#!/usr/bin/env python3
"""Probe CPU do opponent-pool no RLlib 2.10, sem SGD.

Restaura um checkpoint integral, adiciona quatro policies yellow congeladas com
pesos vindos dos policy_blue historicos e amostra episodios para provar mapping,
estabilidade por episodio e preservacao do estado treinavel.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA_VISIBLE_DEVICES deve estar vazio")

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from rewards import DENSE_REWARDS, SPARSE_REWARDS
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from scripts.model.custom_torch_model import CustomFCNet

POOL_IDS = [f"opponent_{index}" for index in range(4)]
BEHAVIOR_COLUMNS = (
    "obs", "new_obs", "actions", "rewards", "terminateds", "truncateds",
    "action_dist_inputs", "action_logp", "vf_preds",
)


def array_hash(value) -> str:
    digest = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            digest.update(array_hash(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(array_hash(item).encode())
    else:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def policy_state(path: str, source_policy: str) -> dict:
    with open(Path(path) / f"policies/{source_policy}/policy_state.pkl", "rb") as stream:
        return pickle.load(stream)


def behavior_hashes(batch) -> dict:
    result = {}
    for policy_id, policy_batch in sorted(batch.policy_batches.items()):
        result[policy_id] = {}
        for column in BEHAVIOR_COLUMNS:
            if column not in policy_batch:
                continue
            value = np.asarray(policy_batch[column])
            if value.dtype == object:
                raise RuntimeError(f"{policy_id}/{column}: dtype object")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise RuntimeError(f"{policy_id}/{column}: NaN/Inf")
            result[policy_id][column] = {
                "hash": array_hash(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    return result


def optimizer_manifest(optimizer_variables) -> dict:
    """Canonicaliza Adam ignorando IDs de parametros especificos do processo."""
    optimizers = []
    tensor_count = element_count = 0
    for optimizer in optimizer_variables:
        param_ids = [param_id for group in optimizer["param_groups"]
                     for param_id in group["params"]]
        canonical_id = {param_id: index for index, param_id in enumerate(param_ids)}
        state = {}
        for param_id, values in optimizer["state"].items():
            entries = {}
            for name, value in sorted(values.items()):
                array = np.asarray(value)
                if array.dtype == object or not np.isfinite(array).all():
                    raise RuntimeError(f"optimizer/{name}: tipo invalido ou NaN/Inf")
                entries[name] = {
                    "shape": list(array.shape), "dtype": str(array.dtype),
                    "hash": array_hash(np.ascontiguousarray(array)),
                }
                tensor_count += 1
                element_count += array.size
            state[str(canonical_id[param_id])] = entries
        groups = []
        for group in optimizer["param_groups"]:
            clean = {key: value for key, value in group.items() if key != "params"}
            clean["params"] = [canonical_id[param_id] for param_id in group["params"]]
            groups.append(clean)
        optimizers.append({"state": state, "param_groups": groups})
    return {"optimizers": optimizers, "tensor_count": tensor_count,
            "element_count": element_count}


class CyclicMapping:
    def __init__(self):
        self.next_slot = 0
        self.episode_slots = {}
        self.agents = defaultdict(set)

    def __call__(self, agent_id, episode, worker=None, **kwargs):
        if agent_id.startswith("blue_"):
            return "policy_blue"
        episode_id = str(episode.episode_id)
        if episode_id not in self.episode_slots:
            self.episode_slots[episode_id] = self.next_slot
            self.next_slot = (self.next_slot + 1) % len(POOL_IDS)
        self.agents[episode_id].add(agent_id)
        return POOL_IDS[self.episode_slots[episode_id]]


def build_algorithm(checkpoint: str, seed: int) -> PPO:
    experiment = Path(checkpoint).parent
    with open(experiment / "params.json", encoding="utf-8") as stream:
        config = json.load(stream)

    register_env("Soccer", lambda env_config: SSLMultiAgentEnv(**env_config))
    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

    env_config = config["env_config"]
    env_config["init_pos"]["blue"] = {
        int(key): value for key, value in env_config["init_pos"]["blue"].items()
    }
    env_config["init_pos"]["yellow"] = {
        int(key): value for key, value in env_config["init_pos"]["yellow"].items()
    }
    env_config["dense_rewards"] = DENSE_REWARDS
    env_config["sparse_rewards"] = SPARSE_REWARDS
    temp_env = SSLMultiAgentEnv(**env_config)
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    config.update({
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 1200,
        "batch_mode": "complete_episodes",
        "seed": seed,
        "callbacks": DefaultCallbacks,
        "env": "Soccer",
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "policy_blue": (None, obs_space, act_space,
                                {"model": {"custom_action_dist": "beta_dist_blue"}}),
                "policy_yellow": (None, obs_space, act_space,
                                  {"model": {"custom_action_dist": "beta_dist_yellow"}}),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs:
                "policy_blue" if agent_id.startswith("blue_") else "policy_yellow",
            "policies_to_train": ["policy_blue"],
        },
    })
    algorithm = PPO(config=config)
    algorithm.restore(checkpoint)
    return algorithm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pool-checkpoint", action="append", required=True)
    parser.add_argument("--source-policy", choices=("policy_blue", "policy_yellow"),
                        default="policy_blue")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.pool_checkpoint) != 4:
        raise SystemExit("informe exatamente quatro --pool-checkpoint")

    ray.init(include_dashboard=False, num_cpus=2, local_mode=True)
    algorithm = None
    try:
        algorithm = build_algorithm(args.checkpoint, args.seed)
        worker = algorithm.workers.local_worker()
        blue_restored = worker.get_policy("policy_blue").get_state()
        serialized_blue = policy_state(args.checkpoint, "policy_blue")
        counters_before = dict(algorithm._counters)
        yellow = worker.get_policy("policy_yellow")

        source_hashes = []
        for policy_id, checkpoint in zip(POOL_IDS, args.pool_checkpoint):
            source = policy_state(checkpoint, args.source_policy)
            state = yellow.get_state()
            state["weights"] = source["weights"]
            algorithm.add_policy(
                policy_id,
                policy_cls=type(yellow),
                observation_space=yellow.observation_space,
                action_space=yellow.action_space,
                config=yellow.config,
                policy_state=state,
                policies_to_train=["policy_blue"],
                evaluation_workers=False,
            )
            loaded = worker.get_policy(policy_id).get_state()["weights"]
            if array_hash(loaded) != array_hash(source["weights"]):
                raise RuntimeError(f"{policy_id}: pesos carregados divergem da fonte")
            source_hashes.append(array_hash(source["weights"]))

        blue_before = worker.get_policy("policy_blue").get_state()

        mapping = CyclicMapping()
        worker.set_policy_mapping_fn(mapping)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        behavior_batches = []
        while len(mapping.episode_slots) < args.episodes:
            batch = worker.sample()
            behavior_batches.append(behavior_hashes(batch))

        blue_after = worker.get_policy("policy_blue").get_state()
        counters_after = dict(algorithm._counters)
        selected = list(mapping.episode_slots.items())[:args.episodes]
        assignments = [
            {
                "episode_id": episode_id,
                "slot": slot,
                "policy_id": POOL_IDS[slot],
                "yellow_agents": sorted(mapping.agents[episode_id]),
            }
            for episode_id, slot in selected
        ]
        expected = [index % 4 for index in range(args.episodes)]
        observed = [row["slot"] for row in assignments]
        all_agents = all(row["yellow_agents"] == ["yellow_0", "yellow_1", "yellow_2"]
                         for row in assignments)
        result = {
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "episodes_requested": args.episodes,
            "assignments": assignments,
            "slot_counts": dict(Counter(observed)),
            "expected_slots": expected,
            "mapping_sequence_ok": observed == expected,
            "all_yellow_agents_same_slot": all_agents,
            "blue_weights_hash_before": array_hash(blue_before["weights"]),
            "blue_weights_hash_after": array_hash(blue_after["weights"]),
            "blue_weights_unchanged": (
                array_hash(blue_before["weights"]) == array_hash(blue_after["weights"])
            ),
            "blue_optimizer_hash_before": array_hash(blue_before["_optimizer_variables"]),
            "blue_optimizer_hash_after": array_hash(blue_after["_optimizer_variables"]),
            "blue_optimizer_unchanged": (
                array_hash(blue_before["_optimizer_variables"])
                == array_hash(blue_after["_optimizer_variables"])
            ),
            "blue_state_keys": sorted(blue_before),
            "optimizer_serialized": optimizer_manifest(
                serialized_blue["_optimizer_variables"]
            ),
            "optimizer_restored": optimizer_manifest(
                blue_restored["_optimizer_variables"]
            ),
            "optimizer_after_add_policy": optimizer_manifest(
                blue_before["_optimizer_variables"]
            ),
            "counters_before": counters_before,
            "counters_after": counters_after,
            "source_weight_hashes": source_hashes,
            "source_policy": args.source_policy,
            "behavior_batches": behavior_batches,
            "policy_ids": sorted(worker.policy_map.keys()),
            "policies_to_train": sorted(worker.get_policies_to_train()),
        }
        if not result["mapping_sequence_ok"] or not all_agents:
            raise RuntimeError("mapping por episodio falhou")
        if not result["blue_weights_unchanged"] or not result["blue_optimizer_unchanged"]:
            raise RuntimeError("pesos/optimizer da policy_blue mudaram durante probe sem SGD")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
