#!/usr/bin/env python3
"""Valida estrutura e finitude dos pesos de um checkpoint RLlib."""

import argparse
import json
from pathlib import Path

import numpy as np
import ray.cloudpickle as cloudpickle


EXPECTED_POLICIES = ("policy_blue", "policy_yellow")


def collect_arrays(value) -> list[np.ndarray]:
    if isinstance(value, dict):
        return [array for item in value.values() for array in collect_arrays(item)]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in collect_arrays(item)]
    if isinstance(value, np.ndarray):
        return [value]
    return []


def validate(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "rllib_checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy_ids = set(metadata.get("policy_ids", []))
    if policy_ids != set(EXPECTED_POLICIES):
        raise ValueError(f"policy_ids invalidas: {sorted(policy_ids)}")

    algorithm_state_path = checkpoint / "algorithm_state.pkl"
    if not algorithm_state_path.is_file():
        raise FileNotFoundError(algorithm_state_path)
    with algorithm_state_path.open("rb") as stream:
        algorithm_state = cloudpickle.load(stream)
    iteration = algorithm_state.get("training_iteration")
    counters = algorithm_state.get("counters", {})
    env_sampled = counters.get("num_env_steps_sampled")
    env_trained = counters.get("num_env_steps_trained")
    agent_sampled = counters.get("num_agent_steps_sampled")
    agent_trained = counters.get("num_agent_steps_trained")
    if not isinstance(iteration, int) or iteration < 1:
        raise ValueError(f"training_iteration invalida: {iteration!r}")
    if not isinstance(env_sampled, int) or env_sampled < 1:
        raise ValueError(f"num_env_steps_sampled invalido: {env_sampled!r}")
    if env_sampled != env_trained:
        raise ValueError(f"env steps sampled={env_sampled!r} e trained={env_trained!r}")
    if agent_sampled != env_sampled * 6 or agent_trained != agent_sampled:
        raise ValueError(
            f"agent steps invalidos: sampled={agent_sampled!r}, trained={agent_trained!r}"
        )

    result = {
        "checkpoint": str(checkpoint),
        "training_iteration": iteration,
        "counters": dict(counters),
        "policies": {},
    }
    for policy_id in EXPECTED_POLICIES:
        state_path = checkpoint / "policies" / policy_id / "policy_state.pkl"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        with state_path.open("rb") as stream:
            state = cloudpickle.load(stream)
        weights = state.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"{policy_id}: weights ausentes")
        arrays = [np.asarray(value) for value in weights.values()]
        non_finite = sum(int((~np.isfinite(value)).sum()) for value in arrays)
        if non_finite:
            raise ValueError(f"{policy_id}: {non_finite} pesos NaN/Inf")
        optimizer_arrays = collect_arrays(state.get("_optimizer_variables"))
        optimizer_non_finite = sum(
            int((~np.isfinite(value)).sum()) for value in optimizer_arrays
        )
        if optimizer_non_finite:
            raise ValueError(
                f"{policy_id}: {optimizer_non_finite} valores NaN/Inf no optimizer"
            )
        if policy_id == "policy_blue" and not optimizer_arrays:
            raise ValueError("policy_blue: estado do optimizer ausente")
        result["policies"][policy_id] = {
            "tensors": len(arrays),
            "parameters": int(sum(value.size for value in arrays)),
            "max_abs": float(max(np.abs(value).max() for value in arrays)),
            "l2_norm": float(np.sqrt(sum(np.square(value).sum() for value in arrays))),
            "optimizer_tensors": len(optimizer_arrays),
            "optimizer_parameters": int(sum(value.size for value in optimizer_arrays)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.checkpoint)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())