#!/usr/bin/env python3
"""Exporta pesos RLlib sem optimizer para inferencia CPU-portavel."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import ray.cloudpickle as cloudpickle


POLICIES = ("policy_blue", "policy_yellow")


def export(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / "rllib_checkpoint.json", destination / "rllib_checkpoint.json")
    params_source = source.parent / "params.json"
    params_target = destination.parent / "params.json"
    if params_source.is_file():
        if params_target.exists() and params_target.read_bytes() != params_source.read_bytes():
            raise ValueError(f"params.json divergente em {params_target}")
        shutil.copy2(params_source, params_target)
    result = {"source": str(source), "destination": str(destination), "policies": {}}
    for policy_id in POLICIES:
        source_policy = source / "policies" / policy_id
        target_policy = destination / "policies" / policy_id
        target_policy.mkdir(parents=True)
        with (source_policy / "policy_state.pkl").open("rb") as stream:
            state = cloudpickle.load(stream)
        weights = {
            name: np.asarray(value).copy()
            for name, value in state["weights"].items()
        }
        portable_state = {"weights": weights}
        with (target_policy / "policy_state.pkl").open("wb") as stream:
            pickle.dump(portable_state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy2(
            source_policy / "rllib_checkpoint.json",
            target_policy / "rllib_checkpoint.json",
        )
        result["policies"][policy_id] = {
            "tensors": len(weights),
            "parameters": int(sum(value.size for value in weights.values())),
        }
    (destination / "inference_export.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())