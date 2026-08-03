#!/usr/bin/env python3
"""Valida estrutura e finitude dos pesos de um checkpoint RLlib."""

import argparse
import json
from pathlib import Path

import numpy as np
import ray.cloudpickle as cloudpickle


EXPECTED_POLICIES = ("policy_blue", "policy_yellow")


def validate(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "rllib_checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy_ids = set(metadata.get("policy_ids", []))
    if policy_ids != set(EXPECTED_POLICIES):
        raise ValueError(f"policy_ids invalidas: {sorted(policy_ids)}")

    result = {"checkpoint": str(checkpoint), "policies": {}}
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
        result["policies"][policy_id] = {
            "tensors": len(arrays),
            "parameters": int(sum(value.size for value in arrays)),
            "max_abs": float(max(np.abs(value).max() for value in arrays)),
            "l2_norm": float(np.sqrt(sum(np.square(value).sum() for value in arrays))),
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