#!/usr/bin/env python3
"""Gate numérico RLlib CustomFCNet versus modelo standalone de deploy."""

import pickle
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.model.action_dists import TorchBetaTest_blue
from scripts.model.action_dists_inferece import InferenceBetaDist
from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.model_inferece import InferenceModel


def remap_weights(weights):
    remapped = {}
    for layer_name, value in weights.items():
        split = layer_name.split(".")
        if split[0] in ("_logits", "_value_branch"):
            new_name = f"{split[0]}.{split[-1]}"
        else:
            new_name = f"{split[0]}.{int(split[1]) * 2}.{split[-1]}"
        remapped[new_name] = torch.as_tensor(value)
    return remapped


def main(checkpoint):
    policy_file = Path(checkpoint) / "policies/policy_blue/policy_state.pkl"
    with policy_file.open("rb") as stream:
        policy_state = pickle.load(stream)
    weights = policy_state["weights"]

    obs_space = gym.spaces.Box(-1.201, 1.201, shape=(616,), dtype=np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)
    model_config = {
        "fcnet_activation": "tanh",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": "relu",
        "no_final_linear": False,
        "free_log_std": False,
    }
    rllib_model = CustomFCNet(
        obs_space,
        action_space,
        8,
        model_config,
        "parity_reference",
        fcnet_hiddens=[300, 200, 100],
        vf_share_layers=False,
    )
    rllib_model.load_state_dict(
        {name: torch.as_tensor(value) for name, value in weights.items()}, strict=True
    )
    rllib_model.eval()

    standalone = InferenceModel(616, 8)
    standalone.load_state_dict(remap_weights(weights), strict=True)
    standalone.eval()

    generator = torch.Generator().manual_seed(20260721)
    corpus = torch.cat(
        [
            torch.zeros((1, 616)),
            torch.ones((1, 616)),
            -torch.ones((1, 616)),
            torch.rand((61, 616), generator=generator) * 2.0 - 1.0,
        ]
    )
    with torch.no_grad():
        reference_logits, _ = rllib_model(
            {"obs": corpus, "obs_flat": corpus}, [], None
        )
        reference_values = rllib_model.value_function().unsqueeze(1)
        deploy_logits, deploy_values = standalone(corpus)

    np.testing.assert_allclose(
        deploy_logits.numpy(), reference_logits.numpy(), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        deploy_values.numpy(), reference_values.numpy(), rtol=0.0, atol=1e-6
    )

    reference_dist = TorchBetaTest_blue(reference_logits, rllib_model)
    deploy_dist = InferenceBetaDist(deploy_logits)
    np.testing.assert_allclose(
        deploy_dist.inputs.detach().numpy(),
        reference_dist.inputs.detach().numpy(),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deploy_dist.deterministic_sample().detach().numpy(),
        reference_dist.deterministic_sample().detach().numpy(),
        rtol=0.0,
        atol=1e-6,
    )

    print(
        "INFERENCE_PARITY_OK",
        f"vectors={len(corpus)}",
        f"max_logits_error={(deploy_logits-reference_logits).abs().max().item():.3e}",
        f"max_value_error={(deploy_values-reference_values).abs().max().item():.3e}",
    )


if __name__ == "__main__":
    main(sys.argv[1])
