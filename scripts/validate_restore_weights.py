"""Restore descartavel do checkpoint cirurgico H-sync (exigencia do debate).

Constroi o Algorithm PPO identico ao RL_train.py (0 workers, CPU), restaura o
checkpoint e prova, ANTES de qualquer train(), que os pesos efetivamente
carregados de policy_yellow == policy_blue tensor a tensor.

Uso: python scripts/validate_restore_weights.py <config.yaml> <checkpoint>
"""
import sys

import numpy as np
import yaml

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray import tune

from RL_train import (
    SelfPlayUpdateCallback,
    ScoreCounter,
    create_rllib_env,
    policy_mapping_fn,
)
from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rewards import DENSE_REWARDS, SPARSE_REWARDS


def main():
    config_path, checkpoint = sys.argv[1], sys.argv[2]
    with open(config_path) as f:
        file_configs = yaml.safe_load(f)

    ray.init(num_cpus=1)
    ScoreCounter.options(name="score_counter").remote(
        maxlen=file_configs["score_average_over"]
    )
    tune.registry.register_env("Soccer", create_rllib_env)
    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    configs["env_config"] = dict(file_configs["env"])
    configs["env_config"]["dense_rewards"] = DENSE_REWARDS
    configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS

    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    configs["callbacks"] = SelfPlayUpdateCallback
    configs["multiagent"] = {
        "policies": {
            "policy_blue": (None, obs_space, act_space, {"model": {"custom_action_dist": "beta_dist_blue"}}),
            "policy_yellow": (None, obs_space, act_space, {"model": {"custom_action_dist": "beta_dist_yellow"}}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["policy_blue"],
    }
    configs["model"] = {
        "custom_model": "custom_vf_model",
        "custom_model_config": file_configs["custom_model"],
        "custom_action_dist": "beta_dist",
    }
    configs["env"] = "Soccer"
    # Restore barato: sem rollout workers e sem GPU
    configs["num_workers"] = 0
    configs["num_gpus"] = 0
    configs["num_cpus"] = 1

    algo = PPO(config=configs)
    algo.restore(checkpoint)

    it = algo.iteration
    w = algo.get_weights(["policy_blue", "policy_yellow"])
    blue, yellow = w["policy_blue"], w["policy_yellow"]
    keys_b = sorted(blue)
    keys_y = sorted(yellow)
    n_equal = n_params = 0
    for kb, ky in zip(keys_b, keys_y):
        a, b = np.asarray(blue[kb]), np.asarray(yellow[ky])
        assert a.shape == b.shape, f"shape diverge: {kb} vs {ky}"
        if not np.array_equal(a, b):
            raise SystemExit(f"FALHA: yellow != blue apos restore em {kb}/{ky}")
        if not np.isfinite(a).all():
            raise SystemExit(f"FALHA: NaN/Inf em {kb}")
        n_equal += 1
        n_params += a.size
    l2 = float(np.sqrt(sum((np.asarray(v) ** 2).sum() for v in blue.values())))
    print(
        f"RESTORE_VALIDATION_OK iteration={it} tensors={n_equal} "
        f"params={n_params} blue_l2={l2:.4f} yellow==blue bit-exato"
    )
    algo.stop()


if __name__ == "__main__":
    main()
