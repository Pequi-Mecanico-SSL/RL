import yaml
import pickle
import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
import numpy as np
from gymnasium.spaces.box import Box

from scripts import CustomFCNet
from scripts import TorchBetaTest_blue, TorchBetaTest_yellow
from rSoccer import SSLMultiAgentEnv

from rewards import DENSE_REWARDS, SPARSE_REWARDS
import time

from torch import nn

with open("config.yaml") as f:
    # use safe_load instead load
    file_configs = yaml.safe_load(f)

CHECKPOINT_PATH_BLUE = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_28c7c_00000_0_2025-03-16_22-36-53/checkpoint_000003"
CHECKPOINT_PATH_YELLOW = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_6a346_00000_0_2025-03-18_02-05-07/checkpoint_000004"
NUM_EPS = 100

with open(f"{CHECKPOINT_PATH_YELLOW}/policies/policy_blue/policy_state.pkl", "rb") as f:
    policy_state = pickle.load(f)

obs_space_size = policy_state["policy_spec"]['observation_space']['space']['shape'][0]
act_space_size = policy_state["policy_spec"]['action_space']['space']['shape'][0]

model = CustomFCNet(
    obs_space=Box(low=-1.2, high=1.2, shape=(obs_space_size,), dtype=np.float64),
    action_space=Box(low=-1.0, high=1.0, shape=(act_space_size,), dtype=np.float64),
    num_outputs=2*act_space_size,
    model_config=policy_state["policy_spec"]['config']["model"],
    name=None,
    **file_configs["custom_model"]
)

model.load_state_dict({k: torch.tensor(v) for k, v in  policy_state["weights"].items()})

