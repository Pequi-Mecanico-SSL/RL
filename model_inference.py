from rsoccer_gym.Entities import Frame, Robot, Ball, Field



from pprint import pprint
import yaml

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
import os
import numpy as np

from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest, TorchBetaTest_blue, TorchBetaTest_yellow
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
import time
import random

#TODO: Some of this code has to be executed before instantiating this class, figure out which lines are 
#important and figure out how to integrate them

#with open("config.yaml") as f:
#    file_configs = yaml.safe_load(f)
#
#configs = {**file_configs["rllib"], **file_configs["PPO"]}
#configs["env_config"] = file_configs["env"]
#configs["env"] = "Soccer"
#configs["env_config"]["match_time"] = 40
#
#
#def create_rllib_env(config):
#    #breakpoint()
#    return GenericSSLMultiAgentEnv(**config)
#
##configs["env_config"]["init_pos"]["ball"] = [random.uniform(-2, 2), random.uniform(-1.2, 1.2)]
#tune.registry._unregister_all()
#tune.registry.register_env("Soccer", create_rllib_env)


class SoccerPolicyWrapper:
    """
    A wrapper
    """
    def __init__(self, 
                 checkpoint_path="/root/ray_results/PPO_selfplay_rec/PPO_Soccer_28842_00000_0_2024-12-06_02-52-40/checkpoint_000007",
                 observation_space=(-1.2009999999999998, 1.2009999999999998, (616,), np.float64),
                 action_space=(-1.0, 1.0, (4,), np.float64), 
                 number_of_blue_robots=3, 
                 number_of_yellow_robots=3,
                 config_path="config.yaml",
                 configs=None
                ):
        """
        checkpoint_path: the path to the model's checkpoint
        observation_space: the shape of the observation space
        action_space: the shape of the action space
        number_of_blue_robots: how many blue robots there are
        number_of_yellow_robots: how many yellow robots there are
        """
        self.checkpoint_path = checkpoint_path
        #print("self.checkpoint_path =",self.checkpoint_path )
        self.obs_space = observation_space
        #print("self.obs_space =",self.obs_space )
        self.act_space = action_space
        #print("self.act_space =",self.act_space )
        self.n_robots_blue = number_of_blue_robots
        #print("self.n_robots_blue =",self.n_robots_blue )
        self.n_robots_yellow = number_of_yellow_robots
        #print("self.n_robots_yellow =",self.n_robots_yellow )
        self.configs=configs

        ray.init()

        with open(config_path) as f:
            self.file_configs = yaml.safe_load(f)

        #TODO: Don't delete this, something outside the class is currently doing it but for the final thing we need it I think
        #self.configs = {**self.file_configs["rllib"], **self.file_configs["PPO"]}

        ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
        ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
        ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

        self.configs["multiagent"] = {
            "policies": {
                "policy_blue": (None, self.obs_space, self.act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}}),
                "policy_yellow": (None, self.obs_space, self.act_space, {'model': {'custom_action_dist': 'beta_dist_yellow'}}),
            },
            "policy_mapping_fn": self.policy_mapping_fn,
            "policies_to_train": ["policy_blue"],
        }
        self.configs["model"] = {
            "custom_model": "custom_vf_model",
            "custom_model_config": self.file_configs["custom_model"],
            "custom_action_dist": "beta_dist",
        }

        #TODO: FIX THE CONFIG SITUATION JESUS CHRIST THIS IS A MESS!
        # this thing is using the configs variable from outside the class and its the only way it works
        self.agent = PPOConfig.from_dict(self.configs).build()

        self.agent.restore(self.checkpoint_path)





        #self.observations = {
        #    **{f'blue_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_blue)},
        #    **{f'yellow_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_yellow)}
        #}



    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if "blue" in agent_id:
            pol_id = "policy_blue"
        elif "yellow" in agent_id:
            pol_id = "policy_yellow"
        return pol_id

    #def add_observation(obs):
    #    pass
    def compute_actions(self, obs='o_blue dict', policy_id='policy_blue', full_fetch=False):
        """
        obs: used to be o_blue or o_yellow, it's the stacked observations (not the frame) of all robots
        policy_id: idk but it somehow distinguishes each team, its a rllib thing I think
        full_fetch: no idea what it is tbh
        """
        return self.agent.compute_actions(obs, policy_id=policy_id, full_fetch=full_fetch)