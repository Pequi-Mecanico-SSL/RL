import argparse
import os
import yaml
from collections import deque
import numpy as np
from typing import Dict

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.evaluation.episode import Episode

from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv, SSLMultiAgentEnv_record

from torch.utils.tensorboard import SummaryWriter
import os

# RAY_PDB=1 python rllib_multiagent.py
# ray debug

def create_rllib_env_recorder(config):
    trigger = lambda t: t % 1 == 0
    config["render_mode"] = "rgb_array"
    ssl_el_env = SSLMultiAgentEnv(**config)
    return SSLMultiAgentEnv_record(ssl_el_env, video_folder="/ws/videos", episode_trigger=trigger, disable_logger=True)

def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        pol_id = "policy_blue"
    elif "yellow" in agent_id:
        pol_id = "policy_yellow"
    return pol_id

@ray.remote
class ScoreCounter:
    def __init__(self, maxlen):
        self.last100 = deque(maxlen=maxlen)
        self.last100.extend([0.0 for _ in range(maxlen)])
        self.maxlen = maxlen

    def append(self, s):
        self.last100.append(s)

    def reset(self):
        self.last100.extend([0.0 for _ in range(self.maxlen)])

    def get_score(self):
        return np.array(self.last100).mean()
    

class SelfPlayUpdateCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):

        super().__init__(legacy_callbacks_dict)

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):

        episode.hist_data["score"] = []

    def on_episode_end(
        self, *, worker, base_env, policies, episode: Episode, **kwargs
    ) -> None:
        info_a = episode.last_info_for("blue_0")
        single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]

        score_counter = ray.get_actor("score_counter")
        score_counter.append.remote(single_score)

    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when score is high enough
        """
        score_counter = ray.get_actor("score_counter")
        current_score = ray.get(score_counter.get_score.remote())

        info["result"]["custom_metrics"]["score"] = current_score

        if current_score > 0.6:
            print("---- Updating Opponent!!! ----")
            algorithm = info["algorithm"]
            algorithm.set_weights(
                {
                    "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                }
            )
            score_counter = ray.get_actor("score_counter")
            score_counter.reset.remote()

parser = argparse.ArgumentParser(description="Treina multiagent SSL-EL.")
parser.add_argument("--evaluation", action="store_true", help="Irá renderizar um episódio de tempos em tempos.")

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    with open("config.yaml") as f:
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}

    counter = ScoreCounter.options(name="score_counter").remote(
        maxlen=file_configs["score_average_over"]
    )
    configs["env_config"] = file_configs["env"]

    tune.registry.register_env("Soccer", create_rllib_env)
    tune.registry.register_env("Soccer_recorder", create_rllib_env_recorder)
    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    # Register the models to use.
    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
    # Each policy can have a different configuration (including custom model).


    configs["callbacks"] = SelfPlayUpdateCallback
    configs["multiagent"] = {
        "policies": {
            "policy_blue": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}}),
            "policy_yellow": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_yellow'}}),
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

    if args.evaluation:
        eval_configs = file_configs["evaluation"].copy()
        env_config_eval = file_configs["env"].copy()
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] =  eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = eval_configs["evaluation_config"].copy()
        configs["evaluation_config"]["env_config"] = env_config_eval

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config=configs,
        stop={
            "timesteps_total": int(file_configs["timesteps_total"]),
        },
        checkpoint_freq=int(file_configs["checkpoint_freq"]),
        checkpoint_at_end=True,
        local_dir=os.path.abspath("volume"),
        #resume=True,
        restore=file_configs["checkpoint_restore"],
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

