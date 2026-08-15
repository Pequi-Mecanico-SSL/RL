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

from rewards import DENSE_REWARDS, SPARSE_REWARDS
import time

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


def validate_train_result(result, expected_batch_size, expected_workers):
    expected_values = {
        "num_env_steps_sampled_this_iter": expected_batch_size,
        "num_env_steps_trained_this_iter": expected_batch_size,
        "num_healthy_workers": expected_workers,
        "num_remote_worker_restarts": 0,
        "num_faulty_episodes": 0,
    }
    errors = []
    for key, expected in expected_values.items():
        if key not in result:
            errors.append(f"{key} ausente")
        elif result[key] != expected:
            errors.append(f"{key}={result[key]!r}, esperado {expected!r}")

    episodes = result.get("episodes_this_iter")
    if episodes is None:
        errors.append("episodes_this_iter ausente")
    elif episodes <= 0:
        errors.append(f"episodes_this_iter={episodes!r}, esperado > 0")

    iteration = result.get("training_iteration")
    if not isinstance(iteration, int) or iteration < 1:
        errors.append(f"training_iteration invalida: {iteration!r}")
    else:
        expected_env_steps = iteration * expected_batch_size
        expected_agent_steps = expected_env_steps * 6
        cumulative_values = {
            "num_env_steps_sampled": expected_env_steps,
            "num_env_steps_trained": expected_env_steps,
            "num_agent_steps_sampled": expected_agent_steps,
            "num_agent_steps_trained": expected_agent_steps,
        }
        for key, expected in cumulative_values.items():
            if key not in result:
                errors.append(f"{key} ausente")
            elif result[key] != expected:
                errors.append(f"{key}={result[key]!r}, esperado {expected!r}")

    if errors:
        iteration = result.get("training_iteration", "desconhecida")
        raise RuntimeError(
            f"resultado de treino invalido na iteracao {iteration}: "
            + "; ".join(errors)
        )


def validate_policy_weights(weights):
    expected_policies = {"policy_blue", "policy_yellow"}
    if set(weights) != expected_policies:
        raise RuntimeError(
            f"policies invalidas nos pesos: {sorted(weights)}, "
            f"esperado {sorted(expected_policies)}"
        )

    for policy_id, policy_weights in weights.items():
        if not policy_weights:
            raise RuntimeError(f"{policy_id}: pesos ausentes")
        non_finite = sum(
            int((~np.isfinite(np.asarray(value))).sum())
            for value in policy_weights.values()
        )
        if non_finite:
            raise RuntimeError(f"{policy_id}: {non_finite} pesos NaN/Inf")

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
        algorithm = info["algorithm"]
        validate_train_result(
            info["result"],
            expected_batch_size=algorithm.config["train_batch_size"],
            expected_workers=algorithm.config["num_workers"],
        )
        validate_policy_weights(
            algorithm.get_weights(["policy_blue", "policy_yellow"])
        )

        score_counter = ray.get_actor("score_counter")
        current_score = ray.get(score_counter.get_score.remote())

        info["result"]["custom_metrics"]["score"] = current_score

        if current_score > 0.6:
            if os.environ.get("FREEZE_OPPONENT") == "1":
                # Braco experimental H-sync: adversario fixo durante o run
                print("---- Sync suprimido (FREEZE_OPPONENT=1) ----")
            else:
                print("---- Updating Opponent!!! ----")
                algorithm.set_weights(
                    {
                        "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                    }
                )
                score_counter = ray.get_actor("score_counter")
                score_counter.reset.remote()

parser = argparse.ArgumentParser(description="Treina multiagent SSL-EL.")
parser.add_argument("--evaluation", action="store_true", help="Irá renderizar um episódio de tempos em tempos.")
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--restore", default=None)
parser.add_argument("--stop-timesteps", type=int, default=None)
parser.add_argument("--local-dir", default="volume")
parser.add_argument("--experiment-name", default="PPO_selfplay_rec")

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    with open(args.config) as f:
        file_configs = yaml.safe_load(f)

    local_dir = os.path.abspath(args.local_dir)
    parent_directory = os.path.join(local_dir, args.experiment_name)
    stop_timesteps = (
        args.stop_timesteps
        if args.stop_timesteps is not None
        else int(file_configs["timesteps_total"])
    )
    restore_checkpoint = args.restore or file_configs["checkpoint_restore"]
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    configs["recreate_failed_workers"] = False
    configs["max_num_worker_restarts"] = 0

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

    configs["env_config"]["dense_rewards"] = DENSE_REWARDS
    configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS
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
        name=args.experiment_name,
        config=configs,
        stop={
            "timesteps_total": stop_timesteps,
        },
        checkpoint_freq=int(file_configs["checkpoint_freq"]),
        checkpoint_at_end=True,
        local_dir=local_dir,
        max_failures=0,
        fail_fast=True,
        #resume=True,
        restore=restore_checkpoint,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

