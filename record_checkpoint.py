"""Load a trained checkpoint and record a few episodes to video.

RL_eval.py as committed doesn't actually save videos -- it uses
render_mode="human" (an interactive display, not a file), calls
breakpoint() every 100 steps, and points at a hardcoded, unrelated
checkpoint path. This script reuses RL_eval.py's checkpoint-loading logic
but wires up the same MyRecordVideo + rgb_array pattern RL_train.py's
create_rllib_env_recorder uses for its --evaluation mode, non-interactively.

Usage (inside the training image):
    python record_checkpoint.py <checkpoint_dir> [num_episodes] [match_time]
"""
import os
import sys
import time

import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from src.judges.ssl_judge import Judge
from src.models.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from src.models.custom_torch_model import CustomFCNet
from src.observations import OBSERVATIONS
from src.rewards import DENSE_REWARDS, SPARSE_REWARDS
from src.simulators import SSLMultiAgentEnv
from src.utils.wrappers import MyRecordVideo, StackWrapper

CHECKPOINT_PATH = sys.argv[1]
NUM_EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MATCH_TIME = int(sys.argv[3]) if len(sys.argv) > 3 else 20

ray.init()


def create_recording_env(config):
    stack_size = config.pop("stack_size")
    config["render_mode"] = "rgb_array"
    video_prefix = f"eval-{os.path.basename(CHECKPOINT_PATH)}-{int(time.time())}"
    base_env = StackWrapper(SSLMultiAgentEnv(**config), stack_size=stack_size, observation_funcs=OBSERVATIONS)
    base_env.render_mode = "rgb_array"
    return MyRecordVideo(
        base_env,
        video_folder="/ws/videos",
        episode_trigger=lambda ep: True,
        name_prefix=video_prefix,
        disable_logger=True,
    )


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_blue" if "blue" in agent_id else "policy_yellow"


with open("config.yaml") as f:
    file_configs = yaml.safe_load(f)

configs = {**file_configs["rllib"], **file_configs["PPO"]}
configs["env_config"] = file_configs["env"]
configs["env_config"]["judge"] = Judge
configs["env_config"]["dense_rewards"] = DENSE_REWARDS
configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS
configs["env_config"]["match_time"] = MATCH_TIME
configs["stack_size"] = file_configs["env"].get("stack_size", 8)

ray.tune.registry._unregister_all()
ray.tune.registry.register_env("Soccer", create_recording_env)

temp_env = create_recording_env(configs["env_config"].copy())
obs_space = temp_env.observation_space["blue_0"]
act_space = temp_env.action_space["blue_0"]
temp_env.close()

ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

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
configs["num_cpus"] = 1
configs["num_workers"] = 0

algo = PPOConfig.from_dict(configs).build()
algo.restore(CHECKPOINT_PATH)
print(f"Restored checkpoint: {CHECKPOINT_PATH}")

env = create_recording_env(configs["env_config"].copy())
obs, _ = env.reset()

for ep in range(NUM_EPS):
    done = {"__all__": False}
    truncated = {"__all__": False}
    while not done["__all__"] and not truncated["__all__"]:
        o_blue = {f"blue_{i}": obs[f"blue_{i}"] for i in range(env.n_robots_blue)}
        o_yellow = {f"yellow_{i}": obs[f"yellow_{i}"] for i in range(env.n_robots_yellow)}
        actions = {}
        if env.n_robots_blue > 0:
            actions.update(algo.compute_actions(o_blue, policy_id="policy_blue", full_fetch=False))
        if env.n_robots_yellow > 0:
            actions.update(algo.compute_actions(o_yellow, policy_id="policy_yellow", full_fetch=False))
        obs, reward, done, truncated, info = env.step(actions)

    print(f"Episode {ep}: score={info.get('blue_0', {}).get('score')}")
    obs, _ = env.reset()

env.close()
print("Done. Videos in /ws/videos")
