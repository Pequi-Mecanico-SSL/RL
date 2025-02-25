import yaml

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
import numpy as np

from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rSoccer.rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

from rewards import DENSE_REWARDS, SPARSE_REWARDS

ray.init()

CHECKPOINT_PATH = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_28842_00000_0_2024-12-06_02-52-40/checkpoint_000007"

def create_rllib_env(config):
    #breakpoint()
    return SSLMultiAgentEnv(**config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        pol_id = "policy_blue"
    elif "yellow" in agent_id:
        pol_id = "policy_yellow"
    return pol_id

with open("config.yaml") as f:
    # use safe_load instead load
    file_configs = yaml.safe_load(f)

configs = {**file_configs["rllib"], **file_configs["PPO"]}


configs["env_config"] = file_configs["env"]
#configs["env_config"]["init_pos"]["ball"] = [random.uniform(-2, 2), random.uniform(-1.2, 1.2)]
ray.tune.registry._unregister_all()
ray.tune.registry.register_env("Soccer", create_rllib_env)
temp_env = create_rllib_env(configs["env_config"])
obs_space = temp_env.observation_space["blue_0"]
act_space = temp_env.action_space["blue_0"]
temp_env.close()

# Register the models to use.
ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
# Each policy can have a different configuration (including custom model).

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

agent = PPOConfig.from_dict(configs).build()

agent.restore(CHECKPOINT_PATH)

configs["env_config"]["match_time"] = 40
configs["env_config"]["dense_rewards"] = DENSE_REWARDS
configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS
env = SSLMultiAgentEnv(**configs["env_config"])
obs, *_ = env.reset()

done= {'__all__': False}
e= 0.0
while True:
    o_blue = {f"blue_{i}": obs[f"blue_{i}"] for i in range(env.n_robots_blue)}
    o_yellow = {f"yellow_{i}": obs[f"yellow_{i}"] for i in range(env.n_robots_yellow)}

    a = {}
    if env.n_robots_blue > 0:
        a.update(agent.compute_actions(o_blue, policy_id='policy_blue', full_fetch=False))

    if env.n_robots_yellow > 0:
        a.update(agent.compute_actions(o_yellow, policy_id='policy_yellow', full_fetch=False))

    if np.random.rand() < e:
         a = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(a)
    env.render()

    if done['__all__'] or truncated['__all__']:

        obs, *_ = env.reset()
    #time.sleep(1)
