import yaml

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog


from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

ray.init()

def create_rllib_env(config):
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

tune.registry.register_env("Soccer", create_rllib_env)
temp_env = create_rllib_env(configs["env_config"])
obs_space = temp_env.observation_space["blue_0"]
act_space = temp_env.action_space["blue_0"]
temp_env.close()

# Register the models to use.
ModelCatalog.register_custom_action_dist("beta_dist", TorchBetaTest)
ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
# Each policy can have a different configuration (including custom model).

configs["multiagent"] = {
    "policies": {
        "policy_blue": (None, obs_space, act_space, {}),
        "policy_yellow": (None, obs_space, act_space, {}),
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
agent.restore('backup_checkpoints/PPO_Soccer_f5487_00000_0_2024-10-23_09-47-27/checkpoint_000000')

env = SSLMultiAgentEnv(**configs["env_config"])
obs, *_ = env.reset()

done= {'__all__': False}
while not done['__all__']:
    a = agent.compute_actions(obs, policy_id='policy_blue', full_fetch=False)
    obs, reward, done, truncated, info = env.step(a)
    env.render()
    print(reward)