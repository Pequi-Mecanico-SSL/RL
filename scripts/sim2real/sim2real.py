import yaml
import json
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
from ray.rllib.models import ModelCatalog

from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

import numpy as np
from sim2real import state_to_obs
from sim2real.config import CHECKPOINT_PATH, N_ROBOTS_BLUE, N_ROBOTS_YELLOW

def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        pol_id = "policy_blue"
    elif "yellow" in agent_id:
        pol_id = "policy_yellow"
    return pol_id

ray.init()

with open("config.yaml") as f:
    file_configs = yaml.safe_load(f)

configs = {**file_configs["rllib"], **file_configs["PPO"]}

configs["env_config"] = file_configs["env"]

tune.registry.register_env("Soccer", create_rllib_env)

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

state = {
    "robots_blue": {
        #"robot_0": [x, y, theta] # formato esperado
        "robot_0": np.zeros(3),
        "robot_1": np.zeros(3),
        "robot_2": np.zeros(3),
    },
    "robots_yellow": {
        #"robot_0": [x, y, theta]
        "robot_0": np.zeros(3),
        "robot_1": np.zeros(3),
        "robot_2": np.zeros(3),
    },
    "ball": [0, 0]
}

stacked_obs = {
    "blue_0": np.zeros(77 * 8), # Agrupa as observações dos ultimos 8 frames do robô para passar para a rede
    "blue_1": np.zeros(77 * 8),
    "blue_2": np.zeros(77 * 8),
    "yellow_0": np.zeros(77 * 8),
    "yellow_1": np.zeros(77 * 8),
    "yellow_2": np.zeros(77 * 8),
}

actions = {
    # "blue_0": [v_x, v_y, v_theta, v_kick]
    "blue_0": np.zeros(4),
    "blue_1": np.zeros(4),
    "blue_2": np.zeros(4),
    "yellow_0": np.zeros(4),
    "yellow_1": np.zeros(4),
    "yellow_2": np.zeros(4),
}
last_actions = actions.copy()

def state_to_action(state):
    global stacked_obs
    stacked_obs = state_to_obs.frame_to_observations(state, last_actions, stacked_obs)

    o_blue = {f"blue_{i}": stacked_obs[f"blue_{i}"] for i in range(N_ROBOTS_BLUE)}
    o_yellow = {f"yellow_{i}": stacked_obs[f"yellow_{i}"] for i in range(N_ROBOTS_YELLOW)}

    actions = {
        **agent.compute_actions(o_blue, policy_id='policy_blue', full_fetch=False),
        **agent.compute_actions(o_yellow, policy_id='policy_yellow', full_fetch=False)
    }

    return  actions


if __name__ == "__main__":
    exemple_state = {
        'robots_blue': {
            'robot_0': [0, 0, 0],
            'robot_1': [0, 0, 0],
            'robot_2': [0, 0, 0]
        },
        'robots_yellow': {
            'robot_0': [0, 0, 0],
            'robot_1': [0, 0, 0],
            'robot_2': [0, 0, 0]
        },
        'ball': [0, 0]
    }
    
    actions = state_to_action(exemple_state)
    actions = {agent:str(list(action)) for agent, action in actions.items()}
    print(json.dumps(actions, indent=4))