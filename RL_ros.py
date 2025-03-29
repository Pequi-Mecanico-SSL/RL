import yaml
import pickle
import torch

import numpy as np
from gymnasium.spaces.box import Box

from scripts import CustomFCNet
from scripts import TorchBetaTest_blue, TorchBetaTest_yellow
from rSoccer import SSLMultiAgentEnv
from rSoccer.rsoccer_gym.Entities.Robot import Robot

from torch import nn

def receive_state() -> dict[str, list[list]]:
    """Essa função deve ser chamada para receber a observação do ambiente indepedente de onde venha (ros ou rSoccer)
    O retorno deve seguir a seguinte estrutura:
    {
        "blue_0":   [ x, y, theta ],
        "blue_1":   [ x, y, theta ],
        "blue_2":   [ x, y, theta ],
        "yellow_0": [ x, y, theta ],
        "yellow_1": [ x, y, theta ],
        "yellow_2": [ x, y, theta ],
        "ball":     [ x, y ]
    }
    *Obs: o valor de theta em Graus
    """
    pass

def state_to_observation(state: dict[str, list], last_action = dict[str, list]) -> dict[str, list]:
    observation = {}
    state_to_obs_func = SSLMultiAgentEnv(file_configs["env"]).robot_observation
    state_blue = {key: value for key, value in state.items() if "blue" in key}
    state_yellow = {key: value for key, value in state.items() if "yellow" in key}
    for key, value in state_blue.items():
        robot_id = int(key.split("_")[1])
        robot = Robot(yellow=False, id=robot_id, x=value[0], y=value[1], theta=value[2])
        robot_allies = [
            Robot(
                yellow=False, id=int(ka.split("_")[1]), x=va[0], y=va[1], theta=va[2]
            ) for ka, va in state_blue.items() if ka != key
        ]
        robot_advs = [
            Robot(
                yellow=True, id=int(ka.split("_")[1]), x=va[0], y=va[1], theta=va[2]
            ) for ka, va in state_yellow.items()
        ]
        robot_action = last_action.get(key)
        allies_actions = {ka: last_action.get(ka) for ka in state_blue.keys() if ka != key}
        ball = state["ball"]

def send_action(action: list) -> bool:
    """Essa função deve ser chamada para enviar a ação para o ambiente indepedente de onde venha (ros ou rSoccer)
    O retorno deve um booleano indicando se a partida acabou ou não"""
    pass

with open("config.yaml") as f:
    # use safe_load instead load
    file_configs: dict = yaml.safe_load(f)

CHECKPOINT_PATH_BLUE = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_28c7c_00000_0_2025-03-16_22-36-53/checkpoint_000003"
CHECKPOINT_PATH_YELLOW = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_6a346_00000_0_2025-03-18_02-05-07/checkpoint_000004"
NUM_EPS = 100

with open(f"{CHECKPOINT_PATH_YELLOW}/policies/policy_blue/policy_state.pkl", "rb") as f:
    policy_state: dict = pickle.load(f)

obs_space_size: int = policy_state["policy_spec"]['observation_space']['space']['shape'][0]
act_space_size: int = policy_state["policy_spec"]['action_space']['space']['shape'][0]

model = CustomFCNet(
    obs_space=Box(low=-1.2, high=1.2, shape=(obs_space_size,), dtype=np.float64),
    action_space=Box(low=-1.0, high=1.0, shape=(act_space_size,), dtype=np.float64),
    num_outputs=2*act_space_size, # é o dorbro porque inclui os dois parametros da distribuição beta por action
    model_config=policy_state["policy_spec"]['config']["model"], #isso é um dicionário com as configurações do modelo padrão do ray
    name=None,
    **file_configs["custom_model"] #isso é um dicionário com as configurações do modelo customizado
)

model.load_state_dict({k: torch.tensor(v) for k, v in  policy_state["weights"].items()})

# loop para inferência
done = False
stack_observations = []
last_actions = [[0, 0, 0, 0] for _ in range(3)]
while not done:
    state = receive_state()
    observation = state_to_observation(state, last_actions)
    model_output, *_ = model(observation)
    distribution = TorchBetaTest_blue(model_output, model)
    # Estudar pra ver qual o ideal: o deterministic ou o sample
    actions = distribution.deterministic_sample().detach().numpy() 
    done = send_action(actions)
    last_action = actions



