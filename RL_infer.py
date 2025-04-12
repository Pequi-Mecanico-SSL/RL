import yaml
import pickle
import torch
import numpy as np
import argparse

from rSoccer import SSLMultiAgentEnv

from scripts import InferenceModel
from scripts import InferenceBetaDist


# Só o receive_state e o send_action devem ser alteradas para o ros

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
    global N_ROBOTS, OBS_SPACE_SIZE
    obs = {}
    for i in range(N_ROBOTS):
        robot_name = f"blue_{i}"
        obs[robot_name] = ENV.observations[robot_name][-OBS_SPACE_SIZE:].copy()
    
    for i in range(N_ROBOTS):
        robot_name = f"yellow_{i}"
        obs[robot_name] = ENV.observations[robot_name][-OBS_SPACE_SIZE:].copy()

    ball = ENV.frame.ball
    obs["ball"] = [ball.x, ball.y]
    
    return obs

def send_action(actions: dict[str, list[float]]) -> bool:
    """Essa função deve ser chamada para enviar a ação para o ambiente indepedente de onde venha (ros ou rSoccer)
    O retorno deve um booleano indicando se a partida acabou ou não
    
    formato da entrada (actions):
    {
        "blue_0":   [ v_x, v_y, v_theta, v_kick],
        "blue_1":   [ v_x, v_y, v_theta, v_kick],
        "blue_2":   [ v_x, v_y, v_theta, v_kick],
        "yellow_0": [ v_x, v_y, v_theta, v_kick],
        "yellow_1": [ v_x, v_y, v_theta, v_kick],
        "yellow_2": [ v_x, v_y, v_theta, v_kick]
    }

    Deve retornar True se a partida acabou e False caso contrário
    """
    _, _, done, truncated, _ = ENV.step(actions)
    ENV.render()
    return done["__all__"] or truncated["__all__"]



# Daqui para baixo não precisa mexer

# -----------------------LOAD ENVIRONMENT (com ros não precisa) -----------------------
with open("config.yaml") as f:
    file_configs: dict = yaml.safe_load(f)
    
# Fiz isso temporariamente, quanto tiver o ros funcionando, não vai precisar
ENV = SSLMultiAgentEnv(**file_configs["env"]) 
ENV.reset()


# -----------------------VARIÁVEIS GLOBAIS-----------------------
CHECKPOINT_PATH_BLUE = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_baseline_2025-03-16/checkpoint_000003"
OBS_SPACE_SIZE = 77
ACT_SPACE_SIZE = 4
N_STACK_OBS = 8
N_ROBOTS = 3

parser = argparse.ArgumentParser(description="Choose which team to control.")
parser.add_argument("--control_yellow", action="store_true", help="Control the yellow team if set, otherwise control the blue team.")
args = parser.parse_args()

TEAM = "yellow" if args.control_yellow else "blue"
SIGNAL = [-1, 1, -1, 1] if args.control_yellow else [1, 1, 1, 1]

STACK_OBSERVATIONS = {
    **{f'blue_{i}': np.zeros(N_STACK_OBS * OBS_SPACE_SIZE, dtype=np.float64) for i in range(N_ROBOTS)},
    **{f'yellow_{i}': np.zeros(N_STACK_OBS * OBS_SPACE_SIZE, dtype=np.float64) for i in range(N_ROBOTS)}
}
LAST_ACTIONS: dict[str, list] = {
    **{f'blue_{i}': [0, 0, 0, 0] for i in range(N_ROBOTS)},
    **{f'yellow_{i}':[0, 0, 0, 0] for i in range(N_ROBOTS)}
}


def state_to_observation(state: dict[str, list]):
    global file_configs, STACK_OBSERVATIONS, LAST_ACTIONS, OBS_SPACE_SIZE, n_steps
    env_config = file_configs["env"].copy()
    
    # Configura ambiente para receber o estado inicial com as posições dos robôs
    for robot_name in state.keys():
        if robot_name == "ball": continue
        team, robot_id = robot_name.split("_")
        robot_id = int(robot_id)
        env_config["init_pos"][team][robot_id] = state[robot_name].copy()
    env_config["init_pos"]["ball"] = state["ball"].copy()

    # Uso o ambiente aqui só para calcular as observações
    env_dummy = SSLMultiAgentEnv(**env_config)
    env_dummy.reset()
    env_dummy.LAST_ACTIONS = LAST_ACTIONS.copy()
    env_dummy.steps = n_steps
    env_dummy._frame_to_observations()
    observation = env_dummy.observations.copy()
    # Faz a lógica de guardar as observações passadas
    for robot_name in state.keys():
        if robot_name == "ball": continue
        STACK_OBSERVATIONS[robot_name] = np.delete(STACK_OBSERVATIONS[robot_name], range(OBS_SPACE_SIZE))
        STACK_OBSERVATIONS[robot_name] = np.concatenate([STACK_OBSERVATIONS[robot_name], observation[robot_name][-OBS_SPACE_SIZE:]], axis=0, dtype=np.float64)
    
    return STACK_OBSERVATIONS.copy()


# -----------------------LOAD MODEL CHECKPOINT-----------------------
with open(f"{CHECKPOINT_PATH_BLUE}/policies/policy_blue/policy_state.pkl", "rb") as f:
    policy_state: dict = pickle.load(f)

# OBS_SPACE_SIZE: int = policy_state["policy_spec"]['observation_space']['space']['shape'][0]
# ACT_SPACE_SIZE: int = policy_state["policy_spec"]['action_space']['space']['shape'][0]
model = InferenceModel(input_size=N_STACK_OBS*OBS_SPACE_SIZE, output_size=2*ACT_SPACE_SIZE)

weights_dict = {}
for layer_name, weights in policy_state["weights"].items():
    split = layer_name.split(".")
    if "_logits" == split[0] or "_value_branch" == split[0]:
        new_layer_name = split[0] + "." + split[-1]
    else:
        new_layer_name = split[0] + "." + str(int(split[1])*2) + "." + split[-1]
    weights_dict[new_layer_name] = torch.tensor(weights)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(weights_dict)
model.to(device)


# -----------------------LOOP DE INFERÊNCIA-----------------------
done = False
n_steps = 0
actions = LAST_ACTIONS.copy()
while not done:
    n_steps += 1
    state = receive_state()
    observations = state_to_observation(state)

    model_input = []
    for i in range(N_ROBOTS):
        model_input.append(observations[f"{TEAM}_{i}"])
    model_input = torch.tensor(np.array(model_input, dtype=np.float32))
    model_output, *_ = model(model_input.to(device))
    distribution = InferenceBetaDist(model_output, signal=SIGNAL)
    dist_output = distribution.sample().detach().to("cpu").numpy()

    for i in range(N_ROBOTS):
        actions[f"{TEAM}_{i}"] = list(dist_output[i])
    done = send_action(actions)
    LAST_ACTIONS = actions.copy()






