import yaml
import pickle
import torch
import numpy as np

from rSoccer import SSLMultiAgentEnv

from scripts import InferenceModel
from scripts import InferenceBetaDist
# import debugpy

# debugpy.listen(("0.0.0.0", 5679))
# input("Aguardando o debugger...")
# Daqui para baixo não precisa mexer

def state_to_observation(state: dict[str, list], last_actions: dict[str, list], obs_space_size: int, n_steps: int) -> dict[str, np.ndarray]:
    """
    Função que transforma o estado do ambiente em observações
    
    :Params: *state* estado do ambiente
        Exemplo:
            state = {
                "blue_0": [x, y, theta],
                "blue_1": [x, y, theta],
                "blue_2": [x, y, theta],
                "yellow_0": [x, y, theta],
                "yellow_1": [x, y, theta],
                "yellow_2": [x, y, theta],
                "ball": [x, y]
            }
    :Params: *last_actions* ações anteriores
        Exemplo:
            last_actions = {
                "blue_0": [0, 0, 0, 0],
                "blue_1": [0, 0, 0, 0],
                "blue_2": [0, 0, 0, 0],
                "yellow_0": [0, 0, 0, 0],
                "yellow_1": [0, 0, 0, 0],
                "yellow_2": [0, 0, 0, 0]
            }
    :Params: *obs_space_size* tamanho do espaço de observações
    :Params: *n_steps* número de passos
    :Params: *env_config* configuração do ambiente
        Exemplo:
            env_config = {
                "init_pos": {
                    "blue": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "yellow": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "ball": [0, 0]
                },
                "other_config": ...

    
    """
    env_config = {
        "init_pos": {
            "blue": {}, 
            "yellow": {}, 
            "ball": {}
        },
        "field_type": 1,
        "fps": 30, # frames por segundo
        "match_time": 40, # duração da partida em segundos
        "render_mode": "human"
    }
    # Configura ambiente para receber o estado inicial com as posições dos robôs
    for robot_name in state.keys():
        if robot_name == "ball": continue
        team, robot_id = robot_name.split("_")
        robot_id = int(robot_id)
        env_config["init_pos"][team][robot_id+1] = state[robot_name].copy()
    env_config["init_pos"]["ball"] = state["ball"].copy()

    # Uso o ambiente aqui só para calcular as observações
    env_dummy = SSLMultiAgentEnv(**env_config)
    env_dummy.reset()
    env_dummy.last_actions = last_actions.copy()
    env_dummy.steps = n_steps
    env_dummy._frame_to_observations()
    raw_observation = env_dummy.observations.copy()

    observation = {}
    for robot_name in state.keys():
        if robot_name == "ball": continue
        observation[robot_name] = raw_observation[robot_name][-obs_space_size:].copy()
    
    return observation


def buffer(stack_observations: dict[str, np.ndarray], observation:  np.ndarray, obs_space_size: int, n_stack: int) -> np.ndarray:
    """
    Função que gerencia o buffer de observações
    :Params:
        stack_observations: dict[str, np.ndarray] - dicionário do formato:
        {
            "blue_0":   np.ndarray, # (shape(n_stack * obs_space_size))
            "blue_1":   np.ndarray, # (shape(n_stack * obs_space_size))
            "blue_2":   np.ndarray, # (shape(n_stack * obs_space_size))
            "yellow_0": np.ndarray, # (shape(n_stack * obs_space_size))
            "yellow_1": np.ndarray, # (shape(n_stack * obs_space_size))
            "yellow_2": np.ndarray  # (shape(n_stack * obs_space_size))
        }
        *obs: Deve ser inicializado com zero no inicio para manter o fomato correto

        observations: np.ndarray - estado atual do ambiente # (shape(obs_space_size))

    """

    for robot_name in observation.keys():
        assert stack_observations[robot_name].shape == (n_stack * obs_space_size,), f"stack_observations[{robot_name}].shape: {stack_observations[robot_name].shape} != ({n_stack * obs_space_size},)"
        assert observation[robot_name].shape == (obs_space_size,), f"state[{robot_name}].shape: {observation[robot_name].shape} != ({obs_space_size},)"

        if robot_name == "ball": continue
        stack_observations[robot_name] = np.delete(stack_observations[robot_name], range(obs_space_size))
        stack_observations[robot_name] = np.concatenate([
            stack_observations[robot_name], 
            observation[robot_name][-obs_space_size:]], 
            axis=0, dtype=np.float64
        )
    return stack_observations


def load_model(checkpoint_path: str, action_space_size:int, obs_space_size: int, n_stack: int) -> InferenceModel:
    """Carrega o modelo a partir do checkpoint especificado
    :Params:
        checkpoint_path: str - caminho do checkpoint
        action_space_size: int - tamanho do espaço de ações
        obs_space_size: int - tamanho do espaço de observações
        n_stack: int - número de observações empilhadas
    :Returns:
        model: InferenceModel - modelo carregado
    """
    model = InferenceModel(input_size= obs_space_size*n_stack, output_size=2 * action_space_size)

    with open(f"{checkpoint_path}/policies/policy_blue/policy_state.pkl", "rb") as f:
        policy_state: dict = pickle.load(f)

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
    
    return model


def policy(stack_observations: dict[str, np.ndarray], model: InferenceModel, team: str) -> dict[str, list]:

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_input = []
    for robot_name in stack_observations.keys():
        if team not in robot_name: continue
        model_input.append(stack_observations[robot_name])
    model_input = torch.tensor(np.array(model_input, dtype=np.float32))

    model_output, *_ = model(model_input.to(device))
    distribution = InferenceBetaDist(
        model_output, 
        signal=[-1, 1, -1, 1] if team=="yellow" else [1, 1, 1, 1]
    )
    dist_output = distribution.sample().detach().to("cpu").numpy()

    actions = {}
    for i, robot_name in enumerate(stack_observations.keys()):
        if team not in robot_name: continue
        _, idx = robot_name.split("_")
        idx = int(idx)
        actions[robot_name] = list(dist_output[idx])

    return actions


if __name__ == "__main__":
    # ---------------------EXEMPLO DE USO---------------------
    with open("config.yaml") as f:
        file_configs: dict = yaml.safe_load(f)
        
    stacked_obs = {
        **{f"blue_{i}": np.zeros(8*77) for i in range(3)},
        **{f"yellow_{i}": np.zeros(8*77) for i in range(3)}
    }

    last_actions = {
        **{f"blue_{i}": [0, 0, 0, 0] for i in range(3)},
        **{f"yellow_{i}": [0, 0, 0, 0] for i in range(3)}
    }

    # Essa parte é só para esse teste pra construir state, não estaria no uso final----
    ENV = SSLMultiAgentEnv(**file_configs["env"])
    ENV.reset()
    state = {}
    for i in range(3):
        robot_name = f"blue_{i}"
        state[robot_name] = ENV.observations[robot_name][-77:].copy()
    
    for i in range(3):
        robot_name = f"yellow_{i}"
        state[robot_name] = ENV.observations[robot_name][-77:].copy()
    ball = ENV.frame.ball
    state["ball"] = [ball.x, ball.y]
    #---------------------------------------------------------------------------------

    # observation = state_to_observation(
    #     state=state, 
    #     last_actions=last_actions,
    #     obs_space_size=77,
    #     n_steps=0,
    # )

    
    model = load_model(
        checkpoint_path="/root/ray_results/PPO_selfplay_rec/PPO_Soccer_baseline_2025-03-16/checkpoint_000003",
        action_space_size=4,
        obs_space_size=77,
        n_stack=8
    )

    done = False
    s = 0
    while not done:
        print(f"{f'step {s}':-^20}")

        observation = state_to_observation(
            state=state, 
            last_actions=last_actions,
            obs_space_size=77,
            n_steps=s,
        )

        stacked_obs = buffer(
            stack_observations=stacked_obs,
            observation=observation,
            obs_space_size=77,
            n_stack=8
        )

        #print("\n".join([f"robot {i+1}: {ENV._friendly_observation(stacked_obs['blue_0'][77*i:77*(1+i)])['pos']['robot']}" for i in range(8)]))

        action = policy(
            stack_observations=stacked_obs,
            model=model,
            team="blue"
        )
        last_actions.update(action)

        obs, _, status, *_ = ENV.step(last_actions)
        state = {
            "blue_0": [ENV.frame.robots_blue[0].x, ENV.frame.robots_blue[0].y, ENV.frame.robots_blue[0].theta],
            "blue_1": [ENV.frame.robots_blue[1].x, ENV.frame.robots_blue[1].y, ENV.frame.robots_blue[1].theta],
            "blue_2": [ENV.frame.robots_blue[2].x, ENV.frame.robots_blue[2].y, ENV.frame.robots_blue[2].theta],
            "yellow_0": [ENV.frame.robots_yellow[0].x, ENV.frame.robots_yellow[0].y, ENV.frame.robots_yellow[0].theta],
            "yellow_1": [ENV.frame.robots_yellow[1].x, ENV.frame.robots_yellow[1].y, ENV.frame.robots_yellow[1].theta],
            "yellow_2": [ENV.frame.robots_yellow[2].x, ENV.frame.robots_yellow[2].y, ENV.frame.robots_yellow[2].theta],
            "ball": [ENV.frame.ball.x, ENV.frame.ball.y]
        }

        # observation = {team: obs[team][-77:].copy() for team in obs}
        done = status["__all__"]
        s += 1
        ENV.render()

        #input()

    print(action)




