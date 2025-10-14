# RL_train_league.py

import argparse
import os
import yaml
import numpy as np
import pickle
from datetime import datetime

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm

from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue
from rSoccer.rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv, SSLMultiAgentEnv_record
from rSoccer.rsoccer_gym.judges.ssl_judge import Judge

from rewards import DENSE_REWARDS, SPARSE_REWARDS

OPPONENT_CHECKPOINT_PATHS = [
    "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_f5006_00000_0_2025-09-27_20-45-52/checkpoint_000010",
    "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c3b63_00000_0_2025-09-26_10-58-42/checkpoint_000010",
]

TRAINEE_CHECKPOINT_PATH = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_c3b63_00000_0_2025-09-26_10-58-42/checkpoint_000010"

OPPONENT_POLICY_IDS = [f"opponent_{i}" for i in range(len(OPPONENT_CHECKPOINT_PATHS))]

def create_rllib_env_recorder(config):
    trigger = lambda t: t % 1 == 0
    config["render_mode"] = "rgb_array"
    ssl_el_env = SSLMultiAgentEnv(**config)
    return SSLMultiAgentEnv_record(ssl_el_env, video_folder="/ws/videos", episode_trigger=trigger, disable_logger=True)

def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

def league_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        return "policy_trainee"
    elif "yellow" in agent_id:
        opponent_index = episode.episode_id % len(OPPONENT_POLICY_IDS)
        return OPPONENT_POLICY_IDS[opponent_index]

def save_trainee_weights(checkpoint_path, policy_id="policy_trainee"):
    policy_state_path = os.path.join(checkpoint_path, "policies", policy_id, "policy_state.pkl")
    policy_weights_path = os.path.join(checkpoint_path, "policies", policy_id, "policy_weights.pkl")
    
    if os.path.exists(policy_state_path):
        with open(policy_state_path, "rb") as f:
            policy_state = pickle.load(f)
        with open(policy_weights_path, "wb") as f:
            pickle.dump(policy_state['weights'], f)
        print(f"Pesos salvos para a política '{policy_id}' em: {policy_weights_path}")
    else:
        print(f"AVISO: Arquivo de estado da política não encontrado em {policy_state_path}")

# <<< MUDANÇA PRINCIPAL: Criar uma classe para os Callbacks >>>
class LeagueCallbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs):
        """
        Callback executado na inicialização do algoritmo para carregar os pesos
        do aprendiz e dos oponentes.
        """
        # --- 1. Carrega os pesos do aprendiz (se especificado) ---
        if TRAINEE_CHECKPOINT_PATH and os.path.isdir(TRAINEE_CHECKPOINT_PATH):
            policy_state_path = os.path.join(TRAINEE_CHECKPOINT_PATH, "policies", "policy_blue", "policy_state.pkl")
            try:
                with open(policy_state_path, "rb") as f:
                    policy_state = pickle.load(f)
                algorithm.get_policy("policy_trainee").set_weights(policy_state["weights"])
                print(f"--- Pesos do APRENDIZ carregados com sucesso de '{policy_state_path}' ---")
            except Exception as e:
                print(f"ERRO ao carregar pesos para 'policy_trainee' de {policy_state_path}: {e}")
        else:
            print("--- 'policy_trainee' iniciando com pesos aleatórios (nenhum caminho especificado ou encontrado) ---")

        # --- 2. Carrega os pesos dos oponentes ---
        for i, policy_id in enumerate(OPPONENT_POLICY_IDS):
            checkpoint_path = OPPONENT_CHECKPOINT_PATHS[i]
            policy_state_path = os.path.join(checkpoint_path, "policies", "policy_blue", "policy_state.pkl")
            try:
                with open(policy_state_path, "rb") as f:
                    policy_state = pickle.load(f)
                policy = algorithm.get_policy(policy_id)
                policy.set_weights(policy_state["weights"])
                print(f"--- Pesos carregados com sucesso para o oponente '{policy_id}' de '{policy_state_path}' ---")
                for param in policy.model.parameters():
                    param.requires_grad = False
            except FileNotFoundError:
                print(f"ERRO: Arquivo de pesos não encontrado para '{policy_id}' em '{policy_state_path}'")
            except Exception as e:
                print(f"ERRO ao carregar pesos para '{policy_id}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um agente contra uma liga de oponentes fixos.")
    parser.add_argument("--evaluation", action="store_true", help="Grava vídeos dos episódios de avaliação.")
    args = parser.parse_args()

    ray.init()

    with open("config.yaml") as f:
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    
    configs["env_config"] = file_configs["env"]
    configs["env_config"]["judge"] = Judge
    configs["env_config"]["dense_rewards"] = DENSE_REWARDS
    configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS

    tune.registry.register_env("Soccer", create_rllib_env)
    tune.registry.register_env("Soccer_recorder", create_rllib_env_recorder)
    
    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

    policies_config = {
        "policy_trainee": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}})
    }
    for opponent_id in OPPONENT_POLICY_IDS:
        policies_config[opponent_id] = (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}})

    configs["multiagent"] = {
        "policies": policies_config,
        "policy_mapping_fn": league_policy_mapping_fn,
        "policies_to_train": ["policy_trainee"],
    }
    
    # <<< MUDANÇA PRINCIPAL: Atribuir a CLASSE de callback, não uma instância >>>
    configs["callbacks"] = LeagueCallbacks
    
    configs["model"] = {
        "custom_model": "custom_vf_model",
        "custom_model_config": file_configs["custom_model"],
    }
    configs["env"] = "Soccer"

    if args.evaluation:
        eval_configs = file_configs["evaluation"].copy()
        env_config_eval = file_configs["env"].copy()
        env_config_eval["dense_rewards"] = DENSE_REWARDS
        env_config_eval["sparse_rewards"] = SPARSE_REWARDS
        
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] =  eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = {
            "env": "Soccer_recorder",
            "env_config": env_config_eval
        }

    try:
        analysis = tune.run(
            "PPO",
            name="PPO_TraineeVsPolicies", 
            config=configs,
            stop={
                "timesteps_total": int(file_configs["timesteps_total"]),
            },
            checkpoint_freq=int(file_configs["checkpoint_freq"]),
            checkpoint_at_end=True,
            local_dir=os.path.abspath("volume"),
            restore=file_configs.get("checkpoint_restore"), # Usar .get() para segurança
        )
        
        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        print("Melhor Trial:", best_trial)
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print("Melhor Checkpoint:", best_checkpoint)
        if best_checkpoint:
            save_trainee_weights(best_checkpoint.path)

    finally:
        print("Treinamento finalizado ou interrompido.")
        if 'analysis' in locals() and hasattr(analysis, 'best_checkpoint') and analysis.best_checkpoint:
            last_checkpoint_path = analysis.get_best_checkpoint(analysis.get_best_trial("episode_reward_mean", mode="max"), "episode_reward_mean", mode="max").path
            if last_checkpoint_path:
                print("Salvando pesos do melhor checkpoint...")
                save_trainee_weights(last_checkpoint_path)
        
        print("Done training")