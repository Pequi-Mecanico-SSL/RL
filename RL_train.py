# RL_train.py

import argparse
import os
import yaml
import numpy as np
import pickle
import re
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog

from scripts.model.custom_torch_model import CustomFCNet
from scripts.model.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from rSoccer.rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv, SSLMultiAgentEnv_record
from rSoccer.rsoccer_gym.judges.ssl_judge import Judge

# --- IMPORTAÇÃO DAS NOVAS RECOMPENSAS ---
from rewards import ATTACKER_DENSE_REWARDS, DEFENDER_DENSE_REWARDS, SPARSE_REWARDS

parent_directory = "/root/ray_results/PPO_selfplay_rec"

def create_rllib_env_recorder(config):
    trigger = lambda t: t % 1 == 0
    config["render_mode"] = "rgb_array"
    ssl_el_env = SSLMultiAgentEnv(**config)
    return SSLMultiAgentEnv_record(ssl_el_env, video_folder="/ws/videos", episode_trigger=trigger, disable_logger=True)

def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

# --- FUNÇÃO DE MAPEAMENTO ATUALIZADA ---
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        # Time azul é o atacante
        return "policy_attack"
    elif "yellow" in agent_id:
        # Time amarelo é o defensor
        return "policy_defense"

# Funções de checkpoint (sem alterações)
def find_latest_experiment(parent_dir):
    pattern = r"PPO_Soccer_\w+_\d+_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    experiments_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    matching_exp = [d for d in experiments_dirs if re.match(pattern, d)]
    matching_exp.sort(key=lambda x: datetime.strptime("_".join(x.split('_')[-2:]), "%Y-%m-%d_%H-%M-%S"), reverse=True)
    latest_exp = matching_exp[0] if len(matching_exp) > 0 else None
    if latest_exp is None: return None
    
    checkpoint_pattern = r"checkpoint_\d{6}"
    checkpoint_dirs = [d for d in os.listdir(os.path.join(parent_dir, latest_exp)) if os.path.isdir(os.path.join(parent_dir, latest_exp, d))]
    matching_checkpoints = [d for d in checkpoint_dirs if re.match(checkpoint_pattern, d)]
    matching_checkpoints.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    latest_checkpoint = matching_checkpoints[0] if len(matching_checkpoints) > 0 else None
    if latest_checkpoint is None: return None

    full_checkpoint_path = os.path.join(parent_dir, latest_exp, latest_checkpoint)
    return full_checkpoint_path
    
def save_checkpoint_weights(checkpoint_path):
    # Salva ambas as políticas agora
    for pol_id in ["policy_attack", "policy_defense"]:
        policy_path = f"{checkpoint_path}/policies/{pol_id}/policy_state.pkl"
        if os.path.exists(policy_path):
            with open(policy_path, "rb") as f:
                policy_state = pickle.load(f)
            # save checkpoint weights
            with open(f"{checkpoint_path}/policies/{pol_id}/policy_weights.pkl", "wb") as f:
                pickle.dump(policy_state, f)

# --- ScoreCounter e SelfPlayUpdateCallback REMOVIDOS ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina multiagent SSL-EL com políticas assimétricas.")
    parser.add_argument("--evaluation", action="store_true", help="Irá renderizar um episódio de tempos em tempos.")
    args = parser.parse_args()

    ray.init()

    with open("config.yaml") as f:
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}

    # --- Passando as recompensas para a config do ambiente ---
    configs["env_config"] = file_configs["env"]
    configs["env_config"]["judge"] = Judge
    configs["env_config"]["attacker_dense_rewards"] = ATTACKER_DENSE_REWARDS
    configs["env_config"]["defender_dense_rewards"] = DEFENDER_DENSE_REWARDS
    configs["env_config"]["sparse_rewards"] = SPARSE_REWARDS

    tune.registry.register_env("Soccer", create_rllib_env)
    tune.registry.register_env("Soccer_recorder", create_rllib_env_recorder)
    
    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)

    # --- CONFIGURAÇÃO MULTI-AGENT ATUALIZADA ---
    configs["multiagent"] = {
        "policies": {
            # Política para o time de ATACANTES
            "policy_attack": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_blue'}}),
            # Política para o time de DEFENSORES
            "policy_defense": (None, obs_space, act_space, {'model': {'custom_action_dist': 'beta_dist_yellow'}}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        # >>>>> MUDANÇA MAIS IMPORTANTE: TREINAR AMBAS AS POLÍTICAS <<<<<
        "policies_to_train": ["policy_attack", "policy_defense"],
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
        # Adiciona as recompensas à config de avaliação também
        env_config_eval["attacker_dense_rewards"] = ATTACKER_DENSE_REWARDS
        env_config_eval["defender_dense_rewards"] = DEFENDER_DENSE_REWARDS
        env_config_eval["sparse_rewards"] = SPARSE_REWARDS
        
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] =  eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = eval_configs["evaluation_config"].copy()
        configs["evaluation_config"]["env_config"] = env_config_eval

    try:
        analysis = tune.run(
            "PPO",
            name="PPO_AttackVsDefense", # Nome do novo experimento
            config=configs,
            stop={
                "timesteps_total": int(file_configs["timesteps_total"]),
            },
            checkpoint_freq=int(file_configs["checkpoint_freq"]),
            checkpoint_at_end=True,
            local_dir=os.path.abspath("volume"),
            restore=file_configs["checkpoint_restore"],
        )
    except Exception as e:
        latest_experiment = find_latest_experiment(parent_directory)
        if latest_experiment:
            save_checkpoint_weights(latest_experiment)
            print(f"Checkpoint weights saved from {latest_experiment}")
        raise e
    finally:
        latest_experiment = find_latest_experiment(parent_directory)
        if latest_experiment:
            save_checkpoint_weights(latest_experiment)
            print(f"Checkpoint weights saved from {latest_experiment}")
            

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")