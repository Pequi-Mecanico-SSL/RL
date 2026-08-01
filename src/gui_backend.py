from __future__ import annotations

import copy
import os
import pickle
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ray
import yaml
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog

from src.models.custom_torch_model import CustomFCNet
from src.models.action_dists import TorchBetaTest_blue, TorchBetaTest_yellow
from src.simulators.rsoccer import SSLMultiAgentEnv
from src.utils.wrappers import StackWrapper
from src.judges.ssl_judge import Judge
from src.rewards import DENSE_REWARDS, SPARSE_REWARDS
from src.observations import OBSERVATIONS

DEFAULT_RAY_RESULTS_DIR = os.environ.get("RL_RAY_RESULTS_DIR", "/root/ray_results/PPO_selfplay_rec")
DEFAULT_VIDEO_DIR = os.environ.get("RL_VIDEO_DIR", os.path.abspath("volumes/videos"))

_COMPONENTS_REGISTERED = False
RAY_NAMESPACE = "pequi_rl_gui"


def _gpu_available() -> bool:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None and cuda_visible_devices.strip() in {"", "-1", "none", "None"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _apply_cpu_fallback(configs: dict[str, Any]) -> None:
    if _gpu_available():
        return
    configs["num_gpus"] = 0
    if "num_gpus_per_worker" in configs:
        configs["num_gpus_per_worker"] = 0


def _ensure_project_paths() -> list[str]:
    project_root = os.path.abspath(os.getcwd())
    src_root = os.path.join(project_root, "src")
    for path_entry in (project_root, src_root):
        if path_entry not in sys.path:
            sys.path.insert(0, path_entry)
    return [project_root, src_root]


def load_yaml_config(config_path: str | os.PathLike[str] = "config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def ensure_ray() -> None:
    if not ray.is_initialized():
        required_paths = _ensure_project_paths()
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [entry for entry in current_pythonpath.split(os.pathsep) if entry]
        for required_path in reversed(required_paths):
            if required_path not in pythonpath_entries:
                pythonpath_entries.insert(0, required_path)
        ray.init(
            ignore_reinit_error=True,
            namespace=RAY_NAMESPACE,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join(pythonpath_entries),
                }
            },
        )


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        return "policy_blue"
    if "yellow" in agent_id:
        return "policy_yellow"
    return "policy_blue"


def create_rllib_env(config: dict[str, Any]):
    env_config = copy.deepcopy(config)
    stack_size = env_config.pop("stack_size", 8)
    env_config.setdefault("render_mode", "human")
    env_config["judge"] = Judge
    return StackWrapper(
        SSLMultiAgentEnv(**env_config),
        stack_size=stack_size,
        observation_funcs=OBSERVATIONS,
    )


def create_rllib_env_recorder(config: dict[str, Any], video_folder: str | os.PathLike[str] | None = None):
    from src.utils.wrappers import MyRecordVideo

    env = create_rllib_env(config)
    env.render_mode = "rgb_array"
    video_root = Path(video_folder or DEFAULT_VIDEO_DIR)
    video_root.mkdir(parents=True, exist_ok=True)
    trigger = lambda t: t % 1 == 0
    video_prefix = f"rl-video-{os.getpid()}"
    return MyRecordVideo(
        env,
        video_folder=str(video_root),
        episode_trigger=trigger,
        name_prefix=video_prefix,
        disable_logger=True,
    )


def register_components() -> None:
    global _COMPONENTS_REGISTERED
    if _COMPONENTS_REGISTERED:
        return

    tune.registry.register_env("Soccer", create_rllib_env)
    tune.registry.register_env("Soccer_recorder", create_rllib_env_recorder)
    ModelCatalog.register_custom_action_dist("beta_dist_blue", TorchBetaTest_blue)
    ModelCatalog.register_custom_action_dist("beta_dist_yellow", TorchBetaTest_yellow)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
    _COMPONENTS_REGISTERED = True


def build_training_configs(file_configs: dict[str, Any], evaluation: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    _apply_cpu_fallback(configs)
    env_config = copy.deepcopy(file_configs["env"])
    env_config["judge"] = Judge
    env_config["dense_rewards"] = DENSE_REWARDS
    env_config["sparse_rewards"] = SPARSE_REWARDS

    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    configs["callbacks"] = None
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
    configs["env_config"] = env_config

    if evaluation:
        eval_configs = copy.deepcopy(file_configs["evaluation"])
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] = eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = copy.deepcopy(eval_configs["evaluation_config"])
        configs["evaluation_config"]["env_config"] = copy.deepcopy(env_config)

    return configs, env_config


def build_eval_configs(file_configs: dict[str, Any], render_mode: str = "rgb_array") -> tuple[dict[str, Any], dict[str, Any]]:
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    _apply_cpu_fallback(configs)
    env_config = copy.deepcopy(file_configs["env"])
    env_config["judge"] = Judge
    env_config["dense_rewards"] = DENSE_REWARDS
    env_config["sparse_rewards"] = SPARSE_REWARDS
    env_config["render_mode"] = render_mode

    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

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
    configs["env_config"] = env_config
    configs["num_cpus"] = 1
    configs["num_workers"] = 0

    return configs, env_config


def build_eval_agent(file_configs: dict[str, Any], render_mode: str = "rgb_array"):
    ensure_ray()
    register_components()
    configs, env_config = build_eval_configs(file_configs, render_mode=render_mode)
    agents = PPOConfig.from_dict(configs).build()
    return agents, env_config


def load_policy_state(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    policy_state_path = Path(checkpoint_path) / "policies" / "policy_blue" / "policy_state.pkl"
    with open(policy_state_path, "rb") as policy_file:
        return pickle.load(policy_file)


def restore_opponent_weights(agents, checkpoint_path_yellow: str | os.PathLike[str]) -> None:
    policy_state = load_policy_state(checkpoint_path_yellow)
    agents.set_weights({
        "policy_yellow": policy_state["weights"],
    })


def run_eval_episode(
    config_path: str | os.PathLike[str],
    checkpoint_path_blue: str | os.PathLike[str],
    checkpoint_path_yellow: str | os.PathLike[str] | None = None,
    episodes: int = 1,
    render_mode: str = "rgb_array",
    max_frames: int | None = None,
) -> dict[str, Any]:
    _ensure_project_paths()
    file_configs = load_yaml_config(config_path)
    agents, env_config = build_eval_agent(file_configs, render_mode=render_mode)
    agents.restore(str(checkpoint_path_blue))
    restore_opponent_weights(agents, checkpoint_path_yellow or checkpoint_path_blue)

    env = create_rllib_env(env_config.copy())
    frames: list[np.ndarray] = []
    episode_scores: list[dict[str, Any]] = []

    try:
        obs, *_ = env.reset()
        for episode_index in range(episodes):
            done = {"__all__": False}
            truncated = {"__all__": False}
            episode_frame_count = 0
            latest_info: dict[str, Any] = {}

            while not done["__all__"] and not truncated["__all__"]:
                o_blue = {f"blue_{i}": obs[f"blue_{i}"] for i in range(env.n_robots_blue)}
                o_yellow = {f"yellow_{i}": obs[f"yellow_{i}"] for i in range(env.n_robots_yellow)}

                actions = {
                    **{f"blue_{i}": [0, 0, 0, 0] for i in range(env.n_robots_blue)},
                    **{f"yellow_{i}": [0, 0, 0, 0] for i in range(env.n_robots_yellow)},
                }
                if env.n_robots_blue > 0:
                    actions.update(agents.compute_actions(o_blue, policy_id="policy_blue", full_fetch=False))
                if env.n_robots_yellow > 0:
                    actions.update(agents.compute_actions(o_yellow, policy_id="policy_yellow", full_fetch=False))

                obs, reward, done, truncated, info = env.step(actions)
                latest_info = info
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                    episode_frame_count += 1

                if max_frames is not None and len(frames) >= max_frames:
                    break

            score = None
            if latest_info:
                score = latest_info.get("blue_0", {}).get("score")
            episode_scores.append(
                {
                    "episode": episode_index,
                    "frames": episode_frame_count,
                    "score": score,
                    "done": done["__all__"],
                    "truncated": truncated["__all__"],
                }
            )

            if episode_index + 1 < episodes:
                obs, *_ = env.reset()
    finally:
        env.close()

    return {
        "frames": frames,
        "episode_scores": episode_scores,
        "checkpoint_blue": str(checkpoint_path_blue),
        "checkpoint_yellow": str(checkpoint_path_yellow or checkpoint_path_blue),
    }


def collect_video_files(video_dir: str | os.PathLike[str], limit: int = 5) -> list[str]:
    video_root = Path(video_dir)
    if not video_root.exists():
        return []
    videos = sorted(video_root.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in videos[:limit]]


def list_checkpoint_paths(parent_dir: str | os.PathLike[str] = DEFAULT_RAY_RESULTS_DIR, limit: int = 50) -> list[str]:
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        return []

    checkpoints = [path for path in parent_path.rglob("checkpoint_*") if path.is_dir()]
    checkpoints.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in checkpoints[:limit]]


def find_latest_experiment(parent_dir: str | os.PathLike[str] = DEFAULT_RAY_RESULTS_DIR) -> str | None:
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        return None

    pattern = re.compile(r"PPO_Soccer_\w+_\d+_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
    experiments_dirs = [path for path in parent_path.iterdir() if path.is_dir() and pattern.match(path.name)]
    if not experiments_dirs:
        return None

    experiments_dirs.sort(key=lambda path: datetime.strptime("_".join(path.name.split("_")[-2:]), "%Y-%m-%d_%H-%M-%S"), reverse=True)
    latest_exp = experiments_dirs[0]

    checkpoint_pattern = re.compile(r"checkpoint_\d{6}")
    checkpoint_dirs = [path for path in latest_exp.iterdir() if path.is_dir() and checkpoint_pattern.match(path.name)]
    if not checkpoint_dirs:
        return None

    checkpoint_dirs.sort(key=lambda path: int(path.name.split("_")[-1]), reverse=True)
    return str(checkpoint_dirs[0])


def save_checkpoint_weights(checkpoint_path: str | os.PathLike[str]) -> None:
    checkpoint_dir = Path(checkpoint_path)
    policy_state_path = checkpoint_dir / "policies" / "policy_blue" / "policy_state.pkl"
    if not policy_state_path.exists():
        return

    with open(policy_state_path, "rb") as policy_file:
        policy_state = pickle.load(policy_file)

    with open(checkpoint_dir / "policies" / "policy_blue" / "policy_weights.pkl", "wb") as weights_file:
        pickle.dump(policy_state, weights_file)


@ray.remote
class ScoreCounter:
    def __init__(self, maxlen):
        self.last100 = deque(maxlen=maxlen)
        self.last100.extend([0.0 for _ in range(maxlen)])
        self.maxlen = maxlen

    def append(self, score):
        self.last100.append(score)

    def reset(self):
        self.last100.extend([0.0 for _ in range(self.maxlen)])

    def get_score(self):
        return np.array(self.last100).mean()


class SelfPlayUpdateCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)

    def on_episode_start(self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs):
        episode.hist_data["score"] = []

    def on_episode_end(self, *, worker, base_env, policies, episode: Episode, **kwargs) -> None:
        info_a = episode.last_info_for("blue_0")
        single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]
        try:
            score_counter = ray.get_actor("score_counter", namespace=RAY_NAMESPACE)
            score_counter.append.remote(single_score)
        except ValueError:
            # Keep training running even if the score actor is not available yet.
            return

    def on_train_result(self, **info):
        try:
            score_counter = ray.get_actor("score_counter", namespace=RAY_NAMESPACE)
        except ValueError:
            return

        current_score = ray.get(score_counter.get_score.remote())
        info["result"]["custom_metrics"]["score"] = current_score

        if current_score > 0.6:
            algorithm = info["algorithm"]
            algorithm.set_weights(
                {
                    "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                }
            )
            score_counter.reset.remote()


def build_training_configs(file_configs: dict[str, Any], evaluation: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    _apply_cpu_fallback(configs)
    env_config = copy.deepcopy(file_configs["env"])
    env_config["judge"] = Judge
    env_config["dense_rewards"] = DENSE_REWARDS
    env_config["sparse_rewards"] = SPARSE_REWARDS

    temp_env = create_rllib_env(env_config)
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    configs["callbacks"] = SelfPlayUpdateCallback
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
    configs["env_config"] = env_config

    if evaluation:
        eval_configs = copy.deepcopy(file_configs["evaluation"])
        configs["evaluation_interval"] = eval_configs["evaluation_interval"]
        configs["evaluation_num_workers"] = eval_configs["evaluation_num_workers"]
        configs["evaluation_duration"] = eval_configs["evaluation_duration"]
        configs["evaluation_duration_unit"] = eval_configs["evaluation_duration_unit"]
        configs["evaluation_config"] = copy.deepcopy(eval_configs["evaluation_config"])
        configs["evaluation_config"]["env_config"] = copy.deepcopy(env_config)

    return configs, env_config


def run_training(
    config_path: str | os.PathLike[str] = "config.yaml",
    evaluation: bool = False,
    stop_timesteps: int | None = None,
    checkpoint_restore: str | None = None,
    name: str = "PPO_selfplay_rec",
    local_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    ensure_ray()
    register_components()
    file_configs = load_yaml_config(config_path)
    configs, env_config = build_training_configs(file_configs, evaluation=evaluation)
    try:
        existing_counter = ray.get_actor("score_counter", namespace=RAY_NAMESPACE)
        ray.kill(existing_counter)
    except ValueError:
        pass
    ScoreCounter.options(name="score_counter", lifetime="detached").remote(maxlen=file_configs["score_average_over"])

    stop_value = stop_timesteps if stop_timesteps is not None else int(file_configs["timesteps_total"])
    analysis = tune.run(
        "PPO",
        name=name,
        config=configs,
        stop={"timesteps_total": stop_value},
        checkpoint_freq=int(file_configs["checkpoint_freq"]),
        checkpoint_at_end=True,
        local_dir=os.path.abspath(local_dir or "volume"),
        restore=checkpoint_restore if checkpoint_restore is not None else file_configs.get("checkpoint_restore"),
    )

    latest_experiment = find_latest_experiment()
    if latest_experiment is not None:
        save_checkpoint_weights(latest_experiment)

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = None
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    if best_checkpoint is None and latest_experiment is not None:
        best_checkpoint = latest_experiment

    return {
        "analysis": analysis,
        "best_trial": best_trial,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "latest_experiment": latest_experiment,
        "env_config": env_config,
    }
