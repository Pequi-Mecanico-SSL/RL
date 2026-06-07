import numpy as np
import os
from gymnasium.spaces import Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.monitoring import video_recorder

class StackWrapper(MultiAgentEnv):
    def __init__(self, base_env, stack_size, observation_funcs, *args, **kwargs):
        #super().__init__()
        self.base_env = base_env
        self.stack_size = stack_size
        self.observation_funcs = observation_funcs

        n_blue = self.base_env.n_robots_blue
        n_yellow = self.base_env.n_robots_yellow
        self.reset()
        self.observation_space = Dict(
            **{f'blue_{i}': Box(low=-1, high=1, shape=(self.stack_size * self.obs_size,), dtype=np.float64) for i in range(n_blue)},
            **{f'yellow_{i}':Box(low=-1, high=1, shape=(self.stack_size * self.obs_size,), dtype=np.float64) for i in range(n_yellow)}
        )
    
    def __getattr__(self, attr):
        return getattr(self.base_env, attr)

    
    def _reset_stack(self, obs_size):
        stack_obs = {
            **{f'blue_{i}': np.zeros(self.stack_size * obs_size, dtype=np.float64) for i in range(self.base_env.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(self.stack_size * obs_size, dtype=np.float64) for i in range(self.base_env.n_robots_yellow)}
        }

        return stack_obs


    def _update_stack(self, stack_obs, observations):
        for agent, obs in observations.items():       
            stack_obs[agent] = np.concatenate([
                np.delete(
                    stack_obs[agent], 
                    range(len(obs))
                ), # remove oldest observation
                obs
            ], axis=0, dtype=np.float64)
        
        return stack_obs
    

    def _calculate_observations(self, raw_observations):
        observations = {
            **{f'blue_{i}': np.zeros(0, dtype=np.float64) for i in range(self.base_env.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(0, dtype=np.float64) for i in range(self.base_env.n_robots_yellow)}
        }
        obs_size = 0
        for observation_func, class_attrs in self.observation_funcs:
            kwargs = {
                attr: (
                    getattr(self.base_env, attr, None) or
                    getattr(self, attr, None)
                )
                for attr in class_attrs
            }
            obs_result = observation_func(
                self.base_env.n_robots_blue, 
                self.base_env.n_robots_yellow, 
                raw_observations,
                self.base_env.field_info, 
                kwargs=kwargs
            )

            for agent, obs in obs_result.items():
                observations[agent] = np.hstack([observations[agent], obs])
            obs_size += len(obs)

        return observations, obs_size


    def reset(self, *args, **kwargs):
        self.last_actions = {
            **{f'blue_{i}': np.zeros(4) for i in range(self.base_env.n_robots_blue)}, 
            **{f'yellow_{i}': np.zeros(4) for i in range(self.base_env.n_robots_yellow)}
        }

        raw_observations, info = self.base_env.reset(*args, **kwargs)
        observations, obs_size = self._calculate_observations(raw_observations)
        stack_obs = self._reset_stack(obs_size)
        self.stack_obs = self._update_stack(stack_obs, observations)

        self.obs_size = obs_size
        return self.stack_obs, info
    

    def step(self, action):

        raw_observations, reward, done, truncated, info = self.base_env.step(action)
        observations, _ = self._calculate_observations(raw_observations)
        self.stack_obs = self._update_stack(self.stack_obs, observations)

        self.last_actions = action.copy()
        return self.stack_obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)
    

class MyRecordVideo(RecordVideo, MultiAgentEnv):
    def reset(self, **kwargs):
        """Reset env without forcing an immediate render/capture.

        rsoccer may fail rendering right after reset; first frame is captured after
        the first environment step instead.
        """
        observations = self.env.reset(**kwargs)
        self.terminated = False
        self.truncated = False

        if not self.recording and self._video_enabled():
            self.start_video_recorder()

        return observations

    def start_video_recorder(self):
        """Start recorder without capturing a frame immediately."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.recorded_frames = 0
        self.recording = True

    def close(self):
        """Close recorder explicitly before environment teardown."""
        self.close_video_recorder()
        try:
            self.env.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds["__all__"] or truncateds["__all__"]:
                    self.episode_id += 1
                    self.terminated = terminateds["__all__"]
                    self.truncated = truncateds["__all__"]
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds["__all__"] or truncateds["__all__"]:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos
        
