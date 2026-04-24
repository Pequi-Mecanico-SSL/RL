from gymnasium.envs.registration import register

register(
    id="SSLMultiAgentEnv", 
    entry_point="src.simulators.rsoccer:SSLMultiAgentEnv", 
    max_episode_steps=1200
)