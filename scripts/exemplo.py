import gymnasium as gym
import rsoccer_gym

# Using VSS Single Agent env
env = gym.make("SSLStaticDefenders-v0", render_mode="human")

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    terminated = False
    truncated = False
    while not (terminated or truncated):
        env.render()
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
    print(reward)