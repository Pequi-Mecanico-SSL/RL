"""Record a video of the SSLEL env with random actions, for a visual sanity
check of the real small SSL-EL field (4.5x3.0m) -- no trained policy needed,
no checkpoint exists for this field yet.

Usage (inside the ssl-el-sslel image):
    python record_sslel_env.py [num_episodes] [match_time]
"""
import sys
import time

from src.judges.ssl_judge import Judge
from src.objects import Config, InitialPosition
from src.rewards import DENSE_REWARDS, SPARSE_REWARDS
from src.simulators import SSLMultiAgentEnv
from src.utils.wrappers import MyRecordVideo

NUM_EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MATCH_TIME = int(sys.argv[2]) if len(sys.argv) > 2 else 20

config = Config(
    init_pos=InitialPosition(
        blue={1: [-0.5, 0.0, 0.0], 2: [-1.0, 0.5, 0.0], 3: [-1.0, -0.5, 0.0]},
        yellow={1: [0.5, 0.0, 180.0], 2: [1.0, 0.5, 180.0], 3: [1.0, -0.5, 180.0]},
        ball=[0, 0],
    ),
    field_type=0,
    fps=30,
    match_time=MATCH_TIME,
    render_mode="rgb_array",
)

base_env = SSLMultiAgentEnv(
    judge=Judge,
    dense_rewards=DENSE_REWARDS,
    sparse_rewards=SPARSE_REWARDS,
    use_sslel=True,
    end_on_offense=False,  # random actions foul constantly on this small field; don't cut episodes short
    **{k: v for k, v in config.model_dump().items() if k != "stack_size"},
)
base_env.render_mode = "rgb_array"

env = MyRecordVideo(
    base_env,
    video_folder="/ws/videos",
    episode_trigger=lambda ep: True,
    name_prefix=f"sslel-env-{int(time.time())}",
    disable_logger=True,
)

obs, info = env.reset()
print(f"field: {base_env.field}")

for ep in range(NUM_EPS):
    done = {"__all__": False}
    truncated = {"__all__": False}
    while not done["__all__"] and not truncated["__all__"]:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
    print(f"Episode {ep}: score={info.get('blue_0', {}).get('score')}")
    obs, info = env.reset()

env.close()
print("Done. Videos in /ws/videos")
