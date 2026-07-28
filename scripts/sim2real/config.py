from collections import namedtuple

CHECKPOINT_PATH = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_95caf_00000_0_2024-11-21_02-23-24/checkpoint_000001"
# field_type=1 no rSoccer pinado pelo treino baseline (c684c2b).
FIELD_LENGTH = 9.0
FIELD_WIDTH = 6.0
MAX_EP_LENGTH = 1200
N_ROBOTS_BLUE = 3
N_ROBOTS_YELLOW = 3
NORM_BOUNDS = 1.2
# max(width/2, length/2 + penalty_length) = max(3.0, 4.5 + 1.0)
MAX_POS = 5.5
MAX_V = 1.5
MAX_W = 10.0

GOAL = namedtuple('goal', ['x', 'y'])
BALL = namedtuple('ball', ['x', 'y', 'v_x', 'v_y'])
ROBOT = namedtuple('robot', ['x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])
