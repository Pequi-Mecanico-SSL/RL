from collections import namedtuple

CHECKPOINT_PATH = "/root/ray_results/PPO_selfplay_rec/PPO_Soccer_95caf_00000_0_2024-11-21_02-23-24/checkpoint_000001"
FIELD_LENGTH = 4.5
FIELD_WIDTH = 3
MAX_EP_LENGTH = 1200
N_ROBOTS_BLUE = 3
N_ROBOTS_YELLOW = 3
NORM_BOUNDS = 1.2
MAX_POS = 3.8
MAX_V = 1.5
MAX_W = 10.0

GOAL = namedtuple('goal', ['x', 'y'])
BALL = namedtuple('ball', ['x', 'y', 'v_x', 'v_y'])
ROBOT = namedtuple('robot', ['x', 'y', 'theta', 'v_x', 'v_y', 'v_theta'])
