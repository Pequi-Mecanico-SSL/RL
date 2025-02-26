import numpy as np
from rSoccer.rsoccer_gym.Entities import Frame, Field
from rSoccer.rsoccer_gym.Utils import Geometry2D
from collections import namedtuple
# kwargs vai ter kick_speed_x, fps, 


def r_speed(field: Field, frame: Frame, last_frame: Frame, **kwargs):
    robots_left = frame.robots_blue
    robots_right = frame.robots_yellow
    if kwargs["right"] == "blue":
        robots_left, robots_right = robots_right, robots_left

    kick_speed_x = kwargs["kick_speed_x"]
    time_per_frame = 1/kwargs["fps"]
    last_ball = last_frame.ball
    ball = frame.ball
    discount_factor = 0.01 # this is to penalize the dead ball
    
    goal_right = np.array([ 0.2 + field.length/2, 0])
    goal_left  = np.array([-0.2 - field.length/2, 0])

    # Calculate previous ball dist
    last_ball_pos = np.array([last_ball.x, last_ball.y])
    last_ball_dist_left = np.linalg.norm(goal_left - last_ball_pos)
    last_ball_dist_right = np.linalg.norm(goal_right - last_ball_pos)
    
    # Calculate new ball dist
    ball_pos = np.array([ball.x, ball.y])
    ball_dist_left  = np.linalg.norm(goal_left  - ball_pos)
    ball_dist_right = np.linalg.norm(goal_right - ball_pos)

    diff_left  = last_ball_dist_left  - ball_dist_left
    diff_right = last_ball_dist_right - ball_dist_right

    max_dist = kick_speed_x * time_per_frame
    ball_dist_left  = min(diff_left  - discount_factor, max_dist)
    ball_dist_right = min(diff_right - discount_factor, max_dist)

    ball_speed_rw_left  = ball_dist_left / time_per_frame
    ball_speed_rw_right = ball_dist_right / time_per_frame

    reward = {
        **{f"{kwargs['left']}_{idx}":  np.clip(ball_speed_rw_left  / kick_speed_x, -1, 1) for idx in range(len(robots_left))},
        **{f"{kwargs['right']}_{idx}": np.clip(ball_speed_rw_right / kick_speed_x, -1, 1) for idx in range(len(robots_right))},
    }
    
    return reward



def r_dist(field: Field, frame: Frame, last_frame: Frame, **kwargs):
    robots_left = frame.robots_blue
    robots_right = frame.robots_yellow
    if kwargs["right"] == "blue":
        robots_left, robots_right = robots_right, robots_left

    ball = frame.ball
    geometry = Geometry2D(-field.length/2, field.length/2, -field.goal_width/2, field.goal_width/2)
    max_dist = 2.5 # this is to put a limit on the reward

    min_dist_left = max_dist
    for idx in range(len(robots_left)):
        robot = robots_left[idx]
        dist = geometry._get_dist_between(robot, ball) # this is already normalized between 0 and 1
        min_dist_left = min(dist, min_dist_left)
    
    min_dist_right = max_dist
    for idx in range(len(robots_right)):
        robot = robots_right[idx]
        dist = geometry._get_dist_between(robot, ball) # this is already normalized between 0 and 1
        min_dist_right = min(dist, min_dist_right)

    reward = {
        **{f"{kwargs['left']}_{idx}":  -min_dist_left  for idx in range(len(robots_left))},
        **{f"{kwargs['right']}_{idx}": -min_dist_right for idx in range(len(robots_right))},
    }
    
    return reward



def r_off(field: Field, frame: Frame, last_frame: Frame, **kwargs):

    robots_left = frame.robots_blue
    robots_right = frame.robots_yellow
    if kwargs["right"] == "blue":
        robots_left, robots_right = robots_right, robots_left

    ball = frame.ball
    goal_template = namedtuple('goal', ['x', 'y'])
    goal_right = goal_template(x=0.2 + field.length/2, y=0)
    goal_left  = goal_template(x=-0.2 - field.length/2, y=0)
    geometry = Geometry2D(-field.length/2, field.length/2, -field.goal_width/2, field.goal_width/2)
    
    left_robots_angle = []
    for idx in range(len(robots_right)):
        robot = robots_right[idx]
        *_, angle = geometry._get_3dots_angle_between(robot, ball, goal_right)
        left_robots_angle.append(angle)
    
    right_robots_angle = []
    for idx in range(len(robots_left)):
        robot = robots_left[idx]
        *_, angle = geometry._get_3dots_angle_between(robot, ball, goal_left)
        right_robots_angle.append(angle)


    reward = {
        **{f"{kwargs['left']}_{idx}":  left_robots_angle[idx] - 1 for idx in range(len(robots_left))},
        **{f"{kwargs['right']}_{idx}": right_robots_angle[idx] - 1 for idx in range(len(robots_right))},
    }
    
    return reward



def r_def(field, frame, last_frame, **kwargs):
    robots_left = frame.robots_blue
    robots_right = frame.robots_yellow
    if kwargs["right"] == "blue":
        robots_left, robots_right = robots_right, robots_left

    ball = frame.ball
    goal_template = namedtuple('goal', ['x', 'y'])
    goal_right = goal_template(x= 0.2 + field.length/2, y= 0)
    goal_left  = goal_template(x=-0.2 - field.length/2, y= 0)
    geometry = Geometry2D(-field.length/2, field.length/2, -field.goal_width/2, field.goal_width/2)
    
    left_robots_angle = []
    for idx in range(len(robots_right)):
        robot = robots_right[idx]
        *_, angle = geometry._get_3dots_angle_between(goal_right, robot, ball)
        left_robots_angle.append(angle)
    
    right_robots_angle = []
    for idx in range(len(robots_left)):
        robot = robots_left[idx]
        *_, angle = geometry._get_3dots_angle_between(goal_left, robot, ball)
        right_robots_angle.append(angle)


    reward = {
        **{f"{kwargs['left']}_{idx}":  left_robots_angle[idx] - 1 for idx in range(len(robots_left))},
        **{f"{kwargs['right']}_{idx}": right_robots_angle[idx] - 1 for idx in range(len(robots_right))},
    }
    
    return reward

DENSE_REWARDS = [
    #(weight, reward_function, [kwargs])
    (0.7, r_speed, ["kick_speed_x", "fps"]),
    (0.1, r_dist,  []),
    (0.1, r_off,   []),
    (0.1, r_def,   []),
]

SPARSE_REWARDS = {
    "GOAL_REWARD": 10, # robot that scored gets this reward and the other team gets negative this reward
    "OUTSIDE_REWARD": -10 # every robot (both teams) get this reward if the ball is outside the field
}