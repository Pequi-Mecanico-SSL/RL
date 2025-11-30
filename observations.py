import numpy as np
from rsoccer_gym.Utils.Utils import Geometry2D, decorator_observations
from rsoccer_gym.Render.ball import Ball
from collections import namedtuple


@decorator_observations
def positions_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    field_hlen, field_hwid =(
        kwargs["field_info"]["length"]/2,
        kwargs["field_info"]["width"]/2
    )
    positions = np.zeros(0)
    for robot in [main] + allys + advs:
        positions = np.hstack([positions, [
            np.clip(robot['x']/field_hlen, -1, 1), 
            np.clip(robot['y']/field_hwid, -1, 1)
        ]])
    positions = np.hstack([positions, [
        np.clip(ball['x']/field_hlen, -1, 1), 
        np.clip(ball['y']/field_hwid, -1, 1)
    ]])    
    return positions


@decorator_observations
def oritations_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    orientations = np.zeros(0)
    for robot in [main] + allys + advs:
        theta, sin, cos = (
            np.deg2rad(robot['theta'])/(2*np.pi), 
            np.sin(np.deg2rad(robot['theta'])), 
            np.cos(np.deg2rad(robot['theta']))
        )
        orientations = np.hstack([orientations, [sin, cos, theta]])
        
    return orientations


@decorator_observations
def distances_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    field_len, field_wid = kwargs["field_info"]["length"], kwargs["field_info"]["width"]
    goal_adv =  {'x': -(-0.2 - field_len/2), 'y': 0}
    goal_ally = {'x': -( 0.2 + field_len/2), 'y': 0}
    geometry = Geometry2D(
        -field_len/2, 
        field_len/2, 
        -field_wid/2, 
        field_wid/2
    )
    distances = np.array([
        geometry._get_dist_between(ball, goal_ally),
        geometry._get_dist_between(ball, goal_adv),
        geometry._get_dist_between(ball, main)
    ])

    for robot in allys + advs:
        distances = np.hstack([
            distances,
            np.array([geometry._get_dist_between(main, robot)])
        ])

    return distances


@decorator_observations
def angles_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    field_len, field_wid = kwargs["field_info"]["length"], kwargs["field_info"]["width"]
    goal_adv =  {'x': -(-0.2 - field_len/2), 'y': 0}
    goal_ally = {'x': -( 0.2 + field_len/2), 'y': 0}
    geometry = Geometry2D(
        -field_len/2, 
        field_len/2, 
        -field_wid/2, 
        field_wid/2
    )
    angles = np.array([
        *geometry._get_2dots_angle_between(goal_ally, ball),
        *geometry._get_2dots_angle_between(goal_adv, ball),
        *geometry._get_2dots_angle_between(ball, main)
    ])
    for robot in allys + advs:
        angles = np.hstack([
            angles,
            *geometry._get_2dots_angle_between(main, robot)
        ])

    return angles


@decorator_observations
def timesteps_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    timesteps = np.array([1 - kwargs["steps"] / kwargs["max_ep_length"]])
    return timesteps


@decorator_observations
def actions_observations(main_idx: str, main: dict, allys: list[dict], advs: list[dict], ball: dict, **kwargs):
    actions = {}
    last_actions = kwargs["last_actions"]

    color_main, idx = main_idx.split('_')
    robot_action = last_actions[main_idx]
    allys_actions = [last_actions[f"{color_main}_{j}"] for j in range(1 + len(allys)) if j != int(idx)]
    actions = np.array(
        [robot_action] + allys_actions
    ).flatten()
    
    return actions


OBSERVATIONS = [
    #(function,                 [required_kwargs,])
    (positions_observations,    ["field_info"]),
    (oritations_observations,   []),
    (distances_observations,    ["field_info"]),
    (angles_observations,       ["field_info"]),
    (timesteps_observations,    ["max_ep_length", "steps"]),
    (actions_observations,      ["last_actions"])
]