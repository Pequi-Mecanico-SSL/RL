
from sim2real.utils import pos, angle_between_two, dist_between, inverted_robot
from sim2real.config import FIELD_LENGTH, FIELD_WIDTH, N_ROBOTS_BLUE, N_ROBOTS_YELLOW, MAX_EP_LENGTH, GOAL, BALL, ROBOT
import numpy as np

def frame_to_observations(frame, last_actions, observations):

    # print("=====================================OBSERVATION===================================================")
    f = lambda x: " ".join([f"{i:.2f}" for i in x])
    create_robot = lambda x: ROBOT(x[0], x[1], x[2], 0, 0, 0) 
    for i in range(N_ROBOTS_BLUE):
        robot = create_robot(frame["robots_blue"][f"robot_{i}"])
        robot_action = last_actions[f'blue_{i}']
        allys = [create_robot(frame["robots_blue"][f"robot_{j}"]) for j in range(N_ROBOTS_BLUE) if j != i]
        
        allys_actions = [last_actions[f'blue_{j}'] for j in range(N_ROBOTS_BLUE) if j != i]

        advs = [create_robot(frame["robots_yellow"][f"robot_{j}"]) for j in range(N_ROBOTS_YELLOW)]

        ball = BALL(x=frame["ball"][0], y=frame["ball"][1], v_x=0, v_y=0)

        goal_adv = GOAL(x=   0.2 + FIELD_LENGTH/2, y=0)
        goal_ally = GOAL(x= -0.2 - FIELD_LENGTH/2, y=0)

        robot_obs = robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)
        observations[f'blue_{i}'] = np.delete(observations[f'blue_{i}'], range(len(robot_obs)))
        observations[f'blue_{i}'] = np.concatenate([observations[f'blue_{i}'], robot_obs], axis=0, dtype=np.float64)

        # if i == 1:
        #     print(f"blue_{i}")
        #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
        #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")

    for i in range(N_ROBOTS_YELLOW):
        robot = inverted_robot(create_robot(frame["robots_yellow"][f"robot_{i}"]))
        robot_action = last_actions[f'yellow_{i}']
        allys = [inverted_robot(create_robot(frame["robots_yellow"][f"robot_{j}"])) for j in range(N_ROBOTS_YELLOW) if j != i]
        allys_actions = [last_actions[f'yellow_{j}'] for j in range(N_ROBOTS_YELLOW) if j != i]
        advs = [inverted_robot(create_robot(frame["robots_blue"][f"robot_{j}"])) for j in range(N_ROBOTS_BLUE)]

        ball = BALL(x=-frame["ball"][0], y=frame["ball"][1], v_x=0, v_y=0)

        goal_adv = GOAL(x=  -(-0.2 - FIELD_LENGTH/2), y=0)
        goal_ally = GOAL(x= -( 0.2 + FIELD_LENGTH/2), y=0)
        
        robot_obs = robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)

        observations[f'yellow_{i}'] = np.delete(observations[f'yellow_{i}'], range(len(robot_obs)))
        observations[f'yellow_{i}'] = np.concatenate([observations[f'yellow_{i}'], robot_obs], axis=0, dtype=np.float64)

        # if i == 1:
        #     print(f"\nyellow_{i}")
        #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
        #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")
    return observations


def robot_observation(robot, allys, adversaries, robot_action, allys_actions, ball, goal_adv, goal_ally, steps=0):

    positions = []
    orientations = []
    dists = []
    angles = []
    last_actions = np.array([robot_action] + allys_actions).flatten()

    x_b, y_b, *_ = pos(ball)
    sin_BG_al, cos_BG_al, theta_BG_al = angle_between_two(goal_ally, ball)
    sin_BG_ad, cos_BG_ad, theta_BG_ad = angle_between_two(goal_adv, ball)
    dist_BG_al = dist_between(ball, goal_ally)
    dist_BG_ad = dist_between(ball, goal_adv)

    x_r, y_r, *_, sin_r, cos_r, theta_r, _  = pos(robot)
    sin_BR, cos_BR, theta_BR = angle_between_two(ball, robot)
    dist_BR = dist_between(ball, robot)

    positions.append([x_r, y_r])
    orientations.append([sin_r, cos_r, theta_r])
    dists.append([dist_BR, dist_BG_al, dist_BG_ad])
    angles.append([
        sin_BR, cos_BR, theta_BR, 
        sin_BG_al, cos_BG_al, theta_BG_al, 
        sin_BG_ad, cos_BG_ad, theta_BG_ad
    ])

    for ally in allys:
        x_al, y_al, *_, sin_al, cos_al, theta_al, _ = pos(ally)
        sin_AlR, cos_AlR, theta_AlR = angle_between_two(ally, robot)
        ally_dist = dist_between(ally, robot)
        positions.append([x_al, y_al])
        orientations.append([sin_al, cos_al, theta_al])
        dists.append([ally_dist])
        angles.append([sin_AlR, cos_AlR, theta_AlR])
    
    for i in range(N_ROBOTS_BLUE - len(allys) - 1):
        print("não é pra entrar aqui")
        x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
        sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
        ally_dist = 0
        positions.append([x_al, y_al])
        orientations.append([sin_al, cos_al, theta_al])
        dists.append([ally_dist])
        angles.append([sin_AlR, cos_AlR, theta_AlR])

    
    for adv in adversaries:
        x_adv, y_adv, *_,  sin_adv, cos_adv, theta_adv, _ = pos(adv)
        sin_AdR, cos_AdR, theta_AdR = angle_between_two(adv, robot)
        adv_dist = dist_between(adv, robot)
        positions.append([x_adv, y_adv])
        orientations.append([sin_adv, cos_adv, theta_adv])
        dists.append([adv_dist])
        angles.append([sin_AdR, cos_AdR, theta_AdR])

    for i in range(N_ROBOTS_YELLOW - len(adversaries)):
        x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
        sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
        adv_dist = 0
        positions.append([x_adv, y_adv])
        orientations.append([sin_adv, cos_adv, theta_adv])
        dists.append([adv_dist])
        angles.append([sin_AdR, cos_AdR, theta_AdR])

    positions.append([x_b, y_b])

    positions = np.concatenate(positions)
    orientations = np.concatenate(orientations)
    dists = np.concatenate(dists)
    angles = np.concatenate(angles)
    time_left = [(MAX_EP_LENGTH - steps)/MAX_EP_LENGTH]

    #print(f"len_pos: {len(positions)} \t len_ori: {len(orientations)} \t len_dist: {len(dists)} \t len_ang: {len(angles)} \t len_last_act: {len(last_actions)} \t len_time_left: {len(time_left)}")
    robot_obs = np.concatenate([positions, orientations, dists, angles, last_actions, time_left], dtype=np.float64)
    return robot_obs