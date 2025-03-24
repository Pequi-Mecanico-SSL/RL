from collections import namedtuple
from scripts.sim2real.utils import pos, angle_between_two, dist_between, inverted_robot
from scripts.sim2real.config import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    N_ROBOTS_BLUE,
    N_ROBOTS_YELLOW,
    MAX_EP_LENGTH,
    GOAL,
    BALL,
    ROBOT,
)
import numpy as np

# Define tuplas nomeadas para facilitar a manipulação dos dados
Angle = namedtuple("Angle", ["sin", "cos", "theta"])
ObsBola = namedtuple(
    "ObsBola",
    [
        "x",
        "y",
        "dist_ally_goal",
        "dist_adv_goal",
        "angle_ally_goal",
        "angle_adv_goal",
    ],
)


# Funções auxiliares
def create_robot(robot_data):
    """Cria um objeto ROBOT a partir dos dados de um robô."""
    return ROBOT(robot_data[0], robot_data[1], robot_data[2], 0, 0, 0)


def update_observation(obs_array, new_data):
    """Atualiza a observação de um robô com novos dados."""
    obs_array = np.delete(obs_array, range(len(new_data)))
    return np.concatenate([obs_array, new_data], axis=0, dtype=np.float64)


# Classe para computar observacao
class ComputeObservation:
    def __init__(self, positions: list, orientations: list, dists: list, angles: list):
        self.positions = positions
        self.orientations = orientations
        self.dists = dists
        self.angles = angles
        self.ball_obs: ObsBola = None
        self.last_actions = None

    def compute_ball_observation(self, ball, goal_ally, goal_adv):
        """Esta função calcula as observações da bola: tempo restante,
        posição da bola, ângulo e distância da bola para cada gol.

        Returns:
            ball_obs (namedtuple): Tupla nomeada contendo posição da bola (x, y),
            distância para gol aliado (dist_ally_goal), distância para gol
            adversário (dist_adv_goal), ângulo para gol aliado (angle_ally_goal),
            ângulo para gol adversário (angle_adv_goal).
        """

        x, y, *_ = pos(ball)
        bg_ally_ang = angle_between_two(
            goal_ally, ball
        )  # angulo entre bola e gol aliado
        bg_adv_ang = angle_between_two(
            goal_adv, ball
        )  # angulo entre bola e gol adversario
        ball_obs = ObsBola(
            x=x,
            y=y,
            dist_ally_goal=dist_between(ball, goal_ally),
            dist_adv_goal=dist_between(ball, goal_adv),
            angle_ally_goal=Angle(
                sin=bg_ally_ang[0], cos=bg_ally_ang[1], theta=bg_ally_ang[2]
            ),
            angle_adv_goal=Angle(
                sin=bg_adv_ang[0], cos=bg_adv_ang[1], theta=bg_adv_ang[2]
            ),
        )

        self.ball_obs = ball_obs

        return ball_obs

    def compute_self_observation(self, robot, ball, goal_ally, goal_adv):
        """Calcula observações de um robô específico em relação a bola
        e aos gols e adiciona às respectivas listas.
        """
        x_r, y_r, *_, sin_r, cos_r, theta_r, _ = pos(robot)
        sin_BR, cos_BR, theta_BR = angle_between_two(
            ball, robot
        )  # seno, cosseno e theta entre bola e robô
        robot_to_ball_distance = dist_between(ball, robot)
        ball_obs: ObsBola = self.compute_ball_observation(ball, goal_ally, goal_adv)

        self.positions.append([x_r, y_r])
        self.orientations.append([sin_r, cos_r, theta_r])
        self.dists.append(
            [robot_to_ball_distance, ball_obs.dist_ally_goal, ball_obs.dist_adv_goal]
        )
        self.angles.append(
            [
                # angulos do robô
                sin_BR,
                cos_BR,
                theta_BR,
                # angulos entre bola e gol aliado
                ball_obs.angle_ally_goal.sin,
                ball_obs.angle_ally_goal.cos,
                ball_obs.angle_ally_goal.theta,
                # angulos entre bola e gol adversario
                ball_obs.angle_adv_goal.sin,
                ball_obs.angle_adv_goal.cos,
                ball_obs.angle_adv_goal.theta,
            ]
        )

    def compute_allies_observation(self, robot, allies):
        """Calcula as observações para cada aliado em relação ao robô e
        adiciona às respectivas listas.
        """
        for ally in allies:
            x_al, y_al, *_, sin_al, cos_al, theta_al, _ = pos(ally)
            sin_AlR, cos_AlR, theta_AlR = angle_between_two(ally, robot)
            ally_dist = dist_between(ally, robot)

            self.positions.append([x_al, y_al])
            self.orientations.append([sin_al, cos_al, theta_al])
            self.dists.append([ally_dist]),
            self.angles.append([sin_AlR, cos_AlR, theta_AlR])

    def compute_adversaries_observation(self, robot, adversaries):
        """Calcula as observações para cada adversário em relação ao robô e
        adiciona às respectivas listas.
        """
        for adv in adversaries:
            x_adv, y_adv, *_, sin_adv, cos_adv, theta_adv, _ = pos(adv)
            sin_AdR, cos_AdR, theta_AdR = angle_between_two(adv, robot)
            adv_dist = dist_between(adv, robot)

            self.positions.append([x_adv, y_adv])
            self.orientations.append([sin_adv, cos_adv, theta_adv])
            self.dists.append([adv_dist])
            self.angles.append([sin_AdR, cos_AdR, theta_AdR])

    def concatenate_observations(
        self,
        steps=0,
        dtype=np.float64,
    ):
        """Concatena todas as observações em um único vetor numpy."""
        self.positions = np.concatenate(self.positions)
        self.orientations = np.concatenate(self.orientations)
        self.dists = np.concatenate(self.dists)
        self.angles = np.concatenate(self.angles)
        time_left = [(MAX_EP_LENGTH - steps) / MAX_EP_LENGTH]

        return np.concatenate(
            [
                self.positions,
                self.orientations,
                self.dists,
                self.angles,
                self.last_actions,
                time_left,
            ],
            dtype=dtype,
        )

    def compute_observations(
        self,
        robot,
        allies,
        adversaries,
        robot_action,
        allies_actions,
        ball,
        goal_adv,
        goal_ally,
        steps=0,
    ):
        """Computa todas as observações para um robô específico.

        Returns:
            obs (np.array): Vetor numpy contendo todas as observações.
        """
        self.positions.append([self.ball_obs.x, self.ball_obs.y])
        self.compute_self_observation(robot, ball, goal_ally, goal_adv)
        self.compute_allies_observation(robot, allies)
        self.compute_adversaries_observation(robot, adversaries)
        self.last_actions = np.array([robot_action] + allies_actions).flatten()

        return self.concatenate_observations(steps=steps)


class ObservationBuilder:
    def __init__(self, field_length, max_ep_length, n_robots_blue, n_robots_yellow):
        self.field_length = field_length
        self.max_ep_length = max_ep_length
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

    def build_observations(self, frame, last_actions, observations):
        """Constrói as observações para todos os robôs em um frame."""
        observations = self._process_team(
            frame=frame,
            team="blue",
            last_actions=last_actions,
            observations=observations,
            invert=False,
        )
        observations = self._process_team(
            frame=frame,
            team="yellow",
            last_actions=last_actions,
            observations=observations,
            invert=True,
        )
        return observations

    def _process_team(self, frame, team, last_actions, observations, invert=False):
        """Processa as observações de um time específico."""
        team_key = f"robots_{team}"
        n_robots = self.n_robots_blue if team == "blue" else self.n_robots_yellow
        for i in range(n_robots):
            robot_data = frame[team_key][f"robot_{i}"]
            robot = create_robot(robot_data)
            if invert:
                robot = inverted_robot(robot)
            robot_action = last_actions[f"{team}_{i}"]
            # Build allies and adversaries lists based on team
            if team == "blue":
                allies = [
                    create_robot(frame[team_key][f"robot_{j}"])
                    for j in range(n_robots)
                    if j != i
                ]
                adversaries = [
                    create_robot(frame["robots_yellow"][f"robot_{j}"])
                    for j in range(self.n_robots_yellow)
                ]
                ball = BALL(x=frame["ball"][0], y=frame["ball"][1], v_x=0, v_y=0)
                goal_adv = GOAL(x=0.2 + self.field_length / 2, y=0)
                goal_ally = GOAL(x=-0.2 - self.field_length / 2, y=0)
            else:
                allies = [
                    inverted_robot(create_robot(frame[team_key][f"robot_{j}"]))
                    for j in range(n_robots)
                    if j != i
                ]
                adversaries = [
                    inverted_robot(create_robot(frame["robots_blue"][f"robot_{j}"]))
                    for j in range(self.n_robots_blue)
                ]
                ball = BALL(x=-frame["ball"][0], y=frame["ball"][1], v_x=0, v_y=0)
                goal_adv = GOAL(x=-(-0.2 - self.field_length / 2), y=0)
                goal_ally = GOAL(x=-(0.2 + self.field_length / 2), y=0)
            # Compute the robot observation vector
            compute_observation = ComputeObservation()
            robot_obs = compute_observation.compute_observations(
                robot,
                allies,
                adversaries,
                robot_action,
                [last_actions[f"{team}_{j}"] for j in range(n_robots) if j != i],
                ball,
                goal_adv,
                goal_ally,
            )
            observations[f"{team}_{i}"] = update_observation(
                observations[f"{team}_{i}"], robot_obs
            )
        return observations
