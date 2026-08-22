import numpy as np
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers.record_video import RecordVideo

from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Entities import Robot as SimRobot
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv


from src.objects import Ball, Frame, Robot
from src.utils.geometry import Geometry2D
from src.judges.ssl_judge import Judge
from src.objects import Robot, Config, InitialPosition


class SSLMultiAgentEnv(SSLBaseEnv, MultiAgentEnv):
    default_players = 3
    def __init__(self,
        judge: Judge,
        init_pos,
        field_type=2, 
        fps=40,
        match_time=40,
        render_mode='human',
        dense_rewards = {},
        sparse_rewards = {},
        possession_radius_scale=3,
        direction_change_threshold=1,
        end_on_offense=True,
        **kwargs
    ):

        self.class_judge = judge
        self.init_pos = init_pos
        if isinstance(init_pos, dict):
            self.init_pos = InitialPosition(**init_pos)
        
        self.possession_radius_scale = possession_radius_scale
        self.direction_change_threshold = direction_change_threshold
        self.n_robots_blue = min(len(self.init_pos.blue), 3)
        self.n_robots_yellow = min(len(self.init_pos.yellow), 3)
        
        self.score = {'blue': 0, 'yellow': 0}
        self.render_mode = render_mode
        super().__init__(
            field_type=field_type, 
            n_robots_blue=self.n_robots_blue,
            n_robots_yellow=self.n_robots_yellow, 
            time_step=1/fps,
            render_mode=render_mode
        )

        self.score = {'blue': 0, 'yellow': 0}
        self.field_info = {
            "length": self.field.length,
            "width": self.field.width,
            "goal_width": self.field.goal_width
        }
        self.dense_rewards = dense_rewards
        self.sparse_rewards = sparse_rewards
        self.render_mode = render_mode
        self.geometry = Geometry2D(
            -self.field.length/2, 
            self.field.length/2, 
            -self.field.width/2, 
            self.field.width/2
        )
        self.goal_template = namedtuple('goal', ['x', 'y'])
        self.ball_template = namedtuple('ball', ['x', 'y', 'v_x', 'v_y'])
        self._agent_ids = [
            *[f'blue_{i}'for i in range(self.n_robots_blue)], 
            *[f'yellow_{i}'for i in range(self.n_robots_yellow)]
        ]
        self.max_ep_length = int(match_time*fps)
        self.fps = fps
        # Limit robot speeds
        self.max_v = 1.5
        self.max_w = 10
        self.kick_speed_x = 3.0

        self.init_pos = init_pos

        self.obs_size = 3 #obs[f'blue_0'].shape[0]
        self.act_size = 4

        self.actions_bound = {"low": -1, "high": 1}

        blue = {f'blue_{i}': Box(
            low=self.actions_bound["low"], 
            high=self.actions_bound["high"], 
            shape=(self.act_size, ), 
            dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(
            low=self.actions_bound["low"], 
            high=self.actions_bound["high"], 
            shape=(self.act_size, ), 
            dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.action_space =  Dict({**blue, **yellow})

        blue = {f'blue_{i}': Box(
                    low=max(-self.field.length/2, -self.field.width/2), 
                    high=min(self.field.length/2, self.field.width/2),
                    shape=(self.obs_size, ), 
                    dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(
                    low=max(-self.field.length/2, -self.field.width/2), 
                    high=min(self.field.length/2, self.field.width/2),
                    shape=(self.obs_size, ), 
                    dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.observation_space = Dict(**blue, **yellow)



        self.judge_last_status, self.judge_last_info = dict(), dict()
        self.judge_status, self.judge_info = dict(), dict()
        self.judge = self.class_judge(
            field=self.field, 
            init_pos=self.init_pos,
            possession_radius_scale=possession_radius_scale, 
            direction_change_threshold=direction_change_threshold
        )

        self.end_on_offense = end_on_offense

    def _get_commands(self, actions):
        commands = []
        for i in range(self.n_robots_blue):
            robot_actions = actions[f'blue_{i}'].copy()
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))
            cmd = SimRobot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x * max(robot_actions[3], 0))
            commands.append(cmd)
        
        for i in range(self.n_robots_yellow):
            robot_actions = actions[f'yellow_{i}'].copy()
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))

            cmd = SimRobot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x * max(robot_actions[3], 0))
            commands.append(cmd)

        return commands
    
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""

        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def convert_frame_to_sim_frame(self, frame: Frame) -> Frame:
        # Convert frame from our format to the simulator format if needed
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            is_blue = i < self.n_robots_blue
            idx = i if is_blue else i - self.n_robots_blue
            color = "blue" if is_blue else "yellow"
            robot = getattr(frame, f"robots_{color}")[idx] 
            getattr(frame, f"robots_{color}")[idx] = SimRobot(
                yellow= not is_blue,
                id=idx,
                x=robot.x, 
                y=robot.y, 
                theta=robot.theta,
                z=0,
                v_x=0,
                v_y=0,
                v_theta=0,
                kick_v_x=0,
                kick_v_z=0,
                dribbler=False,
                infrared=False,
                wheel_speed=False,
                v_wheel0=0, # rad/s
                v_wheel1=0, # rad/s
                v_wheel2=0, # rad/s
                v_wheel3=0 # rad/s
            )
        return frame
    
    def convert_sim_frame_to_frame(self, sim_frame) -> Frame:
        # Convert frame from simulator format to our format if needed
        robots_blue = {i: Robot(
            yellow=False,
            id=i,
            x=sim_frame.robots_blue[i].x, 
            y=sim_frame.robots_blue[i].y, 
            theta=sim_frame.robots_blue[i].theta,
            v_x=sim_frame.robots_blue[i].v_x,
            v_y=sim_frame.robots_blue[i].v_y,
            v_theta=sim_frame.robots_blue[i].v_theta
        ) for i in range(self.n_robots_blue)}
        robots_yellow = {i: Robot(
            yellow=True,
            id=i,
            x=sim_frame.robots_yellow[i].x, 
            y=sim_frame.robots_yellow[i].y, 
            theta=sim_frame.robots_yellow[i].theta,
            v_x=sim_frame.robots_yellow[i].v_x,
            v_y=sim_frame.robots_yellow[i].v_y,
            v_theta=sim_frame.robots_yellow[i].v_theta
        ) for i in range(self.n_robots_yellow)}
        ball = Ball(
            x=sim_frame.ball.x, 
            y=sim_frame.ball.y, 
            v_x=sim_frame.ball.v_x, 
            v_y=sim_frame.ball.v_y
        )
        return Frame(ball=ball, robots_blue=robots_blue, robots_yellow=robots_yellow)

    def _get_offense_counts(self):
        counts = {
            "total_offenses": 0,
            "collision_count": 0,
            "team_defense_area_count": 0,
            "opponent_defense_area_count": 0,
            "double_touch_count": 0,
        }

        for offenses in self.judge_info.get("offenses", {}).values():
            for offense in offenses:
                counts["total_offenses"] += 1
                if offense == "COLLISION":
                    counts["collision_count"] += 1
                elif offense == "TEAM_DEFENSE_AREA":
                    counts["team_defense_area_count"] += 1
                elif offense == "OPPONENT_DEFENSE_AREA":
                    counts["opponent_defense_area_count"] += 1
                elif offense == "DOUBLE_TOUCH":
                    counts["double_touch_count"] += 1

        return counts

    def _get_min_robot_distance(self):
        robots = [*self.frame.robots_blue.values(), *self.frame.robots_yellow.values()]
        if len(robots) < 2:
            return 0.0

        min_dist = float("inf")
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                dx = robots[i].x - robots[j].x
                dy = robots[i].y - robots[j].y
                dist = float(np.hypot(dx, dy))
                if dist < min_dist:
                    min_dist = dist

        return 0.0 if min_dist == float("inf") else min_dist

    def _count_robots_near_ball(self, radius=0.35):
        ball = self.frame.ball
        count = 0
        for robot in [*self.frame.robots_blue.values(), *self.frame.robots_yellow.values()]:
            dist = float(np.hypot(robot.x - ball.x, robot.y - ball.y))
            if dist <= radius:
                count += 1
        return count

    def _build_step_metrics(self):
        offense_counts = self._get_offense_counts()
        return {
            "step": 1,
            "goal_blue": int(self.judge_status == "RIGHT_GOAL"),
            "goal_yellow": int(self.judge_status == "LEFT_GOAL"),
            "is_kickoff": int(getattr(self.judge, "is_kickoff", False)),
            "min_robot_distance": self._get_min_robot_distance(),
            "robots_near_ball_count": self._count_robots_near_ball(),
            **offense_counts,
        }


    def _calculate_reward_done(self):
        self.judge_last_status = self.judge_status
        self.judge_last_info = self.judge_info
        self.judge_status, self.judge_info = self.judge.judge(self.frame)

        done = {'__all__': False}
        truncated = {'__all__': False}

        reward_agents = {
            **{f"blue_{idx}":  0 for idx in range(self.n_robots_blue)},
            **{f"yellow_{idx}": 0 for idx in range(self.n_robots_yellow)},
        }
        for weight, reward_func, list_attr in self.dense_rewards:
            kwargs = {attr: getattr(self, attr) for attr in list_attr}
            reward_result = reward_func(
                self.field_info, self.observation, self.last_observation, 
                left="blue", right="yellow", 
                **kwargs
            )

            for agent, reward in reward_result.items():
                reward_agents[agent] += weight * reward

        ball = self.frame.ball
        last_touch = self.judge_info["last_touch"]
        touch_team = "blue" if not last_touch else last_touch.split("_")[0]
        map_freekick = {
            "RIGHT_BOTTOM_LINE_blue": [self.field.length/2 - 1, (self.field.width/2 - 0.2) * (1 if ball.y > 0 else -1)],
            "RIGHT_BOTTOM_LINE_yellow": [self.field.length/2 - 0.2, (self.field.width/2 - 0.2) * (1 if ball.y > 0 else -1)],
            "LEFT_BOTTOM_LINE_blue": [-self.field.length/2 + 0.2, (self.field.width/2 - 0.2) * (1 if ball.y > 0 else -1)],
            "LEFT_BOTTOM_LINE_yellow": [-self.field.length/2 + 1, (self.field.width/2 - 0.2) * (1 if ball.y > 0 else -1)]
        }

        if self.judge_status == "RIGHT_GOAL":
            done = {'__all__': True}
            self.score['blue'] += 1

            reward_agents.update({f'blue_{i}': self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f'yellow_{i}': -self.sparse_rewards.get("GOAL_REWARD", 0)for i in range(self.n_robots_yellow)})
        
        elif self.judge_status == "LEFT_GOAL":
            done = {'__all__': True}
            self.score['yellow'] += 1

            reward_agents.update({f'blue_{i}': -self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f'yellow_{i}': self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_yellow)})


        elif self.judge_status in ["LOWER_SIDELINE", "UPPER_SIDELINE"]:
            #reward_agents.update({last_touch: self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f"blue_{i}": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f"yellow_{i}": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_yellow)})
                
            limit = self.field.length/2 - 0.2
            dx = max(abs(ball.x) - limit, 0) * (-1 if ball.x > 0 else 1)
            dy = -0.2  if ball.y > 0 else 0.2
            initial_pos_frame: Frame = self.judge._get_initial_positions_frame(
                "freekick", 
                ball_pos=[ball.x + dx, ball.y + dy], 
                team_freekick="yellow" if touch_team == "blue" else "blue"
            )
            self.rsim.reset(self.convert_frame_to_sim_frame(initial_pos_frame))
            self.frame = self.convert_sim_frame_to_frame(self.rsim.get_frame())
        
        elif self.judge_status in ["RIGHT_BOTTOM_LINE", "LEFT_BOTTOM_LINE"]:
            #reward_agents.update({last_touch: self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f"blue_{i}": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f"yellow_{i}": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_yellow)})
        
            initial_pos_frame: Frame = self.judge._get_initial_positions_frame(
                "freekick", 
                ball_pos=map_freekick[self.judge_status + "_" + touch_team],
                team_freekick="yellow" if touch_team == "blue" else "blue"
            )
            self.rsim.reset(self.convert_frame_to_sim_frame(initial_pos_frame))
            self.frame = self.convert_sim_frame_to_frame(self.rsim.get_frame())

        
        double_touch = False
        for robot_name, offenses in self.judge_info["offenses"].items():
            if len(offenses) == 0: continue
            for offense in offenses:
                if offense == "DOUBLE_TOUCH":
                    double_touch = True
                elif offense in ["OPPONENT_DEFENSE_AREA", "TEAM_DEFENSE_AREA"]:
                    done = {'__all__': True and self.end_on_offense} # Analise if it should be done or not
                reward_agents[robot_name] += self.sparse_rewards.get(offense, 0)

        # if double_touch:
        #     initial_pos_frame: Frame = self.judge._get_initial_positions_frame(
        #         "freekick", 
        #         ball_pos=[ball.x, ball.y],
        #         team_freekick="yellow" if "blue" in last_touch else "blue",
        #         use_init_pos=True
        #     )
        #     self.rsim.reset(self.convert_frame_to_sim_frame(initial_pos_frame))
        #     self.frame = self.convert_sim_frame_to_frame(self.rsim.get_frame())
        return reward_agents, done, truncated

    def reset(self, seed=42, options={}):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None
        self.last_observation = None


        self.judge_last_status, self.judge_last_info = dict(), dict()
        self.judge_status, self.judge_info = dict(), dict()
        self.judge = self.class_judge(
            field=self.field, 
            init_pos = self.init_pos,
            possession_radius_scale=self.possession_radius_scale, 
            direction_change_threshold=self.direction_change_threshold
        )
        init_frame = self.judge._get_initial_positions_frame(None)#"kickoff")
        self.rsim.reset(self.convert_frame_to_sim_frame(init_frame))

        # Get frame from simulator
        self.frame = self.convert_sim_frame_to_frame(self.rsim.get_frame())

        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}
        info = {**blue, **yellow}
        self.observation = self._frame_to_observations()
        self.score = {'blue': 0, 'yellow': 0}

        return self.observation.copy(), info
    
    def _frame_to_observations(self):
        rblue = self.frame.robots_blue
        ryellow = self.frame.robots_yellow
        observation = {
            **{f"blue_{i}": {"x": rblue[i].x, "y": rblue[i].y, "theta": rblue[i].theta} for i in range(self.n_robots_blue)},
            **{f"yellow_{i}": {"x": ryellow[i].x, "y": ryellow[i].y, "theta": ryellow[i].theta} for i in range(self.n_robots_yellow)},
            "ball": {"x": self.frame.ball.x, "y": self.frame.ball.y}
        }
        return observation
    
    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        self.last_observation = self.observation
        self.observation = self._frame_to_observations()
        
        reward, done, truncated = self._calculate_reward_done()

        if self.steps >= self.max_ep_length:
            done = {'__all__': False}
            truncated = {'__all__': True}

        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }

        if "blue_0" in infos:
            infos["blue_0"]["metrics_step"] = self._build_step_metrics()

        if done.get("__all__", False) or truncated.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()    

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()
        
        return self.observation.copy(), reward, done, truncated, infos

# class SSLMultiAgentEnv_record(RecordVideo, MultiAgentEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._agent_ids = self.env._agent_ids


if __name__ == "__main__":
    config = Config(
        init_pos=InitialPosition(
            blue={
                0: [ 1.5,  0.0,  0.0],
                1: [-2.0,  1.0,  0.0],
                2: [-2.0, -1.0,  0.0]
            },
            yellow={
                0: [ 1.5,  0.0,  180.0],
                1: [ 2.0,  1.0,  180.0],
                2: [ 2.0, -1.0,  180.0]
            },
            ball=[0, 0]
        )
    )
    from src.rewards import DENSE_REWARDS, SPARSE_REWARDS
    env = SSLMultiAgentEnv(judge=Judge, dense_rewards=DENSE_REWARDS, sparse_rewards=SPARSE_REWARDS, **config.model_dump())
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        done = done.get("__all__", False) or truncated.get("__all__", False)
        env.render()
        print(reward)