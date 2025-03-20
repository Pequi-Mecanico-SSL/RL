from rsoccer_gym.Entities import Frame, Robot, Ball, Field
from model_inference import SoccerPolicyWrapper
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
from ray import tune
import numpy as np
import time
import random
import yaml
from pprint import pprint



#class GenericSSLMultiAgentEnv(SSLMultiAgentEnv):
#    def step(self, action):
#        self.steps += 1
#        # Join agent action with environment actions
#        print()
#        print("ACTIONS TO SEND")
#        print()
#        pprint(action)
#
#        commands = self._get_commands(action)
#        #print()
#        #print("SENDING COMMANDS TO GENERIC ENV")
#        #print()
#        #pprint(commands)
#        self.rsim.send_commands(commands)
#        self.sent_commands = commands
#
#        self.last_actions = action.copy()
#
#        print()
#        print("GETTING FRAME FROM GENERIC ENV")
#        print()
#        self.last_frame = self.frame
#        self.frame = self.rsim.get_frame()
#
#
#
#        # THIS IS ALL YOU NEED AS INPUT:
#        """yellow=None,
#        id=0
#        x=-1.0906827089347646
#        y=0.07882837507505365
#        z=None
#        theta=344.19559380437795
#        v_x=1.1210878735972218
#        v_y=0.28481642634656285
#        v_theta=69.74197133751942"""
#
#        #self.frame.robots_blue[0].yellow=None,
#        #self.frame.robots_blue[0].id=0
#        #self.frame.robots_blue[0].x=-1.0906827089347646
#        #self.frame.robots_blue[0].y=0.07882837507505365
#        #self.frame.robots_blue[0].z=None
#        #self.frame.robots_blue[0].theta=0
#        #self.frame.robots_blue[0].v_x=1.1210878735972218
#        #self.frame.robots_blue[0].v_y=0.28481642634656285
#        #self.frame.robots_blue[0].v_theta=69.74197133751942
#
#        #UNNECESSARY VALUES
#        for i in range(0,len(self.frame.robots_blue)):
#            self.frame.robots_blue[i].kick_v_x=0
#            self.frame.robots_blue[i].kick_v_z=0
#            self.frame.robots_blue[i].dribbler=False
#            self.frame.robots_blue[i].infrared=False
#            self.frame.robots_blue[i].wheel_speed=False
#            self.frame.robots_blue[i].v_wheel0=0
#            self.frame.robots_blue[i].v_wheel1=0
#            self.frame.robots_blue[i].v_wheel2=0
#            self.frame.robots_blue[i].v_wheel3=0
#
#        for i in range(0,len(self.frame.robots_yellow)):
#            self.frame.robots_yellow[i].kick_v_x=0
#            self.frame.robots_yellow[i].kick_v_z=0
#            self.frame.robots_yellow[i].dribbler=False
#            self.frame.robots_yellow[i].infrared=False
#            self.frame.robots_yellow[i].wheel_speed=False
#            self.frame.robots_yellow[i].v_wheel0=0
#            self.frame.robots_yellow[i].v_wheel1=0
#            self.frame.robots_yellow[i].v_wheel2=0
#            self.frame.robots_yellow[i].v_wheel3=0
#
#
#        #self.frame = Frame()
#        #self.frame.ball = Ball(x=0.0, y=0.0, z=0.021499999998607966, v_x=0.0, v_y=0.0, v_z=0.0)
#        #self.frame.robots_blue = {0: Robot(yellow=None,
#        #                  id=0,
#        #                  x=-1.0906827089347646,
#        #                  y=0.07882837507505365,
#        #                  z=None,
#        #                  theta=344.19559380437795,
#        #                  v_x=1.1210878735972218,
#        #                  v_y=0.28481642634656285,
#        #                  v_theta=69.74197133751942,
#        #                  kick_v_x=0,
#        #                  kick_v_z=0,
#        #                  dribbler=False,
#        #                  infrared=False,
#        #                  wheel_speed=False,
#        #                  v_wheel0=-17.063803230518353,
#        #                  v_wheel1=-61.26337622306077,
#        #                  v_wheel2=0.08054853987234788,
#        #                  v_wheel3=58.06685401391549),
#        #         1: Robot(yellow=None,
#        #                  id=1,
#        #                  x=-1.933782947589478,
#        #                  y=0.8587370691447901,
#        #                  z=None,
#        #                  theta=80.92854027928625,
#        #                  v_x=0.15060462457403084,
#        #                  v_y=-0.31318121093131523,
#        #                  v_theta=70.21188927314259,
#        #                  kick_v_x=0,
#        #                  kick_v_z=0,
#        #                  dribbler=False,
#        #                  infrared=False,
#        #                  wheel_speed=False,
#        #                  v_wheel0=7.293957066747546,
#        #                  v_wheel1=57.088664976622546,
#        #                  v_wheel2=1.9271527452067418,
#        #                  v_wheel3=-60.26482213688339),
#        #         2: Robot(yellow=None,
#        #                  id=2,
#        #                  x=-2.090657689071506,
#        #                  y=-0.7794768206011873,
#        #                  z=None,
#        #                  theta=60.345173249602475,
#        #                  v_x=0.0248024518725231,
#        #                  v_y=0.46732122866548415,
#        #                  v_theta=104.23280909182587,
#        #                  kick_v_x=0,
#        #                  kick_v_z=0,
#        #                  dribbler=False,
#        #                  infrared=False,
#        #                  wheel_speed=False,
#        #                  v_wheel0=36.53523032620408,
#        #                  v_wheel1=-11.138640901237029,
#        #                  v_wheel2=-79.10081433664703,
#        #                  v_wheel3=-46.70109303743999)}
#
#        #self.frame.robots_yellow = {0: Robot(yellow=None,
#        #                    id=0,
#        #                    x=1.0383395384129772,
#        #                    y=0.0962684586032043,
#        #                    z=None,
#        #                    theta=168.2854369154206,
#        #                    v_x=-1.0303419484215166,
#        #                    v_y=0.36529245637176005,
#        #                    v_theta=-140.53623012664175,
#        #                    kick_v_x=0,
#        #                    kick_v_z=0,
#        #                    dribbler=False,
#        #                    infrared=False,
#        #                    wheel_speed=False,
#        #                    v_wheel0=-54.86761935247593,
#        #                    v_wheel1=-28.207721689209997,
#        #                    v_wheel2=53.236461272855905,
#        #                    v_wheel3=44.88072603499251),
#        #           1: Robot(yellow=None,
#        #                    id=1,
#        #                    x=1.9768545498241556,
#        #                    y=0.7620475850821781,
#        #                    z=None,
#        #                    theta=352.6401291843325,
#        #                    v_x=-0.2989847590754326,
#        #                    v_y=-0.6443739012868224,
#        #                    v_theta=-227.13263645608154,
#        #                    kick_v_x=0,
#        #                    kick_v_z=0,
#        #                    dribbler=False,
#        #                    infrared=False,
#        #                    wheel_speed=False,
#        #                    v_wheel0=9.319334031902908,
#        #                    v_wheel1=51.39687408905208,
#        #                    v_wheel2=5.265680440005721,
#        #                    v_wheel3=-47.17960880093887),
#        #           2: Robot(yellow=None,
#        #                    id=2,
#        #                    x=1.8099105768338384,
#        #                    y=-0.8528880713876515,
#        #                    z=None,
#        #                    theta=132.8416833813268,
#        #                    v_x=-0.14351698677787667,
#        #                    v_y=0.21824016731207582,
#        #                    v_theta=358.4511452975673,
#        #                    kick_v_x=0,
#        #                    kick_v_z=0,
#        #                    dribbler=False,
#        #                    infrared=False,
#        #                    wheel_speed=False,
#        #                    v_wheel0=19.28139326900321,
#        #                    v_wheel1=46.27544338497431,
#        #                    v_wheel2=44.729267499714766,
#        #                    v_wheel3=17.387722283262224)}
#
#
#
#
#
#
#
#        #pprint(self.frame.__dict__)
#
#        # Calculate environment observation, reward and done condition
#        self._frame_to_observations()
#        reward, done, truncated = self._calculate_reward_done()
#
#        if self.steps >= self.max_ep_length:
#            done = {'__all__': False}
#            truncated = {'__all__': True}
#
#        infos = {
#            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
#            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
#        }  
#
#        if done.get("__all__", False) or truncated.get("__all__", False):
#            for i in range(self.n_robots_blue):
#                infos[f'blue_{i}']["score"] = self.score.copy()    
#
#            for i in range(self.n_robots_yellow):
#                infos[f'yellow_{i}']["score"] = self.score.copy()
#        
#        return self.observations.copy(), reward, done, truncated, infos
#
#
#
#
#
#



#env = GenericSSLMultiAgentEnv(**configs["env_config"])


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading
import time

# Hack
from rclpy.clock import Clock



class GenericSSLMultiAgentEnv(SSLMultiAgentEnv, Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, "ssl_multi_agent_env")
        SSLMultiAgentEnv.__init__(self, *args, **kwargs)

        # Subscribers
        self.ball_sub = self.create_subscription(
            Pose2D, 
            f'/simulator/poses/ball', 
            lambda pose: self.pose_callback(pose, "ball", 0),
            10
        )

        self.blue_subs = [
            self.create_subscription(
                Pose2D, 
                f'/simulator/poses/blue/robot{i}', 
                lambda pose, id=i: self.pose_callback(pose, "blue", id),
                10
            ) for i in range(3)
        ]

        self.yellow_subs = [
            #TODO: fix number of robots
            self.create_subscription(
                Pose2D, 
                f'/simulator/poses/yellow/robot{i}',
                lambda pose, id=i: self.pose_callback(pose, "yellow", id),
                10
            ) for i in range(3)
        ]

        # Publishers
        self.blue_pubs = [
            self.create_publisher(Twist, f'/simulator/cmd/blue/robot{i}', 10) for i in range(3)
        ]

        self.yellow_pubs = [
            self.create_publisher(Twist, f'/simulator/cmd/yellow/robot{i}', 10) for i in range(3)
        ]

        # Storage for robot states
        self.robot_states_previous = {}
        self.robot_states = {}

        ### Start RL model loop in a separate thread
        self.rl_thread = threading.Thread(target=self.rl_loop, daemon=True)
        self.rl_thread.start()
        #self.rl_loop()

    def pose_callback(self, pose, color, id):
        """Callback for updating robot pose."""
        timestamp = Clock().now().to_msg()
        if f"{color}_{id}" in self.robot_states:
            self.robot_states_previous[f"{color}_{id}"] = self.robot_states[f"{color}_{id}"]
        else:
            self.robot_states_previous[f"{color}_{id}"] = {"x": pose.x, "y": pose.y, "theta": pose.theta, "timestamp": timestamp}

        self.robot_states[f"{color}_{id}"] = {"x": pose.x, "y": pose.y, "theta": pose.theta, "timestamp": timestamp}

        robot_prev = self.robot_states_previous[f"{color}_{id}"]
        robot = self.robot_states[f"{color}_{id}"]
        dt = robot["timestamp"].sec - robot_prev["timestamp"].sec + (robot["timestamp"].nanosec - robot_prev["timestamp"].nanosec)/1000000000.0
        if(dt > 0):
            dx = robot["x"] - robot_prev["x"]
            dy = robot["y"] - robot_prev["y"]
            dtheta = robot["theta"] - robot_prev["theta"]
            vx = dx/dt
            vy = dy/dt
            vtheta = dtheta/dt
        else:
            dx = 0
            dy = 0
            dtheta = 0
            vx = 0
            vy = 0
            vtheta = 0
        
        self.robot_states[f"{color}_{id}"].update({"vx": vx, "vy": vy, "vtheta": vtheta})

        print(f"{color}_{id}: x: {pose.x}, y: {pose.y}, theta: {pose.theta}, timestamp: {timestamp}, \ndx: {dx}, dy: {dy}, dtheta: {dtheta}, dt: {dt}, \nvx: {vx}, vy: {vy}, vtheta: {vtheta}")

    def rl_loop(self):
        """Main loop for RL processing."""
        obs, *_ = self.reset()

        done= {'__all__': False}
        e= 0.0

        with open("config.yaml") as f:
            file_configs = yaml.safe_load(f)
        
        configs = {**file_configs["rllib"], **file_configs["PPO"]}
        configs["env_config"] = file_configs["env"]
        configs["env"] = "Soccer"
        configs["env_config"]["match_time"] = 40
        
        



        ssl_policy_wrapper = SoccerPolicyWrapper(observation_space=self.observation_space["blue_0"],
                                                action_space=self.action_space["blue_0"],
                                                number_of_blue_robots=self.n_robots_blue,
                                                number_of_yellow_robots=self.n_robots_yellow,
                                                config_path="config.yaml",
                                                configs=configs)


        while rclpy.ok():
            o_blue = {f"blue_{i}": obs[f"blue_{i}"] for i in range(self.n_robots_blue)}
            o_yellow = {f"yellow_{i}": obs[f"yellow_{i}"] for i in range(self.n_robots_yellow)}

            print("Blue Observation:")
            print(o_blue)

            a = {}
            if self.n_robots_blue > 0:
                a.update(ssl_policy_wrapper.compute_actions(obs=o_blue, policy_id='policy_blue', full_fetch=False))

            if self.n_robots_yellow > 0:
                a.update(ssl_policy_wrapper.compute_actions(obs=o_yellow, policy_id='policy_yellow', full_fetch=False))

            if np.random.rand() < e:
                a = self.action_space.sample()

            obs, reward, done, truncated, info = self.step(a)
            #print("Pure Default Observation:")
            #print(obs)
            self.render()

            if done['__all__'] or truncated['__all__']:

                obs, *_ = self.reset()
            #time.sleep(1)




        while rclpy.ok():
            frame = self.get_frame()  # Get robot states
            actions = self.run_rl_model(frame)  # Run RL model with current state
            
            # Send the actions to the robots
            self.send_commands(actions)

            time.sleep(0.1)  # Control frequency of RL updates

    def get_frame(self):
        """Return the current state of all robots, in the shape of an rsim frame."""
        #TODO: rsim frame format

        frame = Frame()

        ball = self.robot_states["ball_0"]
        frame.ball = Ball(ball["x"], ball["y"], None, ball["vx"], ball["vy"], 0)

        frame.robots_blue = {
            i: Robot(yellow=False,
                  id = i,
                  x = self.robot_states[f"blue_{i}"]["x"],
                  y = self.robot_states[f"blue_{i}"]["y"],
                  #theta = self.robot_states[f"blue_{i}"]["theta"],
                  theta = 1,
                  v_x = self.robot_states[f"blue_{i}"]["vx"],
                  v_y = self.robot_states[f"blue_{i}"]["vy"],
                  v_theta = self.robot_states[f"blue_{i}"]["vtheta"],
            ) for i in range(3)
        }

        frame.robots_yellow = {
            i: Robot(yellow=True,
                  id = i,
                  x = self.robot_states[f"yellow_{i}"]["x"],
                  y = self.robot_states[f"yellow_{i}"]["y"],
                  #theta = self.robot_states[f"yellow_{i}"]["theta"],
                  theta = 1,
                  v_x = self.robot_states[f"yellow_{i}"]["vx"],
                  v_y = self.robot_states[f"yellow_{i}"]["vy"],
                  v_theta = self.robot_states[f"yellow_{i}"]["vtheta"],
            ) for i in range(3)
        }

        return frame

    def send_commands(self, action):
        """Send actions to robots' cmd."""
        print("ACTION = ")
        print(action)

        for i in range(0,self.n_robots_blue):
            print("SENDING ACTION TO TOPIC", self.blue_pubs[i].topic_name)
            robot_action = action[f"blue_{i}"]
            msg = Twist()
            msg.linear.x = float(robot_action[0])  # Linear velocity
            msg.linear.y = float(robot_action[1])  # Linear velocity
            msg.angular.z = float(robot_action[2])  # Angular velocity
            self.blue_pubs[i].publish(msg)
        for i in range(0,self.n_robots_yellow):
            robot_action = action[f"yellow_{i}"]
            msg = Twist()
            msg.linear.x = float(robot_action[0])  # Linear velocity
            msg.linear.y = float(robot_action[1])  # Linear velocity
            msg.angular.z = float(robot_action[2])  # Angular velocity
            self.yellow_pubs[i].publish(msg)

    def run_rl_model(self, states):
        """Mock RL model - replace with actual model inference."""
        return [(0.5, 0.1) for _ in range(6)]  # Placeholder actions

    def step(self, action):
        """
        This function is called by the RL agent to execute an action in the environment.
        It first retrieves the current state, sends actions, and then returns the next state.
        """

        self.steps += 1
        # Join agent action with environment actions
        print()
        print("ACTIONS TO SEND")
        print()
        pprint(action)

        self.send_commands(action)

        #commands = self._get_commands(action)
        #print()
        #print("SENDING COMMANDS TO GENERIC ENV")
        #print()
        #pprint(commands)
        #self.rsim.send_commands(commands)
        #self.sent_commands = commands

        self.last_actions = action.copy()

        print()
        print("GETTING FRAME FROM GENERIC ENV")
        print()
        self.last_frame = self.frame
        self.frame = self.get_frame()


        print("HEREEEEEE\n\n")


        #self.frame.robots_blue[0].yellow=None,
        #self.frame.robots_blue[0].id=0
        #self.frame.robots_blue[0].x=-1.0906827089347646
        #self.frame.robots_blue[0].y=0.07882837507505365
        #self.frame.robots_blue[0].z=None
        #self.frame.robots_blue[0].theta=0
        #self.frame.robots_blue[0].v_x=1.1210878735972218
        #self.frame.robots_blue[0].v_y=0.28481642634656285
        #self.frame.robots_blue[0].v_theta=69.74197133751942

        #UNNECESSARY VALUES
        for i in range(0,len(self.frame.robots_blue)):
            self.frame.robots_blue[i].kick_v_x=0
            self.frame.robots_blue[i].kick_v_z=0
            self.frame.robots_blue[i].dribbler=False
            self.frame.robots_blue[i].infrared=False
            self.frame.robots_blue[i].wheel_speed=False
            self.frame.robots_blue[i].v_wheel0=0
            self.frame.robots_blue[i].v_wheel1=0
            self.frame.robots_blue[i].v_wheel2=0
            self.frame.robots_blue[i].v_wheel3=0

        for i in range(0,len(self.frame.robots_yellow)):
            self.frame.robots_yellow[i].kick_v_x=0
            self.frame.robots_yellow[i].kick_v_z=0
            self.frame.robots_yellow[i].dribbler=False
            self.frame.robots_yellow[i].infrared=False
            self.frame.robots_yellow[i].wheel_speed=False
            self.frame.robots_yellow[i].v_wheel0=0
            self.frame.robots_yellow[i].v_wheel1=0
            self.frame.robots_yellow[i].v_wheel2=0
            self.frame.robots_yellow[i].v_wheel3=0

        # Calculate environment observation, reward and done condition
        self._frame_to_observations()
        reward, done, truncated = self._calculate_reward_done()

        if self.steps >= self.max_ep_length:
            done = {'__all__': False}
            truncated = {'__all__': True}

        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }  

        if done.get("__all__", False) or truncated.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()    

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()
        
        return self.observations.copy(), reward, done, truncated, infos


def main():
    with open("config.yaml") as f:
        file_configs = yaml.safe_load(f)

    configs = {**file_configs["rllib"], **file_configs["PPO"]}
    configs["env_config"] = file_configs["env"]
    configs["env"] = "Soccer"
    configs["env_config"]["match_time"] = 40

    def create_rllib_env(config):
        #breakpoint()
        return SSLMultiAgentEnv(**config)
    
    #configs["env_config"]["init_pos"]["ball"] = [random.uniform(-2, 2), random.uniform(-1.2, 1.2)]
    tune.registry._unregister_all()
    tune.registry.register_env("Soccer", create_rllib_env)

    rclpy.init()
    node = GenericSSLMultiAgentEnv(**configs["env_config"])
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
