from model_inference import SoccerPolicyWrapper
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv
from ray import tune
import numpy as np
import time
import random
import yaml
from pprint import pprint



class GenericSSLMultiAgentEnv(SSLMultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        print()
        print("ACTIONS TO SEND")
        print()
        pprint(action)

        commands = self._get_commands(action)
        #print()
        #print("SENDING COMMANDS TO GENERIC ENV")
        #print()
        #pprint(commands)
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_actions = action.copy()

        print()
        print("GETTING FRAME FROM GENERIC ENV")
        print()
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        pprint(self.frame.ball)


        # THIS IS ALL YOU NEED AS INPUT:
        """yellow=None,
        id=0
        x=-1.0906827089347646
        y=0.07882837507505365
        z=None
        theta=344.19559380437795
        v_x=1.1210878735972218
        v_y=0.28481642634656285
        v_theta=69.74197133751942"""

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
            #self.frame.robots_blue[i].v_x=0
            #self.frame.robots_blue[i].v_y=0
            #self.frame.robots_blue[i].v_theta=0

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
            #self.frame.robots_yellow[i].v_x=0
            #self.frame.robots_yellow[i].v_y=0
            #self.frame.robots_yellow[i].v_theta=0

            self.frame.robots_yellow[i].kick_v_x=0
            self.frame.robots_yellow[i].kick_v_z=0
            self.frame.robots_yellow[i].dribbler=False
            self.frame.robots_yellow[i].infrared=False
            self.frame.robots_yellow[i].wheel_speed=False
            self.frame.robots_yellow[i].v_wheel0=0
            self.frame.robots_yellow[i].v_wheel1=0
            self.frame.robots_yellow[i].v_wheel2=0
            self.frame.robots_yellow[i].v_wheel3=0



        #self.frame = Frame()
        #self.frame.ball = Ball(x=0.0, y=0.0, z=0.021499999998607966, v_x=0.0, v_y=0.0, v_z=0.0)
        #self.frame.robots_blue = {0: Robot(yellow=None,
        #                  id=0,
        #                  x=-1.0906827089347646,
        #                  y=0.07882837507505365,
        #                  z=None,
        #                  theta=344.19559380437795,
        #                  v_x=1.1210878735972218,
        #                  v_y=0.28481642634656285,
        #                  v_theta=69.74197133751942,
        #                  kick_v_x=0,
        #                  kick_v_z=0,
        #                  dribbler=False,
        #                  infrared=False,
        #                  wheel_speed=False,
        #                  v_wheel0=-17.063803230518353,
        #                  v_wheel1=-61.26337622306077,
        #                  v_wheel2=0.08054853987234788,
        #                  v_wheel3=58.06685401391549),
        #         1: Robot(yellow=None,
        #                  id=1,
        #                  x=-1.933782947589478,
        #                  y=0.8587370691447901,
        #                  z=None,
        #                  theta=80.92854027928625,
        #                  v_x=0.15060462457403084,
        #                  v_y=-0.31318121093131523,
        #                  v_theta=70.21188927314259,
        #                  kick_v_x=0,
        #                  kick_v_z=0,
        #                  dribbler=False,
        #                  infrared=False,
        #                  wheel_speed=False,
        #                  v_wheel0=7.293957066747546,
        #                  v_wheel1=57.088664976622546,
        #                  v_wheel2=1.9271527452067418,
        #                  v_wheel3=-60.26482213688339),
        #         2: Robot(yellow=None,
        #                  id=2,
        #                  x=-2.090657689071506,
        #                  y=-0.7794768206011873,
        #                  z=None,
        #                  theta=60.345173249602475,
        #                  v_x=0.0248024518725231,
        #                  v_y=0.46732122866548415,
        #                  v_theta=104.23280909182587,
        #                  kick_v_x=0,
        #                  kick_v_z=0,
        #                  dribbler=False,
        #                  infrared=False,
        #                  wheel_speed=False,
        #                  v_wheel0=36.53523032620408,
        #                  v_wheel1=-11.138640901237029,
        #                  v_wheel2=-79.10081433664703,
        #                  v_wheel3=-46.70109303743999)}

        #self.frame.robots_yellow = {0: Robot(yellow=None,
        #                    id=0,
        #                    x=1.0383395384129772,
        #                    y=0.0962684586032043,
        #                    z=None,
        #                    theta=168.2854369154206,
        #                    v_x=-1.0303419484215166,
        #                    v_y=0.36529245637176005,
        #                    v_theta=-140.53623012664175,
        #                    kick_v_x=0,
        #                    kick_v_z=0,
        #                    dribbler=False,
        #                    infrared=False,
        #                    wheel_speed=False,
        #                    v_wheel0=-54.86761935247593,
        #                    v_wheel1=-28.207721689209997,
        #                    v_wheel2=53.236461272855905,
        #                    v_wheel3=44.88072603499251),
        #           1: Robot(yellow=None,
        #                    id=1,
        #                    x=1.9768545498241556,
        #                    y=0.7620475850821781,
        #                    z=None,
        #                    theta=352.6401291843325,
        #                    v_x=-0.2989847590754326,
        #                    v_y=-0.6443739012868224,
        #                    v_theta=-227.13263645608154,
        #                    kick_v_x=0,
        #                    kick_v_z=0,
        #                    dribbler=False,
        #                    infrared=False,
        #                    wheel_speed=False,
        #                    v_wheel0=9.319334031902908,
        #                    v_wheel1=51.39687408905208,
        #                    v_wheel2=5.265680440005721,
        #                    v_wheel3=-47.17960880093887),
        #           2: Robot(yellow=None,
        #                    id=2,
        #                    x=1.8099105768338384,
        #                    y=-0.8528880713876515,
        #                    z=None,
        #                    theta=132.8416833813268,
        #                    v_x=-0.14351698677787667,
        #                    v_y=0.21824016731207582,
        #                    v_theta=358.4511452975673,
        #                    kick_v_x=0,
        #                    kick_v_z=0,
        #                    dribbler=False,
        #                    infrared=False,
        #                    wheel_speed=False,
        #                    v_wheel0=19.28139326900321,
        #                    v_wheel1=46.27544338497431,
        #                    v_wheel2=44.729267499714766,
        #                    v_wheel3=17.387722283262224)}







        #pprint(self.frame.__dict__)

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






with open("config.yaml") as f:
    file_configs = yaml.safe_load(f)

configs = {**file_configs["rllib"], **file_configs["PPO"]}
configs["env_config"] = file_configs["env"]
configs["env"] = "Soccer"
configs["env_config"]["match_time"] = 40


def create_rllib_env(config):
    #breakpoint()
    return GenericSSLMultiAgentEnv(**config)

#configs["env_config"]["init_pos"]["ball"] = [random.uniform(-2, 2), random.uniform(-1.2, 1.2)]
tune.registry._unregister_all()
tune.registry.register_env("Soccer", create_rllib_env)



env = GenericSSLMultiAgentEnv(**configs["env_config"])
obs, *_ = env.reset()

done= {'__all__': False}
e= 0.0



ssl_policy_wrapper = SoccerPolicyWrapper(observation_space=env.observation_space["blue_0"],
                                         action_space=env.action_space["blue_0"],
                                         number_of_blue_robots=env.n_robots_blue,
                                         number_of_yellow_robots=env.n_robots_yellow,
                                         config_path="config.yaml",
                                         configs=configs)


while True:
    o_blue = {f"blue_{i}": obs[f"blue_{i}"] for i in range(env.n_robots_blue)}
    o_yellow = {f"yellow_{i}": obs[f"yellow_{i}"] for i in range(env.n_robots_yellow)}

    #print("Blue Observation:")
    #print(o_blue)

    a = {}
    if env.n_robots_blue > 0:
        a.update(ssl_policy_wrapper.compute_actions(obs=o_blue, policy_id='policy_blue', full_fetch=False))

    if env.n_robots_yellow > 0:
        a.update(ssl_policy_wrapper.compute_actions(obs=o_yellow, policy_id='policy_yellow', full_fetch=False))

    if np.random.rand() < e:
         a = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(a)
    #print("Pure Default Observation:")
    #print(obs)
    env.render()

    if done['__all__'] or truncated['__all__']:

        obs, *_ = env.reset()
    #time.sleep(1)

