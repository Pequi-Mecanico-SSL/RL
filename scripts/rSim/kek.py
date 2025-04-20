import robosim
from typing import List

field_type = 0  # 0 for Division A, 1 for Division B, 2 Hardware Challenges
n_robots_blue = 2  # number of blue robots
n_robots_yellow = 2  # number of yellow robots
time_step_ms = 25  # time step in milliseconds
# ball initial position [x, y, v_x, v_y] in meters and meter/s
ball_pos = [0.0, 0.0, 0.0, 0.0]

# robots initial positions [[x, y, angle], [x, y, angle]...], where [[id_0], [id_1]...]
# Units are meters and degrees
blue_robots_pos = [[-0.2, 0.0, 0.0], [-0.4, 0.0, 0.0]]
yellow_robots_pos = [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]

# Init simulator
sim = robosim.SSLEL(
    field_type,
    n_robots_blue,
    n_robots_yellow,
    time_step_ms,
    ball_pos,
    blue_robots_pos,
    yellow_robots_pos,
)

# Get dictionary with field parameters
field_params = sim.get_field_params()

# Get simulator state
# Units are meters, meters/s, degrees
# states is [ball_x, ball_y, ball_z, ball_v_x, ball_v_y,
#           blue_0_x, blue_0_y, blue_0_angle, blue_0_v_x, blue_0_v_y, blue_0_v_angle,
#           blue_0_infrared, blue_0_desired_wheel0_speed, blue_0_desired_wheel1_speed,
#           blue_0_desired_wheel2_speed, blue_0_desired_wheel3_speed, ...]
ç = sim.get_state()
print(ç)
# Step simulator
# arguments are actions to robots starting from blue robos and increasing ids
# e.g. [
#       [blue_id_0.has_v_wheel,
#       blue_id_0.wheel_0_speed or v_x, blue_id_0.wheel_1_speed or v_y, blue_id_0.wheel_2_speed or v_angle, blue_id_0.wheel_3_speed,
#       blue_id_0.kick_v_x, blue_id_0.kick_v_y, blue_id_0.dribbler
#       ],
#       ...],
# Units are in rad/s, meters and meters/s and boolean
actions: List[List[float]] = [
    [0.0 for _ in range(6)] for _ in range(n_robots_blue + n_robots_yellow)
]
sim.step(actions)

# Reset simulator, ball and robots are positioned with same argument format as init
sim.reset(ball_pos, blue_robots_pos, yellow_robots_pos)