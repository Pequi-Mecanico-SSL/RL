from src.utils.geometry import Geometry2D

def show_reward(reward_func, robot='blue_0'):
    def wrapper(*args, **kwargs):
        reward = reward_func(*args, **kwargs)
        print(f"{reward_func.__name__} - {robot} {reward[robot]}")
        return reward
    return wrapper


def decorator_observations(obs_func):
    def wrapper(n_blue, n_yellow, raw_observations, field_info, kwargs):
        results = {}
        ball = raw_observations["ball"]
        robot_colors = ['blue'] * n_blue + ['yellow'] * n_yellow
        geometry = Geometry2D(
            -field_info["length"]/2,
            field_info["length"]/2,
            -field_info["width"]/2,
            field_info["width"]/2
        )
        mapper_inverter = {
            'blue': lambda x: x,
            'yellow': lambda x: geometry._invert_coordinates(x, on_x=True)
        } # AI will see yellow robots as if it were blue. Invertion trick
        # BUG FIX: the ball used to be fetched once here and passed
        # unchanged to every color, never going through the same inverter
        # every robot position does. For yellow, that left "my own
        # position" and "ball position" in two different coordinate
        # frames within the same observation vector (mirrored vs. not),
        # so any relative computation (distance/angle to ball, which side
        # of the field it's on) was subtly wrong for yellow specifically.
        # Mirror the ball's x too, for the same reason robots are mirrored
        # (yellow perceives itself as if it were blue, attacking +x).
        # _invert_coordinates itself isn't reused here -- it always reads
        # a 'theta' key, which robots have and the ball doesn't.
        ball_by_color = {
            'blue': ball,
            'yellow': {**ball, 'x': -ball['x']},
        }

        for i, (color_main, color_adv) in enumerate(zip(robot_colors, robot_colors[::-1])):
            idx = i % n_blue
            inverter = mapper_inverter[color_main]
            ball_for_main = ball_by_color[color_main]

            n_main, n_adv = (n_blue, n_yellow) if color_main == 'blue' else (n_yellow, n_blue)
            main_robots = [inverter(robot) for name, robot in raw_observations.items() if name.startswith(color_main)]
            adv_robots = [inverter(robot) for name, robot in raw_observations.items() if name.startswith(color_adv)]

            main = main_robots[idx]
            allys = [main_robots[j] for j in range(n_main) if j != idx]
            advs = [adv_robots[j] for j in range(n_adv)]
            results[f"{color_main}_{idx}"] = obs_func(f"{color_main}_{idx}", main, allys, advs, ball_for_main, **kwargs)

        return results

    return wrapper